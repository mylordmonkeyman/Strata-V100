// SPDX-License-Identifier: MIT
// Standalone prefill crossover: canonical NVFP4 QPN vs actual eligible MMQ
// and dequantize -> FP16 cuBLAS. No engine dispatch changes.
#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace q = strata::kernels::qpn;
namespace k = strata::kernels;
namespace m = strata::prefill::mmq;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void blas_ck(cublasStatus_t e) { if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(e)); }
struct Device {
    void* p = nullptr;
    explicit Device(size_t bytes) { ck(cudaMalloc(&p, bytes)); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
    void put(const void* v, size_t n) { ck(cudaMemcpy(p, v, n, cudaMemcpyHostToDevice)); }
};
struct Stream { cudaStream_t s{}; Stream() { ck(cudaStreamCreate(&s)); } ~Stream() { cudaStreamDestroy(s); } };
struct Graph {
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
};
struct Blas {
    cublasHandle_t h{};
    explicit Blas(cudaStream_t s) { blas_ck(cublasCreate(&h)); blas_ck(cublasSetStream(h, s)); blas_ck(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH)); }
    ~Blas() { cublasDestroy(h); }
    void project(const Device& w, const Device& x, Device& y, int n, int dim, int tokens) {
        const float a = 1, b = 0;
        // Row-major W[n,k] is column-major W^T[k,n]; result is [n,t]
        // column-major, identical bytes to Strata's contiguous [t,n].
        blas_ck(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, tokens, dim,
                             &a, w.p, CUDA_R_16F, dim, x.p, CUDA_R_16F, dim,
                             &b, y.p, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
};
std::vector<double> project_ref(const std::vector<float>& w, const std::vector<double>& x, int n, int dim, int tokens) {
    std::vector<double> y(size_t(n)*tokens);
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < n; ++r) {
            double sum = 0;
            for (int c = 0; c < dim; ++c) sum += double(w[size_t(r)*dim+c])*x[size_t(t)*dim+c];
            y[size_t(t)*n+r] = sum;
        }
    return y;
}
double relative_error(const Device& y, const std::vector<double>& ref) {
    std::vector<float> got(ref.size()); ck(cudaMemcpy(got.data(), y.p, got.size()*4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) return std::numeric_limits<double>::infinity();
        const double d = got[i] - ref[i]; num += d*d; den += ref[i]*ref[i];
    }
    return den ? std::sqrt(num/den) : num ? std::numeric_limits<double>::infinity() : 0;
}
struct Arm {
    const char* name; bool available; std::function<void()> run;
    double error = 0; std::vector<double> us;
};
int positive_integer(const char* value) {
    size_t pos = 0; const std::string v(value); int i = std::stoi(v, &pos);
    if (pos != v.size() || i < 1) throw std::invalid_argument("expected positive integer");
    return i;
}
} // namespace
int main(int argc, char** argv) try {
    std::string path, activation_path; int tokens = 8, iterations = 100, qpn_tile = 0;
    bool validate_graph = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--validate-graph") { validate_graph = true; continue; }
        if (i+1 == argc) throw std::invalid_argument("option needs a value");
        const char* v = argv[++i];
        if (a == "--blob") path = v;
        else if (a == "--activations") activation_path = v;
        else if (a == "--tokens") tokens = positive_integer(v);
        else if (a == "--qpn-tile") {
            const std::string tile(v);
            if (tile != "0" && tile != "8" && tile != "16" && tile != "32")
                throw std::invalid_argument("--qpn-tile must be 0, 8, 16, or 32");
            qpn_tile = std::stoi(tile);
        }
        else if (a == "--iterations") iterations = positive_integer(v);
        else throw std::invalid_argument("unknown option: " + a);
    }
    if (tokens > 512) throw std::invalid_argument("prefill tokens must be 1..512");
    int dev = 0; ck(cudaGetDevice(&dev)); cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, dev));
    if (prop.major != 7 || prop.minor != 0) throw std::invalid_argument("prefill experiment requires sm_70");
    constexpr int N = 2560, FF = 640;
    const auto L = k::native_expert_layout(40, 40, N, FF);
    std::vector<uint8_t> blob(L.bytes); std::mt19937 rng(20261029);
    if (path.empty()) {
        for (size_t b = 0; b < L.tail_off; b += 36) {
            for (int j = 0; j < 4; ++j) blob[b+j] = uint8_t(rng()%128);
            for (int j = 4; j < 36; ++j) blob[b+j] = uint8_t(rng());
        }
        const float scales[] = {1e-4f, 1e-4f, 1e-4f, 0};
        std::copy(reinterpret_cast<const uint8_t*>(scales), reinterpret_cast<const uint8_t*>(scales)+16, blob.begin()+L.tail_off);
    } else {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() != std::streamoff(L.bytes)) throw std::runtime_error("wrong fixture size");
        f.seekg(0); if (!f.read(reinterpret_cast<char*>(blob.data()), blob.size())) throw std::runtime_error("fixture read failed");
    }
    float tail[4]; std::copy(blob.data()+L.tail_off, blob.data()+L.tail_off+16, reinterpret_cast<uint8_t*>(tail));
    for (int i = 0; i < 3; ++i) if (!std::isfinite(tail[i]) || tail[i] <= 0) throw std::runtime_error("invalid scale tail");
    for (size_t b = 0; b < L.tail_off; b += 36)
        for (int i = 0; i < 4; ++i) if (blob[b+i]&128) throw std::runtime_error("signed scale byte");
    std::vector<float> wg(size_t(N)*FF), wu(wg.size()), wd(wg.size());
    const auto* type = ggml_get_type_traits(GGML_TYPE_NVFP4);
    type->to_float(blob.data(), wg.data(), wg.size()); type->to_float(blob.data()+L.up_off, wu.data(), wu.size());
    type->to_float(blob.data()+L.down_off, wd.data(), wd.size());
    std::vector<float> x(size_t(tokens)*N); std::normal_distribution<float> nd;
    if (activation_path.empty()) {
        for (auto& v : x) v = nd(rng);
    } else {
        // This is a prefix of actual routed rows, exported by the pinned M0
        // capture tool. Never repeat or pad a short capture with Gaussian data.
        std::ifstream f(activation_path, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() < std::streamoff(x.size()*sizeof(float)) ||
            f.tellg() % std::streamoff(N*sizeof(float)) != 0)
            throw std::runtime_error("activation file needs complete FP32 rows for requested tokens");
        f.seekg(0);
        if (!f.read(reinterpret_cast<char*>(x.data()), x.size()*sizeof(float)))
            throw std::runtime_error("activation file read failed");
    }
    for (float v : x)
        if (!std::isfinite(v) || std::abs(v) > 65504.0f)
            throw std::runtime_error("nonfinite or FP16-unrepresentable activation");
    const std::vector<double> xr(x.begin(), x.end());
    auto gr = project_ref(wg, xr, FF, N, tokens), ur = project_ref(wu, xr, FF, N, tokens);
    std::vector<double> hr(size_t(tokens)*FF);
    for (size_t i = 0; i < hr.size(); ++i) {
        const double g = gr[i]*tail[0], e = std::exp(-std::abs(g));
        hr[i] = g*(g >= 0 ? 1/(1+e) : e/(1+e))*(ur[i]*tail[1]);
    }
    auto reference = project_ref(wd, hr, N, FF, tokens); for (auto& v : reference) v *= tail[2];
    Stream stream; const auto s = stream.s; Blas blas(s);
    Device w(blob.size()), input(x.size()*4), xh(x.size()*2), gh(hr.size()*4), uh(hr.size()*4), hidden(hr.size()*4),
           hh(hr.size()*2), y(reference.size()*4+256), wgh(wg.size()*2), wuh(wu.size()*2), wdh(wd.size()*2),
           mmgu(size_t(tokens)*2*FF*4), mmx(m::q8_bytes(tokens, N)), mmh(m::q8_bytes(tokens, FF)),
           bounds(8), ids(size_t(tokens)*4), mmwg(2*L.up_off+4096), mmwd(L.up_off+4096);
    w.put(blob.data(), blob.size()); input.put(x.data(), x.size()*4);
    const int limits[] = {0, tokens}; bounds.put(limits, 8); m::iota(ids.as<int32_t>(), tokens, s);
    const auto* scales = reinterpret_cast<const float*>(w.as<uint8_t>()+L.tail_off);
    auto dequant = [&] {
        k::iq_dequant_f16(40, w.p, wg.size(), wgh.as<uint16_t>(), s);
        k::iq_dequant_f16(40, w.as<uint8_t>()+L.up_off, wu.size(), wuh.as<uint16_t>(), s);
        k::iq_dequant_f16(40, w.as<uint8_t>()+L.down_off, wd.size(), wdh.as<uint16_t>(), s);
    };
    dequant();
    auto fp16 = [&] {
        q::stage_f32(input.as<float>(), xh.as<uint16_t>(), x.size(), s);
        blas.project(wgh, xh, gh, FF, N, tokens); blas.project(wuh, xh, uh, FF, N, tokens);
        q::swiglu(gh.as<float>(), uh.as<float>(), scales, hidden.as<float>(), hr.size(), s);
        q::stage_f32(hidden.as<float>(), hh.as<uint16_t>(), hr.size(), s);
        blas.project(wdh, hh, y, N, FF, tokens); q::scale_down(y.as<float>(), scales, reference.size(), s);
    };
    const bool mmq_ok = m::built() && m::nvfp4_mode() == m::Nvfp4Mode::W4A8 && m::fits(40, 2*FF) && m::fits(40, N);
    m::Context mmq;
    auto mmq_run = [&] {
        // Match the canonical prompt route's gather, including its zero guards;
        // include this D2D staging in completed-expert timings.
        m::gather_native(w.p,w.as<uint8_t>()+L.up_off,L.up_off,w.as<uint8_t>()+L.down_off,L.up_off,
                         mmwg.p,mmwd.p,s,nullptr,nullptr,4096);
        m::quantize(input.as<float>(), nullptr, mmx.p, 40, N, N, tokens, s);
        m::Product gu; gu.w=mmwg.p; gu.type=40; gu.w_rows=2*FF; gu.w_cols=N; gu.expert_bytes=2*L.up_off;
        gu.n=1; gu.xq=mmx.p; gu.bounds=bounds.as<int32_t>(); gu.ids=ids.as<int32_t>();
        gu.total_rows=tokens; gu.max_rows=tokens; gu.dst=mmgu.as<float>(); gu.ld_dst=2*FF;
        mmq.run(gu, s);
        m::swiglu_scaled(mmgu.as<float>(), hidden.as<float>(), tokens, FF, bounds.as<int32_t>(), 1, scales, 0, s);
        m::quantize(hidden.as<float>(), nullptr, mmh.p, 40, FF, FF, tokens, s);
        m::Product down=gu; down.w=mmwd.p; down.w_rows=N; down.w_cols=FF;
        down.expert_bytes=L.up_off; down.xq=mmh.p; down.dst=y.as<float>(); down.ld_dst=N;
        mmq.run(down, s); m::scale_down_rows(y.as<float>(), N, bounds.as<int32_t>(), 1, scales, tokens, s);
    };
    const bool qpn_ok = tokens <= 32 &&
        q::supported(70, FF, N, tokens, {4,1,1,qpn_tile}) &&
        q::supported(70, N, FF, tokens, {4,1,1,qpn_tile});
    auto qpn_run = [&] {
        q::stage_f32(input.as<float>(), xh.as<uint16_t>(), x.size(), s);
        q::projection(w.as<uint8_t>(), xh.as<uint16_t>(), gh.as<float>(), FF, N, tokens, {4,1,1,qpn_tile}, s);
        q::projection(w.as<uint8_t>()+L.up_off, xh.as<uint16_t>(), uh.as<float>(), FF, N, tokens, {4,1,1,qpn_tile}, s);
        q::swiglu(gh.as<float>(), uh.as<float>(), scales, hidden.as<float>(), hr.size(), s);
        q::stage_f32(hidden.as<float>(), hh.as<uint16_t>(), hr.size(), s);
        q::projection(w.as<uint8_t>()+L.down_off, hh.as<uint16_t>(), y.as<float>(), N, FF, tokens, {4,1,1,qpn_tile}, s);
        q::scale_down(y.as<float>(), scales, reference.size(), s);
    };
    std::vector<Arm> arms;
    arms.push_back({"qpn_direct",qpn_ok,qpn_run,0,{}});
    arms.push_back({"nvfp4_w4a8_mmq",mmq_ok,mmq_run,0,{}});
    arms.push_back({"dequant_fp16_cublas",true,[&]{dequant();fp16();},0,{}});
    arms.push_back({"cached_fp16_cublas_control",true,fp16,0,{}});
    for (auto& arm : arms) if (arm.available) { arm.run(); ck(cudaStreamSynchronize(s)); arm.error=relative_error(y,reference); }

    // Optional QPN-only CUDA graph replay and out-of-bounds tail canary checks.
    // All graph buffers are stable and allocated before stream capture.
    bool replay_equal=false, changed_input_equal=false, output_guard_ok=false;
    if (validate_graph) {
        if (!qpn_ok) throw std::invalid_argument("graph validation requires QPN schedule support");
        const size_t bytes=reference.size()*sizeof(float);
        auto* guard_ptr=static_cast<std::uint8_t*>(y.p)+bytes;
        ck(cudaMemset(guard_ptr,0xa5,256));
        qpn_run(); ck(cudaStreamSynchronize(s));
        std::vector<float> expected(reference.size());
        ck(cudaMemcpy(expected.data(),y.p,bytes,cudaMemcpyDeviceToHost));
        Graph g;
        ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
        qpn_run();
        ck(cudaStreamEndCapture(s,&g.graph));
        ck(cudaGraphInstantiate(&g.exec,g.graph,nullptr,nullptr,0));
        ck(cudaMemsetAsync(y.p,0xff,bytes,s));
        ck(cudaGraphLaunch(g.exec,s)); ck(cudaStreamSynchronize(s));
        std::vector<float> replay(reference.size());
        ck(cudaMemcpy(replay.data(),y.p,bytes,cudaMemcpyDeviceToHost));
        if (std::memcmp(expected.data(),replay.data(),bytes))
            throw std::runtime_error("QPN CUDA graph replay differs from ordinary execution");
        replay_equal=true;
        auto changed=x;
        for (auto& v:changed) v*=0.5f;
        input.put(changed.data(),changed.size()*sizeof(float));
        qpn_run(); ck(cudaStreamSynchronize(s));
        std::vector<float> changed_expected(reference.size());
        ck(cudaMemcpy(changed_expected.data(),y.p,bytes,cudaMemcpyDeviceToHost));
        ck(cudaMemsetAsync(y.p,0xff,bytes,s));
        ck(cudaGraphLaunch(g.exec,s)); ck(cudaStreamSynchronize(s));
        std::vector<float> changed_replay(reference.size());
        ck(cudaMemcpy(changed_replay.data(),y.p,bytes,cudaMemcpyDeviceToHost));
        if (std::memcmp(changed_expected.data(),changed_replay.data(),bytes))
            throw std::runtime_error("QPN CUDA graph replay ignores changed input");
        if (std::any_of(expected.begin(),expected.end(),[](float v){return v!=0.0f;}) &&
            !std::memcmp(expected.data(),changed_replay.data(),bytes))
            throw std::runtime_error("QPN graph changed-input replay returned stale output");
        changed_input_equal=true;
        input.put(x.data(),x.size()*sizeof(float));
        qpn_run(); ck(cudaStreamSynchronize(s));
        std::uint8_t guard[256];
        ck(cudaMemcpy(guard,guard_ptr,sizeof(guard),cudaMemcpyDeviceToHost));
        if (!std::all_of(std::begin(guard),std::end(guard),[](std::uint8_t v){return v==0xa5;}))
            throw std::runtime_error("QPN output tail canary overwritten");
        output_guard_ok=true;
    }
    cudaEvent_t a{}, b{}; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
    for (size_t rep = 0; rep < 6; ++rep)
        for (size_t j = 0; j < arms.size(); ++j) {
            auto& arm=arms[(rep+j)%arms.size()]; if (!arm.available || arm.error>.03) continue;
            for (int warm=0; warm<5; ++warm) arm.run();
            ck(cudaEventRecord(a,s)); for (int it=0; it<iterations; ++it) arm.run();
            ck(cudaEventRecord(b,s)); ck(cudaEventSynchronize(b)); float ms=0;
            ck(cudaEventElapsedTime(&ms,a,b)); arm.us.push_back(double(ms)*1000/iterations);
        }
    cudaEventDestroy(a); cudaEventDestroy(b);
    bool pass=true;
    std::printf("{\"tokens\":%d,\"qpn_tile\":%d,\"fixture\":\"%s\",\"activation_source\":\"%s\",\"mmq_eligible\":%s,\"fp16_weight_bytes\":%zu,\"arms\":[",tokens,qpn_tile,path.empty()?"synthetic":"real",activation_path.empty()?"gaussian":"captured",mmq_ok?"true":"false",(wg.size()+wu.size()+wd.size())*2);
    for (size_t i=0;i<arms.size();++i) {
        const auto& arm=arms[i]; if (arm.available && arm.error>.03) pass=false;
        std::printf("%s{\"backend\":\"%s\",\"available\":%s,\"expert_l2\":",i?",":"",arm.name,arm.available?"true":"false");
        if (!arm.available || !std::isfinite(arm.error)) std::printf("null"); else std::printf("%.9g",arm.error);
        std::printf(",\"numerics_pass\":%s,\"completed_expert_us\":[",arm.available&&arm.error<=.03?"true":"false");
        for (size_t j=0;j<arm.us.size();++j) std::printf("%s%.6f",j?",":"",arm.us[j]);
        std::printf("]}");
    }
    std::printf("],\\\"graph_checked\\\":%s,\\\"graph_replay_equal\\\":%s,\\\"graph_changed_input_equal\\\":%s,\\\"output_tail_guard_ok\\\":%s}\\n",
                validate_graph?"true":"false",replay_equal?"true":"false",
                changed_input_equal?"true":"false",output_guard_ok?"true":"false");
    return pass?0:1;
} catch (const std::exception& e) { std::fprintf(stderr,"prefill_bench: %s\n",e.what()); return 2; }
