// SPDX-License-Identifier: MIT
// Standalone qualification candidate. Does not change engine dispatch.
#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace q = strata::kernels::qpn;
namespace k = strata::kernels;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { ck(cudaMalloc(&p, bytes)); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
    void put(const void* src, std::size_t bytes) { ck(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice)); }
};
struct Events {
    cudaEvent_t a{}, b{};
    Events() { ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b)); }
    ~Events() { cudaEventDestroy(a); cudaEventDestroy(b); }
};
struct Graph {
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
};
struct Stream {
    cudaStream_t s{};
    Stream() { ck(cudaStreamCreate(&s)); }
    ~Stream() { cudaStreamDestroy(s); }
};
struct Error { double l2, maximum; };
Error error(const std::vector<float>& got, const std::vector<double>& ref) {
    double num = 0, den = 0, maximum = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i]))
            throw std::runtime_error("non-finite output/reference (qualification failure)");
        const double d = double(got[i]) - ref[i];
        num += d*d; den += ref[i]*ref[i]; maximum = std::max(maximum, std::abs(d));
    }
    return {den ? std::sqrt(num / den) : num ? std::numeric_limits<double>::infinity() : 0, maximum};
}
std::vector<float> download(const float* p, std::size_t count) {
    std::vector<float> result(count);
    ck(cudaMemcpy(result.data(), p, count * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}
std::vector<double> download_half(const Device& p, std::size_t count) {
    std::vector<ggml_fp16_t> bits(count);
    ck(cudaMemcpy(bits.data(), p.p, count * sizeof(ggml_fp16_t), cudaMemcpyDeviceToHost));
    std::vector<double> values(count);
    for (std::size_t i = 0; i < count; ++i) values[i] = ggml_fp16_to_fp32(bits[i]);
    return values;
}
std::vector<double> project(const std::vector<float>& w, const std::vector<double>& x, int n, int dim, int tokens) {
    std::vector<double> y(std::size_t(tokens)*n);
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < n; ++r) {
            double sum = 0;
            for (int c = 0; c < dim; ++c) sum += double(w[std::size_t(r)*dim+c])*x[std::size_t(t)*dim+c];
            y[std::size_t(t)*n+r] = sum;
        }
    return y;
}
double silu(double v) {
    const double e = std::exp(-std::abs(v));
    return v * (v >= 0 ? 1/(1+e) : e/(1+e));
}
int integer(const std::string& value) {
    std::size_t end = 0; const int i = std::stoi(value, &end);
    if (end != value.size()) throw std::invalid_argument("expected integer");
    return i;
}
} // namespace

int main(int argc, char** argv) try {
    std::string blob_path, activation_path;
    int tokens = 8, iterations = 100, sk_gu = 4, sk_down = 4, acc = 1, decoder = 0;
    bool input_q8 = false, hidden_q8 = false;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--input-q8") { input_q8 = true; continue; }
        if (arg == "--hidden-q8") { hidden_q8 = true; continue; }
        if (a + 1 >= argc) throw std::invalid_argument("option requires value: " + arg);
        const std::string value = argv[++a];
        if (arg == "--blob") blob_path = value;
        else if (arg == "--activations") activation_path = value;
        else if (arg == "--tokens") tokens = integer(value);
        else if (arg == "--iterations") iterations = integer(value);
        else if (arg == "--split-gu") sk_gu = integer(value);
        else if (arg == "--split-down") sk_down = integer(value);
        else if (arg == "--acc") acc = integer(value);
        else if (arg == "--decoder") decoder = integer(value);
        else throw std::invalid_argument("unknown option: " + arg);
    }
    constexpr int N = 2560, FF = 640, guard = 64;
    constexpr float sentinel = -1234567.0f;
    int device = 0; ck(cudaGetDevice(&device));
    cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, device));
    const int cc = prop.major * 10 + prop.minor;
    const q::Schedule gu{sk_gu, acc, decoder}, down{sk_down, acc, decoder};
    if (!q::supported(cc, FF, N, tokens, gu) || !q::supported(cc, N, FF, tokens, down) || iterations < 1)
        throw std::invalid_argument("requires sm_70, tokens 1..32 and valid projection-specific split/acc settings");
    std::fprintf(stderr, "device=%s tokens=%d fixture=%s input=%s hidden=%s\n", prop.name, tokens,
                 blob_path.empty() ? "synthetic (NOT model qualification)" : blob_path.c_str(),
                 input_q8 ? "Q8_1" : "FP32", hidden_q8 ? "Q8_1" : "FP32");
    const auto L = k::native_expert_layout(40, 40, N, FF);
    if (L.bytes != 2764816 || L.tail_off != 2764800 || L.up_off != 921600 || L.down_off != 1843200)
        throw std::runtime_error("unexpected NVFP4 expert ABI");
    std::vector<std::uint8_t> blob(L.bytes);
    std::mt19937 rng(20261028);
    if (!blob_path.empty()) {
        std::ifstream f(blob_path, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() != std::streamoff(L.bytes)) throw std::runtime_error("--blob must be exactly one canonical expert");
        f.seekg(0); if (!f.read(reinterpret_cast<char*>(blob.data()), blob.size())) throw std::runtime_error("read blob");
    } else {
        for (std::size_t b = 0; b < L.tail_off; b += 36) {
            for (int j = 0; j < 4; ++j) blob[b+j] = std::uint8_t(rng() % 128);
            for (int j = 4; j < 36; ++j) blob[b+j] = std::uint8_t(rng());
        }
        const float tail[4] = {1e-4f, 1e-4f, 1e-4f, 0};
        std::memcpy(blob.data() + L.tail_off, tail, sizeof tail);
    }
    float tail[4]; std::memcpy(tail, blob.data()+L.tail_off, sizeof tail);
    for (int i = 0; i < 3; ++i) if (!(tail[i] > 0) || !std::isfinite(tail[i])) throw std::runtime_error("invalid global scale");
    for (std::size_t b = 0; b < L.tail_off; b += 36)
        for (int j = 0; j < 4; ++j) if (blob[b+j] & 0x80) throw std::runtime_error("noncanonical signed scale byte");
    // Independent ggml dequantizer is the numerical oracle, NOT q::weight_value.
    const auto* tt = ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<float> wg(std::size_t(N)*FF), wu(wg.size()), wd(wg.size());
    tt->to_float(blob.data(), wg.data(), wg.size());
    tt->to_float(blob.data()+L.up_off, wu.data(), wu.size());
    tt->to_float(blob.data()+L.down_off, wd.data(), wd.size());
    for (std::size_t r = 0; r < FF; ++r)
        for (int c = 0; c < N; ++c)
            if (q::weight_value(blob.data()+r*L.gu_row, c) != wg[r*N+c]) throw std::runtime_error("GGUF decoder disagrees with ggml");
    const std::size_t nx = std::size_t(tokens)*N, nh = std::size_t(tokens)*FF;
    std::vector<float> x(nx);
    if (activation_path.empty()) {
        std::normal_distribution<float> normal;
        for (auto& v : x) v = normal(rng);
    } else {
        std::ifstream f(activation_path, std::ios::binary);
        if (!f.read(reinterpret_cast<char*>(x.data()), x.size()*4)) throw std::runtime_error("activation file too short");
    }
    for (float v : x) if (!std::isfinite(v) || std::abs(v) > 65504) throw std::runtime_error("FP16 activation range");
    std::vector<double> xd(x.begin(), x.end());
    auto rg = project(wg, xd, FF, N, tokens), ru = project(wu, xd, FF, N, tokens);
    std::vector<double> rh(nh);
    for (std::size_t i = 0; i < nh; ++i) rh[i] = silu(rg[i]*tail[0]) * (ru[i]*tail[1]);
    auto ry = project(wd, rh, N, FF, tokens);
    for (auto& v : ry) v *= tail[2];

    Device db(blob.size()), dx(nx*4), xq(nx/32*36), xhalf(nx*2), g(nh*4), u(nh*4), h(nh*4),
           hq(nh/32*36), hhalf(nh*2), out((nx+2*guard)*4), legacy(nx*4),
           scratch(k::native_expert_scratch_bytes(tokens, FF)), ptr(8), start(8), ng(4), ids(tokens*4);
    db.put(blob.data(), blob.size()); dx.put(x.data(), nx*4);
    const auto address = reinterpret_cast<unsigned long long>(db.p);
    const int starts[2] = {0, tokens}, groups = 1;
    std::vector<int> idx(tokens); for (int i = 0; i < tokens; ++i) idx[i] = i;
    ptr.put(&address, 8); start.put(starts, 8); ng.put(&groups, 4); ids.put(idx.data(), tokens*4);
    Stream stream; const auto s = stream.s;
    Device probe(128*16*sizeof(float));
    q::decoder_probe(probe.as<float>(), s); ck(cudaStreamSynchronize(s));
    const auto decoded = download(probe.as<float>(), 128*16);
    for (int sc = 0; sc < 128; ++sc)
        for (int code = 0; code < 16; ++code) {
            const float want = q::code_value(code) * q::scale_value(sc), got = decoded[sc*16+code];
            if (got != want || std::signbit(got) != std::signbit(want)) throw std::runtime_error("device decoder/sign-zero parity");
        }
    Device pairs(65536*sizeof(float)); q::decoder_pair_probe(pairs.as<float>(), s); ck(cudaStreamSynchronize(s));
    const auto decoded_pairs=download(pairs.as<float>(),65536);
    for (int sc=0;sc<128;++sc) for (int pair=0;pair<256;++pair) for (int lane=0;lane<2;++lane) {
        const float want=q::code_value((pair>>(lane*4))&15)*q::scale_value(sc);
        const float got_pair=decoded_pairs[(sc*256+pair)*2+lane];
        if (got_pair!=want || std::signbit(got_pair)!=std::signbit(want)) throw std::runtime_error("fast decoder pair/sign-zero parity");
    }
    auto* y = out.as<float>() + guard;
    const auto* scales = reinterpret_cast<const float*>(db.as<std::uint8_t>() + L.tail_off);
    auto quant_input = [&] { k::quantize_q8_1_rows(dx.as<float>(), tokens, N, xq.p, s); };
    auto candidate = [&] {
        if (input_q8) { quant_input(); q::stage_q8_1(xq.p, xhalf.as<std::uint16_t>(), nx, s); }
        else q::stage_f32(dx.as<float>(), xhalf.as<std::uint16_t>(), nx, s);
        q::projection(db.as<std::uint8_t>(), xhalf.as<std::uint16_t>(), g.as<float>(), FF, N, tokens, gu, s);
        q::projection(db.as<std::uint8_t>() + L.up_off, xhalf.as<std::uint16_t>(), u.as<float>(), FF, N, tokens, gu, s);
        q::swiglu(g.as<float>(), u.as<float>(), scales, h.as<float>(), nh, s);
        if (hidden_q8) {
            k::quantize_q8_1_rows(h.as<float>(), tokens, FF, hq.p, s);
            q::stage_q8_1(hq.p, hhalf.as<std::uint16_t>(), nh, s);
        } else q::stage_f32(h.as<float>(), hhalf.as<std::uint16_t>(), nh, s);
        q::projection(db.as<std::uint8_t>()+L.down_off, hhalf.as<std::uint16_t>(), y, N, FF, tokens, down, s);
        q::scale_down(y, scales, nx, s);
    };
    auto baseline = [&] {
        quant_input();
        k::native_expert_grouped(L, ptr.as<unsigned long long>(), start.as<int>(), ng.as<int>(), ids.as<int>(),
                                 ids.as<int>(), 1, tokens, xq.p, scratch.p, legacy.as<float>(), s);
    };
    std::vector<float> guarded(nx+2*guard, sentinel); out.put(guarded.data(), guarded.size()*4);
    candidate(); baseline(); ck(cudaStreamSynchronize(s));
    const auto got = download(y, nx);
    const Error eg = error(download(g.as<float>(), nh), rg), eu = error(download(u.as<float>(), nh), ru),
                ey = error(got, ry), eb = error(download(legacy.as<float>(), nx), ry);
    // Down isolation: compare against the actual FP32 hidden, before its staging,
    // to distinguish propagated gate/up error from down conversion/accumulation.
    const auto actual_h = download(h.as<float>(), nh);
    const auto rd = project(wd, std::vector<double>(actual_h.begin(), actual_h.end()), N, FF, tokens);
    auto scaled_rd = rd; for (auto& v : scaled_rd) v *= tail[2];
    const Error ed = error(got, scaled_rd);
    const auto staged_x = download_half(xhalf, nx), staged_h = download_half(hhalf, nh);
    const Error eg_mma = error(download(g.as<float>(), nh), project(wg, staged_x, FF, N, tokens)),
                eu_mma = error(download(u.as<float>(), nh), project(wu, staged_x, FF, N, tokens));
    auto down_control = project(wd, staged_h, N, FF, tokens);
    for (auto& v : down_control) v *= tail[2];
    const Error ed_mma = error(got, down_control);
    const Error input_rounding = error(std::vector<float>(staged_x.begin(), staged_x.end()), xd),
                hidden_rounding = error(std::vector<float>(staged_h.begin(), staged_h.end()),
                                        std::vector<double>(actual_h.begin(), actual_h.end()));
    // Graph capture/replay exercises caller-owned storage and launches. No malloc,
    // CPU count readback, or synchronization is permitted inside candidate().
    Graph graph;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)); candidate();
    ck(cudaStreamEndCapture(s, &graph.graph)); ck(cudaGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));
    // Poison output before replay so a missing writer cannot pass via stale bytes.
    ck(cudaMemsetAsync(y, 0xff, nx*4, s)); ck(cudaGraphLaunch(graph.exec, s)); ck(cudaStreamSynchronize(s));
    const auto replay = download(y, nx);
    if (std::memcmp(replay.data(), got.data(), nx*4)) throw std::runtime_error("graph replay differs");
    guarded = download(out.as<float>(), nx+2*guard);
    for (int i = 0; i < guard; ++i)
        if (guarded[i] != sentinel || guarded[guard+nx+i] != sentinel) throw std::runtime_error("output guard overwritten");
    Events events;
    auto timed = [&](bool qpn) {
        for (int i = 0; i < 5; ++i) { if (qpn) candidate(); else baseline(); }
        ck(cudaEventRecord(events.a, s));
        for (int i = 0; i < iterations; ++i) { if (qpn) candidate(); else baseline(); }
        ck(cudaEventRecord(events.b, s)); ck(cudaEventSynchronize(events.b));
        float ms = 0; ck(cudaEventElapsedTime(&ms, events.a, events.b)); return double(ms)*1000/iterations;
    };
    double times[2][6];
    for (int rep = 0; rep < 6; ++rep)
        for (int arm = 0; arm < 2; ++arm) { const int which = (rep+arm)%2; times[which][rep] = timed(which == 1); }
    const bool pass = eg.l2 <= .02 && eu.l2 <= .02 && ed.l2 <= .02 && ey.l2 <= .03;
    std::printf("{\"tokens\":%d,\"fixture\":\"%s\",\"input_q8\":%s,\"hidden_q8\":%s,"
                "\"split_gu\":%d,\"split_down\":%d,\"acc\":%d,\"iterations\":%d,\"decoder\":%d,"
                "\"gate_l2\":%.9g,\"up_l2\":%.9g,\"down_l2\":%.9g,\"expert_l2\":%.9g,\"baseline_l2\":%.9g,"
                "\"input_staging_l2\":%.9g,\"hidden_staging_l2\":%.9g,"
                "\"gate_mma_l2\":%.9g,\"up_mma_l2\":%.9g,\"down_mma_l2\":%.9g,"
                "\"expert_max_abs\":%.9g,\"graph_replay_equal\":true,\"output_guards_ok\":true,\"numerics_pass\":%s,",
                tokens, blob_path.empty() ? "synthetic" : "real", input_q8 ? "true" : "false", hidden_q8 ? "true" : "false",
                sk_gu, sk_down, acc, iterations, decoder, eg.l2, eu.l2, ed.l2, ey.l2, eb.l2,
                input_rounding.l2, hidden_rounding.l2, eg_mma.l2, eu_mma.l2, ed_mma.l2,
                ey.maximum, pass ? "true" : "false");
    for (int arm = 0; arm < 2; ++arm) {
        std::printf("\"%s_us\":[", arm ? "qpn_completed_expert" : "legacy_completed_expert");
        for (int rep = 0; rep < 6; ++rep) std::printf("%s%.6f", rep ? "," : "", times[arm][rep]);
        std::printf("]%s", arm ? "}\n" : ",");
    }
    return pass ? 0 : 1;
} catch (const std::exception& e) { std::fprintf(stderr, "qpn_bench: %s\n", e.what()); return 2; }
