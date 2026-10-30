// SPDX-License-Identifier: MIT
// Standalone 16-expert, real nonuniform routed workload. Never linked into engine.
#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
namespace q=strata::kernels::qpn;
namespace m=strata::prefill::mmq;
namespace {
constexpr int E=16,N=2560,F=640,GU=1280;
constexpr size_t MAT=size_t(N)*F/64*36,STRIDE=3*MAT+16;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Dev {
    void* p=nullptr;
    explicit Dev(size_t bytes){ck(cudaMalloc(&p,bytes));}
    ~Dev(){if(p)cudaFree(p);}
    Dev(const Dev&)=delete;Dev& operator=(const Dev&)=delete;
    template<class T>T* as() const{return static_cast<T*>(p);}
    void put(const void* src,size_t bytes){ck(cudaMemcpy(p,src,bytes,cudaMemcpyHostToDevice));}
};
// Persistent page-locked host layouts: the DMA source never changes during
// captured graph replay. No worker, driver, or host memlock settings changed.
struct Pinned {
    void* p=nullptr;
    explicit Pinned(std::size_t bytes){ck(cudaHostAlloc(&p,bytes,cudaHostAllocDefault));}
    ~Pinned(){if(p)cudaFreeHost(p);}
    Pinned(const Pinned&)=delete;Pinned& operator=(const Pinned&)=delete;
    template<class T>T* as() const{return static_cast<T*>(p);}
};
struct Stream {cudaStream_t s{};Stream(){ck(cudaStreamCreate(&s));}~Stream(){cudaStreamDestroy(s);}};
struct Graph {cudaGraph_t g{};cudaGraphExec_t exec{};~Graph(){if(exec)cudaGraphExecDestroy(exec);if(g)cudaGraphDestroy(g);}};
struct Events {cudaEvent_t a{},b{};Events(){ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));}~Events(){cudaEventDestroy(a);cudaEventDestroy(b);}};
std::vector<uint8_t> read_file(const std::string& name,size_t bytes){
    std::ifstream f(name,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(bytes))throw std::runtime_error("private input size mismatch");
    f.seekg(0);std::vector<uint8_t> out(bytes);
    if(!f.read(reinterpret_cast<char*>(out.data()),std::streamsize(bytes)))throw std::runtime_error("read failed");
    return out;
}
std::vector<int> read_counts(const std::string& name){
    std::ifstream f(name);std::string line,extra;
    if(!f||!std::getline(f,line)||std::getline(f,extra))throw std::runtime_error("counts file malformed");
    std::vector<int> c;size_t start=0;
    while(start<line.size()){
        const auto end=line.find(',',start);
        const auto value=line.substr(start,end==std::string::npos?end:end-start);
        if(value.empty()||value.find_first_not_of("0123456789")!=std::string::npos)
            throw std::runtime_error("noninteger count");
        const int n=std::stoi(value);
        if(n<1||n>1024)throw std::runtime_error("group outside valid route bounds");
        c.push_back(n);
        if(end==std::string::npos) break;
        start=end+1;
    }
    if(c.size()!=E||!std::all_of(c.begin(),c.begin()+6,[](int n){return n>32;})||
       !std::all_of(c.begin()+6,c.end(),[](int n){return n<=32;}))
        throw std::runtime_error("invalid nonuniform fallback prefix");
    std::array<int,6> buckets{};
    for(int n:c)buckets[n<=8?0:n<=11?1:n<=16?2:n<=32?3:n<=64?4:5]++;
    if(buckets!=std::array<int,6>{2,2,2,4,3,3})
        throw std::runtime_error("mixed sample stratification changed");
    return c;
}
double l2(const std::vector<float>& a,const std::vector<float>& b){
    if(a.size()!=b.size())throw std::runtime_error("output row shape changed");
    double num=0,den=0;
    for(size_t i=0;i<a.size();i++){
        if(!std::isfinite(a[i])||!std::isfinite(b[i]))return std::numeric_limits<double>::infinity();
        const double d=double(a[i])-b[i];num+=d*d;den+=double(b[i])*b[i];
    }
    return den>0?std::sqrt(num/den):std::numeric_limits<double>::infinity();
}

struct OracleRow {int row;std::vector<float> values;};
std::vector<OracleRow> sampled_oracle(const std::vector<uint8_t>& weights,
            const std::vector<float>& x,const std::array<int,E+1>& bounds){
    const auto* traits=ggml_get_type_traits(GGML_TYPE_NVFP4);
    if(!traits||!traits->to_float) throw std::runtime_error("independent GGUF NVFP4 decode unavailable");
    std::vector<float> gate(size_t(F)*N),up(gate.size()),down(size_t(N)*F);
    std::vector<OracleRow> result;
    for(int e=0;e<E;e++){
        const uint8_t* w=weights.data()+size_t(e)*STRIDE;
        traits->to_float(w,gate.data(),int64_t(gate.size()));
        traits->to_float(w+MAT,up.data(),int64_t(up.size()));
        traits->to_float(w+2*MAT,down.data(),int64_t(down.size()));
        std::array<float,4> scale{};
        std::memcpy(scale.data(),w+3*MAT,16);
        for(int j=0;j<2;j++){
            const int row=j==0?bounds[e]:bounds[e+1]-1;
            if(j==1&&row==bounds[e])continue;
            std::vector<double> hidden(F);
            const auto* xv=x.data()+size_t(row)*N;
            for(int r=0;r<F;r++){
                double g=0,u=0;
                const auto* gw=gate.data()+size_t(r)*N;
                const auto* uw=up.data()+size_t(r)*N;
                for(int k=0;k<N;k++){g+=double(gw[k])*xv[k];u+=double(uw[k])*xv[k];}
                g*=scale[0];u*=scale[1];
                const double z=std::exp(-std::abs(g));
                const double sigmoid=g>=0?1.0/(1.0+z):z/(1.0+z);
                hidden[r]=g*sigmoid*u;
            }
            OracleRow item;item.row=row;item.values.resize(N);
            for(int r=0;r<N;r++){
                double v=0;const auto* dw=down.data()+size_t(r)*F;
                for(int k=0;k<F;k++)v+=double(dw[k])*hidden[k];
                item.values[r]=float(v*scale[2]);
            }
            result.push_back(std::move(item));
        }
    }
    return result;
}
double sampled_l2(const std::vector<float>& output,const std::vector<OracleRow>& oracle){
    double num=0,den=0;
    for(const auto& item:oracle)for(int r=0;r<N;r++){
        const float got=output[size_t(item.row)*N+r],want=item.values[r];
        if(!std::isfinite(got)||!std::isfinite(want))
            return std::numeric_limits<double>::infinity();
        const double delta=double(got)-want;
        num+=delta*delta;den+=double(want)*want;
    }
    return den>0?std::sqrt(num/den):std::numeric_limits<double>::infinity();
}
double med(std::vector<double> a){
    if(a.size()<5)throw std::runtime_error("at least 5 repetitions required");
    std::sort(a.begin(),a.end());
    const size_t mid=a.size()/2;
    return a.size()%2?a[mid]:(a[mid-1]+a[mid])/2;
}
void arr(const std::vector<double>& a){
    std::printf("[");for(size_t i=0;i<a.size();i++)std::printf("%s%.6f",i?",":"",a[i]);std::printf("]");
}
} // namespace
int main(int argc,char** argv)try{
    std::string wp,xp,cp,packed_path;int iterations=10,rounds=6,qpn_min=1,qpn_max=128;bool check_graph=false, fused_gu=false, chunkmajor=false, pinned_stream=false, async_pipeline=false, explicit_range=false, native_decode_control=false;int row_cap=0;
    for(int i=1;i<argc;i++){
        const std::string flag=argv[i];
        if(flag=="--native-decode-control"){native_decode_control=true;continue;}
        if(flag=="--graph-check"){check_graph=true;continue;}
        if(flag=="--qpn-fused-gu"){fused_gu=true;continue;}
        if(flag=="--qpn-chunkmajor"){chunkmajor=true;continue;}
        if(flag=="--qpn-pinned-stream"){pinned_stream=true;continue;}
        if(flag=="--qpn-async-pipeline"){async_pipeline=true;continue;}
        if(++i>=argc)throw std::invalid_argument("missing value");
        const std::string val=argv[i];
        if(flag=="--weights")wp=val;
        else if(flag=="--activations")xp=val;
        else if(flag=="--counts")cp=val;
        else if(flag=="--qpn-packed-file")packed_path=val;
        else if(flag=="--iterations")iterations=std::stoi(val);
        else if(flag=="--rounds")rounds=std::stoi(val);
        else if(flag=="--row-cap")row_cap=std::stoi(val);
        else if(flag=="--qpn-range") {
            const auto colon=val.find(':');
            if(colon==std::string::npos)throw std::invalid_argument("range must be min:max");
            qpn_min=std::stoi(val.substr(0,colon));qpn_max=std::stoi(val.substr(colon+1));explicit_range=true;
        }
        else if(flag=="--qpn-min")qpn_min=std::stoi(val);
        else if(flag=="--qpn-max")qpn_max=std::stoi(val);
        else throw std::invalid_argument("unknown argument");
    }
    if(wp.empty()||xp.empty()||cp.empty()||iterations<1||iterations>1000||rounds<1||rounds>50||
       row_cap<0 || row_cap>512 || (row_cap && !explicit_range) ||
       (native_decode_control && (row_cap<1 || row_cap>16)) ||
       (explicit_range ? (qpn_min<1 || qpn_max<qpn_min || qpn_max>512) :
         ((qpn_min!=1&&qpn_min!=17&&qpn_min!=33&&qpn_min!=129) ||
          (qpn_max!=128&&qpn_max!=512) || (qpn_min==129&&qpn_max!=512) ||
          ((qpn_min==1||qpn_min==17)&&qpn_max!=128))) ||
       (!explicit_range && fused_gu && (qpn_min!=129 || qpn_max!=512)) ||
       (chunkmajor && !fused_gu) ||
       (pinned_stream && !chunkmajor) ||
       (!packed_path.empty() && !chunkmajor) ||
       (async_pipeline && !pinned_stream))
        throw std::invalid_argument("invalid arguments");
    auto counts=read_counts(cp);
    const int original_rows=std::accumulate(counts.begin(),counts.end(),0);
    auto host_w=read_file(wp,E*STRIDE);
    const auto cache_load_begin=std::chrono::steady_clock::now();
    std::vector<uint8_t> disk_prepacked_w;
    if(!packed_path.empty())disk_prepacked_w=read_file(packed_path,E*STRIDE);
    const double cache_load_us=packed_path.empty()?0:
        std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-cache_load_begin).count();
    auto host_x=read_file(xp,size_t(original_rows)*N*4);
    // Decode/verification shapes use genuine prefixes from each expert's own
    // captured activations. Never duplicate rows or upsample tiny groups.
    if(row_cap) {
        std::vector<uint8_t> capped;size_t start=0;
        for(int& n:counts) {
            const int take=std::min(n,row_cap);
            capped.insert(capped.end(),host_x.begin()+start*N*4,host_x.begin()+(start+take)*N*4);
            start+=n;n=take;
        }
        host_x.swap(capped);
    }
    const int rows=std::accumulate(counts.begin(),counts.end(),0);
    // Offline partition of the SAME private real expert rows and weights
    // for both arms: MMQ fallback prefix then selected QPN candidates.
    // This is NOT the full-model production dispatcher.
    const auto qpn_large_candidate=[&](int n) {
        if(explicit_range)return n>=qpn_min&&n<=qpn_max;
        if(qpn_min==17)return n>=17&&n<=32;
        if(qpn_min==33)return n>=33&&n<=qpn_max;
        if(qpn_min==129)return n>=129&&n<=qpn_max;
        return n<=32;
    };
    const int candidates=(!explicit_range&&qpn_min==1)?10:
        int(std::count_if(counts.begin(),counts.end(),qpn_large_candidate));
    if(!explicit_range && qpn_min==129 && candidates!=2)
        throw std::runtime_error("expected exactly 187 and 508 real routed rows");
    if(!explicit_range && qpn_min==33 && qpn_max==512 && candidates!=6)
        throw std::runtime_error("expected exactly six real T33..512 experts");
    if(explicit_range || qpn_min==17 || qpn_min==33 || qpn_min==129){
        std::vector<int> order;
        for(int i=0;i<E;i++)if(!qpn_large_candidate(counts[i]))order.push_back(i);
        for(int i=0;i<E;i++)if(qpn_large_candidate(counts[i]))order.push_back(i);
        if(order.size()!=E)throw std::runtime_error("group permutation misses experts");
        std::array<int,E+1> original{};
        for(int i=0;i<E;i++)original[i+1]=original[i]+counts[i];
        std::vector<uint8_t> new_w(host_w.size()),new_x(host_x.size());
        std::vector<uint8_t> new_prepacked;
        if(!disk_prepacked_w.empty())new_prepacked.resize(disk_prepacked_w.size());
        std::vector<int> reordered;reordered.reserve(E);
        size_t next_row=0;
        for(int src:order){
            const int n=counts[src];
            std::memcpy(new_w.data()+reordered.size()*STRIDE,
                        host_w.data()+size_t(src)*STRIDE,STRIDE);
            if(!disk_prepacked_w.empty())
                std::memcpy(new_prepacked.data()+reordered.size()*STRIDE,
                            disk_prepacked_w.data()+size_t(src)*STRIDE,STRIDE);
            std::memcpy(new_x.data()+next_row*N*4,
                        host_x.data()+size_t(original[src])*N*4,size_t(n)*N*4);
            next_row+=size_t(n);reordered.push_back(n);
        }
        if(next_row!=size_t(rows))throw std::runtime_error("reorder lost or duplicated rows");
        host_w.swap(new_w);host_x.swap(new_x);counts.swap(reordered);
        if(!disk_prepacked_w.empty())disk_prepacked_w.swap(new_prepacked);
    }
    if(candidates==0)throw std::invalid_argument("empty selected row bucket");
    const int fallback_groups=E-candidates;
    std::array<int,E+1> bounds{};
    for(int i=0;i<E;i++)bounds[i+1]=bounds[i]+counts[i];
    const int fallback_rows=bounds[fallback_groups],small=rows-fallback_rows;
    if((explicit_range || qpn_min!=1) &&
       (std::any_of(counts.begin(),counts.begin()+fallback_groups,qpn_large_candidate) ||
        !std::all_of(counts.begin()+fallback_groups,counts.end(),qpn_large_candidate)))
        throw std::runtime_error("selected large-row mixed partition broken");
    std::vector<float> x(size_t(rows)*N);
    std::memcpy(x.data(),host_x.data(),host_x.size());
    for(float v:x)if(!std::isfinite(v)||std::abs(v)>65504.f)
        throw std::runtime_error("invalid captured activation");
    std::array<float,E*4> scales{};
    for(int i=0;i<E;i++){
        const uint8_t* src=host_w.data()+size_t(i)*STRIDE;
        std::memcpy(scales.data()+4*i,src+3*MAT,16);
        for(int z=0;z<3;z++)if(!std::isfinite(scales[4*i+z])||scales[4*i+z]<=0)
            throw std::runtime_error("invalid NVFP4 tail");
        if(scales[4*i+3]!=0)throw std::runtime_error("noncanonical NVFP4 tail");
    }
    int gpu;ck(cudaGetDevice(&gpu));cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop,gpu));
    if(prop.major!=7||prop.minor!=0)throw std::runtime_error("V100 sm_70 required");
    if(!m::built()||m::nvfp4_mode()!=m::Nvfp4Mode::W4A8||
       !m::fits(GGML_TYPE_NVFP4,GU)||!m::fits(GGML_TYPE_NVFP4,N))
        throw std::runtime_error("native NVFP4 W4A8 MMQ is unavailable");
    for(int i=fallback_groups;i<E;i++)if(!q::supported(70,F,N,counts[i],{4,1,1,counts[i]>16?16:0})||
                             !q::supported(70,N,F,counts[i],{4,1,1,counts[i]>16?16:0}))
        throw std::runtime_error("QPN not qualified for selected small group");
    Stream stream;const auto s=stream.s;
    // Experimental private chunk-major repack is intentionally prepared
    // outside the timed expert forward; the repack cost is separately
    // measured and cannot be ignored in model integration decisions.
    double weight_pack_cpu_us=0,weight_pack_upload_us=0;
    std::unique_ptr<Dev> packed_w;
    std::unique_ptr<Pinned> pinned_canonical,pinned_qpn;
    if(chunkmajor) {
        const auto begin=std::chrono::steady_clock::now();
        // The offline cache stores original expert order. It was reordered
        // alongside the canonical source above, and its manifest/source SHA
        // was independently verified by the runner before launching here.
        std::vector<uint8_t> reordered_w =
            disk_prepacked_w.empty()?host_w:std::move(disk_prepacked_w);
        auto repack_projection=[&](int expert,std::size_t mat_offset,int nrows,int kcols){
            if(kcols%64)throw std::runtime_error("invalid NVFP4 repack K");
            const int blocks=kcols/64;
            const uint8_t* src=host_w.data()+std::size_t(expert)*STRIDE+mat_offset;
            uint8_t* dst=reordered_w.data()+std::size_t(expert)*STRIDE+mat_offset;
            for(int st=0;st<blocks;st++)for(int row=0;row<nrows;row++)
                std::memcpy(dst+(std::size_t(st)*nrows+row)*36,
                            src+(std::size_t(row)*blocks+st)*36,36);
        };
        if(packed_path.empty())
            for(int expert=fallback_groups;expert<E;expert++) {
                repack_projection(expert,0,GU,N);
                repack_projection(expert,2*MAT,N,F);
            }
        const auto packed=std::chrono::steady_clock::now();
        packed_w=std::make_unique<Dev>(host_w.size());
        if(pinned_stream) {
            // Exact selected 2*STRIDE compressed expert blobs only, packed
            // ONCE in persistent host memory. The canonical MMQ prefix is
            // transferred separately. Never stage/transmit the unused packed
            // 14 experts. No conversion in the timed forward pass.
            const std::size_t selected=std::size_t(E-fallback_groups)*STRIDE;
            pinned_qpn=std::make_unique<Pinned>(selected);
            std::memcpy(pinned_qpn->p,
                reordered_w.data()+std::size_t(fallback_groups)*STRIDE,selected);
            const auto prepared=std::chrono::steady_clock::now();
            weight_pack_cpu_us=packed_path.empty()?
                std::chrono::duration<double,std::micro>(prepared-begin).count():0.0;
            weight_pack_upload_us=0;
        } else {
            packed_w->put(reordered_w.data(),reordered_w.size());
            const auto uploaded=std::chrono::steady_clock::now();
            weight_pack_cpu_us=packed_path.empty()?std::chrono::duration<double,std::micro>(packed-begin).count():0;
            weight_pack_upload_us=std::chrono::duration<double,std::micro>(uploaded-packed).count();
        }
    }
    if(pinned_stream) {
        pinned_canonical=std::make_unique<Pinned>(host_w.size());
        std::memcpy(pinned_canonical->p,host_w.data(),host_w.size());
    }
    Dev w(host_w.size()),input(host_x.size()),tail(scales.size()*4);
    Dev b(bounds.size()*4),ids(size_t(rows+512)*4),xh(size_t(rows)*N*2);
    Dev gate(size_t(rows)*F*4),up(size_t(rows)*F*4),hidden(size_t(rows)*F*4),hh(size_t(rows)*F*2);
    Dev mmwg(E*2*MAT+4096),mmwd(E*MAT+4096),xq(m::q8_bytes(rows,N)),hq(m::q8_bytes(rows,F));
    Dev gu(size_t(rows)*GU*4),y(size_t(rows)*N*4+256);
    w.put(host_w.data(),host_w.size());input.put(host_x.data(),host_x.size());
    b.put(bounds.data(),bounds.size()*4);tail.put(scales.data(),scales.size()*4);
    ck(cudaMemsetAsync(ids.p,0,size_t(rows+512)*4,s));
    m::iota(ids.as<int32_t>(),rows,s);
    ck(cudaStreamSynchronize(s));
    m::Context ctx;
    auto gather=[&](int n,const uint8_t* source){
        for(int i=0;i<n;i++){
            const auto* src=source+size_t(i)*STRIDE;
            m::gather_native(src,src+MAT,MAT,src+2*MAT,MAT,
                 mmwg.as<uint8_t>()+size_t(i)*2*MAT,
                 mmwd.as<uint8_t>()+size_t(i)*MAT,s,
                 nullptr,nullptr,i==n-1?4096:0);
        }
    };
    auto mmq=[&](int n,int active_rows){
        m::quantize(input.as<float>(),nullptr,xq.p,GGML_TYPE_NVFP4,N,N,active_rows,s);
        m::Product g;
        g.w=mmwg.p;g.type=GGML_TYPE_NVFP4;g.w_rows=GU;g.w_cols=N;
        g.expert_bytes=2*MAT;g.n=n;g.xq=xq.p;g.bounds=b.as<int32_t>();g.ids=ids.as<int32_t>();
        g.total_rows=active_rows;g.max_rows=*std::max_element(counts.begin(),counts.begin()+n);
        g.dst=gu.as<float>();g.ld_dst=GU;
        ctx.run(g,s);
        m::swiglu_scaled(gu.as<float>(),hidden.as<float>(),active_rows,F,
                         b.as<int32_t>(),n,tail.as<float>(),0,s);
        m::quantize(hidden.as<float>(),nullptr,hq.p,GGML_TYPE_NVFP4,F,F,active_rows,s);
        auto down=g;down.w=mmwd.p;down.w_rows=N;down.w_cols=F;
        down.expert_bytes=MAT;down.xq=hq.p;down.dst=y.as<float>();down.ld_dst=N;
        ctx.run(down,s);
        m::scale_down_rows(y.as<float>(),N,b.as<int32_t>(),n,tail.as<float>(),active_rows,s);
    };
    const size_t out_bytes=size_t(rows)*N*4;
    const std::size_t canonical_h2d=std::size_t(E)*STRIDE;
    const std::size_t fallback_h2d=std::size_t(fallback_groups)*STRIDE;
    const std::size_t packed_h2d=std::size_t(E-fallback_groups)*STRIDE;
    // Copy from persistent pinned memory directly into the pre-existing CUDA
    // slots on the SAME stream as gather/compute, so each iteration's event
    // includes the actual PCIe transfer and the graph captures real HtoD nodes.
    // All-MMQ transfers E canonical compressed blobs. Hybrid transfers exactly
    // fallback_groups canonical blobs plus selected QPN packed blobs: identical
    // compressed bytes, without a GPU layout conversion.
    auto stream_weights=[&](bool hybrid){
        if(!pinned_stream)return;
        if(!hybrid || fallback_h2d)ck(cudaMemcpyAsync(w.p,pinned_canonical->p,
            hybrid?fallback_h2d:canonical_h2d,cudaMemcpyHostToDevice,s));
        if(hybrid)
            ck(cudaMemcpyAsync(packed_w->as<uint8_t>()+fallback_h2d,
                               pinned_qpn->p,packed_h2d,
                               cudaMemcpyHostToDevice,s));
    };
    // Actual production decode arithmetic control: linear Q8_1 activations
    // and native grouped DP4A, distinct from the prefill MMQ comparator.
    std::unique_ptr<Dev> decode_xq,decode_scratch,decode_ng;
    std::vector<std::pair<const uint8_t*,std::unique_ptr<Dev>>> decode_ptrs;
    const auto decode_layout=strata::kernels::native_expert_layout(GGML_TYPE_NVFP4,GGML_TYPE_NVFP4,N,F);
    if(native_decode_control) {
        decode_xq=std::make_unique<Dev>(size_t(rows)*N/32*36);
        decode_scratch=std::make_unique<Dev>(strata::kernels::native_expert_scratch_bytes(rows,F));
        decode_ng=std::make_unique<Dev>(4);const int groups=E;decode_ng->put(&groups,4);
    }
    auto decode_table=[&](const uint8_t* native)->const unsigned long long* {
        for(const auto& item:decode_ptrs)if(item.first==native)return item.second->as<unsigned long long>();
        cudaStreamCaptureStatus status{};ck(cudaStreamIsCapturing(s,&status));
        if(status!=cudaStreamCaptureStatusNone)throw std::runtime_error("decode pointer table not prepared before capture");
        std::array<unsigned long long,E> addresses{};
        for(int i=0;i<E;i++)addresses[i]=reinterpret_cast<unsigned long long>(native+size_t(i)*STRIDE);
        auto table=std::make_unique<Dev>(E*8);table->put(addresses.data(),E*8);
        decode_ptrs.emplace_back(native,std::move(table));
        return decode_ptrs.back().second->as<unsigned long long>();
    };
    auto all_on=[&](const uint8_t* native){
        ck(cudaMemsetAsync(y.p,0xff,out_bytes,s));
        if(native_decode_control) {
            const auto* table=decode_table(native);
            strata::kernels::quantize_q8_1_rows(input.as<float>(),rows,N,decode_xq->p,s);
            strata::kernels::native_expert_grouped(decode_layout,table,b.as<int32_t>(),decode_ng->as<int32_t>(),
                ids.as<int32_t>(),ids.as<int32_t>(),E,rows,decode_xq->p,decode_scratch->p,y.as<float>(),s);
        } else { gather(E,native); mmq(E,rows); }
    };
    auto mix_on=[&](const uint8_t* native,const uint8_t* qpn_packed){
        ck(cudaMemsetAsync(y.p,0xff,out_bytes,s));
        if(fallback_groups) { gather(fallback_groups,native); mmq(fallback_groups,fallback_rows); }
        q::stage_f32(input.as<float>()+size_t(fallback_rows)*N,xh.as<uint16_t>()+size_t(fallback_rows)*N,
                     size_t(small)*N,s);
        for(int i=fallback_groups;i<E;i++){
            const int offset=bounds[i],n=counts[i];
            const q::Schedule schedule{4,1,1,n>16?16:0};
            const auto* src=native+size_t(i)*STRIDE;
            auto* xv=xh.as<uint16_t>()+size_t(offset)*N;
            auto* gv=gate.as<float>()+size_t(offset)*F;
            auto* uv=up.as<float>()+size_t(offset)*F;
            auto* hv=hidden.as<float>()+size_t(offset)*F;
            auto* hhalf=hh.as<uint16_t>()+size_t(offset)*F;
            auto* out=y.as<float>()+size_t(offset)*N;
            auto* sc=tail.as<float>()+i*4;
            if (fused_gu) {
                // Both packed and native options perform one 1280-wide GU
                // projection; the packed option changes only compressed
                // block addresses and must pass the same numerical oracle.
                if(chunkmajor)
                    q::projection_chunkmajor_batch(
                        qpn_packed+size_t(i)*STRIDE,0,xv,
                        gu.as<float>()+size_t(offset)*GU,1,GU,N,n,schedule,s);
                else
                    q::projection(src,xv,gu.as<float>()+size_t(offset)*GU,
                                  GU,N,n,schedule,s);
                q::swiglu_stage_half(gu.as<float>()+size_t(offset)*GU,
                                     sc,hhalf,n,F,s);
            } else {
                q::projection(src,xv,gv,F,N,n,schedule,s);
                q::projection(src+MAT,xv,uv,F,N,n,schedule,s);
                q::swiglu(gv,uv,sc,hv,size_t(n)*F,s);
                q::stage_f32(hv,hhalf,size_t(n)*F,s);
            }
            if(chunkmajor)
                q::projection_chunkmajor_batch(
                    qpn_packed+size_t(i)*STRIDE+2*MAT,0,hhalf,
                    out,1,N,F,n,schedule,s);
            else
                q::projection(src+2*MAT,hhalf,out,N,F,n,schedule,s);
            q::scale_down(out,sc,size_t(n)*N,s);
        }
    };
    auto all=[&](){stream_weights(false);all_on(w.as<uint8_t>());};
    auto mix=[&](){stream_weights(true);mix_on(w.as<uint8_t>(),packed_w->as<uint8_t>());};
    auto result=[&](){
        ck(cudaStreamSynchronize(s));
        std::vector<float> out(size_t(rows)*N);
        ck(cudaMemcpy(out.data(),y.p,out_bytes,cudaMemcpyDeviceToHost));
        if(std::any_of(out.begin(),out.end(),[](float v){return !std::isfinite(v);}))
            throw std::runtime_error("unwritten/nonfinite output");
        return out;
    };
    const auto* guard=y.as<uint8_t>()+out_bytes;
    ck(cudaMemsetAsync(const_cast<uint8_t*>(guard),0xa5,256,s));
    auto check_canary=[&](){
        std::array<uint8_t,256> check{};
        ck(cudaMemcpy(check.data(),guard,256,cudaMemcpyDeviceToHost));
        if(!std::all_of(check.begin(),check.end(),[](uint8_t v){return v==0xa5;}))
            throw std::runtime_error("output guard overwritten");
    };
    all();auto baseline=result();
    mix();auto candidate=result();
    const double cross=l2(candidate,baseline);
    const auto oracle=sampled_oracle(host_w,x,bounds);
    const double all_reference_l2=sampled_l2(baseline,oracle);
    const double mixed_reference_l2=sampled_l2(candidate,oracle);
    if(!std::isfinite(cross)||cross>0.06||
       !std::isfinite(all_reference_l2)||all_reference_l2>0.03||
       !std::isfinite(mixed_reference_l2)||mixed_reference_l2>0.03)
        throw std::runtime_error("mixed/MMQ or independent double-precision oracle parity failed");
    check_canary();
    if(check_graph){
        for(int arm=0;arm<2;arm++){
            const std::function<void()> run=arm?std::function<void()>(mix):std::function<void()>(all);
            run();const auto expected=result();
            Graph graph;ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
            run();ck(cudaStreamEndCapture(s,&graph.g));
            ck(cudaGraphInstantiate(&graph.exec,graph.g,nullptr,nullptr,0));
            ck(cudaGraphLaunch(graph.exec,s));auto replay=result();
            if(std::memcmp(expected.data(),replay.data(),out_bytes))
                throw std::runtime_error("graph replay differs from ordinary run");
            auto changed=x;for(float& v:changed)v*=.5f;
            input.put(changed.data(),host_x.size());
            run();const auto changed_ref=result();
            ck(cudaGraphLaunch(graph.exec,s));auto changed_replay=result();
            if(std::memcmp(changed_ref.data(),changed_replay.data(),out_bytes)||
               !std::memcmp(expected.data(),changed_replay.data(),out_bytes))
                throw std::runtime_error("graph did not consume changed input");
            input.put(x.data(),host_x.size());check_canary();
        }
    }
    std::array<std::vector<double>,2> times;Events events;
    for(int rep=0;rep<rounds;rep++)for(int j=0;j<2;j++){
        const int arm=(rep+j)%2;
        const std::function<void()> run=arm?std::function<void()>(mix):std::function<void()>(all);
        run();run();ck(cudaEventRecord(events.a,s));
        for(int i=0;i<iterations;i++)run();
        ck(cudaEventRecord(events.b,s));ck(cudaEventSynchronize(events.b));
        float ms=0;ck(cudaEventElapsedTime(&ms,events.a,events.b));
        if(ms<=0||!std::isfinite(ms))throw std::runtime_error("invalid GPU timing");
        times[arm].push_back(double(ms)*1000/iterations);
    }
    check_canary();
    // Replay the SAME two complete grouped workloads as CUDA graphs to
    // distinguish ordinary host submission gaps from GPU-side work.
    std::array<std::vector<double>,2> graph_times;
    if(check_graph){
        Graph graphs[2];
        for(int arm=0;arm<2;arm++){
            const std::function<void()> run=arm?std::function<void()>(mix):std::function<void()>(all);
            ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
            run();
            ck(cudaStreamEndCapture(s,&graphs[arm].g));
            ck(cudaGraphInstantiate(&graphs[arm].exec,graphs[arm].g,nullptr,nullptr,0));
        }
        for(int rep=0;rep<rounds;rep++)for(int j=0;j<2;j++){
            const int arm=(rep+j)%2;
            for(int warm=0;warm<2;warm++)ck(cudaGraphLaunch(graphs[arm].exec,s));
            ck(cudaEventRecord(events.a,s));
            for(int i=0;i<iterations;i++)ck(cudaGraphLaunch(graphs[arm].exec,s));
            ck(cudaEventRecord(events.b,s));ck(cudaEventSynchronize(events.b));
            float ms=0;ck(cudaEventElapsedTime(&ms,events.a,events.b));
            if(ms<=0||!std::isfinite(ms))throw std::runtime_error("invalid graph timing");
            graph_times[arm].push_back(double(ms)*1000/iterations);
        }
        check_canary();
    }
    // An isolated realistic two-stream *throughput* test. Each arm uses two
    // distinct compressed-weight GPU slots and one compute scratch stream,
    // with copy-stream events guarding reuse. The next iteration's H2D can
    // overlap this iteration's compute. This does NOT capture a two-stream
    // CUDA graph or reproduce the model's layer-ahead scheduling.
    std::array<std::vector<double>,2> pipeline_times;
    bool pipeline_output_equal=false;
    if(async_pipeline) {
        Stream copy_stream;
        std::array<std::unique_ptr<Dev>,2> native_slots;
        std::array<std::unique_ptr<Dev>,2> qpn_slots;
        for(int i=0;i<2;i++) {
            native_slots[i]=std::make_unique<Dev>(canonical_h2d);
            qpn_slots[i]=std::make_unique<Dev>(canonical_h2d);
        }
        std::array<cudaEvent_t,2> ready{}, consumed{};
        for(int i=0;i<2;i++) {
            ck(cudaEventCreateWithFlags(&ready[i],cudaEventDisableTiming));
            ck(cudaEventCreateWithFlags(&consumed[i],cudaEventDisableTiming));
        }
        // Events are explicitly destroyed before stream/buffer destruction.
        auto next_copy=[&](bool hybrid,int slot){
            if(!hybrid || fallback_h2d)ck(cudaMemcpyAsync(native_slots[slot]->p,pinned_canonical->p,
                hybrid?fallback_h2d:canonical_h2d,cudaMemcpyHostToDevice,copy_stream.s));
            if(hybrid)
                ck(cudaMemcpyAsync(qpn_slots[slot]->as<uint8_t>()+fallback_h2d,
                    pinned_qpn->p,packed_h2d,cudaMemcpyHostToDevice,copy_stream.s));
            ck(cudaEventRecord(ready[slot],copy_stream.s));
        };
        auto trial=[&](bool hybrid)->double{
            ck(cudaStreamSynchronize(copy_stream.s));
            ck(cudaStreamSynchronize(s));
            Events timer;
            ck(cudaEventRecord(timer.a,copy_stream.s));
            next_copy(hybrid,0);
            for(int k=0;k<iterations;k++){
                const int current=k&1;
                if(k+1<iterations) {
                    const int upcoming=(k+1)&1;
                    // Do not overwrite weight slots until compute has
                    // finished consuming all of their previous contents.
                    if(k+1>=2)
                        ck(cudaStreamWaitEvent(copy_stream.s,consumed[upcoming],0));
                    next_copy(hybrid,upcoming);
                }
                ck(cudaStreamWaitEvent(s,ready[current],0));
                if(hybrid)
                    mix_on(native_slots[current]->as<uint8_t>(),
                           qpn_slots[current]->as<uint8_t>());
                else
                    all_on(native_slots[current]->as<uint8_t>());
                ck(cudaEventRecord(consumed[current],s));
            }
            ck(cudaEventRecord(timer.b,s));
            ck(cudaEventSynchronize(timer.b));
            float ms=0;
            ck(cudaEventElapsedTime(&ms,timer.a,timer.b));
            if(ms<=0 || !std::isfinite(ms))
                throw std::runtime_error("invalid async pipeline CUDA timing");
            const auto actual=result();
            const auto& expected=hybrid?candidate:baseline;
            if(std::memcmp(actual.data(),expected.data(),out_bytes))
                throw std::runtime_error("two-stream pipeline output differs from serial reference");
            check_canary();
            return double(ms)*1000/iterations;
        };
        // Warm each arm and alternate subsequent observations to reduce
        // thermal, cache, driver and ordering bias.
        trial(false);trial(true);
        for(int rep=0;rep<rounds;rep++)
            for(int j=0;j<2;j++) {
                const int arm=(rep+j)&1;
                pipeline_times[arm].push_back(trial(arm==1));
            }
        pipeline_output_equal=true;
        ck(cudaStreamSynchronize(s));
        ck(cudaStreamSynchronize(copy_stream.s));
        for(int i=0;i<2;i++) {
            ck(cudaEventDestroy(ready[i]));
            ck(cudaEventDestroy(consumed[i]));
        }
    }
    std::printf("{\"scope\":\"16 real experts; row-cap cases are decode-shape proxies, not model decode or TTFT\",");
    std::printf("\"groups\":16,\"fallback_groups\":%d,\"qpn_groups\":%d,\"qpn_min\":%d,\"qpn_max\":%d,\"fused_gu\":%s,\"chunkmajor\":%s,\"pinned_stream\":%s,\"disk_prepacked\":%s,\"cache_load_us\":%.6f,\"repack_cpu_us\":%.6f,\"repack_upload_us\":%.6f,\"control_h2d_bytes\":%zu,\"hybrid_h2d_bytes\":%zu,\"routed_rows\":%d,",
                fallback_groups,E-fallback_groups,qpn_min,qpn_max,fused_gu?"true":"false",chunkmajor?"true":"false",pinned_stream?"true":"false",packed_path.empty()?"false":"true",cache_load_us,weight_pack_cpu_us,weight_pack_upload_us,pinned_stream?canonical_h2d:0,pinned_stream?fallback_h2d+packed_h2d:0,rows);
    std::printf("\"baseline_backend\":\"%s\",",native_decode_control?"native-grouped-dp4a":"prefill-mmq");
    std::printf("\"row_cap\":%d,\"original_routed_rows\":%d,\"explicit_range\":%s,",row_cap,original_rows,explicit_range?"true":"false");
    std::printf("\"weights_bytes\":%zu,\"l2_bytes\":%d,\"weights_exceed_l2\":%s,",
                host_w.size(),prop.l2CacheSize,host_w.size()>size_t(prop.l2CacheSize)?"true":"false");
    std::printf("\"mixed_vs_mmq_output_l2\":%.9g,\"mmq_reference_l2\":%.9g,\"mixed_reference_l2\":%.9g,\"oracle_rows\":%zu,\"numerics_pass\":true,",
                cross,all_reference_l2,mixed_reference_l2,oracle.size());
    std::printf("\"graph_checked\":%s,\"graph_replay_equal\":%s,\"graph_changed_input_equal\":%s,",
                check_graph?"true":"false",check_graph?"true":"false",check_graph?"true":"false");
    std::printf("\"output_tail_guard_ok\":true,\"iterations\":%d,\"rounds\":%d,",iterations,rounds);
    std::printf("\"async_pipeline\":%s,\"pipeline_output_equal\":%s,",async_pipeline?"true":"false",pipeline_output_equal?"true":"false");
    if(async_pipeline) {
        std::printf("\"pipeline_all_mmq_us\":");arr(pipeline_times[0]);
        std::printf(",\"pipeline_mixed_us\":");arr(pipeline_times[1]);
        if(rounds>=5)
            std::printf(",\"pipeline_all_mmq_median_us\":%.6f,\"pipeline_mixed_median_us\":%.6f,\"pipeline_speedup\":%.7f",
                        med(pipeline_times[0]),med(pipeline_times[1]),
                        med(pipeline_times[0])/med(pipeline_times[1]));
        std::printf(",");
    }
    std::printf("\"all_mmq_us\":");arr(times[0]);std::printf(",\"mixed_qpn_mmq_us\":");arr(times[1]);
    if(rounds>=5)std::printf(",\"all_mmq_median_us\":%.6f,\"mixed_median_us\":%.6f,\"mixed_over_mmq_speedup\":%.7f",med(times[0]),med(times[1]),med(times[0])/med(times[1]));
    if(check_graph){
        std::printf(",\"all_mmq_graph_us\":");arr(graph_times[0]);
        std::printf(",\"mixed_graph_us\":");arr(graph_times[1]);
        if(rounds>=5)std::printf(",\"all_mmq_graph_median_us\":%.6f,\"mixed_graph_median_us\":%.6f,\"mixed_graph_speedup\":%.7f",
                                 med(graph_times[0]),med(graph_times[1]),med(graph_times[0])/med(graph_times[1]));
    }
    std::printf("}\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"mixed_gpu_bench: %s\n",e.what());return 2;}
