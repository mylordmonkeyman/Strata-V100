// SPDX-License-Identifier: MIT
// Standalone 16-expert, real nonuniform routed workload. Never linked into engine.
#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
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
        if(end==std::string::npos)break;start=end+1;
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
    std::string wp,xp,cp;int iterations=10,rounds=6;bool check_graph=false;
    for(int i=1;i<argc;i++){
        const std::string flag=argv[i];
        if(flag=="--graph-check"){check_graph=true;continue;}
        if(++i>=argc)throw std::invalid_argument("missing value");
        const std::string val=argv[i];
        if(flag=="--weights")wp=val;
        else if(flag=="--activations")xp=val;
        else if(flag=="--counts")cp=val;
        else if(flag=="--iterations")iterations=std::stoi(val);
        else if(flag=="--rounds")rounds=std::stoi(val);
        else throw std::invalid_argument("unknown argument");
    }
    if(wp.empty()||xp.empty()||cp.empty()||iterations<1||iterations>1000||rounds<1||rounds>50)
        throw std::invalid_argument("invalid arguments");
    const auto counts=read_counts(cp);
    std::array<int,E+1> bounds{};for(int i=0;i<E;i++)bounds[i+1]=bounds[i]+counts[i];
    const int rows=bounds[E],large=bounds[6],small=rows-large;
    const auto host_w=read_file(wp,E*STRIDE);
    const auto host_x=read_file(xp,size_t(rows)*N*4);
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
    for(int i=6;i<E;i++)if(!q::supported(70,F,N,counts[i],{4,1,1,0})||
                             !q::supported(70,N,F,counts[i],{4,1,1,0}))
        throw std::runtime_error("QPN not qualified for selected small group");
    Stream stream;const auto s=stream.s;
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
    auto gather=[&](int n){
        for(int i=0;i<n;i++){
            const auto* src=w.as<uint8_t>()+size_t(i)*STRIDE;
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
    auto all=[&](){ck(cudaMemsetAsync(y.p,0xff,out_bytes,s));gather(E);mmq(E,rows);};
    auto mix=[&](){
        ck(cudaMemsetAsync(y.p,0xff,out_bytes,s));gather(6);mmq(6,large);
        q::stage_f32(input.as<float>()+size_t(large)*N,xh.as<uint16_t>()+size_t(large)*N,
                     size_t(small)*N,s);
        for(int i=6;i<E;i++){
            const int offset=bounds[i],n=counts[i];
            auto* src=w.as<uint8_t>()+size_t(i)*STRIDE;
            auto* xv=xh.as<uint16_t>()+size_t(offset)*N;
            auto* gv=gate.as<float>()+size_t(offset)*F;
            auto* uv=up.as<float>()+size_t(offset)*F;
            auto* hv=hidden.as<float>()+size_t(offset)*F;
            auto* hhalf=hh.as<uint16_t>()+size_t(offset)*F;
            auto* out=y.as<float>()+size_t(offset)*N;
            auto* sc=tail.as<float>()+i*4;
            q::projection(src,xv,gv,F,N,n,{4,1,1,0},s);
            q::projection(src+MAT,xv,uv,F,N,n,{4,1,1,0},s);
            q::swiglu(gv,uv,sc,hv,size_t(n)*F,s);
            q::stage_f32(hv,hhalf,size_t(n)*F,s);
            q::projection(src+2*MAT,hhalf,out,N,F,n,{4,1,1,0},s);
            q::scale_down(out,sc,size_t(n)*N,s);
        }
    };
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
    if(!std::isfinite(cross)||cross>0.06)
        throw std::runtime_error("mixed/all-MMQ numerical parity failed");
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
    std::printf("{\"scope\":\"complete 16 real expert mixed workload, not full-model TTFT\",");
    std::printf("\"groups\":16,\"fallback_groups\":6,\"qpn_groups\":10,\"routed_rows\":%d,",rows);
    std::printf("\"weights_bytes\":%zu,\"l2_bytes\":%d,\"weights_exceed_l2\":%s,",
                host_w.size(),prop.l2CacheSize,host_w.size()>size_t(prop.l2CacheSize)?"true":"false");
    std::printf("\"mixed_vs_mmq_output_l2\":%.9g,\"numerics_pass\":true,",cross);
    std::printf("\"graph_checked\":%s,\"graph_replay_equal\":%s,\"graph_changed_input_equal\":%s,",
                check_graph?"true":"false",check_graph?"true":"false",check_graph?"true":"false");
    std::printf("\"output_tail_guard_ok\":true,\"iterations\":%d,\"rounds\":%d,",iterations,rounds);
    std::printf("\"all_mmq_us\":");arr(times[0]);std::printf(",\"mixed_qpn_mmq_us\":");arr(times[1]);
    if(rounds>=5)std::printf(",\"all_mmq_median_us\":%.6f,\"mixed_median_us\":%.6f,\"mixed_over_mmq_speedup\":%.7f",med(times[0]),med(times[1]),med(times[0])/med(times[1]));
    std::printf("}\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"mixed_gpu_bench: %s\n",e.what());return 2;}
