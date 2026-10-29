// SPDX-License-Identifier: MIT
// Experimental QPN occupancy/layout attribution; no production integration.
#include "strata/kernels/nvfp4_volta_qpn.hpp"
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
namespace {
constexpr int E=6,N=2560,F=640,KBLOCKS=40;
constexpr size_t MAT=size_t(N)*F/64*36,STRIDE=3*MAT+16;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Dev {
    void* p=nullptr;
    explicit Dev(size_t bytes){ck(cudaMalloc(&p,bytes));}
    ~Dev(){if(p)cudaFree(p);}
    Dev(const Dev&)=delete;Dev& operator=(const Dev&)=delete;
    template<class T>T* as() const{return static_cast<T*>(p);}
    void put(const void* x,size_t bytes){ck(cudaMemcpy(p,x,bytes,cudaMemcpyHostToDevice));}
};
struct Stream {cudaStream_t s{};Stream(){ck(cudaStreamCreate(&s));}~Stream(){cudaStreamDestroy(s);}};
struct Events {cudaEvent_t a{},b{};Events(){ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));}~Events(){cudaEventDestroy(a);cudaEventDestroy(b);}};
struct Graph {
    cudaGraph_t g{};cudaGraphExec_t e{};
    ~Graph(){if(e)cudaGraphExecDestroy(e);if(g)cudaGraphDestroy(g);}
    Graph()=default;Graph(const Graph&)=delete;Graph& operator=(const Graph&)=delete;
};
std::vector<uint8_t> read(const std::string& name,size_t bytes){
    std::ifstream f(name,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(bytes))throw std::runtime_error("private file size mismatch");
    f.seekg(0);std::vector<uint8_t> out(bytes);
    if(!f.read(reinterpret_cast<char*>(out.data()),std::streamsize(bytes)))
        throw std::runtime_error("private file read failed");
    return out;
}
std::vector<int> counts(const std::string& filename) {
    std::ifstream f(filename);std::string line,extra;
    if(!f||!std::getline(f,line)||std::getline(f,extra))throw std::runtime_error("invalid group counts");
    std::vector<int> result;size_t first=0;
    while(first<line.size()){
        size_t end=line.find(',',first);
        auto part=line.substr(first,end==std::string::npos?std::string::npos:end-first);
        if(part.empty()||part.find_first_not_of("0123456789")!=std::string::npos)
            throw std::runtime_error("count not numeric");
        int n=std::stoi(part);if(n<1||n>1024)throw std::runtime_error("invalid group length");
        result.push_back(n);
        if(end==std::string::npos)break;
        first=end+1;
    }
    if(result.size()!=16||
       !std::all_of(result.begin(),result.begin()+E,[](int n){return n>32;}))
        throw std::runtime_error("must use first six pinned >32-row groups");
    return result;
}
std::vector<uint8_t> chunk_major(const std::vector<uint8_t>& weights){
    auto packed=weights;
    for(int e=0;e<E;e++)for(int proj=0;proj<2;proj++) {
        const auto* src=weights.data()+size_t(e)*STRIDE+size_t(proj)*MAT;
        auto* dst=packed.data()+size_t(e)*STRIDE+size_t(proj)*MAT;
        for(int st=0;st<KBLOCKS;st++)for(int row=0;row<F;row++)
            std::memcpy(dst+(size_t(st)*F+row)*36,
                        src+(size_t(row)*KBLOCKS+st)*36,36);
    }
    return packed;
}
double l2(const std::vector<float>& a,const std::vector<float>& b){
    if(a.size()!=b.size())throw std::runtime_error("output size mismatch");
    double num=0,den=0;
    for(size_t i=0;i<a.size();i++){
        if(!std::isfinite(a[i])||!std::isfinite(b[i]))
            return std::numeric_limits<double>::infinity();
        double d=double(a[i])-b[i];num+=d*d;den+=double(b[i])*b[i];
    }
    return den?std::sqrt(num/den):num?std::numeric_limits<double>::infinity():0;
}
double med(std::vector<double> data){
    if(data.size()<5)throw std::runtime_error("not enough repetitions");
    std::sort(data.begin(),data.end());
    size_t mid=data.size()/2;
    return data.size()%2?data[mid]:(data[mid-1]+data[mid])/2;
}
void values(const std::vector<double>& v){
    std::printf("[");for(size_t i=0;i<v.size();i++)
        std::printf("%s%.6f",i?",":"",v[i]);
    std::printf("]");
}
} // namespace
int main(int argc,char** argv)try{
    std::string weight_path,x_path,count_path;
    int iterations=30,rounds=6,only_t=0;bool no_graph=false;
    for(int i=1;i<argc;i++){
        std::string flag(argv[i]);
        if(flag=="--no-graph"){no_graph=true;continue;}
        if(++i>=argc)throw std::invalid_argument("missing argument value");
        std::string val(argv[i]);
        if(flag=="--weights")weight_path=val;
        else if(flag=="--activations")x_path=val;
        else if(flag=="--counts")count_path=val;
        else if(flag=="--iterations")iterations=std::stoi(val);
        else if(flag=="--rounds")rounds=std::stoi(val);
        else if(flag=="--only-tokens")only_t=std::stoi(val);
        else throw std::invalid_argument("unknown argument");
    }
    constexpr std::array<int,10> sizes={1,2,4,8,12,16,17,20,24,32};
    if(weight_path.empty()||x_path.empty()||count_path.empty()||
       iterations<1||iterations>400||rounds<1||rounds>20||
       (only_t&&std::find(sizes.begin(),sizes.end(),only_t)==sizes.end()))
        throw std::invalid_argument("invalid profiler arguments");
    auto c=counts(count_path);
    int total=std::accumulate(c.begin(),c.end(),0);
    auto all_w=read(weight_path,16*STRIDE);
    auto all_x=read(x_path,size_t(total)*N*4);
    std::vector<float> routed(size_t(total)*N);
    std::memcpy(routed.data(),all_x.data(),all_x.size());
    std::array<int,E+1> offsets{};
    for(int i=0;i<E;i++)offsets[i+1]=offsets[i]+c[i];
    std::vector<uint8_t> weight(all_w.begin(),all_w.begin()+E*STRIDE);
    auto packed=chunk_major(weight);
    int dev=0;ck(cudaGetDevice(&dev));cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop,dev));
    if(prop.major!=7||prop.minor!=0)throw std::runtime_error("V100 sm_70 required");
    if(2*E*MAT<=size_t(prop.l2CacheSize))throw std::runtime_error("working set not larger than L2");
    Stream stream;auto s=stream.s;
    Dev w(weight.size()),p(packed.size());
    w.put(weight.data(),weight.size());p.put(packed.data(),packed.size());
    std::printf("{\"scope\":\"six real distinct expert gate/up projections, captured row prefixes, not MoE forward\",");
    std::printf("\"experts\":6,\"compressed_weight_bytes_accessed\":%zu,\"l2_bytes\":%d,\"sms\":%d,\"cases\":[",
                2*E*MAT,prop.l2CacheSize,prop.multiProcessorCount);
    bool first=true;
    for(int t:sizes){
        if(only_t&&t!=only_t)continue;
        const q::Schedule fast{4,1,1,t>16?16:0},scalar{4,1,0,t>16?16:0};
        if(!q::supported(70,F,N,t,fast)||!q::supported(70,F,N,t,scalar))
            throw std::runtime_error("unsupported sweep schedule");
        std::vector<float> x(size_t(E)*t*N);
        for(int i=0;i<E;i++)
            std::memcpy(x.data()+size_t(i)*t*N,routed.data()+size_t(offsets[i])*N,size_t(t)*N*4);
        for(float v:x)if(!std::isfinite(v)||std::abs(v)>65504.f)
            throw std::runtime_error("nonfinite / unrepresentable routed activation");
        Dev in(x.size()*4),xh(x.size()*2),g(size_t(E)*t*F*4+256),u(size_t(E)*t*F*4+256);
        in.put(x.data(),x.size()*4);
        q::stage_f32(in.as<float>(),xh.as<uint16_t>(),x.size(),s);
        ck(cudaMemsetAsync(g.as<uint8_t>()+size_t(E)*t*F*4,0xa5,256,s));
        ck(cudaMemsetAsync(u.as<uint8_t>()+size_t(E)*t*F*4,0xa5,256,s));
        auto serial=[&](){
            for(int i=0;i<E;i++){
                auto* X=xh.as<uint16_t>()+size_t(i)*t*N;
                auto* G=g.as<float>()+size_t(i)*t*F;
                auto* U=u.as<float>()+size_t(i)*t*F;
                const auto* W=w.as<uint8_t>()+size_t(i)*STRIDE;
                q::projection(W,X,G,F,N,t,fast,s);
                q::projection(W+MAT,X,U,F,N,t,fast,s);
            }
        };
        auto batch=[&](q::Schedule schedule){
            q::projection_batch(w.as<uint8_t>(),STRIDE,xh.as<uint16_t>(),g.as<float>(),E,F,N,t,schedule,s);
            q::projection_batch(w.as<uint8_t>()+MAT,STRIDE,xh.as<uint16_t>(),u.as<float>(),E,F,N,t,schedule,s);
        };
        auto transposed=[&](){
            q::projection_chunkmajor_batch(p.as<uint8_t>(),STRIDE,xh.as<uint16_t>(),g.as<float>(),E,F,N,t,fast,s);
            q::projection_chunkmajor_batch(p.as<uint8_t>()+MAT,STRIDE,xh.as<uint16_t>(),u.as<float>(),E,F,N,t,fast,s);
        };
        std::array<std::function<void()>,4> arms={
            std::function<void()>(serial),
            std::function<void()>([&](){batch(fast);}),
            std::function<void()>(transposed),
            std::function<void()>([&](){batch(scalar);})
        };
        auto output=[&](){
            ck(cudaStreamSynchronize(s));
            std::vector<float> y(size_t(E)*t*F*2);
            ck(cudaMemcpy(y.data(),g.p,size_t(E)*t*F*4,cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(y.data()+size_t(E)*t*F,u.p,size_t(E)*t*F*4,cudaMemcpyDeviceToHost));
            return y;
        };
        auto canary=[&](){
            std::array<uint8_t,256> check{};
            ck(cudaMemcpy(check.data(),g.as<uint8_t>()+size_t(E)*t*F*4,256,cudaMemcpyDeviceToHost));
            if(!std::all_of(check.begin(),check.end(),[](uint8_t v){return v==0xa5;}))
                throw std::runtime_error("gate canary damaged");
            ck(cudaMemcpy(check.data(),u.as<uint8_t>()+size_t(E)*t*F*4,256,cudaMemcpyDeviceToHost));
            if(!std::all_of(check.begin(),check.end(),[](uint8_t v){return v==0xa5;}))
                throw std::runtime_error("up canary damaged");
        };
        arms[0]();const auto baseline=output();
        std::array<double,4> parity{};
        for(int a=0;a<4;a++){
            arms[a]();const auto got=output();parity[a]=l2(got,baseline);
            if(!std::isfinite(parity[a])||parity[a]>.002)
                throw std::runtime_error("layout or decoder changed numerical output");
        }
        std::array<Graph,4> graph;
        if(!no_graph)for(int a=0;a<4;a++){
            ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
            arms[a]();ck(cudaStreamEndCapture(s,&graph[a].g));
            ck(cudaGraphInstantiate(&graph[a].e,graph[a].g,nullptr,nullptr,0));
            ck(cudaGraphLaunch(graph[a].e,s));
            if(l2(output(),baseline)>.002)
                throw std::runtime_error("diagnostic graph replay differs");
        }
        Events events;
        std::array<std::vector<double>,4> normal_times,graph_times;
        for(int rep=0;rep<rounds;rep++)for(int j=0;j<4;j++){
            int a=(rep+j)%4;
            arms[a]();arms[a]();
            ck(cudaEventRecord(events.a,s));
            for(int i=0;i<iterations;i++)arms[a]();
            ck(cudaEventRecord(events.b,s));ck(cudaEventSynchronize(events.b));
            float elapsed=0;ck(cudaEventElapsedTime(&elapsed,events.a,events.b));
            if(elapsed<=0||!std::isfinite(elapsed))throw std::runtime_error("invalid GPU time");
            normal_times[a].push_back(double(elapsed)*1000/iterations);
            if(!no_graph){
                ck(cudaGraphLaunch(graph[a].e,s));
                ck(cudaEventRecord(events.a,s));
                for(int i=0;i<iterations;i++)ck(cudaGraphLaunch(graph[a].e,s));
                ck(cudaEventRecord(events.b,s));ck(cudaEventSynchronize(events.b));
                elapsed=0;ck(cudaEventElapsedTime(&elapsed,events.a,events.b));
                graph_times[a].push_back(double(elapsed)*1000/iterations);
            }
        }
        canary();
        if(!first)std::printf(",");first=false;
        int row_tile=t>16?16:t>8?16:8,cta=20*((t+row_tile-1)/row_tile);
        std::printf("{\"tokens\":%d,\"cta_per_expert_projection\":%d,\"cta_six_expert_batch\":%d,",
                    t,cta,cta*E);
        std::printf("\"minimum_wasted_mma_row_fraction\":%.5f,",double(((t+row_tile-1)/row_tile)*row_tile-t)/(((t+row_tile-1)/row_tile)*row_tile));
        std::printf("\"serial_us\":");values(normal_times[0]);
        std::printf(",\"batch_us\":");values(normal_times[1]);
        std::printf(",\"chunkmajor_us\":");values(normal_times[2]);
        std::printf(",\"scalar_decoder_us\":");values(normal_times[3]);
        if(rounds>=5){
            const double serial_m=med(normal_times[0]),batched=med(normal_times[1]),packed_m=med(normal_times[2]),scalar_m=med(normal_times[3]);
            std::printf(",\"median_us\":{\"serial\":%.6f,\"batch\":%.6f,\"chunkmajor\":%.6f,\"scalar\":%.6f},",
                        serial_m,batched,packed_m,scalar_m);
            std::printf("\"batch_speedup\":%.6f,\"layout_speedup\":%.6f,\"decoder_speedup\":%.6f",
                        serial_m/batched,batched/packed_m,scalar_m/batched);
        }
        std::printf(",\"relative_l2\":{\"serial\":%.9g,\"batch\":%.9g,\"chunkmajor\":%.9g,\"scalar\":%.9g}",
                    parity[0],parity[1],parity[2],parity[3]);
        if(!no_graph && rounds>=5){
            std::printf(",\"graph_median_us\":{\"serial\":%.6f,\"batch\":%.6f,\"chunkmajor\":%.6f,\"scalar\":%.6f}",
                        med(graph_times[0]),med(graph_times[1]),med(graph_times[2]),med(graph_times[3]));
        }
        std::printf("}");
    }
    std::printf("]}\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"QPN diagnostic: %s\n",e.what());return 2;}
