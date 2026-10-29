// SPDX-License-Identifier: MIT
// Standalone deterministic T33..128 NVFP4 projection qualification on Volta.
// Synthetic data only; model routing remains restricted to T17..32.
#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
namespace q=strata::kernels::qpn;
constexpr int N=1280,K=2560,MAX=512,GUARD=256;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static double duration(cudaStream_t s,const std::function<void()>& f,int reps){
    f();ck(cudaStreamSynchronize(s));
    cudaEvent_t a{},b{};ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    ck(cudaEventRecord(a,s));
    for(int k=0;k<reps;k++)f();
    ck(cudaEventRecord(b,s));ck(cudaEventSynchronize(b));
    float ms=0;ck(cudaEventElapsedTime(&ms,a,b));
    ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
    return double(ms)*1000/reps;
}
int main(int argc,char** argv)try{
    bool san=argc==2&&std::string(argv[1])=="--sanitizer";
    if(argc>2||(argc==2&&!san))throw std::invalid_argument("only --sanitizer");
    cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));
    if(p.major!=7||p.minor!=0)throw std::runtime_error("requires sm_70");
    constexpr std::size_t W=std::size_t(N)*K/64*36;
    std::vector<std::uint8_t> weights(W);
    std::uint32_t seed=20261029u;
    for(std::size_t k=0;k<W;k+=36){
        for(int j=0;j<4;j++)weights[k+j]=8;
        for(int j=4;j<36;j++){
            seed=seed*1664525u+1013904223u;
            weights[k+j]=std::uint8_t(seed>>24);
        }
    }
    std::vector<std::uint16_t> hx(std::size_t(MAX)*K);
    for(std::size_t k=0;k<hx.size();k++)
        hx[k]=((k*13+k/37)%7<3)?0x3800:0xb800;
    std::uint8_t *w=nullptr,*y=nullptr,*ref=nullptr;
    std::uint16_t* x=nullptr;
    constexpr std::size_t ybytes=std::size_t(MAX)*N*sizeof(float);
    ck(cudaMalloc(reinterpret_cast<void**>(&w),W));
    ck(cudaMalloc(reinterpret_cast<void**>(&x),hx.size()*2));
    ck(cudaMalloc(reinterpret_cast<void**>(&y),ybytes+GUARD));
    ck(cudaMalloc(reinterpret_cast<void**>(&ref),ybytes+GUARD));
    cudaStream_t s{};ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    ck(cudaMemcpyAsync(w,weights.data(),W,cudaMemcpyHostToDevice,s));
    const q::Schedule sched{4,1,1,16};
    const std::vector<int> cases=san?std::vector<int>{187,508}:
                                      std::vector<int>{33,40,48,64,96,97,128,187,508,512};
    std::printf("{\"scope\":\"synthetic QPN large-token experiment, not model/TTFT\","
                "\"cuda_sm\":70,\"weight_bytes\":%zu,\"results\":[",W);
    bool first=true;
    for(int t:cases){
        ck(cudaMemcpyAsync(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice,s));
        auto full=[&]{q::projection(w,x,reinterpret_cast<float*>(y),N,K,t,sched,s);};
        auto segmented=[&]{
            for(int start=0;start<t;start+=32){
                const int rows=(t-start<32)?t-start:32;
                q::projection(w,x+std::size_t(start)*K,
                    reinterpret_cast<float*>(ref)+std::size_t(start)*N,
                    N,K,rows,sched,s);
            }
        };
        const auto check=[&](bool graph){
            ck(cudaMemsetAsync(y,0xa5,ybytes+GUARD,s));
            ck(cudaMemsetAsync(ref,0xa5,ybytes+GUARD,s));
            if(graph){
                cudaGraph_t g{};cudaGraphExec_t ex{};
                ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
                full();
                ck(cudaStreamEndCapture(s,&g));
                ck(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0));
                ck(cudaGraphLaunch(ex,s));
                ck(cudaStreamSynchronize(s));
                ck(cudaGraphExecDestroy(ex));ck(cudaGraphDestroy(g));
            }else full();
            segmented();ck(cudaStreamSynchronize(s));
            std::vector<std::uint8_t> a(ybytes+GUARD),b(ybytes+GUARD);
            ck(cudaMemcpy(a.data(),y,a.size(),cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(b.data(),ref,b.size(),cudaMemcpyDeviceToHost));
            const std::size_t size=std::size_t(t)*N*sizeof(float);
            if(std::memcmp(a.data(),b.data(),size))
                throw std::runtime_error("bitwise mismatch full vs segmented at T="+std::to_string(t));
            for(std::size_t i=size;i<a.size();i++)
                if(a[i]!=0xa5||b[i]!=0xa5)
                    throw std::runtime_error("output guard damaged");
            for(std::size_t i=0;i<size;i+=4){
                float v;std::memcpy(&v,a.data()+i,4);
                if(!std::isfinite(v))throw std::runtime_error("nonfinite output");
            }
        };
        check(false);
        // Mutate both an early and an actual late row, including all inputs.
        // This catches accidental truncation at 32 or stale >128 tiles.
        const int late_row=t-1;
        for(int k=0;k<K;k++) {
            hx[std::size_t(32)*K+k]^=0x8000u;
            hx[std::size_t(late_row)*K+k]^=0x8000u;
        }
        ck(cudaMemcpyAsync(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice,s));
        check(true); // CUDA Graph replay also sees changed-input activations.
        for(int k=0;k<K;k++) {
            hx[std::size_t(32)*K+k]^=0x8000u;
            hx[std::size_t(late_row)*K+k]^=0x8000u;
        }
        const int reps=san?1:40;
        const double fused_us=duration(s,full,reps);
        const double segmented_us=duration(s,segmented,reps);
        std::printf("%s{\"tokens\":%d,\"full_us\":%.6f,\"segmented_us\":%.6f,"
                    "\"segmented_over_full\":%.6f,\"bitwise\":true,"
                    "\"changed_input_graph\":true,\"guards\":true}",
                    first?"":",",t,fused_us,segmented_us,segmented_us/fused_us);
        first=false;
    }
    std::puts("]}");
    ck(cudaStreamSynchronize(s));
    ck(cudaFree(ref));ck(cudaFree(y));ck(cudaFree(x));ck(cudaFree(w));
    ck(cudaStreamDestroy(s));
    return 0;
}catch(const std::exception& e){
    std::fprintf(stderr,"QPN long-row probe failed: %s\n",e.what());
    return 1;
}
