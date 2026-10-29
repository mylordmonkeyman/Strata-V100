// SPDX-License-Identifier: MIT
// Experimental standalone ragged expert batching test; no engine dispatch or model assets.
#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

namespace q = strata::kernels::qpn;
constexpr int N=2560, FF=640, PAD=32, GU=2*FF, BLOCK=256;
static void ck(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<typename T> static T* dev(std::size_t count) {
    T* p=nullptr; ck(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T))); return p;
}
__global__ void gather_ragged(const half* input,const int32_t* row_index,
                               const int32_t* bounds,half* padded,int groups,int single) {
    const int idx=int(blockIdx.x)*blockDim.x+threadIdx.x;
    const int group=single<0 ? idx/(PAD*N) : single;
    const int inner=single<0 ? idx%(PAD*N) : idx;
    if(group>=groups || inner>=PAD*N) return;
    const int row=inner/N, col=inner%N;
    const int len=bounds[group+1]-bounds[group];
    padded[std::size_t(group)*PAD*N+inner]=
        row<len ? input[std::size_t(row_index[bounds[group]+row])*N+col] : __float2half_rn(0.f);
}
__global__ void swiglu_ragged(const float* gu,const float* scales,half* hidden,
                              int groups,int single) {
    const int idx=int(blockIdx.x)*blockDim.x+threadIdx.x;
    const int group=single<0 ? idx/(PAD*FF) : single;
    const int inner=single<0 ? idx%(PAD*FF) : idx;
    if(group>=groups || inner>=PAD*FF) return;
    const int row=inner/FF,col=inner%FF;
    const float* r=gu+(std::size_t(group)*PAD+row)*GU;
    const float g=r[col]*scales[group*4], u=r[FF+col]*scales[group*4+1];
    const float e=expf(-fabsf(g));
    const float sig=g>=0 ? 1.f/(1.f+e) : e/(1.f+e);
    hidden[std::size_t(group)*PAD*FF+inner]=__float2half_rn((g*sig)*u);
}
__global__ void scatter_ragged(const float* padded,const int32_t* bounds,float* output,
                               int groups,int single) {
    const int idx=int(blockIdx.x)*blockDim.x+threadIdx.x;
    const int group=single<0 ? idx/(PAD*N) : single;
    const int inner=single<0 ? idx%(PAD*N) : idx;
    if(group>=groups || inner>=PAD*N) return;
    const int row=inner/N;
    if(row>=bounds[group+1]-bounds[group]) return;
    output[(std::size_t(bounds[group])+row)*N+inner%N]=padded[(std::size_t(group)*PAD*N)+inner];
}
struct Report { double ordinary_us, graph_us; };
static Report measure(cudaStream_t s,const std::function<void()>& fn,int iterations) {
    fn();ck(cudaStreamSynchronize(s));
    cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    ck(cudaEventRecord(a,s));
    for(int i=0;i<iterations;++i) fn();
    ck(cudaEventRecord(b,s));ck(cudaEventSynchronize(b));
    float ms=0;ck(cudaEventElapsedTime(&ms,a,b));
    const double ordinary=double(ms)*1000/iterations;
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    fn();
    ck(cudaStreamEndCapture(s,&graph));
    ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    ck(cudaGraphLaunch(exec,s));ck(cudaStreamSynchronize(s));
    ck(cudaEventRecord(a,s));
    for(int i=0;i<iterations;++i) ck(cudaGraphLaunch(exec,s));
    ck(cudaEventRecord(b,s));ck(cudaEventSynchronize(b));
    ck(cudaEventElapsedTime(&ms,a,b));
    const double replay=double(ms)*1000/iterations;
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
    ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
    return {ordinary,replay};
}
int main(int argc,char** argv) try {
    const bool sanitizer=argc>1 && std::string(argv[1])=="--sanitizer";
    if(argc>1 && !sanitizer)throw std::invalid_argument("only --sanitizer accepted");
    cudaDeviceProp device{};ck(cudaGetDeviceProperties(&device,0));
    if(device.major!=7 || device.minor!=0)throw std::runtime_error("V100 sm_70 required");
    constexpr std::size_t WGU=std::size_t(GU)*N/64*36, WD=std::size_t(N)*FF/64*36;
    const int groups=sanitizer?4:16;
    std::vector<int32_t> bounds(groups+1,0);
    for(int g=0;g<groups;++g)bounds[g+1]=bounds[g]+17+g;
    const int rows=bounds.back();
    std::vector<int32_t> rowmap(rows);
    std::iota(rowmap.begin(),rowmap.end(),0);
    std::mt19937 rng(20261029);
    std::shuffle(rowmap.begin(),rowmap.end(),rng);
    std::vector<std::uint16_t> input(std::size_t(rows)*N);
    for(std::size_t i=0;i<input.size();++i)input[i]=((i*17+i/13)%11)<5?0x3800:0xb800; // +/-0.5
    // Canonical 36-byte NVFP4 GGUF blocks: four E4M3 subscales, 32 packed E2M1 bytes.
    auto make_weights=[&](int experts,std::size_t stride) {
        std::vector<std::uint8_t> v(std::size_t(experts)*stride);
        for(std::size_t i=0;i<v.size();i+=36) {
            for(int j=0;j<4;++j)v[i+j]=8; // small positive, finite scale
            for(int j=4;j<36;++j)v[i+j]=std::uint8_t(rng()&0xffu);
        }
        return v;
    };
    std::vector<std::uint8_t> hgu=make_weights(groups,WGU),hd=make_weights(groups,WD);
    std::vector<float> hscales(std::size_t(groups)*4,1.f);
    auto* dgu=dev<std::uint8_t>(hgu.size()),*dd=dev<std::uint8_t>(hd.size());
    auto* dx=dev<half>(input.size());
    auto* ids=dev<int32_t>(rowmap.size());
    auto* db=dev<int32_t>(bounds.size());
    auto* ds=dev<float>(hscales.size());
    auto* px=dev<half>(std::size_t(groups)*PAD*N),*ph=dev<half>(std::size_t(groups)*PAD*FF);
    auto* pgu=dev<float>(std::size_t(groups)*PAD*GU),*py=dev<float>(std::size_t(groups)*PAD*N);
    constexpr int GUARD=256;
    const std::size_t out_bytes=std::size_t(rows)*N*sizeof(float);
    auto* serial=dev<std::uint8_t>(out_bytes+GUARD),*batch=dev<std::uint8_t>(out_bytes+GUARD);
    cudaStream_t s{};ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    ck(cudaMemcpyAsync(dgu,hgu.data(),hgu.size(),cudaMemcpyHostToDevice,s));
    ck(cudaMemcpyAsync(dd,hd.data(),hd.size(),cudaMemcpyHostToDevice,s));
    ck(cudaMemcpyAsync(dx,input.data(),input.size()*2,cudaMemcpyHostToDevice,s));
    ck(cudaMemcpyAsync(ids,rowmap.data(),rowmap.size()*4,cudaMemcpyHostToDevice,s));
    ck(cudaMemcpyAsync(db,bounds.data(),bounds.size()*4,cudaMemcpyHostToDevice,s));
    ck(cudaMemcpyAsync(ds,hscales.data(),hscales.size()*4,cudaMemcpyHostToDevice,s));
    ck(cudaMemsetAsync(serial,0xa5,out_bytes+GUARD,s));
    ck(cudaMemsetAsync(batch,0xa5,out_bytes+GUARD,s));
    const q::Schedule sched{4,1,1,16};
    const auto per_expert=[&] {
        for(int g=0;g<groups;++g) {
            gather_ragged<<<(PAD*N+BLOCK-1)/BLOCK,BLOCK,0,s>>>(dx,ids,db,px,groups,g);
            const int t=bounds[g+1]-bounds[g];
            q::projection(dgu+std::size_t(g)*WGU,
                reinterpret_cast<const std::uint16_t*>(px+std::size_t(g)*PAD*N),
                pgu+std::size_t(g)*PAD*GU,GU,N,t,sched,s);
            q::swiglu_stage_half(pgu+std::size_t(g)*PAD*GU,ds+g*4,
                reinterpret_cast<std::uint16_t*>(ph+std::size_t(g)*PAD*FF),t,FF,s);
            q::projection(dd+std::size_t(g)*WD,
                reinterpret_cast<const std::uint16_t*>(ph+std::size_t(g)*PAD*FF),
                py+std::size_t(g)*PAD*N,N,FF,t,sched,s);
            scatter_ragged<<<(PAD*N+BLOCK-1)/BLOCK,BLOCK,0,s>>>(
                py,db,reinterpret_cast<float*>(serial),groups,g);
        }
        ck(cudaGetLastError());
    };
    const auto grouped=[&] {
        gather_ragged<<<(groups*PAD*N+BLOCK-1)/BLOCK,BLOCK,0,s>>>(
            dx,ids,db,px,groups,-1);
        q::projection_batch(dgu,WGU,reinterpret_cast<const std::uint16_t*>(px),
                            pgu,groups,GU,N,PAD,sched,s);
        swiglu_ragged<<<(groups*PAD*FF+BLOCK-1)/BLOCK,BLOCK,0,s>>>(
            pgu,ds,ph,groups,-1);
        q::projection_batch(dd,WD,reinterpret_cast<const std::uint16_t*>(ph),
                            py,groups,N,FF,PAD,sched,s);
        scatter_ragged<<<(groups*PAD*N+BLOCK-1)/BLOCK,BLOCK,0,s>>>(
            py,db,reinterpret_cast<float*>(batch),groups,-1);
        ck(cudaGetLastError());
    };
    const auto validate=[&] (bool changed) {
        ck(cudaMemsetAsync(serial,0xa5,out_bytes+GUARD,s));
        ck(cudaMemsetAsync(batch,0xa5,out_bytes+GUARD,s));
        per_expert();grouped();ck(cudaStreamSynchronize(s));
        std::vector<std::uint8_t> a(out_bytes+GUARD),b(out_bytes+GUARD);
        ck(cudaMemcpy(a.data(),serial,a.size(),cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(b.data(),batch,b.size(),cudaMemcpyDeviceToHost));
        for(std::size_t j=out_bytes;j<a.size();++j)
            if(a[j]!=0xa5||b[j]!=0xa5)throw std::runtime_error("output guard changed");
        const float *af=reinterpret_cast<const float*>(a.data()),
                    *bf=reinterpret_cast<const float*>(b.data());
        double num=0,den=0;float largest=0;
        for(std::size_t i=0;i<std::size_t(rows)*N;++i) {
            if(!std::isfinite(af[i])||!std::isfinite(bf[i]))
                throw std::runtime_error("nonfinite valid row output");
            const double d=double(af[i])-double(bf[i]);
            num+=d*d;den+=double(af[i])*af[i];
            largest=std::max(largest,std::abs(af[i]-bf[i]));
        }
        const double rel=std::sqrt(num/std::max(den,1e-50));
        if(rel>1e-6 || largest>0.02f)
            throw std::runtime_error("grouped ragged output parity failed: relative L2="+std::to_string(rel));
        std::printf("%s\"parity%s_l2\":%.12g,\"max_abs%s\":%.9g,\"guards%s\":true",
                    changed?",":"",changed?"_changed":"",rel,changed?"_changed":"",largest,changed?"_changed":"");
    };
    std::printf("{\"scope\":\"isolated synthetic ragged QPN, full gate/up SwiGLU down plus gather/scatter\","
                "\"source\":\"synthetic deterministic GGUF blocks, not real model/TTFT\","
                "\"groups\":%d,\"valid_rows\":%d,\"padded_rows\":%d,"
                "\"distinct_weight_bytes\":%zu,",
                groups,rows,groups*PAD,hgu.size()+hd.size());
    validate(false);
    // The same tensors with changed input must still agree; no stale padded rows.
    input[0]=(input[0]==0x3800?0xb800:0x3800);
    ck(cudaMemcpyAsync(dx,input.data(),input.size()*2,cudaMemcpyHostToDevice,s));
    validate(true);
    const int reps=sanitizer?1:12;
    const Report ref=measure(s,per_expert,reps),batched=measure(s,grouped,reps);
    std::printf(",\"serial_ordinary_us\":%.6f,\"batch_ordinary_us\":%.6f,"
                "\"serial_graph_us\":%.6f,\"batch_graph_us\":%.6f,"
                "\"ordinary_speedup\":%.6f,\"graph_speedup\":%.6f}\n",
                ref.ordinary_us,batched.ordinary_us,ref.graph_us,batched.graph_us,
                ref.ordinary_us/batched.ordinary_us,ref.graph_us/batched.graph_us);
    ck(cudaStreamSynchronize(s));
    cudaFree(batch);cudaFree(serial);cudaFree(py);cudaFree(pgu);cudaFree(ph);cudaFree(px);
    cudaFree(ds);cudaFree(db);cudaFree(ids);cudaFree(dx);cudaFree(dd);cudaFree(dgu);
    cudaStreamDestroy(s);
    return 0;
} catch(const std::exception& e) {
    std::fprintf(stderr,"ragged batch probe FAILED: %s\n",e.what());return 1;
}
