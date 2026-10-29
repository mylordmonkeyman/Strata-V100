// SPDX-License-Identifier: MIT
// Standalone synthetic sm_70 diagnostic: no model weights, prompt, or engine dispatch.
#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include "nvfp4_volta_qpn_core.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>

namespace q = strata::kernels::qpn;
namespace d = strata::kernels::qpn::detail;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__global__ void launch_probe(int* x) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(x, 0);
}
struct Timing { double gpu_us, host_enqueue_us; };
static Timing measure(cudaStream_t s, const std::function<void()>& fn, int iterations) {
    cudaEvent_t start{}, end{};
    ck(cudaEventCreate(&start)); ck(cudaEventCreate(&end));
    fn(); ck(cudaStreamSynchronize(s));
    ck(cudaEventRecord(start, s));
    const auto t0 = std::chrono::steady_clock::now();
    for (int i=0;i<iterations;++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    ck(cudaEventRecord(end, s)); ck(cudaEventSynchronize(end));
    float ms=0; ck(cudaEventElapsedTime(&ms,start,end));
    ck(cudaEventDestroy(start)); ck(cudaEventDestroy(end));
    return {double(ms)*1000.0/iterations,
            std::chrono::duration<double,std::micro>(t1-t0).count()/iterations};
}
int main() try {
    cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop,0));
    if (prop.major!=7 || prop.minor!=0) throw std::runtime_error("requires sm_70");
    cudaStream_t s{}; ck(cudaStreamCreate(&s));
    constexpr int FF=640, N=2560, iters=256;
    constexpr int threads=128, tile=16, warps=threads/32;
    using Kernel = decltype(&d::nvfp4_volta_qpn_gemm_kernel<2,4,1,true,false>);
    Kernel kernel = &d::nvfp4_volta_qpn_gemm_kernel<2,4,1,true,false>;
    cudaFuncAttributes attrs{};
    ck(cudaFuncGetAttributes(&attrs,kernel));
    int max_blocks=0;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,kernel,threads,0));
    // All-zero canonical synthetic weights/activations: this probe is about
    // geometry and launch costs, not representative model accuracy or cache residency.
    std::uint8_t *wgu=nullptr,*wd=nullptr;
    std::uint16_t *x=nullptr,*h=nullptr;
    float *gu=nullptr,*y=nullptr,*tail=nullptr;
    int* counter=nullptr;
    const size_t gu_bytes=size_t(2*FF)*N/64*36;
    const size_t down_bytes=size_t(N)*FF/64*36;
    ck(cudaMalloc(reinterpret_cast<void**>(&wgu),gu_bytes));
    ck(cudaMalloc(reinterpret_cast<void**>(&wd),down_bytes));
    ck(cudaMalloc(reinterpret_cast<void**>(&x),size_t(32)*N*2));
    ck(cudaMalloc(reinterpret_cast<void**>(&h),size_t(32)*FF*2));
    ck(cudaMalloc(reinterpret_cast<void**>(&gu),size_t(32)*2*FF*4));
    ck(cudaMalloc(reinterpret_cast<void**>(&y),size_t(32)*N*4));
    ck(cudaMalloc(reinterpret_cast<void**>(&tail),4*sizeof(float)));
    ck(cudaMalloc(reinterpret_cast<void**>(&counter),sizeof(int)));
    ck(cudaMemsetAsync(wgu,0,gu_bytes,s)); ck(cudaMemsetAsync(wd,0,down_bytes,s));
    ck(cudaMemsetAsync(x,0,size_t(32)*N*2,s)); ck(cudaMemsetAsync(h,0,size_t(32)*FF*2,s));
    ck(cudaMemsetAsync(gu,0,size_t(32)*2*FF*4,s)); ck(cudaMemsetAsync(y,0,size_t(32)*N*4,s));
    ck(cudaMemsetAsync(tail,0,4*sizeof(float),s)); ck(cudaMemsetAsync(counter,0,sizeof(int),s));
    ck(cudaStreamSynchronize(s));
    const auto noop=measure(s,[&]{launch_probe<<<1,128,0,s>>>(counter);},4096);
    std::printf("{\"scope\":\"synthetic single-expert CUDA-event diagnostic, not full-model\","
                "\"sm_count\":%d,\"max_warps_per_sm\":%d,\"threads_per_cta\":%d,"
                "\"registers_per_thread\":%d,\"static_shared_bytes\":%zu,"
                "\"occupancy_api_max_blocks_per_sm\":%d,"
                "\"noop_gpu_timeline_us\":%.5f,\"noop_host_enqueue_us\":%.5f,"
                "\"note\":\"CUDA-event timeline includes host enqueue gaps; CTA warp-capacity figures are geometry estimates, NOT observed active occupancy\","
                "\"cases\":[",
                prop.multiProcessorCount,prop.maxThreadsPerMultiProcessor/32,threads,
                attrs.numRegs,size_t(attrs.sharedSizeBytes),max_blocks,
                noop.gpu_us,noop.host_enqueue_us);
    bool first_case=true;
    for (int t : {17,24,32}) for (int token_tile : {8,16}) {
        const q::Schedule sched{4,1,1,token_tile};
        Kernel case_kernel = token_tile==8
            ? &d::nvfp4_volta_qpn_gemm_kernel<1,4,1,true,false>
            : &d::nvfp4_volta_qpn_gemm_kernel<2,4,1,true,false>;
        cudaFuncAttributes case_attrs{};
        ck(cudaFuncGetAttributes(&case_attrs,case_kernel));
        int case_max_blocks=0;
        ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &case_max_blocks,case_kernel,threads,0));
        auto gu_fn=[&]{q::projection(wgu,x,gu,2*FF,N,t,sched,s);};
        auto down_fn=[&]{q::projection(wd,h,y,N,FF,t,sched,s);};
        const Timing g=measure(s,gu_fn,iters), down=measure(s,down_fn,iters);
        const int gu_ctas=(2*FF/32)*((t+token_tile-1)/token_tile);
        const int down_ctas=(N/32)*((t+token_tile-1)/token_tile);
        const auto cap=[&](int ctas) {
            return 100.0*std::min(double(case_max_blocks),double(ctas)/prop.multiProcessorCount)*
                warps/(prop.maxThreadsPerMultiProcessor/32);
        };
        std::printf("%s{\"tokens\":%d,\"tile\":%d,\"tile_max_blocks_per_sm\":%d,"
                    "\"tile_registers_per_thread\":%d,\"tile_static_shared_bytes\":%zu,"
                    "\"gu_ctas\":%d,\"down_ctas\":%d,"
                    "\"gu_warp_capacity_estimate_pct\":%.4f,\"down_warp_capacity_estimate_pct\":%.4f,"
                    "\"gu_gpu_timeline_us\":%.5f,\"down_gpu_timeline_us\":%.5f,"
                    "\"gu_host_enqueue_us\":%.5f,\"down_host_enqueue_us\":%.5f}",
                    first_case?"":",",t,token_tile,case_max_blocks,case_attrs.numRegs,
                    size_t(case_attrs.sharedSizeBytes),gu_ctas,down_ctas,cap(gu_ctas),cap(down_ctas),
                    g.gpu_us,down.gpu_us,g.host_enqueue_us,down.host_enqueue_us);
        first_case=false;
    }
    std::printf("]}\n");
    ck(cudaStreamSynchronize(s));
    ck(cudaFree(counter)); ck(cudaFree(tail)); ck(cudaFree(y)); ck(cudaFree(gu));
    ck(cudaFree(h)); ck(cudaFree(x)); ck(cudaFree(wd)); ck(cudaFree(wgu));
    ck(cudaStreamDestroy(s));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr,"synthetic QPN diagnostic failed: %s\n",e.what());
    return 1;
}
