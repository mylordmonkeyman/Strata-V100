// SPDX-License-Identifier: MIT
#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include "nvfp4_volta_qpn_core.cuh"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels::qpn {
namespace {
void check() {
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("NVFP4 QPN launch: ") + cudaGetErrorString(e));
}
constexpr unsigned threads = 256;
unsigned grid(std::size_t count) {
    if (!count || count > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("QPN staging count must fit a positive int");
    return unsigned((count + threads - 1) / threads);
}
template<bool Q8>
__global__ void stage_kernel(const void* input, half* output, std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float value;
    if constexpr (Q8) {
        const auto* block = static_cast<const std::uint8_t*>(input) + i / 32 * 36;
        const float d = __half2float(*reinterpret_cast<const half*>(block));
        value = d * float(reinterpret_cast<const std::int8_t*>(block + 4)[i % 32]);
    } else value = static_cast<const float*>(input)[i];
    output[i] = __float2half_rn(value);
}
__global__ void swiglu_kernel(const float* gate, const float* up, const float* tail,
                             float* hidden, std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = gate[i] * tail[0], u = up[i] * tail[1];
    // Stable sigmoid in FP32; avoids exp overflow on a large negative g.
    const float e = expf(-fabsf(g));
    const float sigmoid = g >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
    hidden[i] = (g * sigmoid) * u;
}
__global__ void down_kernel(float* output, const float* tail, std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] *= tail[2];
}
__global__ void probe_kernel(float* output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 128*16) {
        const half2 w = __hmul2(__float2half2_rn(detail::decode_code(i % 16)),
                               __float2half2_rn(detail::decode_scale(i / 16)));
        output[i] = __low2float(w);
    }
#endif
}
__global__ void pair_probe_kernel(float* output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<128*256) {
        const unsigned pair=(i&15u)|(((i>>4)&15u)<<16);
        const half2 values=__hmul2(detail::decode_pair(pair),__float2half2_rn(detail::decode_scale(i/256)));
        const float2 v=__half22float2(values); output[2*i]=v.x; output[2*i+1]=v.y;
    }
#endif
}
template<int Tiles, int Split, int Acc>
void launch(const std::uint8_t* w, const std::uint16_t* x, float* y,
            int n, int k, int t, int decoder, int groups, std::size_t weight_stride, cudaStream_t stream) {
    // Only sm_70 code is emitted for this standalone library.
    if (decoder == 1)
        detail::nvfp4_volta_qpn_gemm_kernel<Tiles, Split, Acc, true><<<dim3(n / 32,groups,(t+Tiles*8-1)/(Tiles*8)), Split * 32, 0, stream>>>(w, reinterpret_cast<const half*>(x), n, k, t, y,weight_stride);
    else
        detail::nvfp4_volta_qpn_gemm_kernel<Tiles, Split, Acc, false><<<dim3(n / 32,groups,(t+Tiles*8-1)/(Tiles*8)), Split * 32, 0, stream>>>(w, reinterpret_cast<const half*>(x), n, k, t, y,weight_stride);
}
template<int Tiles, int Split>
void launch_acc(const std::uint8_t* w, const std::uint16_t* x, float* y,
                int n, int k, int t, Schedule s, int groups, std::size_t weight_stride, cudaStream_t stream) {
    if (s.accumulators == 1) launch<Tiles, Split, 1>(w, x, y, n, k, t, s.decoder,groups,weight_stride,stream);
    else launch<Tiles, Split, 2>(w, x, y, n, k, t, s.decoder,groups,weight_stride,stream);
}
template<int Tiles>
void launch_split(const std::uint8_t* w, const std::uint16_t* x, float* y,
                  int n, int k, int t, Schedule s, int groups, std::size_t weight_stride, cudaStream_t stream) {
    if (s.split_k == 4) launch_acc<Tiles, 4>(w, x, y, n, k, t, s,groups,weight_stride,stream);
    else if (s.split_k == 8) launch_acc<Tiles, 8>(w, x, y, n, k, t, s,groups,weight_stride,stream);
    else {
        // 16 warps * 4 tiles needs 64 KiB static shared memory: never instantiate.
        if constexpr (Tiles < 4) launch_acc<Tiles, 16>(w, x, y, n, k, t, s,groups,weight_stride,stream);
    }
}
} // namespace
void projection(const std::uint8_t* w, const std::uint16_t* x, float* y,
                int n, int k, int t, Schedule s, void* stream) {
    projection_batch(w,0,x,y,1,n,k,t,s,stream);
}
void projection_batch(const std::uint8_t* w, std::size_t weight_stride, const std::uint16_t* x, float* y,
                      int groups, int n, int k, int t, Schedule s, void* stream) {
    if (!w || !x || !y || !supported(70, n, k, t, s) || groups<1 || groups>65535 ||
        (reinterpret_cast<std::uintptr_t>(w)&3u) || (reinterpret_cast<std::uintptr_t>(x)&15u) ||
        (reinterpret_cast<std::uintptr_t>(y)&3u) || (weight_stride&3u) ||
        (groups>1 && weight_stride<std::size_t(n)*k/64*36))
        throw std::invalid_argument("unsupported standalone NVFP4 QPN projection");
    auto cs = static_cast<cudaStream_t>(stream);
    const int tile_rows = s.token_tile ? s.token_tile : (t <= 8 ? 8 : t <= 16 ? 16 : 32);
    if (tile_rows == 8) launch_split<1>(w, x, y, n, k, t, s,groups,weight_stride,cs);
    else if (tile_rows == 16) launch_split<2>(w, x, y, n, k, t, s,groups,weight_stride,cs);
    else launch_split<4>(w, x, y, n, k, t, s,groups,weight_stride,cs);
    check();
}
void stage_f32(const float* input, std::uint16_t* output, std::size_t count, void* stream) {
    stage_kernel<false><<<grid(count), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        input, reinterpret_cast<half*>(output), count); check();
}
void stage_q8_1(const void* input, std::uint16_t* output, std::size_t count, void* stream) {
    if (count % 32) throw std::invalid_argument("QPN Q8_1 staging requires whole blocks");
    stage_kernel<true><<<grid(count), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        input, reinterpret_cast<half*>(output), count); check();
}
void swiglu(const float* g, const float* u, const float* tail, float* h, std::size_t count, void* stream) {
    swiglu_kernel<<<grid(count), threads, 0, static_cast<cudaStream_t>(stream)>>>(g, u, tail, h, count); check();
}
void scale_down(float* output, const float* tail, std::size_t count, void* stream) {
    down_kernel<<<grid(count), threads, 0, static_cast<cudaStream_t>(stream)>>>(output, tail, count); check();
}
void decoder_probe(float* output, void* stream) {
    probe_kernel<<<8, 256, 0, static_cast<cudaStream_t>(stream)>>>(output); check();
}
void decoder_pair_probe(float* output, void* stream) {
    pair_probe_kernel<<<128,256,0,static_cast<cudaStream_t>(stream)>>>(output); check();
}
} // namespace strata::kernels::qpn
