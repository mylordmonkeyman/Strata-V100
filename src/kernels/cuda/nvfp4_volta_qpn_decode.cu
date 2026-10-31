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
// Independent diagnostic for ggml's block_q8_1_mmq D4 layout (144 bytes).
// NOT the legacy linear 36-byte Q8_1 block used by stage_q8_1.
struct alignas(4) MmqD4Block {
    float d4[4];
    std::int8_t qs[128];
};
static_assert(sizeof(MmqD4Block) == 144, "MMQ D4 ABI changed");

__global__ void stage_mmq_d4_kernel(const MmqD4Block* input, half* output,
                                     std::size_t count, int cols, int ne1,
                                     unsigned int* invalid_count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const std::size_t row = i / std::size_t(cols);
    const int col = int(i % std::size_t(cols));
    // ggml quantize_mmq_q8_1: ib = (column / 128) * ne1 + row,
    // both for scattered compact X and non-scattered hidden H.
    const MmqD4Block& b = input[std::size_t(col / 128) * ne1 + row];
    const int within = col % 128;
    const float reconstructed = b.d4[within / 32] * float(b.qs[within]);
    if (!isfinite(reconstructed) || fabsf(reconstructed) > 65504.0f) {
        // Fail-closed diagnostic: caller MUST check the pre-zeroed device
        // invalid_count after synchronization; this output is unusable if nonzero.
        atomicAdd(invalid_count, 1u);
        output[i] = __float2half_rn(0.0f);
        return;
    }
    output[i] = __float2half_rn(reconstructed);
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
__global__ void swiglu_stage_half_kernel(const float* gu, const float* tail,
                                       half* hidden, std::size_t count, int ff) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i>=count) return;
    const std::size_t row=i/std::size_t(ff), col=i%std::size_t(ff);
    const float* r=gu+row*std::size_t(ff*2);
    const float g=r[col]*tail[0], u=r[std::size_t(ff)+col]*tail[1];
    const float e=expf(-fabsf(g));
    const float sigmoid=g>=0 ? 1.0f/(1.0f+e) : e/(1.0f+e);
    hidden[i]=__float2half_rn((g*sigmoid)*u);
}
// Ragged staged-prefill experiment: every CTA owns an output index; padded
// 32-row experts have no valid output owner beyond their actual bounds.
__global__ void ragged_gather_half_kernel(const half* source, const int32_t* source_rows,
                                           const int32_t* bounds, half* packed,
                                           int groups, int source_tokens, int total_rows) {
    constexpr int width=2560, pad=32;
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if(i>=std::size_t(groups)*pad*width)return;
    const int expert=int(i/(pad*width)), local=int(i%(pad*width));
    const int row=local/width, col=local%width;
    const int start=bounds[expert],end=bounds[expert+1];
    if(start<0 || end<start || end-start>pad || end>total_rows)
        asm volatile("trap;");
    if(row>=end-start) {
        packed[i]=__float2half_rn(0.f);
        return;
    }
    const int src=source_rows[start+row];
    if(src<0 || src>=source_tokens)asm volatile("trap;");
    packed[i]=source[std::size_t(src)*width+col];
}
__global__ void ragged_swiglu_stage_half_kernel(const float* gu, const float* scales,
                                                 half* hidden, int groups) {
    constexpr int pad=32, ff=640, gu_n=2*ff;
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if(i>=std::size_t(groups)*pad*ff)return;
    const int expert=int(i/(pad*ff)), inner=int(i%(pad*ff));
    const int row=inner/ff,col=inner%ff;
    const float* r=gu+(std::size_t(expert)*pad+row)*gu_n;
    const float g=r[col]*scales[expert*4],u=r[ff+col]*scales[expert*4+1];
    const float ex=expf(-fabsf(g));
    const float sigmoid=g>=0 ? 1.0f/(1.0f+ex) : ex/(1.0f+ex);
    hidden[i]=__float2half_rn((g*sigmoid)*u);
}
__global__ void ragged_scatter_f32_kernel(const float* packed,const int32_t* bounds,
                                           float* output,int groups,int total_rows) {
    constexpr int width=2560,pad=32;
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if(i>=std::size_t(groups)*pad*width)return;
    const int expert=int(i/(pad*width)),inner=int(i%(pad*width));
    const int start=bounds[expert],end=bounds[expert+1],row=inner/width;
    if(start<0 || end<start || end-start>pad || end>total_rows)
        asm volatile("trap;");
    if(row<end-start)output[(std::size_t(start)+row)*width+inner%width]=packed[i];
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
void projection_chunkmajor_batch(const std::uint8_t* w, std::size_t stride,
                                 const std::uint16_t* x, float* y,
                                 int groups, int n, int k, int t, Schedule s, void* stream) {
    if (!w || !x || !y || !supported(70,n,k,t,s) || groups<1 || groups>65535 ||
        s.split_k!=4 || s.accumulators!=1 || s.decoder!=1 ||
        (reinterpret_cast<std::uintptr_t>(w)&3u) ||
        (reinterpret_cast<std::uintptr_t>(x)&15u) ||
        (reinterpret_cast<std::uintptr_t>(y)&3u) ||
        (stride&3u) || (groups>1 && stride<std::size_t(n)*k/64*36))
        throw std::invalid_argument("unsupported chunkmajor diagnostic projection");
    const int tile=s.token_tile?s.token_tile:(t<=8?8:t<=16?16:32);
    const cudaStream_t cs=static_cast<cudaStream_t>(stream);
    if(tile==8)
        detail::nvfp4_volta_qpn_gemm_kernel<1,4,1,true,true>
            <<<dim3(n/32,groups,(t+7)/8),128,0,cs>>>(
                w,reinterpret_cast<const half*>(x),n,k,t,y,stride);
    else if(tile==16)
        detail::nvfp4_volta_qpn_gemm_kernel<2,4,1,true,true>
            <<<dim3(n/32,groups,(t+15)/16),128,0,cs>>>(
                w,reinterpret_cast<const half*>(x),n,k,t,y,stride);
    else throw std::invalid_argument("chunkmajor accepts only 8/16-row tiles");
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
void stage_mmq_d4_q8_1_to_fp16(const void* input, std::size_t input_bytes,
                                  std::uint16_t* output, int rows, int columns,
                                  int ne1, unsigned int* invalid_count, void* stream) {
    if (!input || !output || !invalid_count || rows < 1 || columns < 128 ||
        columns % 128 != 0 || ne1 < rows ||
        (reinterpret_cast<std::uintptr_t>(input) & 3u) ||
        (reinterpret_cast<std::uintptr_t>(output) & 1u) ||
        (reinterpret_cast<std::uintptr_t>(invalid_count) & 3u) ||
        std::size_t(rows) > std::size_t(std::numeric_limits<int>::max()) / std::size_t(columns))
        throw std::invalid_argument("invalid experimental MMQ D4 stage shape/pointers");
    const std::size_t blocks = std::size_t(columns) / 128;
    if (std::size_t(ne1) > std::numeric_limits<std::size_t>::max() / (blocks * 144) ||
        input_bytes != blocks * std::size_t(ne1) * 144)
        throw std::invalid_argument("MMQ D4 byte count/pitch mismatch");
    const std::size_t count = std::size_t(rows) * std::size_t(columns);
    stage_mmq_d4_kernel<<<grid(count), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const MmqD4Block*>(input), reinterpret_cast<half*>(output),
        count, columns, ne1, invalid_count);
    check();
}
void swiglu(const float* g, const float* u, const float* tail, float* h, std::size_t count, void* stream) {
    swiglu_kernel<<<grid(count), threads, 0, static_cast<cudaStream_t>(stream)>>>(g, u, tail, h, count); check();
}
void swiglu_stage_half(const float* gu, const float* tail, std::uint16_t* hidden_half,
                       int rows, int ff, void* stream) {
    if (!gu || !tail || !hidden_half || rows<1 || rows>max_tokens || ff!=640)
        throw std::invalid_argument("invalid QPN widened GU stage");
    const std::size_t count=std::size_t(rows)*std::size_t(ff);
    swiglu_stage_half_kernel<<<grid(count),threads,0,static_cast<cudaStream_t>(stream)>>>(
        gu,tail,reinterpret_cast<half*>(hidden_half),count,ff);
    check();
}
void ragged_gather_half(const std::uint16_t* source,const int32_t* source_rows,
                        const int32_t* bounds,std::uint16_t* packed,
                        int groups,int source_tokens,int total_rows,void* stream) {
    if(!source||!source_rows||!bounds||!packed||groups<1||groups>16||
       source_tokens<1||total_rows<1)
        throw std::invalid_argument("invalid grouped QPN gather shape");
    constexpr std::size_t count_per_group=32u*2560u;
    ragged_gather_half_kernel<<<grid(std::size_t(groups)*count_per_group),threads,0,
        static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const half*>(source),source_rows,bounds,
        reinterpret_cast<half*>(packed),groups,source_tokens,total_rows);
    check();
}
void ragged_swiglu_stage_half(const float* gu,const float* scales,
                              std::uint16_t* hidden,int groups,void* stream) {
    if(!gu||!scales||!hidden||groups<1||groups>16)
        throw std::invalid_argument("invalid grouped QPN SwiGLU shape");
    ragged_swiglu_stage_half_kernel<<<grid(std::size_t(groups)*32*640),threads,0,
        static_cast<cudaStream_t>(stream)>>>(
        gu,scales,reinterpret_cast<half*>(hidden),groups);
    check();
}
void ragged_scatter_f32(const float* packed,const int32_t* bounds,
                        float* output,int groups,int total_rows,void* stream) {
    if(!packed||!bounds||!output||groups<1||groups>16||total_rows<1)
        throw std::invalid_argument("invalid grouped QPN scatter shape");
    ragged_scatter_f32_kernel<<<grid(std::size_t(groups)*32*2560),threads,0,
        static_cast<cudaStream_t>(stream)>>>(
        packed,bounds,output,groups,total_rows);
    check();
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
