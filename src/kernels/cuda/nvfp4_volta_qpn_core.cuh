// SPDX-License-Identifier: Apache-2.0
// Adapted from geoffwatts/ninfer-v100 b37d0dd3e1163b9d802d8bccfa89918bf68d793e
// src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh and ops/common/volta_mma.cuh.
// QPN schedule/fragment geometry originates in dnv2003/v100-skinny.
// Modified 2026-10-28: GGUF interleaved decoder, FP16-only activations,
// FP32 outputs, no ninfer dependencies, no prepacked mode or global divisor.
// See NOTICE and LICENSES/Apache-2.0.txt. Original algorithm comments retained below.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
namespace strata::kernels::qpn::detail {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
struct Nvfp4VoltaQpnSchedule {
    static constexpr int kColsPerCta = 32, kRowsPerTile = 8, kChunkK = 64, kGroupK = 16;
};
// ggml UE4M3 semantics; exact FP32 values. Maximum decoded weight = 6*448=2688,
// which is representable in FP16; tensor-global scales remain FP32 outside MMA.
__device__ __forceinline__ float decode_scale(std::uint8_t b) {
    if (b == 0 || b == 0x7f) return 0;
    const unsigned exp = (b >> 3) & 15, man = b & 7;
    return exp == 0 ? float(man) / 512.0f
                    : __uint_as_float(((exp + 120u) << 23) | (man << 20));
}
__device__ __forceinline__ float decode_code(unsigned b) {
    // Exact E2M1 -> FP16 bit pattern, including negative zero. Unlike TurboMind's
    // rebias decoder this uses no copied shift+rebias logic or global divisor.
    const unsigned mag = b & 7;
    const unsigned bits = mag * 0x3800u;
    // For mag>=2: exponent=(mag/2)+14; mantissa=(mag&1)*512.
    const unsigned normal = ((mag / 2 + 14) << 10) | ((mag & 1u) << 9);
    return __half2float(__ushort_as_half((b & 8 ? 0x8000u : 0u) | (mag < 2 ? bits : normal)));
}
// Independent E2M1 lookup in registers. LUT bytes are the upper bytes of the
// eight positive FP16 bit patterns; their low bytes are all zero. PRMT inserts
// literal zero bytes via LUT entry 0 and signs are ORed into FP16 sign bits.
// Input nibbles occupy bits 0..3 and 16..19. No FP32 conversion or rebias.
__device__ __forceinline__ half2 decode_pair(unsigned pair) {
    const unsigned selectors = ((pair & 7u) << 4) | ((pair >> 4) & 0x7000u);
    unsigned bits = __byte_perm(0x3e3c3800u, 0x46444240u, selectors) | ((pair & 0x00080008u) << 12);
    return *reinterpret_cast<half2*>(&bits);
}
__device__ __forceinline__ void stage_pairs(const uint4& v, half2* a) {
    unsigned bits[4] = {__byte_perm(v.x,v.z,0x5410), __byte_perm(v.x,v.z,0x7632),
                        __byte_perm(v.y,v.w,0x5410), __byte_perm(v.y,v.w,0x7632)};
#pragma unroll
    for (int j=0;j<4;++j) a[j]=*reinterpret_cast<half2*>(&bits[j]);
}
__device__ __forceinline__ void volta_mma_qp_n(float (&d)[8], unsigned a0, unsigned a1,
                                               unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
template <int kTiles, int SPLITK, int NACC, bool Fast>
__global__ __launch_bounds__(
    SPLITK * 32, (kTiles == 1 ? 32 : kTiles == 2 ? 16 : 4) / SPLITK < 1
        ? 1
        : (kTiles == 1 ? 32 : kTiles == 2 ? 16 : 4) / SPLITK) void nvfp4_volta_qpn_gemm_kernel(
    const std::uint8_t* __restrict__ weights, const half* __restrict__ x,
    int n, int k, int t, float* __restrict__ output, std::size_t weight_stride) {
    using S = Nvfp4VoltaQpnSchedule;
    weights += std::size_t(blockIdx.y)*weight_stride;
    x += std::size_t(blockIdx.y)*t*k;
    output += std::size_t(blockIdx.y)*t*n;
    // grid.z owns disjoint token rows; unchanged when one tile covers T.
    const int row_base = int(blockIdx.z) * kTiles * S::kRowsPerTile;

    __shared__ float cs[SPLITK][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col  = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const int good = col < n;
    const int use_col = good ? col : 0;

    const int scale_tiles = k / S::kChunkK;
    const int stq          = scale_tiles / SPLITK;
    // For K=2560 / 64=40, split16 must not put the ten remainder chunks
    // in the last warp (the previous schedule's longest path stayed at ten).
    // Balanced ranges cover each chunk once and differ by at most one chunk.
    // Retain the old partition in the scalar control.
    const int st0 = Fast ? scale_tiles * warp / SPLITK : warp * stq;
    const int stend = Fast ? scale_tiles * (warp+1) / SPLITK : (warp == SPLITK-1 ? scale_tiles : st0+stq);

    // GGUF row pitch is K/64 * 36, NOT a separate code plane.
    const std::uint8_t* wrow = weights + static_cast<std::int64_t>(use_col) * (k / 64) * 36;

    float c[kTiles][NACC][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 0; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { c[tile][a][i] = 0.0f; }
        }
    }

    for (int st = st0; st < stend; ++st) {
        const std::uint8_t* block = wrow + static_cast<std::int64_t>(st) * 36;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            // Each 16-value sub-block has eight bytes: low nibbles k0..7,
            // high nibbles k8..15. Construct (j,j+4) pairs for BOTH halves,
            // matching ninfer's activation permutation without repacking slots.
            const half2 sc2 = __float2half2_rn(decode_scale(block[g]));
            const std::uint8_t* qs = block + 4 + g * 8;
            half2 b[8];
            if constexpr (Fast) {
                // A GGUF sub-block is 4-byte aligned, not necessarily 8-byte
                // aligned: use two U32 loads, never an unaligned uint2/uint4.
                const unsigned lo=__ldg(reinterpret_cast<const unsigned*>(qs));
                const unsigned hi=__ldg(reinterpret_cast<const unsigned*>(qs+4));
#pragma unroll
                for (int j=0;j<4;++j) {
                    const unsigned pair=__byte_perm(lo,hi,unsigned(j)|((unsigned(j)+4)<<8));
                    b[j]=__hmul2(decode_pair(pair),sc2);
                    b[j+4]=__hmul2(decode_pair(pair>>4),sc2);
                }
            } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int shift = j < 4 ? 0 : 4;
                const int q = j & 3;
                b[j] = __hmul2(__floats2half2_rn(decode_code((qs[q] >> shift) & 15),
                                                   decode_code((qs[q+4] >> shift) & 15)), sc2);
            }
            }
            const unsigned* B = reinterpret_cast<const unsigned*>(b);
            const int kbase   = st * S::kChunkK + g * S::kGroupK;

#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                const int row = row_base + tile * S::kRowsPerTile + r;
                half2 a[8];
                if (row < t) {
                    const half* xrow = x + static_cast<std::int64_t>(row) * k + kbase;
                    if constexpr (Fast) {
                        const uint4 v0=*reinterpret_cast<const uint4*>(xrow);
                        const uint4 v1=*reinterpret_cast<const uint4*>(xrow+8);
                        stage_pairs(v0,&a[0]); stage_pairs(v1,&a[4]);
                    } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int base = j < 4 ? 0 : 8;
                        a[j] = __halves2half2(xrow[base + (j & 3)], xrow[base + (j & 3) + 4]);
                    }
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { a[j] = __half2half2(__ushort_as_half(0)); }
                }
                const unsigned* A = reinterpret_cast<const unsigned*>(a);
                volta_mma_qp_n(c[tile][0 % NACC], A[0], A[1], B[0], B[1]);
                volta_mma_qp_n(c[tile][1 % NACC], A[2], A[3], B[2], B[3]);
                volta_mma_qp_n(c[tile][2 % NACC], A[4], A[5], B[4], B[5]);
                volta_mma_qp_n(c[tile][3 % NACC], A[6], A[7], B[6], B[7]);
            }
        }
    }

    // Fold the independent accumulator chains together before the cross-warp reduce.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 1; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { c[tile][0][i] += c[tile][a][i]; }
        }
    }

    // C map (v100-skinny mma8_probe.cu, roles swapped): identical to the other three siblings.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(tile * S::kRowsPerTile + row) * S::kColsPerCta + qp * 8 + cl] = c[tile][0][i];
        }
    }
    __syncthreads(); // the only barrier: cross-warp K reduce

    constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
    for (int e = static_cast<int>(threadIdx.x); e < kOut; e += SPLITK * 32) {
        const int row  = e / S::kColsPerCta;
        const int cl   = e % S::kColsPerCta;
        const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + cl;
        if (row_base + row < t && ocol < n) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < SPLITK; ++w) { v += cs[w][e]; }
            output[static_cast<std::int64_t>(row_base + row) * n + ocol] = v;
        }
    }
}


#endif
} // namespace strata::kernels::qpn::detail
