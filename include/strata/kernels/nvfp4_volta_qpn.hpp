#pragma once

// Standalone experimental interface. No engine dispatch or cache mutation.
// Host interfaces intentionally do not expose CUDA headers.
#include <cstddef>
#include <cstdint>

namespace strata::kernels::qpn {
constexpr int block_values = 64;
constexpr int block_bytes = 36;
// Standalone T>32 up to 512 rows qualification only: the full-model experimental routing
// policy still selects 17..32 experts. The Volta grid.z handles row tiles.
constexpr int max_tokens = 512;

// Canonical GGUF NVFP4: [4 scale bytes | 4 * 8 split-half code bytes].
// UE4M3 matches Strata/ggml, including 0x7f -> zero and ignored sign bit.
// The converter emits only sign-clear scales; audit real artifacts before use.
inline float scale_value(std::uint8_t code) {
    if (code == 0 || code == 0x7f) return 0;
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    return exponent == 0 ? float(mantissa) / 512.0f
                         : float(8 + mantissa) * float(1u << exponent) / 1024.0f;
}
inline float code_value(unsigned code) {
    constexpr float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return code & 8 ? -values[code & 7] : values[code & 7];
}
inline float weight_value(const std::uint8_t* row, int k) {
    const auto* block = row + std::size_t(k / 64) * 36;
    const int sub = (k % 64) / 16, j = k % 16;
    const unsigned code = (block[4 + sub * 8 + j % 8] >> (j < 8 ? 0 : 4)) & 15;
    return code_value(code) * scale_value(block[sub]);
}

struct Schedule {
    int split_k = 4;
    int accumulators = 1;
    int decoder = 0;
    // 0 keeps the old 8/16/32-row schedule. Explicit row tiles use grid.z.
    int token_tile = 0;
};
inline bool supported(int cc, int n, int k, int tokens, Schedule s) {
    return cc == 70 && n > 0 && n % 128 == 0 && k > 0 && k % 64 == 0 &&
           tokens >= 1 && tokens <= max_tokens &&
           (s.split_k == 4 || s.split_k == 8 || s.split_k == 16) &&
           (s.token_tile == 0 || s.token_tile == 8 || s.token_tile == 16 || s.token_tile == 32) &&
           k / 64 >= s.split_k && (s.accumulators == 1 || s.accumulators == 2) && (s.decoder == 0 || s.decoder == 1) &&
           s.split_k * (s.token_tile ? s.token_tile / 8 :
                        tokens <= 8 ? 1 : tokens <= 16 ? 2 : 4) * 1024 <= 49152;
}

// All pointers device-resident, caller-owned; stream is cudaStream_t as void*.
// projection consumes canonical GGUF [n,k] bytes and contiguous FP16 [tokens,k].
// Output [tokens,n] is FP32, with NO tensor-global scale folded into weights.
// Call supported() with the active device capability before any launch. Invalid
// host dimensions/schedules throw; CUDA launch failures throw, never retry.
void projection(const std::uint8_t* weights, const std::uint16_t* x_half,
                float* output, int n, int k, int tokens, Schedule schedule, void* stream);
// Standalone uniform multi-expert experiment, not engine grouped dispatch.
// Distinct matrices start weight_stride bytes apart; X/Y are contiguous
// [groups,tokens,k/n]. No slot rewrite, host count readback, or scatter.
void projection_batch(const std::uint8_t* weights, std::size_t weight_stride,
                      const std::uint16_t* x_half, float* output, int groups,
                      int n, int k, int tokens, Schedule schedule, void* stream);
// Standalone diagnostic only: identical 36-byte compressed blocks, privately
// rearranged [row][K chunk] -> [K chunk][row]. Never engine-dispatched.
void projection_chunkmajor_batch(const std::uint8_t* w, std::size_t weight_stride,
                                 const std::uint16_t* x_half, float* output,
                                 int groups, int n, int k, int tokens,
                                 Schedule schedule, void* stream);
// Explicit input conversion; q8_1 is [half d, half sum, int8 qs[32]], 36 bytes.
void stage_f32(const float* input, std::uint16_t* output, std::size_t count, void* stream);
void stage_q8_1(const void* input, std::uint16_t* output, std::size_t count, void* stream);
// Global scales stay FP32, applied before FP32 SiLU and on the down output.
void swiglu(const float* gate, const float* up, const float* tail,
            float* hidden, std::size_t count, void* stream);
 // Opt-in widened [rows, 2*ff] FP32 GU -> FP16 [rows,ff] hidden.
void swiglu_stage_half(const float* gu, const float* tail, std::uint16_t* hidden_half,
                       int rows, int ff, void* stream);

// Opt-in experimental 16-expert ragged pack/compute/scatter support. The
// caller preserves host-validated exclusive routing ownership. Bounds are
// absolute offsets for the group (groups+1); padded rows never scatter.
// All outputs are caller-owned, no allocations or cache/pack changes.
void ragged_gather_half(const std::uint16_t* source, const int32_t* source_rows,
                        const int32_t* absolute_bounds, std::uint16_t* packed,
                        int groups, int source_token_count, int total_routed_rows,
                        void* stream);
void ragged_swiglu_stage_half(const float* packed_gu, const float* per_expert_scales,
                              std::uint16_t* packed_hidden, int groups, void* stream);
void ragged_scatter_f32(const float* packed, const int32_t* absolute_bounds,
                        float* output, int groups, int total_routed_rows, void* stream);
void scale_down(float* output, const float* tail, std::size_t count, void* stream);
// Standalone qualification probe: output[scale*16+code], 128 canonical scales.
void decoder_probe(float* output, void* stream);
// Exhaustive fast decoder: output[(scale*256+pair)*2+lane], 65536 floats.
void decoder_pair_probe(float* output, void* stream);
} // namespace strata::kernels::qpn
