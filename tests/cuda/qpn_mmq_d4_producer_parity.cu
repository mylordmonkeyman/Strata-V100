// Manual-only M3 qualification: actual pinned ggml MMQ D4 producer to QPN stage.
// All inputs synthetic; no model, expert weights, route state, or benchmark.
// No automatic CTest registration: existing V100 gpu.lock/idleness gate required.
#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include "../host/qpn_d4_source_check.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;
namespace qpn = strata::kernels::qpn;

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    void* ptr = nullptr;
    explicit Buffer(std::size_t count) { ck(cudaMalloc(&ptr, count)); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
using Block = qpn_m3::D4Block;

std::uint16_t half_bits(float f) {
    const half result = __float2half_rn(f);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &result, 2);
    return bits;
}
std::vector<float> activation(int rows, int columns) {
    std::vector<float> v(std::size_t(rows) * columns);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < columns; ++col) {
            // Nonzero, exactly representable source values. Every 32 values
            // include differing magnitudes; no all-zero quantizer blocks.
            const int scaled = ((col * 17 + row * 29) % 251) - 125;
            v[std::size_t(row) * columns + col] = (scaled == 0 ? 1 : scaled) / 16.0f;
        }
    return v;
}
std::vector<std::uint16_t> expected_from_producer(const std::vector<std::uint8_t>& q8,
                                                    int rows, int columns) {
    std::vector<std::uint16_t> out(std::size_t(rows) * columns);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < columns; ++col) {
            const std::size_t offset = (std::size_t(col / 128) * rows + row) * 144;
            Block b{};
            std::memcpy(&b, q8.data() + offset, sizeof(b));
            const int j = col % 128;
            const float value = b.d4[j / 32] * float(b.qs[j]);
            if (!std::isfinite(value) || std::abs(value) > 65504.0f)
                throw std::runtime_error("actual Q8 producer contained invalid logical D4");
            out[std::size_t(row) * columns + col] = half_bits(value);
        }
    return out;
}
void run(int tokens, int columns, bool scattered) {
    constexpr int k = 2;
    const int rows = scattered ? tokens * k : tokens;
    const auto source = activation(tokens, columns);
    const std::size_t input_bytes = source.size() * sizeof(float);
    const std::size_t q8_bytes = mmq::q8_bytes(rows, columns);
    const std::size_t used = std::size_t(((columns + 511) / 512) * 512 / 128) *
                             rows * sizeof(Block);
    if (q8_bytes < used + 128 * sizeof(Block))
        throw std::runtime_error("MMQ quantized buffer guard/size ABI changed");

    Buffer x(input_bytes), packed(q8_bytes), h(std::size_t(rows) * columns * 2 + 32),
           invalid(sizeof(unsigned)), slots(std::size_t(rows) * sizeof(std::int32_t)),
           source_rows(std::size_t(rows) * sizeof(std::int32_t));
    Stream stream;
    ck(cudaMemcpyAsync(x.ptr, source.data(), input_bytes, cudaMemcpyHostToDevice, stream.value));
    ck(cudaMemsetAsync(packed.ptr, 0xa5, q8_bytes, stream.value));
    ck(cudaMemsetAsync(h.ptr, 0xa5, std::size_t(rows) * columns * 2 + 32, stream.value));
    ck(cudaMemsetAsync(invalid.ptr, 0, sizeof(unsigned), stream.value));

    if (scattered) {
        if (tokens != 3) throw std::invalid_argument("scatter control requires three tokens");
        const std::int32_t order[6] = {0, 3, 2, 4, 1, 5};
        const std::int32_t inverse[6] = {0, 2, 1, 0, 1, 2};
        ck(cudaMemcpyAsync(slots.ptr, order, sizeof(order), cudaMemcpyHostToDevice, stream.value));
        ck(cudaMemcpyAsync(source_rows.ptr, inverse, sizeof(inverse),
                           cudaMemcpyHostToDevice, stream.value));
        // The temporary host mapping arrays are stack-allocated. Finish their
        // H2D transfers before they go out of scope (not a timing benchmark).
        ck(cudaStreamSynchronize(stream.value));
        mmq::quantize_scatter(static_cast<const float*>(x.ptr),
                              static_cast<const std::int32_t*>(slots.ptr),
                              static_cast<const std::int32_t*>(source_rows.ptr), packed.ptr,
                              int(GGML_TYPE_NVFP4), columns, columns, tokens, k, stream.value);
    } else {
        mmq::quantize(static_cast<const float*>(x.ptr), nullptr, packed.ptr,
                      int(GGML_TYPE_NVFP4), columns, columns, tokens, stream.value);
    }
    qpn::stage_mmq_d4_q8_1_to_fp16(packed.ptr, q8_bytes,
        static_cast<std::uint16_t*>(h.ptr), rows, columns, rows,
        static_cast<unsigned*>(invalid.ptr), stream.value);
    ck(cudaStreamSynchronize(stream.value));

    unsigned error_count = 0;
    ck(cudaMemcpy(&error_count, invalid.ptr, sizeof(error_count), cudaMemcpyDeviceToHost));
    if (error_count != 0) throw std::runtime_error("unexpected invalid D4 code on real producer");

    std::vector<std::uint8_t> actual_q8(used);
    ck(cudaMemcpy(actual_q8.data(), packed.ptr, used, cudaMemcpyDeviceToHost));
    qpn_m3::verify_quantized_source(actual_q8, source, tokens, rows, columns, scattered);
    const auto want = expected_from_producer(actual_q8, rows, columns);
    std::vector<std::uint16_t> got(want.size() + 16);
    ck(cudaMemcpy(got.data(), h.ptr, got.size() * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));
    if (!std::equal(want.begin(), want.end(), got.begin()))
        throw std::runtime_error("ggml-generated D4 vs QPN staging mismatch");
    for (std::size_t i = want.size(); i < got.size(); ++i)
        if (got[i] != 0xa5a5) throw std::runtime_error("D4 stage wrote past output");
    if (scattered) {
        const auto segment = [&](int index) { return got.data() + std::size_t(index) * columns; };
        // token0 -> compact 0/3; token1 -> 2/4; token2 -> 1/5.
        if (!std::equal(segment(0), segment(0) + columns, segment(3)) ||
            !std::equal(segment(2), segment(2) + columns, segment(4)) ||
            !std::equal(segment(1), segment(1) + columns, segment(5)) ||
            std::equal(segment(0), segment(0) + columns, segment(2)))
            throw std::runtime_error("ggml scatter D4 duplicate-source mapping changed");
    }
}
} // namespace

int main() try {
    int id = 0;
    ck(cudaGetDevice(&id));
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, id));
    if (prop.major != 7 || prop.minor != 0) {
        std::fprintf(stderr, "ggml D4 producer parity requires sm_70\n");
        return 77;
    }
    // Fail closed on a contaminated runtime. W4A4 produces another format.
    if (mmq::nvfp4_mode() != mmq::Nvfp4Mode::W4A8)
        throw std::runtime_error("actual D4 producer requires W4A8 mode");
    const char* gather = std::getenv("STRATA_QUANT_GATHER");
    if (gather && std::atoi(gather) != 0)
        throw std::runtime_error("scatter qualification requires native scatter mode");

    run(3, 640, false);
    run(3, 2560, false);
    run(3, 640, true);
    std::puts("{\"schema\":1,\"test\":\"real_ggml_mmq_d4_producer_to_qpn_stage\",\"status\":\"passed\",\"cases\":3,\"synthetic_only\":true}");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "ggml-produced D4 parity failed: %s\n", e.what());
    return 1;
}
