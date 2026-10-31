#pragma once
// Independent source-to-quantized-D4 validation for synthetic M3 fixtures.
// Host-only C++ (also included by the manual CUDA parity test).
// No model, weight, expert residency or private activation data.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qpn_m3 {
struct D4Block {
    float d4[4];
    std::int8_t qs[128];
};
static_assert(sizeof(D4Block) == 144, "MMQ D4 ABI changed");

inline void verify_quantized_source(const std::vector<std::uint8_t>& packed,
                                    const std::vector<float>& source, int tokens,
                                    int rows, int columns, bool scattered) {
    constexpr std::array<int, 6> compact_to_token = {0, 2, 1, 0, 1, 2};
    if (tokens <= 0 || columns <= 0 || columns % 128 ||
        (scattered ? (tokens != 3 || rows != 6) : (rows != tokens)))
        throw std::invalid_argument("invalid synthetic D4 reference shape");
    const auto source_elems = std::size_t(tokens) * std::size_t(columns);
    const auto padded_cols = (std::size_t(columns) + 511) / 512 * 512;
    const auto physical_bytes = padded_cols / 128 * std::size_t(rows) * sizeof(D4Block);
    if (source.size() != source_elems || packed.size() < physical_bytes)
        throw std::invalid_argument("incomplete synthetic D4 source/buffer");
    for (int row = 0; row < rows; ++row) {
        const int token = scattered ? compact_to_token[std::size_t(row)] : row;
        for (int begin = 0; begin < columns; begin += 32) {
            const auto* input = source.data() + std::size_t(token) * columns + begin;
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(input[j]));
            if (!(amax > 0.0f) || !std::isfinite(amax))
                throw std::runtime_error("synthetic input has invalid Q8 group");
            D4Block block{};
            const std::size_t off = (std::size_t(begin / 128) * rows + row) * sizeof(D4Block);
            std::memcpy(&block, packed.data() + off, sizeof(block));
            const int quarter = (begin % 128) / 32;
            const float scale = block.d4[quarter];
            const float expected_scale = 1.0f / (127.0f / amax);
            if (!std::isfinite(scale) ||
                std::abs(scale - expected_scale) > std::max(1e-6f, expected_scale * 0.0001f))
                throw std::runtime_error("MMQ D4 scale disagrees with original FP32");
            for (int j = 0; j < 32; ++j) {
                const int code = int(block.qs[begin % 128 + j]);
                if (code < -127 || code > 127)
                    throw std::runtime_error("invalid MMQ D4 Q8 code");
                const float reconstructed = float(code) * scale;
                if (!std::isfinite(reconstructed) ||
                    std::abs(reconstructed - input[j]) > amax / 254.0f + 0.001f)
                    throw std::runtime_error("MMQ D4 quantization differs from original FP32");
            }
        }
    }
}
} // namespace qpn_m3
