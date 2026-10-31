// Host-only regression for the SAME validator used by the device parity test.
#include "qpn_d4_source_check.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>

static std::vector<float> source_values(int tokens, int cols) {
    std::vector<float> source(std::size_t(tokens) * cols);
    for (int t = 0; t < tokens; ++t) for (int c = 0; c < cols; ++c) {
        const int scaled = ((c * 17 + t * 29) % 251) - 125;
        source[std::size_t(t) * cols + c] = (scaled == 0 ? 1 : scaled) / 16.f;
    }
    return source;
}
static std::vector<std::uint8_t> synth_d4(const std::vector<float>& source,
                                           int tokens, int cols, bool scattered) {
    const int rows = scattered ? tokens * 2 : tokens;
    const int padded = (cols + 511) / 512 * 512;
    std::vector<std::uint8_t> bytes(std::size_t(padded / 128) * rows * 144, 0xa5);
    constexpr int map[6] = {0, 2, 1, 0, 1, 2};
    for (int row = 0; row < rows; ++row) {
        const int token = scattered ? map[row] : row;
        for (int col = 0; col < cols; col += 32) {
            float amax = 0.f;
            for (int j = 0; j < 32; ++j)
                amax = std::max(amax, std::abs(source[std::size_t(token) * cols + col + j]));
            const float d_inv = 127.f / amax;
            const float d = 1.f / d_inv;
            const std::size_t off = (std::size_t(col / 128) * rows + row) * 144;
            std::memcpy(bytes.data() + off + ((col % 128) / 32) * 4, &d, 4);
            for (int j = 0; j < 32; ++j) {
                const int code = int(std::round(source[std::size_t(token) * cols + col + j] * d_inv));
                if (code < -127 || code > 127)
                    throw std::runtime_error("invalid source reference Q8 code");
                bytes[off + 16 + (col % 128) + j] = std::uint8_t(std::int8_t(code));
            }
        }
    }
    return bytes;
}
static bool rejects(const std::vector<std::uint8_t>& q8,
                    const std::vector<float>& source, int tokens,
                    int rows, int cols, bool scattered) {
    try { qpn_m3::verify_quantized_source(q8, source, tokens, rows, cols, scattered); }
    catch (const std::runtime_error&) { return true; }
    catch (const std::invalid_argument&) { return true; }
    return false;
}
int main() {
    for (int cols : {128, 256, 640, 2560}) {
        for (bool scattered : {false, true}) {
            const int tokens = 3, rows = scattered ? 6 : 3;
            const auto source = source_values(tokens, cols);
            auto q8 = synth_d4(source, tokens, cols, scattered);
            qpn_m3::verify_quantized_source(q8, source, tokens, rows, cols, scattered);
            auto invalid = q8;
            invalid[0] = 0; invalid[1] = 0; invalid[2] = 0; invalid[3] = 0;
            if (!rejects(invalid, source, tokens, rows, cols, scattered))
                throw std::runtime_error("corrupted synthetic D4 was accepted");
            invalid = q8;
            invalid[16] = 127; // first code should be negative, not +127
            if (!rejects(invalid, source, tokens, rows, cols, scattered))
                throw std::runtime_error("corrupted synthetic D4 was accepted");
            invalid = q8;
            invalid.resize(invalid.size() - 144);
            if (!rejects(invalid, source, tokens, rows, cols, scattered))
                throw std::runtime_error("corrupted synthetic D4 was accepted");
            invalid = q8;
            const int physical = (cols + 511) / 512 * 512;
            if (physical > cols) {
                // Poison extra column only; logical decode must NOT read it.
                const std::size_t off = std::size_t(cols / 128) * rows * 144;
                invalid[off] = 0xff;
                qpn_m3::verify_quantized_source(invalid, source, tokens, rows, cols, scattered);
            }
            if (scattered) {
                invalid = q8;
                invalid[16 + 144] = 127; // mismatched compact row 1
                if (!rejects(invalid, source, tokens, rows, cols, scattered))
                throw std::runtime_error("corrupted synthetic D4 was accepted");
            }
        }
    }
    std::puts("M3 synthetic source-value verifier: 8 cases and corruption guards passed");
}
