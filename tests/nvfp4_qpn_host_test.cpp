#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace q = strata::kernels::qpn;
void require(bool b, const char* what) { if (!b) throw std::runtime_error(what); }
int main() {
    // Independent arithmetic oracle, not a call to the decoder under test.
    constexpr double magnitudes[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    for (unsigned i = 0; i < 16; ++i) {
        const double want = std::copysign(magnitudes[i & 7], i & 8 ? -1 : 1);
        require(q::code_value(i) == want && std::signbit(q::code_value(i)) == std::signbit(want), "E2M1");
    }
    for (unsigned i = 0; i < 256; ++i) {
        const int exp = (i >> 3) & 15, man = i & 7;
        const double want = i == 0 || i == 127 ? 0 : exp == 0 ? std::ldexp(double(man), -9)
                                                      : std::ldexp(1.0 + man / 8.0, exp - 7);
        require(q::scale_value(i) == want, "UE4M3");
    }
    // Different sentinels in every sub-block, row, and super-block. Set every
    // nibble in turn and compare the whole row; catches split-half confusion
    // (including the easily mistaken 32-element half split).
    std::array<std::uint8_t, 3 * 2 * 36> data{};
    for (int row = 0; row < 3; ++row)
        for (int pos = 0; pos < 128; ++pos)
            for (unsigned code = 0; code < 16; ++code) {
                data.fill(0);
                for (int r = 0; r < 3; ++r)
                    for (int b = 0; b < 2; ++b)
                        for (int g = 0; g < 4; ++g) data[r * 72 + b * 36 + g] = 8 * (r + b + g + 1);
                const int base = row * 72 + pos / 64 * 36, sub = pos % 64 / 16;
                data[base + 4 + sub * 8 + pos % 8] = std::uint8_t(code << (pos % 16 < 8 ? 0 : 4));
                for (int r = 0; r < 3; ++r)
                    for (int j = 0; j < 128; ++j) {
                        const double expected = r == row && j == pos
                            ? (code & 8 ? -1 : 1) * magnitudes[code & 7] * std::ldexp(1.0, row + pos/64 + sub - 6) : 0;
                        require(q::weight_value(data.data() + r * 72, j) == expected, "GGUF boundary/nibble order");
                    }
            }
    for (int t : {1, 2, 3, 4, 8, 16, 17, 32})
        for (int sk : {4, 8})
            for (int acc : {1, 2}) require(q::supported(70, 2560, 640, t, {sk, acc}), "down eligibility");
    require(!q::supported(70, 2560, 640, 8, {16, 1}), "down split16 must fail");
    require(!q::supported(70, 640, 2560, 32, {16, 1}), "64 KiB static shared memory must fail");
    require(q::supported(70, 640, 2560, 16, {16, 2}), "gate split16");
    for (int cc : {0, 60, 75, 80, 90}) require(!q::supported(cc, 640, 2560, 8, {}), "non-Volta");
    for (int t : {-1, 0, 33}) require(!q::supported(70, 640, 2560, t, {}), "token bounds");
    require(!q::supported(70, 639, 2560, 8, {}), "N geometry");
    require(!q::supported(70, 640, 639, 8, {}), "K geometry");
    require(!q::supported(70, 640, 2560, 8, {3, 1}), "split bounds");
    require(!q::supported(70, 640, 2560, 8, {4, 3}), "accumulator bounds");
    require(q::supported(70, 640, 2560, 8, {4, 1, 1}), "fast decoder eligibility");
    require(!q::supported(70, 640, 2560, 8, {4, 1, 2}), "decoder bounds");
    std::puts("NVFP4 QPN host format/schedule tests passed");
}
