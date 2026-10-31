#include "../../src/prefill/qpn_r0_intervals.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
using strata::prefill::r0::Window;
using strata::prefill::r0::duration;
using strata::prefill::r0::overlap;
static bool nearly(double a, double b) { return std::fabs(a - b) < 1e-9; }
int main() {
    assert(nearly(duration({}), 0));
    assert(nearly(duration({{2, 9}, {1, 4}, {12, 13}, {3, 5}}), 9));
    assert(nearly(overlap({{1, 4}, {3, 7}}, {{2, 5}, {6, 8}}), 4));
    assert(nearly(overlap({{0, 3}}, {{4, 8}}), 0));
    assert(nearly(overlap({{0, 1}}, {{1, 2}}), 0));
    assert(nearly(overlap({{0, 1}, {1, 2}}, {{0, 2}}), 2));
    for (const auto& bad : {Window{2, 1}, Window{0, std::numeric_limits<double>::infinity()},
                            Window{std::numeric_limits<double>::quiet_NaN(), 2}}) {
        try { (void) duration({bad}); assert(false); }
        catch (const std::invalid_argument&) {}
    }
}
