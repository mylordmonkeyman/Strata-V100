#pragma once
// Experimental, host-only interval arithmetic used by aggregate R0 diagnostics.
// Inputs are CUDA event elapsed-time envelopes, NOT measured exposed compute or
// hardware occupancy. No route identities, prompts, activations or weight bytes.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace strata::prefill::r0 {
struct Window { double start_ms, end_ms; };

inline std::vector<Window> merged(std::vector<Window> windows) {
    for (const auto& w : windows)
        if (!std::isfinite(w.start_ms) || !std::isfinite(w.end_ms) ||
            w.start_ms > w.end_ms)
            throw std::invalid_argument("invalid diagnostic interval");
    std::sort(windows.begin(), windows.end(), [](const Window& a, const Window& b) {
        return a.start_ms < b.start_ms ||
               (a.start_ms == b.start_ms && a.end_ms < b.end_ms);
    });
    std::vector<Window> out;
    for (const auto& w : windows) {
        if (!out.empty() && w.start_ms <= out.back().end_ms)
            out.back().end_ms = std::max(out.back().end_ms, w.end_ms);
        else out.push_back(w);
    }
    return out;
}
inline double duration(const std::vector<Window>& spans) {
    double total = 0;
    for (const auto& w : merged(spans)) total += w.end_ms - w.start_ms;
    return total;
}
inline double overlap(const std::vector<Window>& a, const std::vector<Window>& b) {
    const auto x = merged(a), y = merged(b);
    std::size_t i = 0, j = 0;
    double result = 0;
    while (i < x.size() && j < y.size()) {
        result += std::max(0.0, std::min(x[i].end_ms, y[j].end_ms) -
                                 std::max(x[i].start_ms, y[j].start_ms));
        if (x[i].end_ms < y[j].end_ms) ++i;
        else ++j;
    }
    return result;
}
} // namespace strata::prefill::r0
