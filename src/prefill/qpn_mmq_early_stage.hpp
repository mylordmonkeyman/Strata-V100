#pragma once
// Source-only, opt-in MMQ stream-all staging predicate.
// A worker may skip a CPU expert-blob preparation only AFTER the router's
// release-store published this layer's routing flags. The worker must acquire
// that publication BEFORE reading the non-atomic bool. Unpublished jobs are
// always prepared to preserve transfer lookahead and correctness.
// This helper has no CUDA, weight layout, allocator, or model dependencies.
#include <atomic>
#include <cstdint>
namespace strata::prefill::detail {
inline bool qpn_mmq_early_stage_skip(const std::atomic<int64_t>& published_layer,
                                     int64_t layer, const bool* routed) {
    if (!routed || layer < 0) return false;
    if (published_layer.load(std::memory_order_acquire) < layer) return false;
    return !*routed; // read only AFTER acquire, not as an argument to a function
}
} // namespace strata::prefill::detail
