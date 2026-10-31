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
// A stream-all request may finish with a shorter staged tail chunk.
// The opt-in remains configured for the whole request, but has no
// staging job to skip on that tail. Do not reject or modify its path.
enum class EarlyStageDecision { Inactive, Active, Invalid };

constexpr EarlyStageDecision qpn_mmq_early_stage_phase(
        bool requested, bool stream_all, bool staged_after_stream_all,
        bool matched_mmq_selective, bool mmq_skip) {
    if (!requested) return EarlyStageDecision::Inactive;
    if (stream_all)
        return (matched_mmq_selective && mmq_skip)
            ? EarlyStageDecision::Active : EarlyStageDecision::Invalid;
    return staged_after_stream_all ? EarlyStageDecision::Inactive
                                   : EarlyStageDecision::Invalid;
}
inline bool qpn_mmq_early_stage_skip(const std::atomic<int64_t>& published_layer,
                                     int64_t layer, const bool* routed) {
    if (!routed || layer < 0) return false;
    if (published_layer.load(std::memory_order_acquire) < layer) return false;
    return !*routed; // read only AFTER acquire, not as an argument to a function
}
} // namespace strata::prefill::detail
