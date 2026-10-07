#pragma once
#include "strata/telemetry/compare_telemetry.hpp"
#include <atomic>
#include <optional>

namespace strata::telemetry {
inline std::atomic<std::uint64_t> sequence{0};
// One owner-local record per native round; existing error returns remain failures.
class RoundScope {
public:
    explicit RoundScope(const char* phase) {
        if (v100_compare::level() && !v100_compare::active)
            round_.emplace("strata", "strata:" + std::to_string(sequence.fetch_add(1)), phase);
    }
    ~RoundScope() { if (round_ && !success_) round_->fail(); }
    void success() noexcept { success_ = true; }
private:
    std::optional<v100_compare::Round> round_;
    bool success_ = false;
};
} // namespace strata::telemetry
