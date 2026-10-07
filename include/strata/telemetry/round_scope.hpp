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
    template<class Token> void inputs(const Token* tokens, unsigned columns, std::int64_t first_index,
                                     const char* mode = "unknown") {
        if (!round_) return;
        v100_compare::InputContext context;
        context.input_columns = columns;
        context.spans.push_back({0, columns, first_index, std::nullopt, std::nullopt});
        context.execution_mode = mode;
        if (tokens && v100_compare::level() >= 2)
            context.input_token_ids.emplace(tokens, tokens + columns);
        round_->context(std::move(context));
    }
    template<class Token> void sampled_tokens(const Token* tokens, unsigned columns) {
        if (round_) round_->sampled_tokens(std::span<const Token>(tokens, columns));
    }
    void success() noexcept { success_ = true; }
private:
    std::optional<v100_compare::Round> round_;
    bool success_ = false;
};
} // namespace strata::telemetry
