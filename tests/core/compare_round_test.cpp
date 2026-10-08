#include "strata/telemetry/round_scope.hpp"
#include <array>
#include <stdexcept>
int main() {
    const std::array<int32_t, 2> input{101, 202}, sampled{303, 404};
    {
        v100_compare::RoundLink link("strata");
        strata::telemetry::RoundScope owner("verify");
        owner.inputs(input.data(), 2, 42, "cuda_graph");
        {
            strata::telemetry::RoundScope nested("verify");
            nested.inputs(input.data(), 1, 99);
            nested.success();
        }
        owner.sampled_tokens(sampled.data(), 2);
        owner.success();
    }
    {
        strata::telemetry::RoundScope prefill("prefill");
        prefill.inputs(input.data(), 2, 100, "eager");
        // GPU router shape is observable without reading or synchronizing ids.
        // Residency classes are not available on fused prefill; do not assert
        // or serialize fabricated zeros for those fields.
        if (v100_compare::active) {
            v100_compare::active->counter(0, v100_compare::Counter::routed_tokens, 2);
            v100_compare::active->counter(0, v100_compare::Counter::total_routes, 20);
        }
        prefill.success();
    }
    { strata::telemetry::RoundScope failed("verify"); }
    try {
        strata::telemetry::RoundScope unwind("verify");
        unwind.success();
        throw std::runtime_error("fixture");
    } catch (const std::runtime_error&) {}
    return v100_compare::active ? 1 : 0;
}
