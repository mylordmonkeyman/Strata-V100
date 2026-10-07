#include "strata/telemetry/round_scope.hpp"
#include <array>
#include <stdexcept>
int main() {
    const std::array<int32_t, 2> input{101, 202}, sampled{303, 404};
    {
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
    { strata::telemetry::RoundScope failed("verify"); }
    try {
        strata::telemetry::RoundScope unwind("verify");
        unwind.success();
        throw std::runtime_error("fixture");
    } catch (const std::runtime_error&) {}
    return v100_compare::active ? 1 : 0;
}
