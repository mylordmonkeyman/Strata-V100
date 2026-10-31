// Host-only runtime shim test of the actual opt-in R0 collector; NO GPU.
// Host build uses -DSTRATA_QPN_PREFILL_EXPERIMENTAL and fake CUDA includes.
#include "../../src/prefill/qpn_supply_trace.hpp"
#include <cassert>
#include <cstdlib>

int main() {
    ::setenv("STRATA_QPN_SUPPLY_R0_AGGREGATE", "1", 1);
    ::unsetenv("STRATA_QPN_SUPPLY_TRACE");
    int dummy = 0;
    int product_calls = 0;
    {
        strata::prefill::QpnSupplyTrace trace(true, nullptr, nullptr);
        assert(trace.enabled());
        for (int i = 0; i < 20000; ++i) {
            assert(trace.h2d(0, &dummy, &dummy, 2764816, nullptr) == cudaSuccess);
            trace.products(0, nullptr, [&] { ++product_calls; });
        }
        assert(trace.h2d(47, &dummy, &dummy, 2764816, nullptr) == cudaSuccess);
        trace.products(47, nullptr, [&] { ++product_calls; });
        assert(trace.report(0, 1024, 24, true, nullptr, nullptr));
    }
    assert(product_calls == 20001);
    assert(fake_cuda::copies == 20001);
    // 1 time origin + 4 category/layer pairs * 2 events.
    assert(fake_cuda::events_created == 9);
    assert(fake_cuda::event_records > 40000);
    ::unsetenv("STRATA_QPN_SUPPLY_R0_AGGREGATE");
    {
        strata::prefill::QpnSupplyTrace off(true, nullptr, nullptr);
        assert(!off.enabled());
        int once = 0;
        assert(off.h2d(0, &dummy, &dummy, 16, nullptr) == cudaSuccess);
        off.products(0, nullptr, [&] { ++once; });
        assert(once == 1);
        assert(off.report(0, 1, 1, false, nullptr, nullptr));
    }
    assert(fake_cuda::events_created == 9);
    assert(fake_cuda::copies == 20002);
}
