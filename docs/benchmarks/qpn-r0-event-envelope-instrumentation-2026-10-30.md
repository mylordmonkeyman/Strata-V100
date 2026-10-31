# R0 diagnostic instrumentation — all-layer event envelopes (2026-10-30)

**Scope:** Rev4.2 R0 bounded bottleneck attribution, instrumentation *only*; no performance or numerical-quality claim. The prior QPN static-admission screen stays STOPPED and full-request R1 feasibility stays UNKNOWN. No dual-format residency, expert conversion/cache or GPU benchmark was introduced.

## New opt-in / source boundaries

- In the already experimental `QpnSupplyTrace`, `STRATA_QPN_SUPPLY_R0_AGGREGATE=1` records H2D, slot/ready wait, gather, and expert-product **CUDA event envelopes** across all 48 model layers reached by the existing supply callbacks (not necessarily every complete model operation). The prior `STRATA_QPN_SUPPLY_TRACE=1` three-layer trace remains unchanged. The normal unset/default/non-CUDA behavior is unchanged.
- R0 does not collect routes, labels, weights, activations, logits, prompts, or token geometry. It does not read back device data. It prints only one aggregate JSON record per reported chunk, tagged `strata qpn-r0:`. Device event pairs have a strict 32,768-per-chunk cap; a cap or CUDA event error invalidates the diagnostic rather than silently calling it a measurement. The wrapped original CUDA operation is still executed exactly once.
- `qpn_r0_intervals.hpp` unions overlapping intervals and determines H2D/product overlap without double-counting coincident event spans; the C++ host-only test covers overlap, adjacent intervals, and invalid/NaN input.
- `tools/qpn_r0_aggregate.py` accepts **private** existing-process logs, validates a strict field allowlist, and writes only chunk-aggregate sum-of-envelopes metadata. Unknown fields and values are rejected. Never publish input logs; delete private raw logs through the existing runner retention policy.

## Local host-only checks (executed independently, not a model run)

```bash
c++ -std=c++17 -Wall -Wextra -Werror -pedantic -O2 \
  tests/host/qpn_r0_intervals_test.cpp -o /tmp/qpn_r0_intervals_test
/tmp/qpn_r0_intervals_test
python3 -m unittest discover -s tools -p 'test_qpn_r0_aggregate.py' -v
```

No Docker build or CI/workflow change is required. The experimental runtime opt-in is **not** enabled in the normal runner and does not authorize a new GPU job. For a subsequently authorized, nonduplicate original-MMQ R0 diagnostic, a private runner may use `STRATA_QPN_SUPPLY_R0_AGGREGATE=1` with the existing pinned model, four mandatory MMQ flags and `STRATA_QPN_PREFILL_EXPERIMENTAL=0`; the summary parser is:

```bash
python3 tools/qpn_r0_aggregate.py --private-log /private/path/original-mmq.log \
  --report /private/path/r0-aggregate.json
```

Private paths above are placeholders, **not** instructions to run a duplicate capture.

## Critical limitations and next gate

**Do not interpret these envelopes as removable time.** CUDA event intervals can include instrumentation, submission gaps, scheduling, or overlap with other work. Product wrappers cover the *instrumented supply callback operations*, not necessarily all executed kernels and are not QPN eligibility labels. The parser explicitly emits `eligible_exposed_critical_path_ms: null`, `full_request_latency_measured: false`, and `r1_full_request_feasibility: UNKNOWN`. Summing per-chunk GPU envelopes is not full-request latency. Host CPU staging, cache state, accepted decode, trace-on/off parity and complete critical-path exposure still require separate evidenced controls. R1 remains UNKNOWN pending these data and a viable single-representation transition strategy; no R2 prototype is authorized.

No CUDA compilation or device execution of this new instrumentation has been verified yet. Subsequent verification must use the existing pinned CUDA12.8 setup, `gpu.lock` and unchanged `strata-qpn-v100` noncancelling concurrency, without Docker rebuild or altered privileges. Publish aggregates only.
