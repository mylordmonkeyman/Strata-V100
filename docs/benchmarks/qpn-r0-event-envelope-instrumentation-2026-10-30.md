# R0 diagnostic instrumentation — coalesced all-layer event envelopes (2026-10-30)

**Scope:** Rev4.2 R0 bounded bottleneck attribution, instrumentation *only*; no performance or numerical-quality claim. The prior QPN static-admission screen stays STOPPED and full-request R1 feasibility stays UNKNOWN. No dual-format residency, expert conversion/cache or GPU benchmark was introduced.

## New opt-in / source boundaries

- In the already experimental `QpnSupplyTrace`, `STRATA_QPN_SUPPLY_R0_AGGREGATE=1` records H2D, slot/ready wait, gather, and expert-product **first/last CUDA-event envelopes** across up to 48 layers reached by the existing supply callbacks (not necessarily every model operation). Re-recording each category/layer end event makes a bounded collector feasible even when the original MMQ path schedules tens of thousands of callbacks. The prior `STRATA_QPN_SUPPLY_TRACE=1` three-layer trace remains unchanged. The normal unset/default/non-CUDA behavior is unchanged.
- R0 does not collect routes, labels, weights, activations, logits, prompts, or token geometry. It does not read back device data. It prints only one **schema 2** aggregate JSON record per reported chunk, tagged `strata qpn-r0:`. Event allocations are capped at 5 operation categories × 48 layers = **240 pairs per chunk**; callback counts and H2D bytes are separately accumulated. The category/layer first/last envelopes contain **all intervening idle gaps**, unlike the historical individually timed event union. Any CUDA/event/overflow failure invalidates the diagnostic; the wrapped original CUDA operation still executes exactly once.
- `qpn_r0_intervals.hpp` unions overlapping first/last envelopes and determines their apparent overlap without double-counting intersecting spans. These are **upper envelopes**, not true H2D/product occupancy or exposed latency. The C++ host-only test covers overlap, adjacent intervals, and invalid/NaN input.
- `tools/qpn_r0_aggregate.py` accepts **private** original-MMQ process logs, validates schema 2 and a strict field allowlist, and writes only chunk-aggregate sum-of-envelopes metadata. It does not confuse event pairs (≤240 per chunk) with the many operations represented by a pair. Legacy schema-1 diagnostics are rejected, not silently mixed. Unknown fields and values are rejected. Never publish input logs; delete private raw logs through the existing runner retention policy.

## Local host-only checks (executed independently, not a model run)

```bash
c++ -std=c++17 -Wall -Wextra -Werror -pedantic -O2 \
  tests/host/qpn_r0_intervals_test.cpp -o /tmp/qpn_r0_intervals_test
/tmp/qpn_r0_intervals_test
python3 -m unittest discover -s tools -p 'test_qpn_r0_aggregate.py' -v
c++ -std=c++17 -Wall -Wextra -Werror -pedantic -O2 \\
  -DSTRATA_QPN_PREFILL_EXPERIMENTAL -I tests/host/qpn_fake_cuda \\
  tests/host/qpn_r0_trace_mock_test.cpp -o /tmp/qpn-r0-trace-mock
/tmp/qpn-r0-trace-mock 2>/tmp/qpn-r0-synthetic-only.jsonl
```

No Docker build or CI/workflow change is required. The experimental runtime opt-in is **not** enabled in the normal runner and does not authorize a new GPU job. For a subsequently authorized, nonduplicate original-MMQ R0 diagnostic, a private runner may use `STRATA_QPN_SUPPLY_R0_AGGREGATE=1` with the existing pinned model, four mandatory MMQ flags and `STRATA_QPN_PREFILL_EXPERIMENTAL=0`; the summary parser is:

```bash
python3 tools/qpn_r0_aggregate.py --private-log /private/path/original-mmq.log \
  --report /private/path/r0-aggregate.json
```

Private paths above are placeholders, **not** instructions to run a duplicate capture.

## Critical limitations and next gate

**Do not interpret these envelopes as removable time.** Each first/last CUDA-event envelope contains unmeasured idle gaps and may greatly overstate the actual duration of a category. It also includes instrumentation, submission gaps, scheduling and other work. Product wrappers cover the *instrumented supply callback operations*, not necessarily all executed kernels and are not QPN eligibility labels. The parser explicitly emits `eligible_exposed_critical_path_ms: null`, `full_request_latency_measured: false`, and `r1_full_request_feasibility: UNKNOWN`. Summing per-chunk GPU envelopes is not full-request latency. Host CPU staging, cache state, accepted decode, trace-on/off parity and complete critical-path exposure still require separate evidenced controls. R1 remains UNKNOWN pending these data and a viable single-representation transition strategy; no R2 prototype is authorized.

No CUDA compilation or device execution of this new instrumentation has been verified yet. Subsequent verification must use the existing pinned CUDA12.8 setup, `gpu.lock` and unchanged `strata-qpn-v100` noncancelling concurrency, without Docker rebuild or altered privileges. Publish aggregates only.

## R0 event-bound correction (2026-10-30)

The previous event-pair-per-operation design was at risk of exceeding its 32,768-pair cap on the pinned original-MMQ workload: prior aggregate H2D accounting alone observed **15,347 first-chunk entries**, before considering waits, gathers or products. That is not proof the cap was reached, but was a credible source-grounded failure mode. The new opt-in R0 collector uses at most **240** reusable event pairs per chunk, independently of callback count, and reports total original H2D operations separately. No request, GPU job, or model was run for this change. R0 remains a feasibility **diagnostic** only, not exposed critical-path evidence or a candidate benchmark.

### Host-only bounded-collector regression

The source-pinned fake-CUDA test compiled with host `g++ -std=c++17 -Wall -Wextra -Werror -pedantic`, without a driver or GPU. It exercised **20,001 H2D callbacks plus 20,001 product callbacks** across two synthetic layer numbers. The collector created **four reusable event pairs** (nine total fake CUDA events including its origin), invoked each original operation exactly once, and produced schema-2 aggregate output. The sanitizer accepted the synthetic output while keeping R1 UNKNOWN; a separate default-off path created no events and executed callbacks exactly once. The five schema-2 sanitizer unit tests passed. These are mocked host semantics, not CUDA toolkit compilation, GPU correctness, or measured latency.
