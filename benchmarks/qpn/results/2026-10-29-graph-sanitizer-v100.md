# QPN V100 graph replay and Compute Sanitizer qualification — 2026-10-29

## Exact provenance

- Branch `qpn/prefill-tiles`, kernel/harness commit `f867667b3343adaffda85110b58dd16d6783258c` (narrow legacy MMQ index-padding fix).
- [V100 Actions run 40895971474](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40895971474), **success**; artifact `qpn-prefill-real-40895971474-1`, **ID 12906374152**, contains `candidates/*/raw.jsonl`, per-expert `summary.json`, `candidates/summary.json`, `job.log`, `sanitizer-t17.log`, `sanitizer-t31.log` and sanitizer JSON output.
- [Independent host, CUDA and Docker CI run 40895971416](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40895971416), **success**.
- Pinned baseline M0 [40815727472](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40815727472) and same nine real routed expert fixtures from layers 0/24/47; checkpoint revision `b9ce18a6887e18c089fdea8509a33e63429a3300`. Fixture index captures SHA-256s for all weights and FP32 activation prefixes.
- Measurements: six alternating completed-expert CUDA event timings per arm, 35 iterations per repetition. Reported figures are medians per arm/fixture, then medians over experts for each (T, tile). Weight supply/cache misses and end-to-end TTFT are excluded.

## Correctness and safety

- **82/82** standalone cases exit 0 and pass QPN/MMQ numerical gates.
- **14/14** requested QPN checks with `--validate-graph` have all four flags true: `graph_checked`, `graph_replay_equal`, `graph_changed_input_equal`, `output_tail_guard_ok`. Actual test sizes are T=17,24,31,32 using explicit 16-row CTA token tiles and genuine routed FP32 activations. The additional 256-byte output-tail canary remained intact.
- `compute-sanitizer --tool memcheck --error-exitcode 98 --leak-check no` was run separately against real L0 E43 fixture at T=17 and T=31 with tile=16 plus graph validation. Both `sanitizer-t17.log` and `sanitizer-t31.log` end in **`ERROR SUMMARY: 0 errors`**. This does **not** establish all GPU memory safety for untested shapes/backends.
- Maximum standalone QPN relative expert-output L2 error across the 82 cases: **0.000375535** (threshold 0.03), against the independent GGUF-dequantized double-precision reference.
- Initial sanitizer run [40891051732](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40891051732) failed: ggml's comparison MMQ backend read inactive tile index lanes beyond a 17-element index allocation. Commit `f867667` padded the **standalone comparison harness** index buffer by 256 entries and initialized inactive entries to zero; active 0..T-1 routing and QPN kernel were unchanged. The rerun above passes.

## Matched completed-expert performance

| Real routed T | Experts | Original QPN tile 0 (us) | New tile 16 (us) | W4A8/MMQ (us) | New QPN/MMQ speedup |
|---:|---:|---:|---:|---:|---:|
| 12 | 1 | 57.666 | — | 61.250 | Original 1.062x |
| 16 | 4 | 59.948 | — | 63.049 | Original 1.052x |
| 17 | 4 | 122.594 | **61.374** | 69.303 | **1.129x** |
| 24 | 4 | 126.610 | **61.696** | 70.796 | **1.147x** |
| 29 | 1 | 146.754 | **62.624** | 77.765 | **1.242x** |
| 31 | 3 | 147.661 | **62.976** | 77.854 | **1.236x** |
| 32 | 3 | 142.687 | **63.167** | 77.779 | **1.231x** |

The measured 16-row independent CTA token tiles eliminate the 17+ token single-CTA schedule regression while maintaining the graph and guard gates. Default tiles remain better at 12..16; at T<=8, W4A8/MMQ remains better.

## Qualification scope and next gates

**The standalone graph and sampled memcheck gate passes.** No production backend is enabled. Before considering any engine integration, separately measure real **nonuniform multiple-expert** routing, memory residency and supply (>L2 distinct weight working sets), group ownership/graph-stable scratch, and end-to-end full-model TTFT and acceptance/quality; validate any chosen path with more extensive memcheck/racecheck and numerical parity. A single-expert microbenchmark speedup must not be extrapolated to model tokens/s.

`main`, model packs, expert-cache slots and unrelated host services remain unchanged.
