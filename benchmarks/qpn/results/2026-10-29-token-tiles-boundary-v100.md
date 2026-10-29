# V100 real-activation QPN token-tile boundary qualification (2026-10-29)

## Reproducibility

- **Commit:** `2459dc76fac880378c48faf29e47259c741bd1c1` on `qpn/prefill-tiles`.
- **V100 test:** [run 40885449866](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40885449866), success; artifact `qpn-prefill-real-40885449866-1` (ID **12903146974**), including `candidates/*/raw.jsonl`, `summary.json`, per-fixture metadata, process list, and job log.
- **Host, CUDA and Docker checks:** [run 40885450020](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40885450020), success. The transient immediately preceding host-test failure (40885442857) was superseded by the fixed boundary expectation tests in the final commit.
- **Original control:** [run 40879696635](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40879696635), report `2026-10-29-token-tiles-v100.md`. Original unchanged model baseline [40815727472](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40815727472).
- **Inputs:** nine pinned checkpoint-matched experts from layers 0/24/47, native `NVFP4-QAD` weights, real routed FP32 activation prefixes. The `candidates/fixture-index.json` SHA-256 is `ce53e96787476f7bac217809269ff4f9333ccffb3d9ff4d0dab6c2338019b2ee`, identical to the earlier real-activation reports. Original model revision `b9ce18a6887e18c089fdea8509a33e63429a3300`.
- **Measurement:** 35 iterations per repetition; six alternating recorded completed-expert CUDA event timings per arm. Values below are medians within each expert then medians across experts at that T/tile. All three MoE projections, staging, SwiGLU and scale applications are included; not any mixed-expert cache, layer scheduler, or end-to-end model TTFT. GPU clock and thermal state are not guaranteed identical between runs: use *within-run* paired comparisons.

## Verified results

**82/82 cases succeeded**, with zero fixture failures and all QPN/MMQ numerical gates passing. Maximum QPN relative output `expert_l2` = **0.000375535** across the entire study (the threshold is 0.03). W4A8/MMQ was selected in 47 cases; QPN in 35 cases when each tile alternative is counted as a separate case. This is not a workload-weighted choice frequency.

| Actual routed rows (T) | Experts | Original QPN (µs) | QPN tile 8 (µs) | QPN tile 16 (µs) | W4A8 MMQ (µs) | tile-16 / MMQ speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1 | 57.78 | 63.78 | — | 61.15 | original schedule wins 1.058× |
| 16 | 4 | 59.98 | 64.20 | — | 63.25 | original schedule wins ~1.05× |
| **17** | **4** | **122.34** | **66.18** | **61.31** | **69.23** | **1.128×** |
| **24** | **4** | **126.68** | **67.09** | **61.83** | **71.01** | **1.147×** |
| 29 | 1 | 147.00 | 69.72 | 62.54 | 77.79 | 1.244× |
| **31** | **3** | **147.91** | **70.64** | **63.06** | **77.79** | **1.234×** |
| 32 | 3 | 143.01 | 69.82 | 63.05 | 77.82 | 1.233× |

At T=17/24/31, every tested expert with a 16-row candidate passed numerics and outperformed W4A8/MMQ. The 16-row schedule also cuts the old QPN runtime by about **1.99× / 2.05× / 2.35×** at T=17/24/31, respectively. All T<=8 observations still favored W4A8/MMQ, and T=12/16 favors the default 16-row single-CTA QPN over splitting into 8-row CTAs.

**Supported standalone scheduling hypothesis:** keep existing default token tile for T<=16; for T=17..32, use independent **16-row CTA** parcels via `grid.z`. Boundary-tail cases 17, 24, and 31 now demonstrate that partial final token tiles produce valid output numerics against the GGUF/FP64 reference. No standalone winner implies production dispatch eligibility.

## Outstanding engineering gates

1. **CUDA graph replay and changed-input correctness** for multi-CTA token tiles; explicit output scratch and write guards; Compute Sanitizer checks. Current 82-case benchmark compares ordinary invocation outputs, not graph captures or memcheck.
2. Nonuniform **mixed-expert** real routing and total grouped workload throughput, with realistic >L2 weight working sets and GPU residency/PCIe supply, rather than selecting a single expert and replaying it.
3. End-to-end fixed-seed full-model numerical/quality and prompt prefill TTFT testing with a reversible, mutually exclusive, opt-in experimental dispatch. This remains prohibited until gates 1–2 pass.

**Production dispatch, model/pack layout, cache slots, and `main` are unchanged.**
