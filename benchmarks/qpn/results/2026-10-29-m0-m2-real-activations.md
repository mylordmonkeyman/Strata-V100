# M0 / M2 real-activation qualification: 2026-10-29

## Provenance

- Hardware: Tesla V100 PCIe 32 GB, CUDA 12.8.1, driver 580.178.04; baseline runner memory limit 48 GiB.
- Baseline: [Actions run 40815727472](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40815727472), commit `1642f7b70ec6b1ceb7083323ddc2a42d911fa19d`; artifact `qpn-m0-40815727472-1` (ID 12872852053). Six full-model invocations exited 0; actual routed activations captured at layers 0, 24, and 47. Baseline uses `--spec 2`, `--suffix-draft 0`, `--adapt-every 0`.
- Matched standalone test: [Actions run 40820436578](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40820436578), commit `60959dd0cf5e144cff76e245f498d6c54f12e10b`; artifact `qpn-real-40820436578-1` (ID 12873614385). The workflow verified all three prepared checkpoint asset SHA-256 hashes before extracting fixtures.
- The artifact includes per-fixture `raw.jsonl`, `summary.json`, provenance metadata and fixture index. Raw activations and weights remain on the private runner, not in the public artifact.

## Verified observations

Nine routed experts (three each from layers 0/24/47), with 1–32 actual activation rows per expert, produced **84/84 passing candidate configurations**. All 84 had exit code 0 and `numerics_pass`, `graph_replay_equal`, `graph_changed_input_equal`, `scratch_guards_ok` and `output_guards_ok` set to true. Maximum reported completed-expert absolute error was **0.00984736** and maximum reported `expert_l2` was **0.01480865**; these are standalone numerical metrics, not full-model logit or output quality equivalence.

**Speed gate fails for every tested configuration.** The reported `completed_expert_speedup` is legacy time divided by QPN time, so less than 1 means QPN is slower. Across 84 candidate configurations: minimum **0.259x**, median **0.337x**, maximum **0.889x** (even the best is ~12.4% slower by elapsed time). The fastest observed ratio was a 16-row FP32-staging case; Q8 reconstruction did not improve the median.

| Routed fixture | Cases | Best completed-expert ratio |
|---|---:|---:|
| L0 E43 | 14 | 0.889x |
| L0 E186 | 14 | 0.889x |
| L0 E453 | 2 | 0.286x |
| L24 E229 | 14 | 0.883x |
| L24 E209 | 12 | 0.755x |
| L24 E509 | 2 | 0.287x |
| L47 E157 | 14 | 0.887x |
| L47 E369 | 10 | 0.460x |
| L47 E507 | 2 | 0.283x |

By token bucket, median ratio was **0.273x at T=1** (18 cases), **0.366x at T=4** (12 cases), **0.519x at T=8** (10 cases), **0.854x at T=16** (8 cases), and **0.780x at T=32** (6 cases). FP32-staging configurations had a median ratio of 0.351x versus 0.326x with reconstructed Q8 staging (42 cases each). These are standalone *single-expert* measurements; the 2,764,816-byte expert working set may fit in V100 L2, and the measured `--groups 1` results do not establish a realistic multi-expert cache residency or end-to-end performance outcome.

## Gate decisions and next engineering steps

1. **M0:** The reproducible baseline and three-layer capture milestone is achieved, but the specification's representative routing/headroom, residency, and model-quality gates are **not** complete. Six runs of one prompt in a 48 GiB cgroup are not an all-RAM/128 GiB qualification.
2. **M2 numerical subset:** Real-activation standalone parity, scratch guards, and graph replay passed for the nine selected experts and 84 tested configurations. This is not grouped dispatch, full-model logits, mixed residency, or acceptance/quality parity.
3. **M2 decode speed:** **FAIL** on these measurements. Do **not** enable production QPN dispatch, infer a model-level speedup, or repack cache slots. Retain canonical legacy `dp4a` fallback.
4. **Next decode experiments, only if justified:** profile stage-by-stage latency and launch overhead; compare larger **mixed-group** working sets with realistic routed entry histograms and non-L2-resident cases; sweep alternate split-K/accumulator/decoder schedules with the full completed-expert path timed. Require a statistically defensible completed-expert win before integration. A `T=16` single-expert near-break-even result is insufficient.
5. **Prefill track is independent:** separately qualify the existing NVFP4 W4A8/MMQ and correctly scaled FP16 crossover using real routed batches, then measure end-to-end TTFT and resource contention. Decode failure must not suppress useful prefill work.
6. **M3/M4:** Defer device-side exclusive grouped dispatch, graph-stable scratch, and runtime opt-in until the corresponding independent track passes correctness, throughput, headroom, and end-to-end gates.

No production engine dispatch, checkpoint layout, `main` branch, or host services were modified for this report.
