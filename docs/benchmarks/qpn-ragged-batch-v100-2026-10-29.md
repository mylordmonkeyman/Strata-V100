# Ragged 16-expert QPN batching — standalone V100 result (2026-10-29)

## Result and scope

**The first isolated GPU batch experiment passed**, with a **1.68098× lower ordinary completed-workload latency** and **1.58655× CUDA-graph replay speedup** relative to separate per-expert QPN execution. The benchmark is deterministic *synthetic GGUF NVFP4*, **not a full-model or real-routed-activation benchmark**, and no inference gain is yet established.

Pinned source [commit dd54541b8130c5c57b624abbb8ac40114a179720](https://github.com/mylordmonkeyman/Strata-V100/commit/dd54541b8130c5c57b624abbb8ac40114a179720), [successful V100 Actions run 40965835267](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40965835267), artifact ID **12937848016** `qpn-ragged-40965835267-1`. Source SHA-256 `02e31213b60d17fbac88a64621a6ab9a03b6dc092d5b837c9e19fee75245ae2c`; binary SHA-256 `47e026609898b68ac227bb79ebf78ce8d8c50a671bac7ec4b1db6b00fc924a27`.

## Matched synthetic workload

- Single Tesla V100, 16 experts with distinct row counts T=17,18,...,32, totaling **392 actual routed rows**. Expert inputs use a seeded **permutation of unique source-row indices**; outputs scatter into exactly those 392 rows. The grouped QPN projection uses a padded uniform 32-row stride (512 rows), and masks padded rows on scatter. The source expert IDs, weights and activations are purely generated for the test, **not** taken from the prepared private checkpoint.
- Independent GGUF-scale/code expert weights: 44,236,800 bytes across gate/up and down, exceeding the V100 6,291,456-byte L2. Distinct compressed bytes do **not** prove measured DRAM-miss traffic. Each path performs complete input gathering, 1280-wide gate/up NVFP4 projection, SwiGLU to FP16, down projection, and output scatter, with the same QPN 16-row CTA schedule.
- Serial path issues five launches per expert (80 total). Grouped path issues five launches total: one ragged gather/pad, uniform `projection_batch` across all experts, one fused SwiGLU, uniform down `projection_batch`, one masked output scatter.
- Ordinary timings are 12 repeated complete workloads with CUDA events and include host-issued launch gaps; graph timings are 12 replays of each captured completed workload. **Neither is an end-to-end model TTFT or comparison against W4A8/MMQ.**

| Full workload | Per-expert serial | 16-expert grouped | Serial/grouped ratio |
|---|---:|---:|---:|
| Ordinary completed workload | 778.325 µs | **463.019 µs** | **1.68098×** |
| Captured CUDA Graph replay | 726.613 µs | **457.984 µs** | **1.58655×** |

## Correctness and instrumentation

- Valid-row output parity: **relative L2 = 0**, maximum absolute difference = 0.
- Repeated after changing a true input value: **relative L2 = 0**, maximum difference = 0.
- Both 256-byte output guards intact before and after input change.
- Sampled four-expert sanitizer run: T17..20, 74 valid rows, 128 padded rows, 11,059,200 distinct weight bytes. Compute Sanitizer `memcheck`: **0 errors**; `racecheck`: **0 hazards, 0 errors, 0 warnings**. Sanitizer-mode timings are meaningless for speed comparisons.
- The successful result applies to the isolated benchmark with this synthetic data and sampled sanitizer coverage; additional real-capture numerical tests and full-model safety/quality evaluation remain required.

## Engineering decision and remaining gates

The positive result justifies adapting this same batch strategy to the **off-by-default experimental staged prefill path**, but does not authorize enabling production dispatch. Reuse existing 16-expert MMQ group ownership and native compressed weight strides; do not alter the model pack or expert cache. A batch-specific opt-in should pack real group inputs from device routing bounds into padded 32-row FP16 buffers, invoke batched GU/down, fuse per-expert scale/SwiGLU, then scatter only valid rows. Compare against *the existing fused per-expert QPN path* in a matched real-model ABBA and preserve original MMQ fallback. Require end-to-end numerical finiteness, all routed rows written, CUDA graph replay, and additional memcheck/racecheck before any production consideration.

The improved standalone kernel execution is promising but untested in actual full-model prefill. No changes to `main`, production defaults, pack, cache format or host services.
