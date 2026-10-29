# Real V100 QPN batching and compressed NVFP4 layout attribution — 2026-10-29

## Independent result and provenance

- Isolated branch: qpn/mixed-route-bench. Measured source commit: 889c41856e8912ef73cb803fdd2fed81f2fc694e.
- [Successful independent V100 Actions run 40917603036](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40917603036); [metadata-only artifact 12917565745](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40917603036/artifacts/12917565745), SHA-256 ceb4d8d657b6d9b3775adb47339d98b8fcaad9f04d9e06fd33adb3f886e0dc4a.
- Real private source data pinned to fixture run 40908922475, original M0 40815727472, baseline manifest SHA-256 218ae1f7fa5bb03ee01aba98278848e13ce17478957ec2300a0e58272af8270d.
- Independent private verifier passed before profiling. Uploaded only summary JSON, hashes, timing aggregates, profiler availability and sanitized logs. Never uploaded weights, activations or full route plans.
- Hardware: sm_70 V100, 80 SMs, 6,291,456-byte L2. Gate+up weight bytes actually accessed per complete two-projection sweep: 11,059,200 across six distinct real experts (>L2 size; NOT a claim of measured miss traffic).
- Exact input: for each of six actually routed experts with >32 rows, use first 1,2,4,8,12,16,17,20,24,32 different real captured rows; no repeated or invented activations. Same experts and data across four arms; online H2D transfers and model-format repack excluded.
- Gate+up projections only: 640x2560 weights, three supported QPN fast cases and a scalar decode control, FP32 results. This is NOT a complete MoE forward, end-to-end TTFT, or comparison to MMQ.
- Six alternating CUDA-event timing repetitions of 30 complete two-projection workloads each per arm and T. Separately measured replay timings with CUDA Graphs.

## Measured median latency (microseconds, lower is better)

| Actual T | Serial native (12 launches) | Batched native (2) | Batched chunk-major (2) | Batched scalar decoder | Serial/batched speedup | Chunk-major/native speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 339.61 | 79.92 | 65.47 | 133.14 | 4.25x | 1.22x |
| 2 | 316.47 | 74.77 | 61.53 | 127.57 | 4.23x | 1.22x |
| 4 | 318.40 | 75.71 | 62.04 | 138.77 | 4.21x | 1.22x |
| 8 | 321.79 | 78.71 | 61.88 | 167.90 | 4.09x | 1.27x |
| 12 | 289.21 | 71.19 | 54.82 | 221.24 | 4.06x | 1.30x |
| 16 | 299.13 | 76.92 | 56.39 | 249.16 | 3.89x | 1.36x |
| 17 | 265.86 | 92.02 | 62.81 | 303.38 | 2.89x | 1.47x |
| 20 | 269.81 | 93.71 | 63.33 | 315.61 | 2.88x | 1.48x |
| 24 | 274.01 | 97.23 | 64.02 | 333.82 | 2.82x | 1.52x |
| 32 | 287.35 | 104.47 | 70.02 | 355.58 | 2.75x | 1.49x |

## What is isolated, what is not

1. **Small kernel launch/grouping geometry** — batching identical fast-decoder, canonical GGUF operations reduces measured time 64-77% versus issuing gate and up separately for six experts (12 to 2 launches); 2.75-4.25x faster. CUDA Graph results show nearly the same improvement (e.g., T4 serial 310.9, batched 74.4 us), so it is not explained merely by CPU dispatch submission. Changes combine GPU kernel-launch count, SM exposure and caching; exact tensor-pipe utilization is NOT measured. Gate/up grid is 20 CTAs per one expert for T<=16 (at most 25% of 80 SMs occupied concurrently); explicitly tiled T17..32 uses 40 CTAs (at most 50%). Six expert batched grids use 120/240 CTAs respectively. These are geometric upper bounds, not measured active SM/cycle percentages.
2. **Compressed NVFP4 layout** — identical 36-byte scale/code blocks privately rearranged from GGUF row/K-block order to K-block/row order; same FP16 staging, PRMT/half2 decoder, MMA, outputs and two batched launches. Layout change reduces batched latency ~18% at T4 (75.71 -> 62.04 us) and ~34% at T24 (97.23 -> 64.02 us), a 1.22x-1.52x throughput gain. This is a hypothetical private repack: engine-pack conversion cost and supply are excluded. The comparison also changes indexing math, so no exclusive DRAM-transaction causal percentage is asserted.
3. **Decoder code choice** — existing fast half2/PRMT decoder is 1.67x-3.43x faster than the legacy scalar decoder with native GGUF layout when batched. This measures implementation alternatives, NOT the cost of all NVFP4 decoding.
4. **Masked small MMA tiles** — theoretical minimum inactive row fraction of an 8-row token tile at T=1 is 87.5%, at T=4 50%; for two 16-row CTA tiles T=17 gives 46.875% and T=24 25%. This is arithmetic masking, not measured hardware tensor-core occupancy.

## Safety and outstanding gates

- 10/10 diagnostic sizes produced finite, parity-matched results. Relative FP32 output L2 was zero for the four arms under this test; current test threshold was 0.002, with separate 256-byte canaries intact and CUDA graph replay parity checked.
- Compute Sanitizer memcheck T=4 and T=24 both returned ERROR SUMMARY: 0 errors. Wider racecheck coverage remains outstanding.
- Existing Nsight Compute version 2025.1.1 was present, but this probe reported the requested tensor-pipe metric unavailable. Hardware counters (tensor-pipe active cycles, SM active cycles, DRAM/L2 sectors) have NOT been obtained. No driver or runner privilege changes were attempted.
- Before production decisions: profile utilization counters if permitted, quantify weight-supply/hot-vs-cold effects and actual grouped MoE MMQ fallback costs, test numerical/graph/race safety for full expert, and run fixed-seed full-model quality and TTFT comparisons. A gate/up projection speedup is not an end-to-end model speedup.

Production paths, main, model packs, expert cache and unrelated host services remain untouched.
