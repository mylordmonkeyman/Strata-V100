# Equal-byte pinned-host prepacked NVFP4 QPN vs MMQ — real V100 result

**Decision:** Offline preconverted chunk-major weights streamed directly from pinned host RAM to V100 remain faster than all-MMQ on the pinned real 16-expert sample, despite identical compressed bytes transferred. With 44.24 MB H2D included, the improvement is **+2.667% ordinary / +2.552% CUDA Graph** (vs +13.16% / +12.92% when weights were already GPU-resident). This demonstrates real weight-supply impact, NOT full-model TTFT or proof of model quality.

## Provenance and controlled comparison
- Source commit `e680c1bde114ebcf6f2b20885274f58b9791cf06` (offline source also `84f0b5fdca6e5491508ff41fca2647b20086b60e`), successful [V100 run 40981028186](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40981028186), aggregate artifact ID **12944179143**.
- Private SHA-verified 16-expert M0 fixture (source original weights SHA256 `ce605a8b34aa54c0dc266aa38be9560f9d8ecb2f4179111e8bf4575fa82801a7`; activations SHA256 `0625460d391179bbc4d2ce56259d6886c2d62ae0aa51dd69eb741093e5afe2a8`). No private source model, weights, activations or raw outputs uploaded.
- Same exact **1100 real routed rows**. QPN uses the T187 and T508 experts (695 rows) with prepacked compressed-block chunk-major GU/down, directly streamed from persistent pinned host memory; all other 14 experts use original canonical GGUF with MMQ. Baseline uses original GGUF for all 16 with MMQ. Both arms transfer **44,237,056** compressed-weight bytes **each iteration**; hybrid partitions 38,707,424 canonical MMQ bytes + 5,529,632 prepacked QPN bytes. A single stream orders actual pinned H2D before weight gathering and compute, so event timing includes all DMA and kernels; the CUDA Graph also captures actual H2D copies.
- Repacking **before** the benchmark reported **40,046.149 µs (~40ms)**; separate extra GPU upload **0 µs**, since the packed bytes are directly staged to persistent pinned host memory. Neither initial conversion nor pinned allocation is included in repeated-workload latency. The offline cache host tests (run 40981355433) passed independently.

| Real 16-expert workload | All-MMQ | Prepacked+MMQ hybrid | MMQ/hybrid ratio |
|---|---:|---:|---:|
| Ordinary median (6 alternating rounds ×10 workloads) | 5208.883095 µs | **5073.561478 µs** | **1.0266719× (+2.667%)** |
| CUDA Graph median | 5175.807953 µs | **5046.988869 µs** | **1.0255239× (+2.552%)** |

Ordinary MMQ individual rounds: [5326.332855,5216.051102,5216.767883,5195.059204,5201.715088,5195.772934] µs. Hybrid: [5170.278549,5146.009445,5073.305511,5064.704132,5073.817444,5062.348938] µs.
Graph MMQ rounds: [5175.500870,5175.603104,5176.012802,5172.326279,5201.100922,5181.030273] µs. Hybrid: [5049.036789,5042.687988,5046.886444,5042.694473,5047.091293,5047.398376] µs. All twelve within-round comparisons favor hybrid.

## Safety and precise limitations
- Independent GGUF dequantized sampled FP64 oracle L2: MMQ **0.0112966272**, hybrid **0.0101718218** (32 sampled oracle rows). Whole-output hybrid-MMQ L2 **0.011639834**. All values finite and within existing benchmark thresholds, but these do **not** qualify the model’s first-logit KL.
- CUDA Graph replay exactly reproduced ordinary results and consumed altered inputs. Output-tail canary intact. Compute Sanitizer **memcheck zero errors** and **racecheck zero hazards/errors/warnings**. Sanitizer sample timings are not performance evidence.
- The workload simulates mandatory retransmission of all **44.24MB** every iteration, sequentially on the compute stream. Real Strata uses a separate DMA stream with lookahead, cache-resident hits, and per-expert reordering. This matched direct-stream test does NOT model those asynchronous pipeline and cache effects, full model layer latency, K-expert routing distribution, PCIe contention, or repeated-request amortization. The ~40ms one-time pack is an opportunity to use an offline SHA-verified compressed-block cache rather than repack each process start.
- A separate persistent-cache build with exact GGUF round-trip/checksums is in `tools/qpn_prepack_expert_cache.py` (host test 40981355433). A packed cache must keep canonical original available for MMQ fallback and be keyed by model, source SHA256, output layout version and expert dimensions. No model loader integration yet.

**Next stage:** Verify real pinned H2D directly from a *disk prebuilt*, manifest-verified chunk-major cache to avoid the 40ms process-start pack; then test copy/compute overlap and realistic per-expert staged slots with identical baseline policies. Only after safety, meaningful model-level ABBA improvements and quality checks should any new QPN route be enabled.

Production `main`, default MMQ, original model/source/pack, expert VRAM cache, user server/services, and runner privileges remain unchanged.
