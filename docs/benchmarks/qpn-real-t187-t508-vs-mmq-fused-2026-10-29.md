# Real T187/T508 fused QPN against MMQ — matched V100 study (2026-10-29)

**Decision:** The fused (single 1280-wide gate/up, direct FP16 SwiGLU stage) QPN path is **slower than MMQ** for real 187/508-row experts in this pinned 16-expert workload. Do not extend the experimental model routing policy to large experts or enable QPN in production on this evidence. Continue with compressed-weight layout testing.

## Immutable inputs and provenance

- Branch `qpn/large-route-probe` source commit `7fa7d23610bc05d68e52c882f3fbe04e248e2a8c`, successful [V100 run 40974894319](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40974894319), aggregate artifact **12942831267** (`qpn-huge-mixed-40974894319-1`).
- Private pinned 16-expert mixture: **1,100 routed entries**, **44,237,056 distinct compressed GGUF expert bytes**, V100 L2 **6,291,456 bytes**. Hybrid QPN computes the exact **187- and 508-row** expert groups (695 routed rows) with original contiguous GGUF weights; the other fourteen groups remain W4A8 MMQ. Both arms share a fixed group permutation, inputs, weights, pinned private fixture SHA-verification and one V100. No model or raw activations/weights uploaded.
- `--qpn-min 129 --qpn-max 512 --qpn-fused-gu` uses one 1280-wide GU projection followed by direct FP16 SwiGLU, then QPN down; this removes two per-expert launches relative to the previously qualified unfused path. **The model engine remains limited to 17..32-row QPN eligibility and off by default.**
- Six interleaved CUDA-event timings, ten full mixed expert workloads per sample. CUDA Graph trials test full captured workloads. Ordinary timing includes CPU-to-GPU launch submission gaps; neither metric is full-model TTFT or end-to-end model prefill.

| Complete real 16-expert sample | All MMQ | Fused large QPN + MMQ fallback | MMQ / hybrid ratio |
|---|---:|---:|---:|
| Ordinary median | **1450.905609 µs** | 1476.355219 µs | **0.9827619×** (~1.75% slower) |
| CUDA Graph median | **1326.540804 µs** | 1353.062391 µs | **0.9803988×** (~1.96% slower) |

Ordinary MMQ per round: `[1451.418,1451.315,1450.496,1450.291,1451.725,1349.530]` µs. Fused QPN per round: `[1476.710,1476.099,1476.611,1477.424,1423.357,1368.883]` µs.
Graph MMQ: `[1329.254,1330.483,1329.357,1323.827,1313.587,1306.010]` µs. Fused QPN: `[1360.077,1358.541,1355.162,1350.963,1338.163,1335.808]` µs. All six graph pairs favor MMQ. Do not infer exact fused-vs-unfused improvement from separate runs with differing baseline timing distributions.

## Safety and numerical gates

- Whole output relative L2, hybrid vs MMQ: **0.011639834**, finite.
- Independent GGUF dequantized double-precision sampled oracle L2: **0.0112966272 MMQ** and **0.0101718218 hybrid** (32 intentionally sampled oracle rows, not exhaustive).
- Both arms' CUDA Graph replay exactly equals ordinary output and consumes changed input; all output guards intact.
- Compute Sanitizer **memcheck 0 errors** and **racecheck 0 hazards, 0 errors, 0 warnings**. These are sanitizer outcomes, not performance measurements.
- Full model first-logit KL and broad quality *have not been qualified* for large-QPN routing. Last earlier model QPN-vs-MMQ first-logit KL was 0.03134775 for small group QPN; it is not a result for these large experts.

## Follow-up

A separately gated, offline-only chunk-major compressed-weight experiment now tests a private host repack of GU and down blocks on the same real sample. It must report **both repack CPU time and compressed-weight transfer/upload cost**, as well as within-run full mixed-workload ordinary, graph, oracle, guards, and sanitizer results. Any performance with prepacked persistent weights is an **upper bound for model integration** unless realistic supply and repack costs are amortized. Preserve canonical GGUF fallback, exact routing, `main`, production MMQ, expert cache and host services.
