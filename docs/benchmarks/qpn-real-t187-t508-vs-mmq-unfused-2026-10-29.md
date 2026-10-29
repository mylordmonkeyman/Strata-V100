# Real 187- and 508-row QPN versus MMQ: unfused control (2026-10-29)

**Result:** The isolated, unfused canonical GGUF QPN path is **2.35% slower in ordinary execution** and **2.02% slower under CUDA Graph replay** than within-run all-MMQ on the pinned 16-expert real-weight/activation sample. Therefore **no large-route full-model integration** is justified. The follow-up fused-GU experiment is distinct and requires new measurements.

## Provenance and exact controlled scope

- Source `9a6bbe2698b7bffff65f8b0536845975e17f0a1f`, successful [V100 run 40973590842](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40973590842), artifact ID **12941542472**, pinned artifact `qpn-huge-mixed-40973590842-1`.
- Privately retained SHA-verified 16-expert sample from fixed model M0, 1,100 genuine routed entries and 44,237,056 bytes expert weights (V100 L2 = 6,291,456 bytes). Exact actual group sizes: **187 and 508** QPN, totaling **695 routed entries**; remaining 14 groups stay MMQ. The all-MMQ control processes the exact same 16 experts, same private weights/activations and expert permutation. No expert row is dropped or duplicated; private raw input/output remains on the V100 runner.
- `--qpn-min 129 --qpn-max 512`; standalone candidate **unfused** gate projection, up projection, FP32 SwiGLU, FP16 stage, down projection, and down scaling. Original model `qpn_candidate` routing still only T17–32; this test is not an engine implementation or a full-model prefill/TTFT measurement.
- Six alternating rounds with 10 completed workloads each; ordinary CUDA-event timing includes launch gaps. Separate CUDA Graph replay measures captured complete GPU workload including staging and matched fallback. Source uses canonical compressed layout and no pack mutation.

| Metric | All-MMQ control | Hybrid QPN 187+508 | MMQ/hybrid ratio |
|---|---:|---:|---:|
| Ordinary median | **1399.194 µs** | 1432.934 µs | **0.976453×** (hybrid 2.35% slower) |
| CUDA Graph median | **1314.509 µs** | 1341.645 µs | **0.979774×** (hybrid 2.02% slower) |

Per-round ordinary MMQ `[1449.885,1449.677,1449.882,1348.710,1347.891,1347.891]` µs; hybrid `[1483.264,1481.219,1482.134,1384.650,1377.382,1378.304]` µs. Both early and late rounds show a regression.

Per-round CUDA Graph MMQ `[1328.640,1326.797,1317.274,1311.744,1311.232,1306.630]` µs; hybrid `[1358.950,1359.770,1340.621,1337.139,1342.669,1332.736]` µs.

## Safety, numerical and interpretation gates

- Real selected expert output versus all-MMQ whole-output relative L2 **0.011639834**. Independent sampled GGUF double-precision oracle L2: MMQ **0.0112966272**, mixed **0.0101718218**; sampled oracle rows 32. The benchmark numerical gate passed but this does **not** qualify model-level first-logit KL or complete quality.
- All outputs finite; complete output-tail guard unchanged; CUDA Graph replay bitwise equals ordinary run and consumes changed inputs.
- Compute Sanitizer **memcheck 0 errors**, **racecheck 0 hazards, errors or warnings** for this exact test.
- This is a direct real-weight comparison against MMQ, unlike earlier synthetic T187/T508 tests which compared **QPN single projection versus segmented QPN** and misleadingly suggested a gain if extrapolated.
- Next isolated change: opt-in **fused gate/up projection + direct FP16 SwiGLU stage**, removing two per-expert launches while retaining the same real fixture, all-MMQ control, changed-input, numerical oracle and sanitizer acceptance. The fused candidate source is `7fa7d23610bc05d68e52c882f3fbe04e248e2a8c`; performance remains **unmeasured** until separate green GPU run.

No updates to `main`, production, default MMQ routing, model packs, expert cache, runner privileges, or unrelated services.
