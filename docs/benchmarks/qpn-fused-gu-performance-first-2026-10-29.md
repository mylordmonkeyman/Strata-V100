# Performance-first widened GU + fused SwiGLU staging: matched full-model ABBA

Date: 2026-10-29 (UTC). **Experimental branch only**: `qpn/performance-first`.
Source commit: `11bdf8d3b1e5c9530ef10cd5f3b5ec83d1d0b641`.
Successful [full-model run](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40960280590) and [standalone qualification](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40960280646).
Raw prompt, token IDs, raw logits, and full model logs remain private on the runner; GitHub artifact contains only aggregate `summary.json` and a process snapshot.

## Configuration and provenance

- Same built binary for all four trials: SHA-256 `f9222999ad8322d42fb3f71bd5b4c587690146139d7329611dbbfacab323bc47`.
- Pinned prepared expert pack SHA-256: `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`.
- Pinned prompt SHA-256: `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`.
- Matched A/B/B/A order: original QPN, fused QPN, fused QPN, original QPN; 2047 reported prefill tokens per run, `--prefill 768`, 4096 context, 16 max new, same routing plan and resident expert-cache settings.
- Common env: `STRATA_PREFILL_NVFP4=w4a8`, `STRATA_QPN_PREFILL_PLAN_CONTROL=1`, `STRATA_PREFILL_GROUP_GATHER=1`, `STRATA_PREFILL_STREAM_MIN=1024`, `STRATA_PREFILL_CPU_SHARE=0`, and MMQ direct/resident-sort/group-sort all 0. `STRATA_QPN_PREFILL_EXPERIMENTAL=1` for both; `STRATA_QPN_PREFILL_FUSED_GU` 0 (A) vs 1 (B).
- All runs executed 144 QPN plan layers, 362 QPN groups, and 4863 QPN experts. No nonzero exits. This tests **staged 768-row prefill**, not default 1024 stream-all or broad model quality.

## Matched results

| Pair (control index / fused index) | Original prefill ms | Fused prefill ms | Original tok/s | Fused tok/s | Prefill speedup | Original TTFT ms | Fused TTFT ms | First-logit KL | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 / 1 | 14173.0 | 13937.7 | 144.43 | 146.87 | 1.01688x | 55929.9 | 54684.0 | 0 | match (760) |
| 3 / 2 | 14147.0 | 13993.9 | 144.69 | 146.28 | 1.01094x | 54734.3 | 54550.3 | 0 | match (760) |
| Median per mode | 14160.0 | 13965.8 | 144.56 | 146.57 | **1.01391x (+1.39%)** | — | — | — | — |

Prefill throughput = 2047 / (prefill milliseconds / 1000). Median throughput shown from the median prefill latency, not a startup-to-first-token throughput estimate. TTFT varies due to startup and model/expert loading; **do not attribute TTFT changes to fusion**. Decode was 29.13–29.43 tok/s, not a meaningful improvement target for this staging-only change.

Both first-logit vectors matched exactly within the QPN-base/fused pairs (SHA-256 `be65c49bdea5749a305c928d971a91f187ed3d0e16cb5e75450fdbb84a2dcfa7`; 248,320 logits; relative L2=0, KL=0, top-1 same). This is **not** a claim that either QPN variant aligns with the original MMQ fallback or passes broader numerical qualification. Only two matched pairs were measured, so +1.39% is a small observed improvement, not a confidence-bound claim.

## Next controlled experiment

Use the **same branch and pinned model/prompt** to run a second A/B/B/A with **original MMQ fallback as A** and **fused QPN as B**, explicitly `--performance-only` (omit `--fusion-ab`). Preserve nonfinite-output, process-exit, missing-logit, and actual-QPN-execution hard gates; record KL/argmax honestly without rejecting a performance-only result solely for KL > 0.03. Keep the original MMQ path and all default-off dispatch unchanged. Compare paired prefill only; no cross-run TTFT attribution.
