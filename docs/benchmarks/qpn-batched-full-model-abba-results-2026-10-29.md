# Grouped ragged QPN versus fused QPN: staged full-model V100 ABBA result (2026-10-29)

## Scope and provenance
- **Experimental branch only:** `qpn/performance-first`, tested source commit `a12702107ce29d4dc4c865d3640674a50fe82778`.
- [GitHub Actions V100 run 40968226415](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40968226415), model-ab and host-policy **success**; [host/CUDA run 40968226388](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40968226388) **success**.
- Private-safe metadata artifact `qpn-full-model-40968226415-1`, ID `12939855987`, SHA256 digest `0e919210b8bc43b9f9e7f3c4575215e30130c1ccfceaf88f83df9c16449a40b5`; no raw model, prompt, logits or runner logs exported.
- Binary SHA256 `fb26a3e2ab4c2b496d1a5f0bb5e1b74590733a3eb8ac52232f00f0e351e1d601`; expert pack SHA256 `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`; prompt SHA256 `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`.
- Same prepared full model, 2047 prefill tokens, pinned staged `--prefill 768`, ABBA order fused / batched / batched / fused; control flags `STRATA_PREFILL_NVFP4=w4a8`, `STRATA_PREFILL_GROUP_GATHER=1`, `STRATA_PREFILL_STREAM_MIN=1024`, `STRATA_QPN_PREFILL_PLAN_CONTROL=1`; MMQ direct/group sort/resident sort and CPU share all 0.
- 144 selected layers, 362 QPN groups and 4,863 QPN experts in **each** arm. All four runs returned 0; first-logit files identical SHA256 `be65c49bdea5749a305c928d971a91f187ed3d0e16cb5e75450fdbb84a2dcfa7` (248,320 finite float32 logits; top-1 index 760).

## Matched performance (ms; lower is better)
| Trial | Mode | Prefill ms | TTFT ms | Decode tok/s |
|---|---|---:|---:|---:|
| 0 | fused QPN | 14159.0 | 56094.2 | 29.17 |
| 1 | batched QPN | 13998.7 | 54665.1 | 29.33 |
| 2 | batched QPN | 14086.0 | 54597.0 | 29.26 |
| 3 | fused QPN | 14122.3 | 54684.2 | 29.47 |

- Pair 0→1: **1.011451x** prefill (+1.145%); pair 3→2: **1.002577x** (+0.258%).
- Median fused prefill **14140.65 ms**, batched **14042.35 ms**, median speedup **1.007000x** (+0.700%). This is a small two-pair observation, not a statistical performance guarantee.
- First-logit KL **0** and relative L2 **0** in both pairs, with identical argmax. This is **between the two QPN variants only**; it does not establish MMQ/model-reference numerical quality.
- TTFT includes startup and weight/model overhead; **do not** use its difference to infer kernel speedup. Prior standalone synthetic 1.68x also cannot be extrapolated to full model.
- Workflow records `performance_only=true`; the summary's `numerical_quality_qualified=true` refers to the between-QPN threshold only, **not broad model quality certification**.

## Decision and next gate
- Grouped QPN wins narrowly over fused QPN in both matched pairs, warranting a **new, direct MMQ → grouped QPN → grouped QPN → MMQ** ABBA under identical pinned supplies.
- The separate direct-MMQ workflow and plan have been committed on this branch; do not infer an MMQ advantage until that run completes and its metadata are inspected.
- Keep grouped QPN **off by default**, original MMQ fallback and per-expert QPN unchanged. Do not modify main, model packs, expert cache, native runner or host services.
