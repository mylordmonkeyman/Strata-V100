# Original MMQ fallback vs fused QPN: performance-only full-model ABBA

Date: 2026-10-29 UTC. **Experimental branch only**: `qpn/performance-first`.
Source commit: `e405d51507046bc46f1d44b6328b226f5ae34465`.
[Successful model run](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40962557826); summary artifact `qpn-full-model-40962557826-1` (aggregate only).
Previous [QPN-base vs fused-QPN report](qpn-fused-gu-performance-first-2026-10-29.md) is a **separate A/B experiment**, not a cross-run comparator.

## Pinned setup and comparison semantics

- Four runs in the same job/binary, **MMQ / fused QPN / fused QPN / MMQ**, using the exact same prepared model and prompt, `--prefill 768`, 2047 reported prefill tokens per trial, 4096 context, 16 new tokens.
- Source SHA `e405d51507046bc46f1d44b6328b226f5ae34465`; built binary SHA-256 `1c303837cf68124699322b7cf25862c263bfc73ad36bf000b7a3d689be04a497`.
- Expert pack SHA-256 `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`; prompt SHA-256 `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`.
- Common flags: `STRATA_PREFILL_NVFP4=w4a8`, `STRATA_QPN_PREFILL_PLAN_CONTROL=1`, `STRATA_PREFILL_GROUP_GATHER=1`, `STRATA_PREFILL_STREAM_MIN=1024`, `STRATA_PREFILL_CPU_SHARE=0`, MMQ direct/group-sort/resident-sort all 0.
- MMQ arm: `STRATA_QPN_PREFILL_EXPERIMENTAL=0` and `STRATA_QPN_PREFILL_FUSED_GU=0`; fused arm: both 1. MMQ executed zero QPN groups; fused arm executed 362 QPN groups and 4863 QPN experts across 144 layers. All exits were 0 and all logits finite.
- Both MMQ runs reported host plan 368 groups / 4855 experts; both fused runs reported 362 groups / 4863 experts. **Actual routed work is not perfectly identical across backends** because numerical differences propagate into routing. Thus this is a matched end-to-end comparison of same inputs, not a kernel-only timing experiment.

## Results

| Pair (MMQ index / fused index) | MMQ prefill ms | Fused prefill ms | MMQ tok/s | Fused tok/s | Fused speedup | MMQ TTFT ms | Fused TTFT ms | First-logit KL | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 / 1 | 14060.5 | 13981.4 | 145.58 | 146.41 | 1.00566x | 55846.8 | 54792.7 | 0.03134775 | match (760) |
| 3 / 2 | 14056.8 | 14174.8 | 145.62 | 144.41 | 0.99168x | 54660.5 | 54748.5 | 0.03134775 | match (760) |
| Median per mode | 14058.65 | 14078.10 | 145.60 | 145.40 | **0.99862x (−0.14%)** | — | — | — | — |

Throughput is calculated from 2047 / prefill seconds, not TTFT. TTFT includes variable startup/loading and cannot be used to claim a kernel win. The paired prefill results disagree in sign and the median is effectively parity: **no established fused-QPN end-to-end speed advantage over original MMQ fallback** at this staged workload.

MMQ first-logit SHA-256 `ea59f668290eb6e98db07e7ab3f910636080b108e626ad9fed11fc470e14f178`; fused QPN `be65c49bdea5749a305c928d971a91f187ed3d0e16cb5e75450fdbb84a2dcfa7`. Relative L2 `0.0737182921`, KL(MMQ||fused) `0.0313477502` (exceeds 0.03), argmax same. This run is **performance-only; numerical quality is explicitly unqualified**, not silently waived. Nonfinite and execution/safety gates remain hard failures.

## Next steps

Do **not** enable fused QPN by default or declare it faster. First measure launch overhead independently on the V100 and investigate per-expert kernel occupancy (prefer Nsight Compute if installed, otherwise clearly label occupancy estimates). Focus on per-group launch count, small T=17..32 expert occupancy, and actual grouped workload. Only pursue a new grouped/ragged kernel after preserving routing and sanitizer gates. Keep original MMQ fallback untouched.
