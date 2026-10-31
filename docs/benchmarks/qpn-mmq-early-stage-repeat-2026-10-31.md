# Matched-H2D MMQ early-staging repeat: full-request materiality not met (2026-10-31)

**Decision: STOP this early-CPU-staging optimization subtrack at the rev4.2 >=5% full-request gate.** The default-off canonical MMQ candidate genuinely omitted CPU staging and improved prefill/TTFT on the measured prompt, but did not meet the declared repeatable full-process materiality target. No QPN arithmetic ran. Do not run another early-stage GPU sweep to rescue this result. Production remains unchanged.

## Provenance and exact scope

- [Successful one-shot V100 run 41180483211](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41180483211), source `b0fbc2ed1fb508c8e5d084de32010df27b00fdd9`, [allowlisted aggregate artifact 13035137913](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41180483211/artifacts/13035137913), ZIP SHA256 `e3b41e62cb7aacb782002289d6a1d0b27311c094a979d3f14ccb1f53c7725608`.
- Model SHA256 `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`; binary SHA256 `5c1e88868dcfa957ea6ae9a745defa85c95d4bef575f292ac600961bf063f860`. Original pack and model remain unchanged.
- Host and GPU workflow jobs passed; two fresh-process warmups (one per arm), then three balanced four-run blocks (six measured fresh processes per arm). One fixed 2,047-token prefill prompt, 128 requested generated tokens, same canonical NVFP4/MMQ source and 1024 stream-all first chunk plus staged 1023-token tail. `--spec` target-pass/accepted-token counters unavailable.
- A: `STRATA_QPN_MMQ_SELECTIVE_SUPPLY=1`, `STRATA_QPN_MMQ_EARLY_STAGE=0`. B: identical with early stage `=1`. Both have `STRATA_QPN_MMQ_MATCHED_H2D=1`, which deliberately waits for current-layer route publication before H2D decisions and **removes speculative copy lookahead**. This is a diagnostic policy, not the original asynchronous production policy.
- Shared controls include `STRATA_MMQ_KQ=1`, `STRATA_PREFILL_MMQ=1`, `STRATA_ARENA_LOCK=0`, `STRATA_ARENA_PIN_GIB=1`, `STRATA_PREFILL_NVFP4=w4a8`, `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_QPN_PREFILL_EXPERIMENTAL=0`, `STRATA_PREFILL_STREAM_MIN=1024`, `STRATA_PREFILL_ISSUER=1`. Single GPU representation per expert throughout.

## Measured six-per-arm aggregate

| Metric | A selective H2D | B + early CPU stage | B vs A |
|---|---:|---:|---:|
| Median prefill latency | 17,085.25 ms | 12,510.60 ms | **-26.78%** |
| Median TTFT | 58,119.8 ms | 53,790.2 ms | **-7.45%** |
| Median full-process wall | 84.5988 s | 80.8110 s | **-4.48%** |
| Median decode rate | 27.405 tok/s | 26.585 tok/s | **-2.99%** |
| p95 prefill latency (reported six-sample order statistic) | 17,283.6 ms | 18,158.1 ms | worse |
| p95 TTFT | 58,569.5 ms | 58,937.7 ms | worse |
| p95 full-process wall | 85.5514 s | 86.9063 s | worse |

The balanced four-run block full-process latency reductions were **5.14%, 4.40%, and 1.37%**; only one exceeded 5%. `all_blocks_full_process_5pct=false`. The candidate had one 18,158.1-ms prefill / 86.9063-s wall outlier, so even its p95 wall result is worse. The decode-rate change (-2.99%) does **not** cross the predeclared 5% regression stop, but remains a measured unfavorable shift; no accepted-draft or target-pass normalization was available.

**Exact first-chunk supply matched in all twelve timed runs:** 15,347 entries, **8,880 H2D copies**, 6,467 H2D skips, **24,551,566,080 H2D bytes**, 6,027 staging jobs, 2,713 skipped-copy jobs. B omitted **2,452–2,526 CPU staging jobs**, **6,779,328,832–6,983,925,216 bytes (6.31–6.50 GiB)** of real CPU preparation; A omitted none. Thus the saved host work is real and is not an artifact of fewer H2D transfers.

All measured processes returned 0; all first-logit comparisons were bitwise identical across 248,320 FP32 logits; no QPN groups executed. First-logit identity does not establish generated-token parity, held-out quality or independent FP64 oracle equivalence.

## Interpretation, limits and action

The CPU staging omission reduces measured prefill latency materially, but median **whole-request** improvement is only 4.48%, and one of three blocks reaches the 5% threshold. The 2K-only, fresh-process, forced-route-publication diagnostic has uncontrolled page-cache/clocks and no warm-engine sessions or longer-context/accepted-token quality; do not claim a general performance gain or transfer this result to the normal asynchronous issuer. Additional samples might change an estimate, but the rev4.2 **stop rule** is to archive a negative/uncertain gate rather than keep tuning the same subtrack.

**STOP:** no additional early-stage benchmark, no production enablement, no change to original async/default behavior, and no reinterpretation of these canonical-MMQ savings as QPN tensor-core gains. Preserve source and artifact for reproducibility, default-off.

**Return to the R1 decision boundary:** prior held-out static packed-residency admission covered at most 3.132% of routed entries under a 2-GiB replacement budget (optimistic complete-group bound, not a timing measurement), with substantial canonical fallback/verification demand. Exposed critical-path timing remains unproven. No QPN packed-cache prototype is authorized by these data. A separate **host-only Q4TC source/ABI audit** is permissible; GPU kernel/model experiments require their own gate and must never hold two GPU formats for one expert.
