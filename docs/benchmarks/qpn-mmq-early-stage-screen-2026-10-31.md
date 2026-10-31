# MMQ early CPU-stage A/B/B/A: promising but H2D-unmatched (2026-10-31)

**Decision: exploratory positive signal; FAILED comparability gate; no full-request performance qualification or production promotion.** This is a canonical-weight MMQ supply experiment, not a QPN-kernel speed result. The experimental early-stage path remains default-off and uses one GPU weight representation per expert.

## Provenance and execution

- [V100 one-shot run 41168604717](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41168604717), source `de8b6147cdd33f12abb71fb1701b57607f71d2f1`, [sanitized summary artifact 13030112209](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41168604717/artifacts/13030112209), ZIP SHA256 `6ac5bca2e0e599bbfcbf9cccbd897066e5501549c137a461a42d73edf4370a7d`.
- Model SHA256 `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`, binary SHA256 `26637d9684555a8c6ac5d1986f0ca7a13afaf7aed0eb48d82cf38fc9b2fc2025`. Private prompt/model/logits remain private; only the published allowlisted aggregate is used.
- CUDA 12.8 build and host validation passed. All four *model* processes completed with return code 0 and 2,047 prefill tokens each, first 1,024-token stream-all chunk followed by a staged 1,023-token tail. The A/B/B/A harness then returned exit code **3** for `failed_mismatched_H2D`, **not** an execution, build, numerical or GPU error.
- A = selective H2D skipping only (`STRATA_QPN_MMQ_EARLY_STAGE=0`); B = same selective H2D plus published-route CPU staging omission (`=1`). Common controls: `STRATA_MMQ_KQ=1`, `STRATA_PREFILL_MMQ=1`, `STRATA_ARENA_LOCK=0`, `STRATA_ARENA_PIN_GIB=1`, native NVFP4 W4A8, no QPN execution, no CPU share, `--prefill 1024`, `--spec 2`, 16 generated tokens. No second packed weight representation or sidecar.

## Sanitized measurements

| Arm / order | First-chunk H2D copies | CPU jobs omitted | CPU bytes omitted | Prefill ms | TTFT ms | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| A1, H2D-only | 10,420 | 0 | 0 | 17,236.3 | 65,995.6 | 30.95 |
| B1, early stage | 10,413 | 2,447 | 6,765,504,752 | 12,556.8 | 54,183.8 | 28.83 |
| B2, early stage | 10,422 | 2,443 | 6,754,445,488 | 12,346.5 | 53,477.8 | 28.84 |
| A2, H2D-only | 10,289 | 0 | 0 | 27,499.3 | 68,637.6 | 31.02 |
| **A median** | — | — | — | **22,367.8** | **67,316.6** | **30.985** |
| **B median** | — | — | — | **12,451.65** | **53,830.8** | **28.835** |

- **CPU staging suppression actually executed:** 2,443–2,447 jobs and 6.29–6.30 GiB omitted in the first streamed chunk. Both arms scheduled the same 6,027 eligible host jobs and 15,347 stream entries.
- **First-logit equality:** all six within- and cross-arm comparisons were bitwise identical for 248,320 FP32 logits (reported KL and relative L2 both zero). This is not independent FP64 or multi-prompt model quality qualification, and does not prove every generated token/accepted draft matched.
- **Exploratory median timing:** B prefill 44.33% lower and TTFT 20.03% lower than A; even the slower B TTFT is 17.90% below the faster A TTFT. **Do not report this as a qualified 20% speedup.** Only two fresh processes per arm; uncontrolled cache/clocks, cold-start effects, and prefill variability.
- **Decode caution:** B median reported decode rate 28.835 tok/s versus A 30.985 tok/s (6.94% lower). Only 16 generated tokens per process; no accepted-token TPS or warm-engine qualification. A decode regression cannot be ruled out.
- **Comparability failure:** H2D copies were `[10420,10413,10422,10289]`, spread **133 experts / 367,720,528 bytes / 1.28% of mean copied bytes**. Harness explicitly allows at most 16-copy spread, so this is a real failed gate. Baseline A2 alone differs by 131 from A1; this is consistent with timing-dependent published-route lookahead, not proof of different routing. A2 copied *less* yet ran slower, so the observed latency advantage cannot be explained simply by candidate copying fewer experts; the effects of scheduling, page cache and clocks remain unresolved.
- **No QPN executed:** all four `qpn_groups=0`; do not conflate this CPU supply optimization with NVFP4 QPN acceleration.

## Next bounded decision

Retain the default-off path as a **promising unqualified hypothesis**, not an accepted optimization. Do not waive or silently loosen the H2D gate. First investigate how the asynchronous issuer/worker scheduling causes the same-arm 131-copy variation; consider a narrowly scoped, source-controlled *measurement* that separates publication timing from host-copy savings without changing the model or GPU weight representation. A follow-up GPU experiment is justified only with an explicitly predeclared comparable H2D policy, >=5 timed repetitions per arm after warmup, representative longer-context prefill, matched memory and output, and accepted-token decode. Avoid a speculative packed-cache/QPN rerun or a broad sweep. If this cannot establish >=5% repeatable **full-request** benefit without a material decode regression, stop this supply subtrack.

**Scope/controls:** experimental branch only; production `main`, default MMQ/native dispatch, original model pack, cache, runner security and GPU concurrency untouched. No raw private data is included.
