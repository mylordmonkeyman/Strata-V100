# V100 MMQ stream-all grouped-gather repeatability isolation — 2026-10-30

## Scope and provenance

This is a **numerical repeatability failure investigation**, not a QPN performance or full-model quality qualification. Only the isolated `qpn/prepacked-stream` branch was used; no production/default routing or original model/pack/cache changes are authorized or implied. Source for the decisive forced-staged versus stream-all comparison: `214bb6c4d9118f3e5f2553b2f8bf02d0c3e253b4`, GitHub Actions [run 41007057377](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41007057377), aggregate-only artifact `12957080766`. Both host jobs at this source passed; GPU profile intentionally failed its strict parity gate.

- Immutable expert-pack SHA256: `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`.
- Private prompt SHA256: `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`. Prompt contents were never uploaded.
- Executable SHA256: `d67f4228c86e49caf430108347632452883c59d810f625a201345ebd52a42ccf`.
- Both fresh-process comparisons: `--prefill 1024 --max-context 4096 --max-new 16 --greedy --expert-cache auto --resident-budget-gib 24 --pool-workers 16 --spec 2 --suffix-draft 0 --adapt-every 0 --vram-reserve-mib 1536`; native/dense/PLE paths point only to the runner's private immutable prepared model.
- Control environment: `STRATA_MMQ_KQ=1`, `STRATA_PREFILL_MMQ=1`, `STRATA_ARENA_LOCK=0`, `STRATA_ARENA_PIN_GIB=1`, `STRATA_PREFILL_NVFP4=w4a8`, `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_QPN_PREFILL_EXPERIMENTAL=0`, `STRATA_PREFILL_ISSUER=1`, `STRATA_PREFILL_GROUP_GATHER=1`; tracing off for every arm.
- A run processes 2047 prefill tokens (the 2048-token pinned prompt minus one), including the second chunk; it is **not** a single 1024-token-only inference.

## Pinned aggregate observations

| Path | `STRATA_PREFILL_STREAM_MIN` | First-logit repeat relative L2 | First-logit KL | Argmax equal | Status |
|---|---:|---:|---:|---|---|
| Forced staged, grouped gather enabled | 4096 | **0.0** | 0.0 | yes | repeatability passes |
| Stream-all first chunk, grouped gather enabled | 1024 | **0.3045612925** | 0.0809781119 | yes | repeatability fails |

All four executions exited 0 and reported 2047 prefill tokens. Fresh-process wall/prefill timings are not used for performance comparisons: no ABBA, cold-cache control or warm-engine equivalence.

Earlier [run 41002088301](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41002088301), artifact `12954402732`, source `f4712fdfd5ce33d540eebc18ed320a0787c28b37`: disabling grouped gather in stream-all restored exact repeatability (relative L2 = 0) with **both** threaded and inline issuers. Earlier [run 41004178389](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41004178389), artifact `12955423052`, source `77582eef00dfd7bfa478c70db453d814a6466a30`: grouped gather still failed with `STRATA_PF_STEP_SYNC=1` (threaded L2 `0.4613410`, inline `0.3163787`). That synchronization applies between steps, not necessarily between individual ring-slot operations.

## Interpretation and next isolated test

The observed failure requires the **stream-all + grouped-gather combination** in these controls. It is not reproduced by forced-staged grouped gather or by stream-all ungrouped. This narrows the search to differences in stream-all ring ownership, grouped gather's partial flush/ready waits, and slot reuse/event lifetimes. It does **not** yet prove which event or kernel is faulty, nor establish independent numerical correctness.

Experimental source `6b5e5910d18c553be081b7d8d24ab5fd41574ef6` adds two strictly opt-in, stream-all-grouped-only controls, default **off**:
- `STRATA_QPN_DIAG_GG_WAIT_EACH=1`: wait on every held slot's copy-ready event rather than only the last slot's event.
- `STRATA_QPN_DIAG_GG_RELEASE_EACH=1`: record distinct compute-consumed events for held slots rather than aliasing all to the last slot's event.

Script `54e947c098d1d90bb63b0ecd1eb3420785914895` and workflow `0bf7995ee701ce0d00822e7c00ac04f93d1a3d44` test ready-each, release-each, and both in fresh-process pairs, using the same SHA-pinned private model/prompt, output finite/size guard, strict first-logit repeatability, and aggregate-only upload. A successful repeatability test is necessary but **not sufficient** for quality qualification. Do not infer QPN gains or full-model speedup from these controls.

No private weights, activations, logits, or prompt bytes are in this report.

## Independent per-slot event controls — completed 2026-10-30

Source `0bf7995ee701ce0d00822e7c00ac04f93d1a3d44`, [GPU run 41012520964](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41012520964), aggregate-only artifact `12958624403`; host qualification passed and GPU parity gate failed as designed. Same pinned expert-pack and prompt SHA256 as above; executable SHA256 `218d63b5e1e0218c1d4a1ef1915d265a50acfbf167827cdbc24aac4ee411dd65`. All six fresh-process runs exited 0, reported 2047 prefill tokens, used `STRATA_QPN_PREFILL_EXPERIMENTAL=0` and tracing off. Strict first-logit repeat comparisons:

| Stream-all grouped diagnostic | Relative L2 | KL between repeats | Repeatability |
|---|---:|---:|---|
| Wait on each held copy-ready event | 0.2803881675 | 0.1257967615 | fail |
| Record each held consumed event | 0.2676527389 | 0.1908974743 | fail |
| Both independent per-slot events | 0.2872697500 | 0.3303256883 | fail |

Neither individual nor combined event changes restored repeatability. These tests do not rule out other races, but weaken the hypothesis that the *last-event aggregation alone* is responsible. Next opt-in test retains grouped stream-all staging/slot protocol but replaces the batched `gather_native_group` launch with its existing serial `gather_native` fallback. If that restores repeatability, inspect batched kernel correctness and its handling of partial flushes; otherwise inspect stream-all grouped ownership and early-flush behavior. No performance or independent numerical quality claim is supported.
