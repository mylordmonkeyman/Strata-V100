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

## Serial gather fallback control — completed 2026-10-30

Source `d49359275c18ca4e2fc5dd3e298f5199fe86fa0e`, [GPU run 41018252300](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41018252300), aggregate-only artifact `12960389110`; host qualification passed and GPU parity gate failed as designed. Immutable expert-pack SHA256 `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`; private prompt SHA256 `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`; executable SHA256 `c2afe0e0582662d3b1a2241bce0ef0d83fe9e7692a1483e51f3e65c53d74de7d`.

Both fresh-process runs exited 0, processed 2047 prefill tokens, reported zero QPN layers/groups/experts, and used `STRATA_PREFILL_GROUP_GATHER=1`, `STRATA_QPN_DIAG_GG_SERIAL_GATHER=1`, `STRATA_PREFILL_STREAM_MIN=1024`, `STRATA_PREFILL_ISSUER=1`, tracing off, and the common control environment and CLI above. Their first-logit repeat comparison over 248,320 elements gave **relative L2 0.2436912115**, **KL 0.0834915281**, argmax equal. Substituting serial `gather_native` for batched `gather_native_group` while retaining stream-all grouped slot ownership **did not** restore repeatability. Thus the batched gather kernel alone does not explain the failure. This does not prove the precise ring-lifetime defect or independent numerical correctness.

Next isolation source `d569bb797fbb5f4e46ef5c5a264b1ca3b54a8a68` introduces default-off `STRATA_QPN_DIAG_GG_DISABLE_STREAM` and `STRATA_QPN_DIAG_GG_DISABLE_STAGED` flags, permitting the same binary to run grouped gather only in the stream-all first chunk, only in staged continuation, in neither, or in both. The script and workflow commits `27a4f46f3cacffeec24f30d58c5da55014baf7aa` and `ca3b27639a21be5326e88483f957e26ae1eb1b28` perform four strict fresh-process repeatability pairs. These controls are diagnostic only, with no production default changes, speedup claim, or QPN numerical qualification.

## Chunk-specific grouping isolation — completed 2026-10-30

Source `ca3b27639a21be5326e88483f957e26ae1eb1b28`, [GPU run 41024609613](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41024609613), aggregate-only artifact `12964136922`; host qualification succeeded, and GPU job intentionally failed the strict repeatability gate. Model SHA256 `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`; prompt SHA256 `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`; executable SHA256 `7abb4794b21e688c609c1683202a33069957b066deb49fcdd7fc1ae93724d867`. Each of eight fresh-process executions exited 0, processed 2047 prefill tokens, reported zero QPN execution, and used tracing off, the same pinned native/dense/PLE model and control CLI above. Both per-arm runs were made with `--prefill 1024` and `STRATA_PREFILL_STREAM_MIN=1024`, so the first 1024-token chunk uses stream-all and the 1023-token continuation uses staged supply.

| Grouped gather in first stream-all chunk | Grouped gather in staged continuation | First-logit repeat relative L2 | Repeat KL | Repeatability |
|---|---|---:|---:|---|
| enabled | disabled | **0.0** | 0.0 | pass |
| disabled | enabled | **0.2833384451** | 0.1719622213 | fail |
| disabled | disabled | **0.0** | 0.0 | pass |
| enabled | enabled | **0.4051989626** | 0.1543223575 | fail |

**Decisive refinement:** In this 2047-token two-chunk workload, the failure tracks **grouping in the staged continuation following a stream-all first chunk**; grouping only the first stream-all chunk is repeatable. This corrects the earlier provisional attribution to stream-all grouped gather itself. Prior forced-staged grouping with `STRATA_PREFILL_STREAM_MIN=4096` was also repeatable, so the interaction appears to involve the transition into staged grouping, not staged grouping in isolation. The exact cause remains unproven (cross-chunk ring state, reused events/slots, or related buffer lifetime are hypotheses). These comparisons are **within-arm repeatability**, not cross-arm numerical equality or an independent GGUF reference check. First-logit argmax matched within all four pairs, which does not qualify numerical quality. No full-model QPN performance claim is supported.

An additional source-only run `41024539299` at `d569bb797fbb5f4e46ef5c5a264b1ca3b54a8a68` (artifact `12962734969`) used serial gather with first-logit repeat L2 `0.2646828598`, consistent with the earlier serial-gather failure. It is not the decisive four-way experiment. The next validation should compare the two repeatable paths **across** arms using only aggregate L2/KL/bitwise status (no raw logits uploaded), before considering a scoped fallback. Preserve production MMQ defaults and all existing GPU/sanitizer gates.
