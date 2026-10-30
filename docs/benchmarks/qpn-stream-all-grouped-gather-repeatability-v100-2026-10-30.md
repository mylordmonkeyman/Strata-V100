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


## Same-binary cross-arm first-logit parity — completed 2026-10-30

Source `634c37fd410e20ea393f885c0cc05b8c297d038c`, [GPU run 41029328581](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41029328581), aggregate-only artifact `12966645331`. Host qualification passed. The GPU workflow failed the intentional **both-on / staged-only repeatability** gate; this does not invalidate the separately recorded passing cross-arm comparisons. The SHA256-verified expert pack was `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`, private prompt `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`, executable `893be8a494c75732eb5928021f9ede902d6bd3249cd5166185127e9c8c598039`.

The `grouped-stream-only` (grouping first 1024-token stream-all chunk, ungrouped 1023-token staged continuation) and `grouped-both-off` controls each repeated bitwise exactly within their two fresh-process pairs. Their **cross-arm** comparisons also matched bitwise in *both* trials: 248,320 first-logit elements per comparison, relative L2 `0.0`, KL `0.0`, argmax equal. Conversely, `grouped-staged-only` repeat L2 was `0.2882365537` (KL `0.3535674901`) and `grouped-both-on` repeat L2 `0.4086562706` (KL `0.1583166533`). All eight processes exited zero, processed 2047 prefill tokens, reported zero QPN execution, and ran with tracing off.

**Interpretation:** This supplies a bitwise *within-model MMQ* reference for an experimental transition-scoped fallback: retain grouped gather in the stream-all chunk and use the existing ungrouped staged path afterward. It is not an independent FP64 GGUF oracle, does not establish numerical quality, and does not qualify QPN or full-model speed. A default-OFF fallback tracking the preceding stream-all chunk *within one run_impl invocation* was committed at `8a7c437429d34dd2ea66147ed43dbc1446857587`; follow-up diagnostic counter validation at `054049a84627d51c12382969e8cba5d9701bdf72`. The fallback requires new pinned V100 repeat/cross-arm testing before any broader qualification. Neither production defaults nor caches/model packs were changed.


## Transition-scoped fallback V100 validation — completed 2026-10-30

Source `cb3abb7cfac28474bf00f669efd3860893a6b4d1`, [GPU run 41046702760](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41046702760), aggregate-only artifact `12973173125`. Both host qualification and V100 profile jobs completed **successfully**. The immutable expert-pack SHA256 was `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`, private prompt SHA256 `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`, and executable SHA256 `1b8cf9f5ec02c92e4e4a18e2bbe430b15e723bfc10fbb435f8714b5db927b908`.

The same-binary experiment ran **two fresh-process repetitions each** of (i) the default-OFF opt-in transition fallback, (ii) grouped stream-all only, and (iii) grouped gather disabled for both chunks. Each of six runs exited zero, processed all **2047 prefill tokens**, reported **zero QPN execution**, and had tracing disabled. Both transition-fallback runs recorded `disabled_staged_layers=48`, proving the intended bypass was exercised in every staged MoE layer.

All three configurations had **bitwise-identical first-logit vectors within each pair**, with relative L2 `0`, KL `0`, and matching argmax over **248,320** float elements. The transition fallback also matched the both-off reference **bitwise in both cross-arm trials**, as did grouped-stream-only versus both-off (four cross-arm comparisons; each L2 `0`, KL `0`, argmax equal). Transition fallback prefill times were 17401.2 and 17423.9 ms; grouped-stream-only 17087.7 and 17198.9 ms; both-off 17737.6 and 17146.1 ms. These **unpaired fresh-process** timings are *not* a credible speedup measurement; no cold-cache eviction, matched ABBA, or warm-engine control was performed.

**Qualification boundary:** This proves the opt-in fallback actually executes and reproduces the previously stable **MMQ first-logit output** for this pinned two-chunk workload. It does **not** prove independent GGUF FP64 correctness, output guards for all intermediate kernels, full-model CUDA Graph replay, sanitizer cleanliness, QPN numerical quality, or QPN speedup. The fallback remains disabled by default, and no production code, default MMQ route, original model/pack/cache, drivers, host services, or runner privileges were changed. Next: controlled MMQ-only A/B/B/A first-logit and timing comparison with the same pinned inputs, followed by independent kernel/oracle and sanitizer qualification before full-model QPN claims.


## Pinned MMQ-only transition fallback A/B/B/A — completed 2026-10-30

Source `20ae4b3fe4bd9608c059d0e4f7e0e996a8cbb432`, [GPU Actions run 41054076458](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41054076458), **aggregate-only artifact `12976497015`**. Both host and V100 jobs completed successfully. Expert-pack SHA256 `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`, private prompt SHA256 `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`, executable SHA256 `f4049fc495b8a494bd72e0b16671134d6fb52df5c8bd1cfda304f661c7c82085`.

The same binary executed **A/B/B/A = both-off / transition-fallback / transition-fallback / both-off**, with `--prefill 1024`, `STRATA_PREFILL_STREAM_MIN=1024`, `STRATA_PREFILL_GROUP_GATHER=1`, `STRATA_PREFILL_ISSUER=1`, `STRATA_QPN_PREFILL_EXPERIMENTAL=0`, `STRATA_QPN_SUPPLY_TRACE=0`, and the pinned 2048-token prompt. The both-off arm additionally set `STRATA_QPN_DIAG_GG_DISABLE_STREAM=1` and `STRATA_QPN_DIAG_GG_DISABLE_STAGED=1`; the fallback arm instead set `STRATA_QPN_DIAG_GG_TRANSITION_FALLBACK=1`. All four fresh processes exited zero, processed **2047 prefill tokens**, and reported **zero QPN execution**. Each fallback execution recorded `transition_disabled_staged_layers=48`.

All **six** comparisons (two within-arm and four cross-arm) were bitwise identical over **248,320 first-logit elements** each: relative L2 `0`, KL `0`, argmax equal. Private logits, raw logs, weights, activations, and prompts were not uploaded.

| Metric | A: both-off runs | B: transition-fallback runs | A median | B median | A/B median ratio |
|---|---|---|---:|---:|---:|
| Prefill (ms) | 17336.1, 17401.1 | 17373.9, 17042.4 | 17368.6 | 17208.15 | **1.009324** |
| TTFT (ms) | 63568.7, 58818.7 | 59311.0, 58509.0 | 61193.7 | 58910.0 | **1.038766** |

**Interpretation:** The opt-in transition fallback is repeatable and bitwise MMQ-first-logit equivalent to the ungrouped reference under this pinned two-chunk test. The ratios above are *descriptive only*, **not validated speedups**: just two fresh-process samples per arm, uncontrolled page-cache temperature, and no warm-engine equivalence. The large TTFT spread and variable decode rates further caution against performance inference. No independent GGUF FP64 full-model oracle, CUDA Graph replay, full-model sanitizer, or QPN model-quality qualification was performed. The existing separate real 16-expert fixture tests do not establish full-model correctness or speed. Preserve the default-OFF flag and production MMQ routing; next validate independent fixture/kernel FP64, output guards, changed-input Graph, and three-iteration two-slot memcheck/racecheck before any QPN speed claims.
