# Rev 4.2 R0 / R1: verification routing measured; held-out and critical-path gates open

**Governing user specification:** `Strata-V100-NVFP4-QPN-Implementation-Specification-rev4.2.md`, SHA256 `a4aa24a64c835b55596ad20664b69a4b96ded6e465f45f5fd3aa7d2f3bfe9d3e`; especially §§1.3, 4.2–4.5 and 11. This is an evidence/status addendum, **not an independent architecture specification**.

## Completed read-only R0 memory measurement

[GitHub Actions run 41094045957](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41094045957), artifact `12994948457`, source `781b2467ab9efc803dd88c46123e1de8f023af1b`, binary SHA256 `c50cc04cfb0d53f3bd17e4cd5cd29cdde553f4579075322de80dc882225a6cba`, original model SHA256 `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`, prompt SHA256 `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`. Original MMQ only (QPN groups 0), staged 1024/1023, 2,047 prefill tokens, all four known NaN workaround flags. The job completed successfully.

- NVML memory samples: **214**, across startup, prefill and decode. Single V100 total **32,768 MiB**; **maximum used 31,246 MiB**, minimum free **1,249 MiB**, last reported free **28,209 MiB**. With the probe's 512-MiB safety guard, its optimistic instantaneous headroom bound is **737 MiB**. Sampling is not an allocator guarantee and cannot establish warm serving or longer-context headroom.
- **Additional VRAM sidecar:** at 1 GiB or 2 GiB, not supported by this observed guarded headroom. Even 512 MiB leaves only 225 MiB beyond the guard, before KV growth and allocation peaks; it is **not a recommended cache allocation**. Do not launch such a prototype on capacity alone.
- **Single-representation replacement** is now the preferred *unproven* hypothesis, consistent with rev4.2; original canonical identities and native decode compatibility must still be accounted for. A 24-GiB configured canonical cache budget is not automatically 24 GiB physically allocated or free.
- The sampled headroom is not a kernel performance result and cannot prove that a fixed-budget format replacement provides any benefit.

## R1 host evaluator (new; no model/cache mutation)

`tools/qpn_resident_r1.py` consumes **private, separately collected** training and held-out route-count snapshots. An input JSON has `schema:1`, `prefill:[{layers:[{layer:0, counts:[[expert_id, routed_rows], ...]}, ...]}, ...]` and `decode:[{layers:[...]}]`. Each `prefill` entry represents one model chunk; each `decode` entry represents one real decode/verification window. Only `--report` aggregates may leave the private runner. Do **not** upload these input JSONs or raw expert IDs; avoid storing prompts/weights/logits in them. Use at least one independently held-out prompt and actual decode windows. This tool does not create trace inputs; their collection must be separately qualified and kept private.

Example with existing private inputs (placeholder local paths, **not** known to exist on the runner):

```bash
python3 tools/qpn_resident_r1.py \
  --profile data/expert-profile.bin \
  --private-training /srv/strata-qpn/private/r1-train.json \
  --private-heldout /srv/strata-qpn/private/r1-heldout.json \
  --budget-mib 128 256 512 737 1024 2048 \
  --report /srv/strata-qpn/aggregates/r1-heldout.json
python3 -m unittest discover -s tools -p test_qpn_resident_r1.py -v
```

The tested **static single-representation policy** picks candidate sets in 16-expert layer-local units from the *currently profile-resident* canonical identities, using training-only row-count frequency. It replaces, rather than duplicates, their payloads, so this bound assumes no canonical identities evicted. Held-out scores count only 16 simultaneously eligible packed experts in the same layer; a grouping upper bound may rearrange members and **does not prove actual Strata group eligibility**. It explicitly counts (a) packed experts routed into groups too small for QPN, requiring canonical fallback, and (b) distinct packed experts needed by the original decode backend, requiring canonical restoration. Restore *bytes* are necessary-cost proxies, **not** measured DMA, wall-time stalls, Graph safety, or a feasible state-transition implementation. Merely installing an extra packed representation cannot be called free. The evaluator rejects identical training and heldout files. It retains no route identities in output, and reports SHA256 of the private sources/profile, budgets and aggregate totals only.

**R1 does not yet pass:** no actual held-out full-model routing, complete group membership, time-critical kernel envelopes, exposed CPU/slot waits, decode-restoration latency, cache occupancy under KV growth, or predictive benefit have been measured. Do not implement `qpn-packed-slot` or queue a costly full-model QPN cache experiment yet.

## Verified R0 verification-window capture (2026-10-30)

The opt-in, read-only GPU `Verifier::run` route snapshot now **works**. [Private verification run 41111215170](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41111215170) succeeded at source `6bb92c47ccaa564d524d5025c11599ac1cba03d5`, aggregate-only [artifact 13004920440](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41111215170/artifacts/13004920440), binary SHA256 `63a8d13d05f9208e72fd0629f79a3287d2f2c614ec00952484949bde916d3c78`, private capture SHA256 `3d112426f23d5049379b8d5517f6e63d4b639b6fd682ae366f597f815de5b81d`. The model/prompt SHA256s and all four mandatory MMQ workaround flags match the above pinned R0 experiment. Both the host preflight and GPU model arms passed. No original weights, prompts, route IDs, logits, or raw traces were uploaded.

| Aggregate verification metric | Measured |
|---|---:|
| Complete `Verifier::run` windows after 2,047-token prefill | 16 |
| Layer records | 768 = 16 × 48 |
| Routed entries | 14,880 |
| Active layer/expert uses across windows | 13,901 |
| Distinct layer/expert identities across windows | 4,160 |
| Largest verification window | 2 tokens |
| Largest per-expert row count within a window | 2 |
| Eligible 33–512-row QPN groups | **0** |
| First-logit MMQ trace-off/trace-on parity | **bitwise equal, 248,320 FP32 elements; L2 = KL = 0** |

The 14,880 routed entries correspond to **31 verification input-token positions** across the 16 windows (15 two-token windows and one one-token window), each across 48 layers with top-10 routing. This is a *single pinned request*, not a sample distribution. Since the largest window contains only two tokens, no expert can have 33 rows in that window; therefore the current **33–512-row QPN dispatch is structurally ineligible for all captured verification windows**. This is stronger than a timing loss for this precise path, but says nothing about alternative cross-request batching or independent prefill routing. The previously tested 1–16-row QPN decode-shaped fixtures lost to native DP4A; do not re-run them or infer an actual QPN decode win.

**Phase and safety qualification:** these are genuine on-GPU verification-router IDs, *not* CPU miss-doorbell IDs. Post-prefill `Verifier::run` windows may contain speculative verification or prompt-tail work, **not necessarily accepted autoregressive token routes**. Capture adds device-to-host copies and is not a performance comparison. This experiment established first-logit parity only, not generated-token parity, independent held-out performance, cache-slot residency, canonical restoration, or QPN model quality. No QPN decode was executed; original MMQ/native decode remained in place.

**R1 decision at this checkpoint:** reject 33–512-row QPN for this observed verification-window geometry. Do **not** build a packed decode/cache-slot prototype on these data. The *prefill-only* single-representation hypothesis remains unqualified, not approved: the previous 21.1% same-trace optimistic eligible-entry bound is not a held-out cache hit rate or time saved; nonresident prepacked full-model QPN already lost; guarded additional VRAM is only 737 MiB. Proceed only after independently held-out prefill and genuinely phase-classified decode samples, measured canonical fallback/restore and exposed critical-path waits, and a plausible ≥5% full-request benefit including transition costs. Otherwise close the R1 subtrack as no-go.

## Decision/next gate

1. Verified: the pinned engine captures genuine post-prefill verifier routing without changing first logits. Still obtain bounded **independent** training and held-out *prefill plus phase-classified actual decode* samples privately, and measure trace overhead; do not relabel the single pinned verification trace as held-out or accepted-token decode.
2. Run this host evaluator and the existing dual-representation capacity analyzer against private data; record useful full 16-expert group counts, mispredictions, and restoration byte bounds. Add canonical-cache miss/slot lifetime changes if selection extends beyond already-resident identities.
3. Measure exposed stage/copy/compute critical path. A cache proposal must show plausible **≥5% full-workload gain** after decode restoration and fallback, with uncertainty excluding parity. If those observations are unavailable or negative, record R1 as unqualified/stop and do not write GPU cache code.

All modifications remain experimental/default-off on the authorized branch. Main, canonical MMQ/default routing, original expert pack and cache, user services and driver/runner permissions are untouched.
