# Rev 4.2 R0 / R1: headroom measured, actual routing/critical-path gates open

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

## Decision/next gate

1. Confirm actual route-dump collection support for the pinned engine. Obtain a bounded training and independent held-out *prefill plus actual decode* sample privately; instrumented/no-trace parity and overhead are required. Do not assume trace flags from an unrelated older build still work.
2. Run this host evaluator and the existing dual-representation capacity analyzer against private data; record useful full 16-expert group counts, mispredictions, and restoration byte bounds. Add canonical-cache miss/slot lifetime changes if selection extends beyond already-resident identities.
3. Measure exposed stage/copy/compute critical path. A cache proposal must show plausible **≥5% full-workload gain** after decode restoration and fallback, with uncertainty excluding parity. If those observations are unavailable or negative, record R1 as unqualified/stop and do not write GPU cache code.

All modifications remain experimental/default-off on the authorized branch. Main, canonical MMQ/default routing, original expert pack and cache, user services and driver/runner permissions are untouched.
