# Pinned private mixed-routing ownership verification — 2026-10-29

## Reproducibility

- Isolated source branch: `qpn/mixed-route-bench`, commit `f8e13fc54f311c1955d79a7662d657089488dc7a`.
- Pinned unchanged-engine M0 run [40815727472](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40815727472), metadata artifact **12872852053**, baseline manifest SHA-256 `218ae1f7fa5bb03ee01aba98278848e13ce17478957ec2300a0e58272af8270d`.
- Captured checkpoint revision `b9ce18a6887e18c089fdea8509a33e63429a3300`.
- Independent V100 runner [verification run 40901602807](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40901602807): **success**. Uploaded metadata-only artifact **12909816673** (`summary.json`). Five negative/positive unit tests passed, and the actual private capture SHA-256, exact per-token routing uniqueness, group membership, full slot coverage, manifest expert-count histogram and per-layer ownership checks all passed.
- Raw activations, full token-to-expert plans and model weights remained private under the runner's /srv/strata-qpn state. The published summary contains only aggregate counts and SHA-256 digests.

## Verified exact routing-plan counts

| Layer | Active expert groups | Routed token-expert slots | Complete coverage | Private plan SHA-256 |
|---|---:|---:|---|---|
| 0 | 204 | 10,240 | Yes | `434d810bb4cbd542407652b6d26b5379cb6c8debc71188bae4a736d4ac77d140` |
| 24 | 336 | 10,240 | Yes | `cb2c5889ece5c85ffee88c96e231e6d05118e75b6877dc6009103ffb0be47327` |
| 47 | 219 | 10,240 | Yes | `ebab5dc2e6137eb4ecbd445c3fdcb26e24eef281604e8eea7b06c546e9470cd1` |
| **Total** | **759** | **30,720** | **Yes** | Per-layer plans |

Each layer is a **1,024-token first-prefill chunk with top-k 10**. Every slot was assigned exactly once to its true expert and its correct token row; there were no duplicate expert IDs within any token and no missing or duplicate slots. The verified aggregate size buckets match the earlier host-only [routing-headroom audit](2026-10-29-routing-headroom.md): 17..32 tokens = 113 groups/2,841 entries; >=33 tokens = 249 groups/25,941 entries.

## Scope and outstanding gates

**Host routing-plan verification passes.** This is not GPU grouped execution. No grouped QPN/MMQ throughput, grouped numerical parity, >L2 weight-supply timing, graph replay, Compute Sanitizer, or end-to-end TTFT was measured by this run. Do not infer a model speedup.

Next: build a **private stratified multi-expert fixture pack** from these verified plans and canonical expert weights, preserving exact nonuniform group sizes and slot ownership. Measure two matched standalone GPU workloads: (A) all eligible groups through the existing W4A8/MMQ comparator; (B) QPN only where independently qualified, with identical MMQ fallback for >32 and other ineligible groups. Include group preparation and launch overhead, distinct weights exceeding V100 L2, numerical parity, CUDA graph changed-input checks and Compute Sanitizer before any opt-in runtime dispatch. Never write private weights/activations into Actions artifacts or alter production engine paths.
