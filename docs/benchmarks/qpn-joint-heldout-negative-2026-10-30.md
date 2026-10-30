# Static single-representation admission: negative held-out routing screen

Source `2e79bd5ccef53066f706cb321366a2d04084aac7`, [successful V100 run
41133416862](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41133416862),
artifact **13015890822**, ZIP SHA256
`38704f3eb26f1acd671206d00ef6b8fd118779dc5260638c48efbd16bbee7790`.
The committed `benchmarks/qpn/results/2026-10-30-joint-heldout-routing.json`
is the unmodified aggregate member. Original model/source, canonical cache,
production defaults and services were unchanged.

Four original-MMQ processes completed: capture off/on for a 1024-token synthetic
compiler prompt used for admission training, then off/on for an independently
authored 2048-token synthetic garden prompt. Actual prefill work was 1023/2047
tokens. Both first-logit comparisons were bitwise equal over the required finite
248320-element arrays; zero QPN execution, all four workaround flags, and complete
prefill layers were required. Each request captured 16 native post-prefill
verification windows. The 9293 observed resident identities were unchanged across
chunks and requests. Raw prompts, logits and routes were destroyed after scoring.

Training-only row-frequency admission selected currently resident identities in
layer-local units of 16. The held-out score optimistically permits regrouping
selected eligible experts. These are replacement payload budgets, **not extra
VRAM allocations**; the previous 737-MiB spare-memory observation is not a cap on
how much canonical payload could be replaced.

| Replacement budget MiB | Selected experts | Held-out eligible complete groups | Routed-entry coverage | Canonical fallback expert uses in prefill | Distinct selected experts consumed by verification |
|---:|---:|---:|---:|---:|---:|
| 128 | 48 | 0 | 0% | 78 | 9 |
| 256 | 96 | 2 | 0.656% | 138 | 14 |
| 512 | 192 | 6 | 1.629% | 245 | 39 |
| 737 | 272 | 7 | 1.871% | 374 | 67 |
| 1024 | 384 | 10 | 2.645% | 525 | 93 |
| 2048 | 768 | 12 | 3.132% | 1198 | 203 |

Coverage denominator: **982560 held-out prefill routed entries**; maximum covered
30774. Native verification consumed 14880 routed entries. The low coverage and
frequent canonical consumer demand do not support committing GPU-cache engineering
to this particular static admission policy. The earlier 21.1% same-trace optimistic
bound used different selection/workload assumptions and is not a comparable
predictive hit-rate baseline.

## Decision and interpretation

**Stop this static single-representation admission subtrack at screening.** Do not
launch a packed resident model cache, conversion benchmark, or repeated kernel
sweep on the strength of these results. Archive the workflow for manual dispatch.
This is an engineering investment decision based on weak measured coverage, prior
streamed full-model losses, and absent critical-path evidence; it is **not** a
mathematical proof that a 5% latency gain is impossible. R1's timing-based decision
remains UNKNOWN because routed-entry shares cannot substitute for exposed time.

The scorer's `optimistic_prefill_restore_bytes` multiplies per-chunk canonical
fallback uses by payload bytes. At 2 GiB it reports 3,312,249,568 bytes; verification
reports 561,257,648 bytes from 203 distinct identities. These are **format-demand
accounting proxies**, not measured DMA, conversion latency, or necessary minimum
transfer traffic. The evaluator does not simulate layout state: an expert restored
to canonical could stay canonical, reducing subsequent restores while also losing
later packed eligibility. Prefill and verification proxies cannot simply be added
as the transfer cost of a realizable policy. Future reuse requires explicit state
simulation or actual measurements; preserve original field names/provenance.

Other limits: one training/held-out pair, synthetic repetitive subject matter,
different prompt lengths, fixed resident identities, and optimistic regrouping.
Counts do not establish actual Strata group membership. There is no representative
prompt distribution, generated-token parity, QPN decode, candidate timing,
conversion/scratch memory measurement, or model-quality qualification. Diagnostic
trace-off/on times are fresh-process observations, not a performance A/B.

## Further work

No automatic new QPN variant follows. A future exact per-layer selector or
packed-compatible native consumer would be a materially different hypothesis and
must first show exposed resident-compute opportunity and account for lost copy
lookahead, activation preparation, regrouping and canonical consumers. Do not
implement these merely to avoid accepting the static-policy negative result.

The independent Q4TC track may proceed with the rev4.2 host-only type/scale/artifact
and licensing audit before any new GPU kernels or model conversion. Alternatively,
measure the existing MMQ/native critical path if continuing broad performance
optimization. Preserve the separately gated within-Q4 kernel comparison and
cross-quantization quality test; no below-four-bit quantization or NVFP4 byte
reinterpretation is authorized.
