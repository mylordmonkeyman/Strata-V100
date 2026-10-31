# R1 aggregate-evidence cross-check (2026-10-30)

**Result: static single-representation admission STOPPED; full-request R1 timing UNKNOWN.**
This is a reproducible host-only audit of **two preexisting, safe aggregate JSON files**, not another model run or a QPN speed claim. No GPU run, Docker rebuild, cache write, raw routing records, or alternate quantization artifact is needed.

Sources: `benchmarks/qpn/results/2026-10-30-joint-heldout-routing.json` (original-MMQ joint training/held-out routing, run [41133416862](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41133416862), source `2e79bd5ccef53066f706cb321366a2d04084aac7`, ZIP artifact `13015890822`, SHA256 `38704f3eb26f1acd671206d00ef6b8fd118779dc5260638c48efbd16bbee7790`) and `benchmarks/qpn/results/2026-10-30-selective-mmq-supply-abba.json` (MMQ-only selective supply experiment, run [41066208726](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41066208726), source `d777b85efadb5a359ebbb7ebb68e88e475f20cc9`, artifact `12982683441`).

Run in the repository with Python 3; no GPU/CUDA required:

```bash
python3 tools/qpn_r1_aggregate_audit.py \
  --heldout-aggregate benchmarks/qpn/results/2026-10-30-joint-heldout-routing.json \
  --supply-aggregate benchmarks/qpn/results/2026-10-30-selective-mmq-supply-abba.json \
  --report /tmp/qpn-r1-aggregate-audit.json
python3 -m unittest discover -s tools -p 'test_qpn_r1_aggregate_audit.py'
```

The audit validates source model SHA, four mandatory MMQ controls, QPN execution disabled, original-MMQ first-logit parity, two distinct independently authored synthetic prompt hashes, complete held-out routed-entry denominator, all six prefill replacement budgets, mathematical share consistency, and matched four-process MMQ-only ABBA. It rejects corrupt claims of positive QPN execution, mismatched prompt identities, changed mandatory controls or manipulated budget arithmetic. Its JSON output is constructed from an explicit scalar allowlist; arbitrary input keys are not propagated. It emits only input file SHA256 digests and safe aggregate numbers.

## Why the existing data do not resolve timing

| Observation | Existing measurement | What it does **not** establish |
|---|---:|---|
| Synthetic held-out, 737 MiB **replacement payload** | 1.8707% optimistic routed-entry coverage | Full-request time share, real packed group eligibility, allocator fit |
| Synthetic held-out, 2048 MiB **replacement payload** | 3.1320% optimistic routed-entry coverage | Representative independent workload distribution or 5% impossibility |
| Selective MMQ first-chunk H2D | About 32.1% of bytes skipped | Same fraction of exposed PCIe wait, or QPN compute savings |
| Selective MMQ prefill | 17,191.85 ms baseline vs 17,070.40 ms selective (medians); ~0.7065% median latency reduction | Reproducible improvement; cache temperature uncontrolled, small ABBA |
| QPN fixture performance | 1.377x isolated resident fixture | Effective full-model eligible-time acceleration |

The selective MMQ experiment used a different source/binary, mode, and prompt from the joint held-out capture. **Never combine their timings or fractions into a claimed QPN speedup**. In the 1.377x *hypothetical* with no overhead, removable eligible work would need to occupy 18.26% of full-request critical-path latency to meet the 5% target. This is a sensitivity condition, **not** an inference that eligible routed entries account for the same fraction of time.

The existing `tools/qpn_supply_profile.py` / `src/prefill/qpn_supply_trace.hpp` already measure selected layers' H2D, waits, gathers and product envelopes, but the instrumentation is intrusive, samples only layers 0/24/47, and its product envelopes are not hardware occupancy. The event intervals cannot currently establish **eligible, exposed, full-request** compute time. Repeating the existing selective transfer experiment or QPN variant benchmarks would not close that gap.

## Decision / next evidence

Preserve static-admission STOP and R1 UNKNOWN. A future bounded, native-MMQ-only full-critical-path measurement would need to distinguish actually *exposed* and *eligible* expert compute from overlapped CPU/H2D/cache work, attribute complete request and accepted decode separately, and pair with realistic format-transition and memory-state bounds. The small synthetic held-out pair does not qualify a representative distribution. Do not treat format-demand restore-byte counters as necessary or measured DMA. A packed resident-cache/Q4 conversion/GPU kernel prototype requires its own supported feasibility gate and may not be triggered by this document.
## Evidence-test hardening

The audit now requires the exact pinned source commits for both runs, checks
each selection against its **replacement** payload budget using the verified
2,764,816-byte native expert size, reconciles copied/skipped expert counts
with transferred/skipped bytes and staging-job counts, and independently
recomputes the published ABBA prefill medians. The associated unit tests
**fail closed** if either committed aggregate file is absent; they no longer
replace missing benchmark evidence with lookalike synthetic fixtures.
In-memory mutated copies remain useful for negative tests only.
These are integrity checks of the available aggregates, not independent
authentication of the original private GPU measurements or representative
heldout/performance qualification. The R1 decision remains UNKNOWN.

## Cross-phase packed-only static compatibility screen (2026-10-30)

The aggregate audit now distinguishes an optimistic prefill routing
opportunity from a **whole-request, unchanged-native-MMQ packed-only
plan without format changes**. For every published nonzero selected
budget, the held-out routing screen reports both selected packed-expert
prefill canonical-format demands and distinct native-verification
consumers. Accordingly every scenario is labeled
`CONFLICTS_WITH_REFERENCE_NATIVE_CONSUMERS`. This is a conditional
logical incompatibility with *unchanged* native readers and no format
transition. It does NOT prove that an on-the-fly packed-layout decoder,
a safe strictly sequential conversion, or a different one-copy format
is impossible or slower.

The original synthetic held-out screen at 2048 MiB selected 768
experts, observed 1,198 prefill canonical-format demand instances
and 203 distinct selected experts actually consumed during native
verification. The demand counts are **not** DMA/conversion counts and
must never be added to compute latency as though transfers occurred.
The audit uses the previously committed immutable aggregate files,
not a new model run or private expert identities.

A future R1 prototype-candidate decision requires a **source-grounded,
bounded** single-format native-access plan as well as independent
held-out, memory and exposed critical-path evidence. The corresponding
R3 execution-quality gate is separate. No mixed-format same-expert
double residency is permitted, even temporarily. Static admission
remains STOPPED; R1 remains UNKNOWN.
