# Existing MMQ supply diagnostic (revision 4.1 continuation)

Branch: `qpn/prepacked-stream`; parent `528105efb41e67435697235df50ee3802d632ed6`.
This is an opt-in measurement step, not a new expert dispatch implementation.

## What is measured

Build with `STRATA_QPN_PREFILL_EXPERIMENTAL=ON`; enable
`STRATA_QPN_SUPPLY_TRACE=1` on a single GPU. The runtime switch is off by
default and does nothing without the existing experimental build flag.
The hooks execute the original CUDA operation exactly once on its original
stream; no source weights, cache entries, routing decisions or ring fences change.

Sample layers **0, 24, 47**, at both **768 staged** and **1024 stream-all**.
CUDA event timestamps share a copy-stream origin per chunk. Intervals include:

- Actual H2D submissions with exact compressed byte counts.
- Copy-stream waits for consumed staging slots (reuse count and interval).
- Compute-stream waits for copy-ready events, preserving grouped last-copy waits.
- The grouped `gather_native_group` call and its existing fallback gathers.
- Expert product submission envelopes (including any gathers or host submission
  gaps inside that call). Compact FP16-TC work is not instrumented as products;
  this job uses the original NVFP4/MMQ path.
- Routed group/row counts and resident group/row counts from already available
  host routing metadata. GPU-only fused routing stays explicitly **null**;
  no device routing readback is introduced for the diagnostic.

The copy issuer owns copy/reuse event vectors; the compute thread owns all other
vectors. The issuer joins before events are inspected or destroyed. Event and
stream errors fail the diagnostic. Graph capture is rejected for tracing. The
tracer adds no event allocation, recording, synchronization, or output when off.

## Interpretation and limits

Timelines are **instrumented diagnostics**. Event and host submission overhead can
perturb scheduling, especially with many small groups. H2D interval bandwidth
includes that overhead; it is not PCIe link saturation. Product envelopes are not
hardware occupancy or exact kernel busy time. Ready-wait intervals include event
overhead; they approximate exposed copy dependency time rather than pure PCIe
latency. Intersections merge nested intervals to avoid counting them twice.
Only the sampled layers have transfer accounting; do not call their totals a
whole-model transfer total. Nsight hardware counters remain permission-blocked.

The runner keeps the native pack's required `--spec 2` and the previously
qualified 16-token generation bound. The first attempt at `--spec 0` was
refused by CLI validation before model loading (run 40992134275); it produced
no performance or numerical result.

The runner launches the same pinned model/prompt with tracing off and on for
each supply regime. Off runs provide uninstrumented timing references, but this
four-run diagnostic does **not** establish a statistical speedup. All QPN execution
is disabled. First-logit finiteness and trace-off/on L2 <= 1e-7 plus identical
argmax are mandatory; logits and raw logs stay private. This does not qualify
QPN model numerics, and the earlier KL ~0.03135 remains unqualified.

These are fresh engine processes. Host page-cache temperature is uncontrolled;
there is no cache eviction or warmed-engine repeated-request experiment. Describe
these runs accordingly. A follow-up warm-engine and naturally cold-start study
is required before attributing a cache-temperature effect.

## Runner and privacy

`.github/workflows/qpn-supply-profile.yml` shares `strata-qpn-v100` concurrency,
uses `gpu.lock`, refuses active compute clients and reuses the existing CUDA 12.8
full-model build directory. No Docker builds, package installs, privilege changes,
service changes, cache mutations, new automation, or main-branch changes.

Raw output remains in a mode-0700 temporary directory under existing runner state.
Only a typed allowlist of metadata, mathematical comparisons and aggregate
per-layer event timelines is uploaded. No expert identifiers, weights,
activations, prompts, token IDs, or logit arrays are published.

Next: inspect supply gaps and residency before introducing any packed model
staging. If matched realistic supply remains at parity, stop further QPN layout
variants. D4 numerical reconstruction and the independently gated Q4TC track
remain separate work; this diagnostic implements neither.
