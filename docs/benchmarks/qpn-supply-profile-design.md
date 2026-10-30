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

## First real run: stream-all output parity failed (do not promote)

Run **40992803235**, source `e50f9ec0a2457ee86fdfc4a5c61f59bea8cd0ef5`,
aggregate artifact **12949774401**: all four model processes exited successfully
and completed 2,047 prefill tokens; QPN executed zero groups throughout.
The 768 trace-off/on first-logit comparison was **exact** (L2=0, KL=0).
The 1024 comparison **failed**: relative L2 **0.2793418918**,
KL **0.1090079737**, same argmax. Finite output and same argmax do not waive
this failure. These data do not establish timing-transparent tracing or qualified
stream-all numerics. No QPN throughput result was measured.

Diagnostic context only: traced first 1024-token chunk transferred **760,324,400 /
846,033,696 / 948,331,888 bytes** at sampled layers 0/24/47. Routed groups were
204/336/219; resident routed groups 110/138/113. These are sampled instrumented
values from the failed parity experiment, not a trustworthy performance comparison.
The stream-all walk issued 275/306/343 nonresident expert copies, including experts
not selected by that layer's routing; this is existing behavior, not introduced by
tracing. The staged 768 walk supplied only its actual nonresident routed experts.

Next diagnostic (workflow uses `--repeatability`): two uninstrumented threaded
1024 runs, then inline-issuer trace-off/on at 1024. This distinguishes ordinary
run-to-run variation from a threaded-issuer/timing interaction before attributing
cause. The inline issuer is an existing experimental runtime control applied only
to that diagnostic pair; default dispatch and synchronization are not patched.
Both comparisons retain the strict L2 <= 1e-7 / same-argmax gate. A failure remains
visible even if the other pair passes. No speed claim or dispatch integration until
this is resolved. Do not infer a confirmed ring race from the first mismatch alone.

## Conditional transfer-opportunity calculation (diagnostic only)

The failed-parity stream-all trace nevertheless exposes a **hypothesis to test**, not a
qualified speedup: unlike the 768 staged walk, the 1024 stream-all walk issues all
nonresident experts before routing is known. The opt-in
`STRATA_PF_SKIP_UNROUTED` path is explicitly restricted to the WMMA walk
(`volta_wmma && m.Xq == nullptr`), not the native NVFP4 W4A8 MMQ path
under test. See `src/prefill/prefill.cpp` around lines 3187–3197, 3237–3266
and 3285–3305. This corrects the overly broad statement that the model
always transfers only routed experts.

For the **failed** run 40992803235, sampled layers report the following
arithmetic. Each compressed expert blob in those layers was 2,764,816 bytes;
`routed nonresident = routed groups - resident routed groups`. The
hypothetical reduction assumes every such expert is known before issuing DMA,
with no lookahead or ring-order penalty. It is **not** a measured attainable
reduction.

| Sample layer | Stream-all copies | Routed nonresident | Copies beyond routed nonresident | Hypothetical H2D bytes avoidable |
|---|---:|---:|---:|---:|
| 0 | 275 | 94 | 181 | 500,431,696 (65.8%) |
| 24 | 306 | 198 | 108 | 298,600,128 (35.3%) |
| 47 | 343 | 106 | 237 | 655,261,392 (69.1%) |

The original trace's 1024 output parity failed, so these are **diagnostic
counts only**. They do not establish correct results, PCIe saturation, a
speedup, or a viable scheduling policy. In particular, waiting for routing
may destroy copy/compute overlap; stream-all deliberately starts transfers
ahead of routing, and the copy/compute ring has strict lifetime fences.
Do **not** enable `STRATA_PF_SKIP_UNROUTED` on W4A8 or change production
dispatch on the strength of these counts.

**Gate:** resolve the pending threaded off/off and inline off/on repeatability
experiment first. Only then compare a separate opt-in, isolated
routing-aware supply diagnostic against unmodified MMQ with equal numerical
output, explicit byte accounting, stalls, and matched timing. Do not
conflate this potential MMQ-side transfer reduction with a QPN-specific gain.

## Repeatability result and localization gate

Run **40994067899**, experimental source `5a4c390c17fb2a09882f1d79ec0065725c7b8171`,
aggregate artifact **12950233696** completed four MMQ-only model invocations:

| Control | First-logit relative L2 | KL(A||B) | Argmax same |
|---|---:|---:|---|
| Threaded issuer, tracing off in both fresh processes | 0.2608480841 | 0.1136804738 | yes |
| Inline issuer, tracing off vs on | 0.3355579349 | 0.2330369091 | **no** |

All processes returned zero and completed 2,047 prefill tokens; the diagnostic
correctly failed. The first pair establishes nonrepeatability **without tracing in
this experimental binary**. It does not establish that unmodified `main` shares
the defect. The second pair shows that disabling the threaded issuer alone does
not establish parity. Do not assign this solely to instrumentation or a confirmed
copy-ring race, and do not interpret any timings as a speedup.

Next, `--layer-repeat` reuses the **existing** `STRATA_DBG_FEAT` and
`STRATA_PREFILL_DUMP_R_ALL` diagnostic hooks in two trace-off 1024 runs. It changes
no engine code or production switches. Capture all 48 router inputs/IDs/weights
in both chunks (1024 + 1023 tokens), plus the chunk-end residual snapshots.
The parser verifies complete record geometry, finite values and matched ordering;
exports only hashes, mismatch counts and mathematical comparisons. Raw features,
expert IDs, routing weights and residual arrays stay in the private mode-0700
runner directory. Original source/model and prompt digests are enforced.

On the pinned Volta build, `mixed_bf` holds IEEE FP16 bits; route weights and
chunk residual dumps are explicitly BF16. Keep these encodings distinct.
The existing feature hooks synchronize the compute stream. If they stabilize
execution, that is evidence of timing sensitivity, **not** proof that the
uninstrumented path is repaired. If captured inputs first diverge after a given
MoE layer, investigate that boundary; if every prefill capture matches while
first logits differ, investigate post-prefill state/cache handoff and the first
verification window. No QPN or Q4TC promotion until the blocker is resolved.

### Chunk-size and issuer isolation controls completed

Run **40995236933**, source `d670c8ec74466bc1702f71d7b3e72af8229bd49a`,
artifact **12951610808**, six trace-off MMQ-only processes all returned zero:

| Repeated supply regime | Relative L2 | KL(A||B) | Argmax same |
|---|---:|---:|---|
| 768 staged (`stream_min=1024`) | **0** | **0** | yes |
| 1024 forced staged (`stream_min=4096`) | **0** | **0** | yes |
| 1024 stream-all, inline issuer | **0.2723905036** | **0.1733464362** | yes |

This narrows the current experimental-binary failure to the stream-all path,
rather than 1024-token geometry or the threaded issuer alone. It does not locate
the defective operation. The next layer-repeat diagnostic targets stream-all;
forced-staged controls and the earlier 768 result are preserved in the harness.
No performance comparison is qualified while stream-all outputs are unstable.

## Layer comparison: first difference in the second chunk at layer 3

Run **40996673563**, source `f62fda17b31d6cc84ae028983788f2ed10303e3a`,
artifact **12951971448**, completed both synchronized private-capture runs and
failed repeatability. All **48 first-chunk router inputs, routed IDs and rounded
routing weights matched bitwise**; the first 1024-token chunk's BF16 final residual
snapshot also matched. The second chunk's layers 0/1/2 matched; its first hidden
and routed-ID differences were **layer 3** (hidden L2 **0.0070176157**, 12 routed-ID
entries different). The second chunk's residual L2 was **0.3322306134**. First-logit
L2 **0.2457868780**, KL **0.1368310186**, same argmax; still a hard failure.

These captures narrow the earliest visible divergence to work between the layer-2
MoE input and layer-3 MoE input of the 1023-token tail, or state carried from the
first chunk. They do **not** prove that layer-3 attention is the cause: layer-2 MoE,
HC write/normalization, and unobserved persistent state remain candidates. The
matching first-chunk residual is a BF16 snapshot, not proof that every FP32
residual or persistent KV/GDN state byte matched.

### Narrow boundary snapshots (default off)

The follow-up adds `STRATA_QPN_BOUNDARY_DUMP`, under the existing experimental
prefill build gate, to read back only the relevant layer-2/layer-3 boundaries.
It records FP32 layer-2 MoE output and HC injection, its exit residual, layer-3
entry residual/norm scales, its **FP16** norm image, FP32 attention input/output
and injection, and its post-attention residual. Both chunks are covered. Raw
snapshots stay private; only hashes and mathematical comparisons are exported.

This is an observational hook, with no new arithmetic kernels, dispatch, source,
cache or ring changes. It rejects unsupported norm/encoding preconditions and
CUDA Graph capture; CUDA and file errors terminate the diagnostic. Runtime off
allocates no snapshot buffers, writes no files and adds no synchronization.
The selected readbacks synchronize their stream and can perturb scheduling; even
complete snapshot parity would not repair or qualify the uninstrumented path.
