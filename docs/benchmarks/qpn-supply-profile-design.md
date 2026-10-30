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

## Corrected benchmark environment: prior NaN workaround had been omitted

The user identified a previously successful local benchmark recipe:

```
STRATA_MMQ_KQ=1
STRATA_PREFILL_MMQ=1
STRATA_ARENA_LOCK=0
STRATA_ARENA_PIN_GIB=1
```

The QPN full-model A/B and supply-diagnostic harnesses sanitized inherited
`STRATA_*` variables and failed to restore this recipe. Therefore all preceding
results in this report used a different configuration from that known-good local
setup. The historical first-logit hash differences remain observations of those
specific builds/configurations; they do not establish that the issue persists
with the user's working flags. Attribution to an upstream defect was premature.

The shared `benchmark_env` helper now restores all four settings explicitly for
both compared arms after sanitization. A regression test verifies that stale
inherited settings cannot remove them, unrelated experimental settings remain
excluded, and the caller's environment is not mutated. Every report records the
actual restored values. No host/service configuration is changed.

The first corrected control (`--workaround-repeat`) runs two uninstrumented
1024 stream-all invocations, with QPN and all private capture hooks disabled.
Strict first-logit repeatability and finite-output gates remain unchanged. Do not
assume the recipe solves this NVFP4 failure before that run completes: in the
pinned source, `STRATA_MMQ_KQ` enables K-quant MMQ types, whereas NVFP4 has its own
W4A8 eligibility path (`moe_mmq.cu::supported`). The arena flags can also change
host pinning/supply behavior. All four are retained to reproduce the agreed setup.
Historical M0 artifacts are immutable and are not relabeled as using these flags.


## Restored benchmark recipe: stream-all still nonrepeatable

Run **40999522670**, source `1300198d2b3a96a580b32a243bfa8ab67132d66f`,
artifact **12952604371**, executed two fresh trace-off MMQ-only 1024
stream-all processes with the previously omitted four-flag NaN recipe
restored. Both exited zero and completed 2,047 prefill tokens, but
first-logit relative L2 was **0.2645154883**, KL **0.1768648700**,
same argmax. The recipe does **not** repair this experimental-binary
stream-all nonrepeatability. Prefill timing differed (27,693.1 vs
17,283.1 ms) and is not a valid performance A/B.

The preceding synchronized boundary capture run **40998104339**,
source `411e8d898fc6`, artifact **12952442605**, narrowed the earliest
recorded second-chunk divergence to **layer 2 MoE output**, relative
L2 **0.0246083454**, while the earlier layer-2 router input matched.
Layer-2 exit residual L2 was **0.0051048843** and the first-logit
L2 **0.2651216535**. **Important chunk-boundary distinction:** with `STRATA_PREFILL_STREAM_MIN=1024`,
the first chunk (T=1024) uses stream-all but the second (T=1023) uses
the *staged* walk. Therefore the layer-2 MoE mismatch is observed in a
staged second chunk **after** a stream-all first chunk. It may reflect
state carried across chunks or a previous asynchronous operation,
not necessarily an error in the second chunk's stream-all gather
(which is not selected). This makes first-chunk stream-all supply,
slot-lifetime, persistent state and cross-chunk handoff candidates;
layer-3 attention is downstream, but no root cause is established.
Readback synchronization may perturb scheduling.

Next isolated control: `--gather-repeat` disables the existing
`STRATA_PREFILL_GROUP_GATHER` optimization only in the diagnostic
process and compares two tracing-off fresh processes for each
threaded and inline issuer. The original stream-all ring, compressed
expert blobs, source, model, cache and MMQ arithmetic remain intact.
This tests whether grouped-gather/event lifetime is necessary for the
failure. If parity remains broken, investigate stream-all per-expert
ready/reuse fences, especially the layer-2 second-chunk boundary.
A passing synchronized or ungathered control is **not** numerical
qualification of the default path or a performance result. Never
enable QPN model-level optimization on these failed traces.

## Performance work resumed with repeatable controls

The MMQ-only ABBA at source `20ae4b3fe4bd9608c059d0e4f7e0e996a8cbb432`,
[run 41054076458](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41054076458),
artifact `12976497015`, passed all six within/cross-arm first-logit comparisons
bitwise (248,320 FP32 elements, L2/KL zero). Median prefill was 17,368.60 ms
for both-off and 17,208.15 ms for transition fallback: about 0.93% throughput
improvement, not a material QPN result. All four prior NaN-workaround flags were
present. QPN executed zero groups. Fresh-process cache temperature was uncontrolled.

User direction is to resume performance work and defer further root-cause debugging.
`tools/qpn_transition_abba.py --supply-profile` now runs a four-process diagnostic:
forced-staged tracing off/on, then transition-fallback tracing off/on. The existing
locked/cached V100 workflow runs this mode; no duplicate automation or GPU workflow
is introduced. The original ABBA CLI mode remains available without the option.
Both supply controls retain the known NaN recipe, SHA-pinned model/prompt,
identical 1024/1023 chunk geometry, full token count, zero QPN execution, and
strict off/on **bytewise** first-logit parity. Each trace must contain both chunks
and all sampled layers 0/24/47 with transfer/product/residency accounting. Missing
work, malformed traces, missing fallback counters, nonfinite output and CUDA errors
remain hard failures. Logs/logits are private temporary files, removed on exit.

The exported sampled timelines provide compressed expert bytes, interval bandwidth,
copy/product overlap, ready-event waits, slot reuse waits, gather envelopes, and
routed group/row residency fractions. Copies beyond routed nonresident groups are
an opportunity count, not achievable transfer savings: stream-all starts before
routing is known, and selective lookahead could lose overlap. Hardware occupancy
and true PCIe saturation are not inferred from CUDA event envelopes. Instrumented
timings are not a speedup result; warm/cold and realistic lookahead studies remain
pending. The experiment changes no engine kernels, ring/cache policy, production
defaults, source model/cache, host services or runner privileges.

Next decision: use the completed trace to identify exposed supply waits and choose
one isolated selective-supply experiment with a matched MMQ control. Stop QPN layout
iterations if realistic supply still erases the material benefit. No numerical-quality
or production promotion is implied by successful MMQ tracing.
