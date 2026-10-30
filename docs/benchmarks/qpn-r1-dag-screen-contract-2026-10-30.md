# R1 native-MMQ fixed-DAG exposed-time screen (host only, 2026-10-30)

**Engineering status:** static single-representation packed admission **STOPPED**;
request-latency R1 **UNKNOWN**. This is a **measurement contract and host-only
sensitivity evaluator**, not a newly measured exposed-time result. It does not
run a model, modify inference dispatch, authorize a resident cache, or provide a
QPN quality qualification. No private per-operation graph is committed or uploaded.

## Existing evidence and why it is insufficient

* Joint MMQ-only synthetic train/held-out routing, source `2e79bd5c`,
  [run 41133416862](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41133416862),
  immutable safe aggregate `benchmarks/qpn/results/2026-10-30-joint-heldout-routing.json`:
  **1.8707%** and **3.1320%** *optimistic routed-entry* coverage at 737 and
  2048 MiB of **replacement payload**, respectively. Neither is exposed time.
* Matched selective-MMQ supply ABBA, [run 41066208726](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41066208726):
  32.1% first-chunk H2D byte omission but about 0.71% median prefill-time
  reduction; fresh-process cache was uncontrolled. Do not convert transfer
  bytes, trace event spans or heterogeneous route shares to achievable time.
* `src/prefill/qpn_supply_trace.hpp` / `tools/qpn_supply_profile.py` already
  instrument **only layers 0/24/47**; the `products` spans include host
  submission/gather gaps and can overlap other CUDA work. `ready_wait` event
  spans are not pure exposed DMA latency. They are not a complete full-request
  dependency graph. Existing MMQ supply tracing is default-off.
* Existing verification captures have at most **two rows per expert**, so the
  33–512-row QPN selector is structurally ineligible there. Verification
  consumes native canonical weights even for rejected speculative tokens;
  it must not be relabeled accepted-token decode.

## Safe host-only tool

`tools/qpn_r1_dag_screen.py` reads a **local owner-only** causal graph and
emits just an aggregate report. It deliberately never includes node/route IDs,
per-operation times, tensor contents or event intervals in the report. It
requires the original SHA-pinned model, all four agreed MMQ control flags and
QPN dispatch disabled. No experiment is initiated. Its script may be tested
using synthetic DAGs with:

```bash
python3 -m unittest discover -s tools -p test_qpn_r1_dag_screen.py -v
```

A future authorized local-only graph file can be assessed by:

```bash
python3 tools/qpn_r1_dag_screen.py \
  --private-dag /owner-only/local/full-request-causal-dag.json \
  --aggregate-report /owner-only/local/qpn-r1-conditional-report.json \
  --assumed-speed-factor 1.377
```

Do **not** commit either private DAG or raw event timestamps; the aggregate
report should be reviewed against the pinned provenance/allowlist before any
publication. The 1.377x assumption comes from an isolated resident fixture
and is **not** a full-model effective speed factor.

The input schema is intentionally narrow: `schema=1`, `model_sha256`,
`control_flags` (four mandatory values and
`STRATA_QPN_PREFILL_EXPERIMENTAL="0"`), `full_request_latency_ms` (excluding
startup and describing the same complete request), `evidence_flags`, `nodes`,
and `edges`. Every node is a private locally assigned consecutive integer `id`
with integer `lane`, per-lane consecutive `order`, phase
(`prefill`/`verification`/`accepted_decode`/`other`), kind
(`expert_compute`/`transfer`/`gather`/`cpu_stage`/`other_compute`/`other`),
**non-overlapping active-work** `duration_ms`, and boolean `eligible`.
Only `expert_compute` in **prefill** may be marked `eligible`, and only when
static packed admission and actual group membership are independently known.
The `edges` are local integer pairs of true cross-resource causal dependencies;
resource-lane serialization edges are generated from lane order. Do not use
expert/routing IDs as node IDs. Device waits belong in dependency edges; counting
the wait as active work as well would double-count stalls. Resource constraints
and host submission gaps must be represented by actual task durations and edges,
not inferred from event timestamps.

The required evidence assertions (all unverified by the calculator itself) are:
`complete_full_request_work`, `all_cpu_gpu_work_and_wait_edges`,
`exact_static_group_eligibility`, `trace_off_parity_and_time_control`,
`phase_classification_validated`, `observational_timing_calibrated`,
`representative_heldout`. An asserted flag **is not proof**. The script also
requires the modeled native-MMQ critical-path duration to agree with an
independently measured same-request wall latency within a diagnostic 2% tolerance.
A discrepancy, missing evidence or false assertion forces **UNKNOWN**.
Reconciliation alone cannot guarantee the causal graph is complete.

## Computation and decisions

For a fixed causal DAG with measured active task durations, the calculator
finds the longest resource-constrained path under three **hypothetical**
conditions: unchanged native MMQ, eligible prefill nodes shortened by an
assumed speed factor (default fixture-only 1.377x), and eligible nodes reduced
to **zero duration**. The latter is an idealized fixed-dependency ceiling, not
a performance measurement. Recomputing the longest path captures the fact that
other CPU/GPU/copy work may become critical, so summing eligible kernel durations
would overstate potential gain.

If the complete, corroborated fixed-DAG model shows less than 5% maximum
zero-time savings, `graph_screen` is
`CONDITIONAL_FIXED_DAG_BELOW_FIVE_PERCENT`. If not, it is
`CONDITIONAL_FIXED_DAG_POSSIBLE`. Incomplete evidence yields
`UNKNOWN_INCOMPLETE_CAUSAL_EVIDENCE`. **Every outcome retains
`r1_feasibility=UNKNOWN`**: no static-DAG sensitivity result authorizes R2
without measured post-implementation overhead bounds, allocator safety and
independent held-out critical-path qualification. Observational instrumentation
and schedule changes can alter real critical paths; a fixed-DAG result does not
prove the same bound applies after redesign.

## Why no GPU workflow is launched yet

The existing supply tracer lacks a complete causally ordered, phase-classified
CPU/GPU work graph and actual static-packed eligibility. Adding CUDA events
alone would not cure its product-envelope and wait-overlap ambiguity; joining
private routing IDs with timing events requires owner-only local state and
explicit lifecycle controls. The independent held-out screen has already
rejected this particular static admission policy at screening, and prior
nonresident full-model QPN ABBA results were negative. The immediate scope is
therefore **host-only tool validation**. Do not reuse the old QPN variants,
rerun joint private captures, build a Q4TC conversion or implement a packed
resident slot based on the existence of this script.

**Future bounded measurement proposal (not scheduled):** first establish a
full-request native-MMQ resource/causality timeline with events for *actual*
active GPU kernels, ordered host enqueue/supply work, copy completions and
ready/reuse fences, and compare matched trace-off runs for numerical and timing
perturbation. Attribute exact held-out eligible group membership locally,
account for accepted-token generation separately, and emit **only aggregate
bounds**. Any GPU runtime profiling proposal needs a new explicit bounded
rationale and must preserve `gpu.lock`, shared concurrency, persistent CUDA
12.8 build, model/cache/privacy and default-off production boundaries.
## Host-only graph parser hardening (2026-10-30)

The DAG parser now validates lane-order density in O(nodes), not by
iterating up to an untrusted maximum order number. A malformed sparse
rank fails closed without a potentially unbounded loop. Nonfinite
accumulated critical-path durations and nonfinite wall reconciliation
are rejected, as are unrepresentably large integers and malformed
non-string node categories. The CLI also refuses to overwrite the
private graph through an output alias or symlink.

When required evidence assertions are incomplete or the modeled baseline
does not reconcile with wall time, both **conditional latency reduction
fractions** are now null. Diagnostic graph metadata remains explicitly
nonqualifying; R1 is still UNKNOWN and static admission remains STOPPED.
No GPU run or new profiler was authorized by this change.
