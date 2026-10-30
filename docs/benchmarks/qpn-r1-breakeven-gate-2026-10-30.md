# R1 full-workload break-even gate — aggregate-only checkpoint (2026-10-30)

This is a host-only evidence gate, not a measured QPN speedup. Source: `tools/qpn_r1_breakeven.py`.
Use `python3 tools/qpn_r1_breakeven.py --aggregate-input docs/benchmarks/qpn-r1-breakeven-known-evidence-2026-10-30.json --report /tmp/qpn-r1-break-even.json`.
The committed aggregate input contains the **737 MiB optimistic guarded headroom upper bound** from [run 41094045957](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41094045957); no private routing IDs or prompts. It does **not** assert that 737 MiB can be allocated safely, nor that any packed working set fits.

The result must remain **UNKNOWN** because independent training/heldout routing, accepted-token decode qualification, actual packed live-memory fit, numerical quality, and exposed full-workload timings are unavailable. The existing [verification route run 41111215170](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41111215170) establishes **zero 33–512-row eligible QPN groups** for the pinned verifier windows (maximum 2 rows/expert), not accepted-token decode. No decode QPN gain may be inferred. Previous [nonresident full-model ABBA run 41075728954](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41075728954) lost to MMQ; those timings are **not** valid resident-prefill candidate timings and must not be inserted as such.

For future private aggregate inputs, supply paired `mmq_prefill_s`, `mmq_decode_s`, `qpn_prefill_exposed_s`, `qpn_decode_s`, `regroup_s`, `transition_s`, `restore_s`, `dispatch_s` for the **same complete workload**, with nonoverlapping exposed critical-path components, plus incremental peak `packed_bytes` (including scratch) and a measured `safe_headroom_bytes`. Negative, nonfinite, and boolean numeric values are rejected. An over-budget configuration or measured timing deficit is **FAIL** even if other measurements are missing. **PASS** requires a >=5% full-workload gain, headroom upper-bound compliance, and explicitly verified independent-heldout, actual-decode, numerical-quality, critical-path-measurement, and allocator-fit qualifications; boolean assertions are not independently authenticated by the calculator.

**Decision: no resident packed-slot GPU prototype, no repeated losing ABBA/1–16-row experiments, and no fabricated heldout or decode timings.** Production MMQ and caches remain untouched.

## Corrected stage boundary: feasibility before qualification

The original calculator remains available as the default `--stage qualification`
check, now labeled **R3**. Its model-quality and measured candidate-memory gates
must not be used as prerequisites for constructing an R2 prototype. New
`--stage feasibility` implements R1 with measured baseline evidence and explicit
optimistic/conservative net-savings bounds. It does not fabricate candidate times.

Required aggregate fields: `baseline_request_s`, `optimistic_saved_s`,
`conservative_saved_s`, `minimum_overhead_s`, `maximum_overhead_s`,
`additional_peak_bytes`, `available_incremental_bytes`. Evidence flags are
`baseline_measured`, `heldout_routes_measured`, `exposed_compute_measured`,
`transition_cost_bounded`, `memory_budget_bounded`. All evidence flags are supplied
assertions that must be backed by pinned reports; the calculator cannot verify them.

- STOP: even optimistic net savings miss 5%, or incremental memory exceeds budget.
- UNKNOWN: evidence is missing or unverified.
- MEASURE_UNCERTAINTY: optimistic bounds pass but conservative bounds do not.
- PROTOTYPE_CANDIDATE: conservative bounds pass; DDR-1 and implementation safety
  obligations still apply. This is never production or numerical qualification.

Both stages use **5% request latency reduction** (candidate <= 0.95 baseline),
which corresponds to >=1.05263x throughput for fixed work. Include the same prompt,
output work and cache state; no double counting of overlapped intervals. Additional
memory is incremental peak including transient conversion and retirement, not the
entire packed payload replacing existing canonical storage.

## Bounded measurement contract for the next GPU work

Before dispatch, prepare a private manifest with distinct training and held-out
prompt hashes, exact token counts, model/binary/profile hashes and common flags.
The existing pinned prompt alone cannot supply independent held-out evidence.
Use a modest short/long prompt pair first and keep startup versus warmed-engine
measurements separate. Changing seeds or copying the same trace is not held-out.

Capture prefill and verification routing in the same private request lifetime;
join on request/chunk/layer/window identities, validate complete work, and feed
the host evaluator before temporary traces are deleted. Current separate probe
scripts delete their captures at exit and cannot be joined retroactively. Preserve
aggregate provenance only outside the runner. Collect actual group membership;
allowing arbitrary regrouping gives an upper bound, not executable group coverage.

Record native verification work even for rejected speculative tokens: its weights
were consumed and matter for restoration. Accepted-token counts are separately
required for useful decode TPS and phase interpretation, not for deciding whether
an executed verification window consumed canonical weights. First-logit parity
alone does not validate the later generation sequence.

Time eligible resident work, gathers, exposed CPU staging and copy/slot waits with
trace-off controls. Use temporary isolated buffers to bound canonical reload or
lossless inverse-layout conversion; include completion fences, scratch and all
canonical fallbacks. Do not mutate the persistent cache for this measurement.
Report the removable exposed time and full overhead range for the same request.
No new packed model cache or full-cache rebuild is justified by the current data.
