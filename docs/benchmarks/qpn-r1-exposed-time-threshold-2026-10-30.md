# R1 conditional exposed-time hurdle — host-only analysis (2026-10-30)

**Status: R1 timing feasibility UNKNOWN; static single-representation admission remains STOPPED.**
This is a mathematical sensitivity analysis of existing aggregate-only evidence,
not a measured full-model speedup, a candidate dispatch, or authority to build
a packed cache/kernel. No GPU work, private inputs, model conversion or new
automated workflow was used.

## Provenance and scope

- Source routing screen: [successful joint trace run 41133416862](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41133416862),
  source `2e79bd5ccef53066f706cb321366a2d04084aac7`,
  artifact `13015890822`, ZIP SHA256
  `38704f3eb26f1acd671206d00ef6b8fd118779dc5260638c48efbd16bbee7790`.
  Immutable aggregate: `benchmarks/qpn/results/2026-10-30-joint-heldout-routing.json`.
- Earlier isolated resident kernel fixture: **1.377x** versus MMQ; **not**
  a full-model, held-out, or exposed-critical-path speedup.
- Synthetic 2048-token held-out prefill optimistic regroupability:
  737-MiB **replacement payload**: 18,381 / 982,560 routed entries
  (**1.8707%**); 2,048-MiB replacement payload: 30,774 / 982,560
  (**3.1320%**). These are not static actual grouping, true executed
  QPN work, or a timing share.
- The separately measured **737 MiB spare guarded VRAM** is an optimistic
  upper bound on *additional live allocation*, **not** a cap on how many
  canonical payload bytes could be replaced in a single-representation plan.
  Nor is either number an allocator fit guarantee.

## Necessary latency concentration, conditional on an assumed kernel gain

For a *single complete request* let `T` be measured MMQ request latency,
`f` be the fraction of that full-request critical-path time that can genuinely
be removed or accelerated by QPN (including the effect of overlap), `s > 1`
be the effective speed factor **on that removable time**, and `h` be
nonoverlapping exposed regrouping/transition/restore/dispatch penalties divided
by `T`. The optimistic request-latency reduction is

`R = f * (1 - 1/s) - h`.

A >=5% request-latency reduction necessarily requires

`f >= (0.05 + h) / (1 - 1/s)`.

These are **conditional arithmetic thresholds**, not bounds on actual
resident-QPN performance: `s` is unmeasured on eligible full-model work,
`f` is unmeasured, and penalties are unmeasured. The isolated 1.377x
fixture is deliberately used only as a *what-if scenario*, not as proof that
this speed factor transfers to real prefill. A complete-request baseline must
include native verification/decode time.

| Assumed effective eligible-time speed factor | Zero exposed overhead | Overhead = 1% of full request | Overhead = 2% |
|---|---:|---:|---:|
| 1.377x **fixture scenario only** | 18.26% | 21.92% | 25.57% |
| 2x hypothetical | 10.00% | 12.00% | 14.00% |
| 4x hypothetical | 6.67% | 8.00% | 9.33% |
| Infinite-speed mathematical ceiling | 5.00% | 6.00% | 7.00% |

For the **1.377x, zero-overhead hypothetical**, genuine eligible work
would need to occupy **18.26% of total request latency**. Compared to
*held-out routed-entry shares only*, that implies the following
**required time-share / route-share ratios** (not measured costs):

| Replacement payload | Optimistic route share | Required full-request time share | Required share ratio |
|---|---:|---:|---:|
| 737 MiB | 1.8707% | 18.26% | 9.76x |
| 2,048 MiB | 3.1320% | 18.26% | 5.83x |

Do **not** interpret these ratios as measurements of relative per-entry cost:
routing entries have heterogeneous shapes, may overlap transfers, and may
require canonical fallback. They merely identify how concentrated *exposed*
eligible time would have to be for this hypothetical to meet the target.
Neither low route share nor these large ratios mathematically proves the
5% goal impossible.

## Next evidence gate; no new GPU variants authorized

Only an existing-MMQ, phase-classified critical-path profile could establish
`f`. It must distinguish compute that truly lies on the full-request critical
path from overlapped CPU/H2D/cache activity, and account for actual static group
membership (not ideal regrouping), native verification canonical consumers,
warm/cold residency, cache misses and format-state churn. Separately bound
exposed regrouping, dispatch, restoration and scratch/allocator peaks to
estimate `h`. Existing trace-off/on fresh-process times, restore-byte
*demand proxies*, and routed-entry shares cannot be substituted for these.

Prior [nonresident full-model ABBA losses](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41075728954)
are negative evidence, not measurements of this hypothetical resident
speed factor. Verification windows with <=2 routed rows/expert have zero
33–512-row QPN eligible groups and do not constitute accepted-token decode
qualification; QPN numerical quality also remains unqualified.

**Decision:** Preserve no-go for the static admission and packed-resident
prototype. Keep R1 `--stage feasibility` as UNKNOWN until there is
independent, representative exposed-time evidence sufficient to populate the
existing bounded gate without converting route shares into time. No repeated
QPN/ABBA/decode GPU experiments or automatic Q4TC pivot are justified.


## Cross-phase full-prefill ceiling from the 2026-10-31 diagnostic (host-only)

This is a **new conditional arithmetic constraint**, not a new GPU measurement,
QPN performance result, actual R1 exposed-time attribution, or approval to
resume a stopped subtrack. It combines the pre-existing isolated resident
packed-fixture **1.377x** speed-factor scenario with the completed
[matched-H2D original-MMQ diagnostic](qpn-mmq-early-stage-repeat-2026-10-31.md)
([run 41180483211](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41180483211),
source `b0fbc2e`). The latter's **control arm A** had a 17.08525-second
median prefill and 84.5988-second median fresh-process wall time on one
2,047-prefill/128-requested-decode workload. It deliberately forced route
publication before H2D, losing normal async lookahead; it is **not** the
production async or warm-engine MMQ baseline. The two medians need not
come from the same invocation, so this is a sensitivity calculation rather
than a paired latency bound.

If the isolated 1.377x factor hypothetically applied to **every millisecond
of that diagnostic's prefill**, with zero new overhead and no other phase
changed, the maximum modeled wall reduction would be
`17.08525 * (1 - 1/1.377) = 4.67766 s`, or **5.529%** of the
84.5988-second process. Reaching 5% would require at least
`(0.05 * 84.5988) / 4.67766 = 90.43%` of **all prefill time** to
be accelerated by that effective factor. Any newly exposed overhead of
**0.44772 seconds** (0.529% of full-process wall) or more would exhaust
this *all-prefill* zero-overhead margin. Actual 33–512-row eligible QPN
compute is only a subset of prefill; no eligible-time fraction has been
measured. The existing 3.132% held-out route share cannot be substituted
for it. The 1.377x factor is from an isolated resident fixture and has
never been demonstrated on full-model eligible time.

**Decision:** this tighter diagnostic-policy hurdle reinforces the
requirement to measure *eligible exposed critical-path time* before any
resident-QPN GPU prototype. It does **not** prove a 5% gain impossible
under normal async operation, a warm serving session, longer prompts, or
an alternative Q4TC format; those have different phase fractions and
unmeasured effective speed factors. Static packed admission and early
CPU staging remain STOPPED; R1 exposed-time remains UNKNOWN. No new GPU
work, cache, model conversion, dual-format VRAM, or production change.
