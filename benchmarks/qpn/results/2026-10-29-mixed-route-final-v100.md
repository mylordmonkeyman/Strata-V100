# V100 mixed-expert NVFP4 QPN qualification — final standalone report (2026-10-29)

## Decision

**Standalone mixed-route qualification passes for the pinned 16-expert sample.**
Selective QPN on actual routed group sizes **17–32** reduces *sample-workload*
median latency by **5.17%** relative to all-W4A8/MMQ, or 1.05451x throughput,
with a **6.13% CUDA Graph latency reduction** in the latest independently
successful run. Widening QPN to **1–32** worsens sample-workload latency by
**12.80%**. This is a conditional, standalone experimental result only.
**Do not enable engine dispatch or claim full-model TTFT/tokens/s gain.**

## Immutable provenance and CI

- Source branch: `qpn/mixed-route-bench`, measured commit
  [`349cca12476558c694e6fe87bc7239b2003f08be`](https://github.com/mylordmonkeyman/Strata-V100/commit/349cca12476558c694e6fe87bc7239b2003f08be)
  (isolated racecheck-output validation fix). **Production `main` unchanged.**
- [Complete V100 run **40938946514**](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40938946514):
  **success**, CUDA-parity, sampled independent double-precision oracle,
  CUDA Graph replay/changed-input, 256-byte output canary, memcheck and
  racecheck for **both routing policies**. Artifact
  [**12926157681**](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40938946514/artifacts/12926157681)
  named `qpn-mixed-gpu-40938946514-1`, ZIP SHA-256
  `a769ecbf5fbbcea0ad6e25611808c8bd4a8ee3a8da914eb08f1c7be0d5b9f54c`.
  Matching standalone host/CUDA qualification
  [40938946597](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40938946597):
  **success**.
- Earlier independent matched run
  [**40919607665**](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40919607665):
  **success**, artifact **12918140836**, SHA-256
  `3c8a0b74c9678858130098400c76265dd1ac8e11eda4484879da850c04834f7f`.
  It predated graph timing/racecheck but reproduced the ordinary-workload
  latency improvement under identical inputs.
- Pinned model M0 [40815727472](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40815727472),
  fixture builder [40908922475](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40908922475),
  metadata artifact 12913635075.
  Original M0 manifest digest
  `218ae1f7fa5bb03ee01aba98278848e13ce17478957ec2300a0e58272af8270d`.
- Independently verified private 16-expert weight SHA-256
  `ce605a8b34aa54c0dc266aa38be9560f9d8ecb2f4179111e8bf4575fa82801a7`;
  routed FP32 activation digest
  `0625460d391179bbc4d2ce56259d6886c2d62ae0aa51dd69eb741093e5afe2a8`;
  expert/route-selection digest
  `2f4df5612b75e70aed808af916578d2babf5db108f20d00e40d2b1044c01caf0`.
  All raw model weights, activations and complete routing plans stayed private.
- Host [routing-plan verifier](2026-10-29-mixed-route-host-verified.md)
  independently established complete routing ownership (no duplicate or missed
  token/expert slots) for all 759 active groups and 30,720 routed entries
  across original layers 0/24/47 (each T=1024, top-k=10).
  The 16-expert benchmark uses a **real stratified subsample**, not all 759
  groups; the independent sample has 1,100 rows, 44,237,056 unique resident
  expert-weight bytes versus a 6,291,456-byte V100 L2.
  Its six fallback groups >32 rows account for most sampled work, and its
  ten <=32 groups include buckets 1–8, 9–11, 12–16, and 17–32.
- Full no-miss routing invariant belongs to the host captured plan; the GPU
  harness independently verifies fixture digests and nonuniform bucket geometry,
  and checks every output finite/written. It is **not** a full routing-engine
  implementation or a full-model run.

## Matched GPU performance (microseconds, lower is better)

Six alternating CUDA-event timing repetitions, 10 completed workloads per
repetition, identical sample weights and routed activations in both arms.
Each policy compares against a corresponding **all-MMQ control in the same
permutation**. The 17–32-only experiment reorders the **same actual experts
and real rows** offline into contiguous MMQ and QPN partitions, and uses
exactly the same order for the matching control. Do not compare absolute
controls across the two permutations.

| Policy | All MMQ ordinary, us | Hybrid ordinary, us | Latency change | Throughput ratio | All MMQ graph, us | Hybrid graph, us | Graph latency change |
|---|---:|---:|---:|---:|---:|---:|---:|
| QPN on all 1–32-row groups (10), MMQ for >32 (6) | 1455.358 | 1641.574 | **+12.80%** (regression) | 0.88656x | 1331.507 | 1486.643 | **+11.65%** (regression) |
| **QPN only on 17–32-row groups (4)**, MMQ on other 12 | **1332.224** | **1263.360** | **-5.17%** | **1.05451x** | **1313.896** | **1233.306** | **-6.13%** |

Independent earlier [40919607665](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40919607665)
gave 1455.771 → 1638.963 us for the broad policy (regression), and
**1345.792 → 1276.363 us for 17–32 only (5.16% lower latency)**.
This reproducibility is limited to the same pinned sample and same hardware;
it is not an estimate over all prefill chunks or production workload types.

## Numerical and safety qualification — run 40938946514

| Gate | QPN 1–32 | QPN 17–32 | Requirement |
|---|---|---|---|
| Independent GGUF-dequantized sampled reference, all-MMQ L2 | 0.0112966272 | 0.0112966272 | <= 0.03 |
| Independent GGUF-dequantized sampled reference, hybrid L2 | 0.0089092935 | 0.0110498529 | <= 0.03 |
| Whole-output hybrid/MMQ relative L2 | 0.00211041585 | 0.00117852742 | <= 0.06 |
| Oracle row coverage | 32 sample first/last rows | 32 sample first/last rows | Deliberately sampled; not exhaustive FP64 |
| CUDA Graph replay matches ordinary execution | Pass | Pass | Exact replay |
| Graph replay uses changed inputs | Pass | Pass | Exact changed-input replay |
| 256-byte output-tail guard and finite output | Pass | Pass | Untouched guard; all rows written |
| Compute Sanitizer memcheck | **0 errors** | **0 errors** | Fail closed |
| Compute Sanitizer racecheck | **0 errors, 0 warnings, 0 hazards** | **0 errors, 0 warnings, 0 hazards** | Fail closed |

Memcheck and racecheck ran independently on both policies (one measured
iteration each under instrumentation). Sanitizer-instrumented timings **must
not** be interpreted as performance measurements. The earlier
[run 40934193943](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40934193943)
was marked failed after a **racecheck log-parser false negative**:
the command expected `ERROR SUMMARY: 0 errors` but valid racecheck logs say
`RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`. The
source fix [349cca12](https://github.com/mylordmonkeyman/Strata-V100/commit/349cca12476558c694e6fe87bc7239b2003f08be)
corrected the parser. The successful rerun confirmed both policies.

## Limitations, disposition, and later work

**This specific experimental standalone mixed-route benchmark qualification
is complete for its stated sampled scope**, and the recurring qualification
watch can be disabled. It must **not** authorize production/model changes.

Remaining prerequisites for any separately authorized full-model integration:

1. Independently decide whether to build a reversible, mutually exclusive,
   **off-by-default experimental QPN dispatch** behind a feature gate; do
   not silently alter existing MMQ routing or expert-cache semantics.
2. Quantify real first-prefill and subsequent-chunk end-to-end TTFT,
   tokens/s and quality with identical prompts, graph behavior and pinned
   model bytes, including host/device expert supply and cache misses.
3. Determine whether a grouped QPN launch and/or private chunk-major
   compressed layout benefit persists after actual per-expert rearrangement
   costs, fallback scheduling and memory residency. Earlier standalone
   [layout diagnostic](2026-10-29-layout-attribution-v100.md) showed
   batching benefits, but **not** full-model throughput.
4. Validate any **newly implemented** grouped/repacked kernel or dispatch
   under broader memcheck, racecheck and numerical/correctness sweeps.
   The current green sanitizer result applies only to the candidate as
   presently implemented.

`main`, production engine dispatch, cache slots, model files, runner
registration, and unrelated host services have not been modified.
