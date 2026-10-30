# Full-model offline-prepacked QPN: no material prefill benefit

Pinned source `c8d4d2fe4335e3feb7e9d39b96c675402dda79a9`, [Actions run
41075728954](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41075728954),
aggregate artifact **12990608578**. Host checks, V100 compilation, offline cache
creation/verification, and both full-model ABBA cases completed successfully.
A successful workflow is not a performance or numerical-quality qualification.

The experiment compared MMQ / QPN / QPN / MMQ in fresh processes on the pinned
2048-token prompt (2047 prefill tokens), with staged 1024/1023-token chunks,
`STRATA_PREFILL_STREAM_MIN=4096`, CPU expert share zero, native NVFP4 W4A8,
fixed expert profile/resident budget, and all four known NaN-workaround settings.
QPN used prepacked weights only for nonresident experts in wholly QPN-owned groups;
canonical resident experts remained on MMQ. Both arms used the same partition
policy and scratch budget. This does not test a resident packed GPU cache.

## Measured results

| Selected row range | MMQ prefill runs (ms) | QPN prefill runs (ms) | Median MMQ (ms) | Median QPN (ms) | MMQ/QPN throughput ratio |
|---|---|---|---:|---:|---:|
| 9–32 | 11982.9, 12093.2 | 18242.8, 15139.3 | 12038.05 | 16691.05 | **0.72123x** |
| 33–512 | 12236.5, 14124.7 | 15768.3, 12491.3 | 13180.60 | 14129.80 | **0.93282x** |

Both 9–32 QPN observations lost to their paired controls. The 33–512 paired ratios
were **0.77602x** and **1.13076x**, so its one faster observation does not establish
a reproducible gain. Its median was slower. Cache temperature was uncontrolled;
there were only two samples per arm. Do not describe these median losses as precise
population effects or extrapolate to warm-engine serving, other prompts or models.
Neither range supplies evidence of a material, repeatable full-model improvement.

All **eight** processes exited zero, completed all 2047 prefill tokens, produced
finite correctly sized first-logit dumps and passed the harness's visible CUDA/
nonfinite error checks. MMQ executed zero QPN groups. Candidate 9–32 executed
**306 groups / 4127 experts** in each repetition; candidate 33–512 executed
**308 groups / 4244 experts** in each repetition. All four within-arm comparisons
were **bytewise equal** over 248320 FP32 first-logit elements (L2/KL zero).

| Range | QPN/MMQ first-logit relative L2 | KL(MMQ||QPN) | Argmax equal |
|---|---:|---:|---|
| 9–32 | 0.1329086860 | 0.0155085661 | yes |
| 33–512 | 0.1433772720 | 0.0440409440 | yes |

**Numerical quality remains unqualified for both.** The 33–512 KL exceeds 0.03 and
is explicitly retained as performance-only. The 9–32 first-logit threshold passes,
but that is not independent GGUF model correctness or broader quality acceptance.
The D4/raw-FP16 input mismatch remains unresolved. Approximation also changes later
routing: control planned-expert counts were 4071 versus candidate 4127 (9–32), and
4499 versus candidate 4244 (33–512). Thus identical partition policy does not mean
identical route IDs or transfer bytes across the complete approximate model.
Full-model H2D, memory residency and exposed stalls were not separately measured
in this run. Do not attribute the result solely to arithmetic or solely to supply.

## Cache/provenance and limits

- Model SHA256: `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`.
- Prompt SHA256: `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`.
- Executable SHA256: `089b21cf6ae24f9391969d62a3dc60606debb01563847d4341f192c6d199dc3c`.
- Offline cache SHA256: `24910343739a15562c50d087350d04d20dfca5624d6f70503d2f1edabc845eaf`.
- Cache: 24576 experts, **67948118016 bytes** of additional temporary disk storage;
  bounded-memory lossless preparation plus verification took **1520.762 s (~25.35 min)**.
  That one-time cost is excluded from prefill timing and cannot be hidden in a
  deployment/startup comparison. Original source SHA remained unchanged.
- All four workaround flags were explicit in every arm: `STRATA_MMQ_KQ=1`,
  `STRATA_PREFILL_MMQ=1`, `STRATA_ARENA_LOCK=0`, `STRATA_ARENA_PIN_GIB=1`.
- New source maps the separate packed file and uses the pinned host Stager for
  selected entries; this adds cache I/O/staging and host-footprint considerations.
  CPU staging, cache temperature and routing changes can confound attribution.
- The workflow cleans its private temporary cache; private logs/logits are removed
  by the harness. Only aggregate metadata is committed/uploaded.
- No full-model packed-path sanitizer, intermediate guard or changed-input model
  Graph qualification was performed. Model Graph capture is explicitly refused.
  The separate 45-case prepacked/MMQ and 15-case native-decode fixture sweeps passed
  their stated oracle/guard/Graph/sanitizer scope; that does not qualify this model path.

## Reporting correction

The original 9–32 artifact incorrectly contains `numerical_quality_qualified=true`
because the old reporter assigned a **first-logit KL/argmax threshold boolean** to
that broader field. It does not establish model qualification. The original artifact
is retained unchanged. Committed aggregate copies under
`benchmarks/qpn/results/2026-10-30-full-model-packed-*.json` move that boolean to
`first_logit_acceptance_passed`, explicitly set model qualification false, and record
this metadata correction and original member/run/artifact provenance. All timing,
logit-comparison and execution measurements remain unchanged. The reporter and its
regression test now prevent that overstatement in future experiments.

## Decision

Archive this default-off nonresident prepacked supply prototype as a **negative
full-model performance result**. Do not promote it or spend further GPU time
qualifying this losing configuration. Its workflow is retained for manual dispatch,
with automatic push runs disabled to avoid wasteful repetitions.

The requested expanded row-bucket and decode arithmetic retests are complete in
their stated scopes. Prepacked QPN lost every 1–16-row capped-input comparison
against production native grouped DP4A; these were **decode-shaped fixture tests**,
not actual QPN autoregressive model decode. The full-model run used the original
decode backend in both arms (`qpn_decode_executed=false`); its logged decode TPS
must not be described as a QPN decode comparison. No winning decode selector exists
to justify integrating a new decode backend from these results.

There is still no demonstrated meaningful full-model QPN speedup. This does not
prove every possible resident packed-cache design cannot help, but the isolated
kernel wins alone do not justify new cache complexity or production changes.
Stop further QPN variants under the performance decision rule unless new evidence
or a new user instruction changes the experiment. The independently gated Q4 track
in rev4.1 remains separate; do not reinterpret NVFP4 as Q4 or go below four bits.
Main, production/default MMQ, original model packs, persistent expert caches,
services, driver and runner privileges remain unchanged.
