# Expanded prepacked row sweep and full-model prefill/decode follow-up

User direction on October 30 explicitly reopens QPN testing after the selective
MMQ supply negative result: retest **prepacked** weights across row groups, then
measure actual full-model prefill **and decode**. Do not pivot to Q4 or apply the
previous stop decision before this requested coverage is evaluated.

## Implemented first stage

The existing `qpn-prepacked-stream-real.yml` workflow now runs the same immutable,
SHA-verified private 16-expert fixture with chunk-major prepacked weights across:

- Natural routed-row buckets 1–8, 9–16, 17–32, 33–64, 65–128, 129–256, 257–512,
  plus combined ranges 1–32, 17–128, 33–512. Empty buckets are omitted explicitly
  by constructing the plan from the original verified counts.
- Decode/verification-shape controls cap each expert at 1, 2, 4, 8, or 16 rows.
  These select only existing prefixes of that expert's real captured activations;
  no rows are duplicated or synthesized. They are **decode-shape proxies**, not
  real model decode routing, sequential generation, speculative acceptance or TPS.
- Every case compares MMQ and the selected QPN hybrid with identical rows, expert
  order, source weights, scales and byte counts. Resident, equal-byte serial H2D,
  and two-slot/two-stream overlapped H2D are measured separately. All cache creation
  is offline and outside timed forwards, with no hot-path conversion. Resident
  timings now load the same verified offline cache too.

`--qpn-range min:max` bypasses the previous large-group-only restriction without
changing old CLI selection semantics. `--row-cap` is explicit and exported in
aggregate metadata along with the original 1100-row count. An all-QPN selection
avoids empty MMQ launches and zero-byte copy submissions. Kernel arithmetic and
production routing are unchanged.

Every case keeps the independent GGUF dequantization / FP64 oracle, finite and
missing-output checks, output guards, changed-input graph replay, alternating
six-round timings and matched transfer-byte checks. Representative 1-row,
17–32-row and 129–256-row cases run memcheck and racecheck with three iterations,
exercising two-slot reuse. Sanitizer raw logs are private temporary files; only
aggregate status leaves the runner. Existing GPU lock, shared concurrency and
persistent CUDA build cache remain in force. No duplicate workflow or automation.

## Full-model work required after the sweep

The existing model hook is canonical-layout, staged-prefill only (17–32 rows).
It cannot test prepacked decode simply by setting a flag. Neither the row sweep
nor the old model hook is evidence that this request's model-facing work is done.
The sweep's report has explicit `full_model_prefill_tested=false` and
`full_model_decode_tested=false` fields until real model tests exist.

Use measured candidates to construct an isolated prepacked source/slot path with
an unchanged canonical MMQ fallback, SHA/version ownership, source/cache guards,
and no per-forward repacking. Account for additional host/VRAM residency and all
transfers fairly. Extend shape selection only to qualified ranges, and preserve
exactly-once routed ownership. Larger prefill groups need correctly sized scratch;
the old 32-row scratch and ragged kernels cannot be reused blindly. Decode needs
its own native expert dispatch hook and graph-stable persistent buffers: the
prefill hook does not alter decode. Direct raw-FP16 QPN input also does not repair
the outstanding D4/Q8_1 MMQ numerical alignment task.

Actual model comparison must pin the same pack/prompt/supply configuration and use
matched-order ABBA with QPN-off controls, all four known NaN-workaround settings,
full token-work counters and explicit QPN execution counters for each phase.
Report prefill time/TTFT separately from decode TPS, target passes and MTP acceptance.
Include changed-input graphs, independent oracle/guards and sanitizer gates in the
applicable scope. Preserve finite-output/guard/crash/missing-work hard failures.
Record first-logit KL/argmax even in performance-only mode; do not certify quality.
Use the repeatable MMQ configuration for the selected regime and keep stream-all
and staged results separate; they previously produced different first logits.

## Current status

Implementation of the expanded fixture sweep is ready for V100 testing. Actual
full-model prepacked prefill and decode integration/measurements remain pending.
Original model packs, persistent expert caches, main, default MMQ dispatch, host
services, driver and runner privileges remain unchanged.


## Decode comparator extension

Production `src/core/verify.cpp` invokes `native_expert_grouped` with a **linear
36-byte Q8_1 image**, rather than prefill MMQ's transposed 144-byte D4 image.
The sweep now has an additional `--native-decode-control` arm for capped rows:
actual native grouped DP4A gate/up + SwiGLU + down, including its input quantization.
Its existing canonical weights, per-expert scales, row bounds and output layout
are compared to the same prepacked QPN path. `baseline_backend` identifies the
comparator; legacy timing-key names containing `mmq` remain for compatibility
and must not be interpreted as MMQ when that field says `native-grouped-dp4a`.

Native pointer tables for each of the resident/two streaming weight slots are
prepared during warmup, retained for graph lifetime, and cannot allocate during
capture. The native control uses separate persistent linear-Q8_1/scratch buffers;
its changing-input graph and oracle checks are identical to the MMQ control's.
`--native-only` schedules only these 15 decode-shape/supply cases, with 1-row and
8-row memcheck/racecheck checks. This avoids rerunning the first prefill/MMQ
matrix when qualifying the additional production-decode arithmetic comparator.
The original full sweep remains available by omitting this option.

This remains fixture-level qualification, with capped **prefill-captured** inputs,
not a fresh capture of actual autoregressive decode routing or full-model TPS.
Real prepacked model prefill and decode still need the isolated hooks above.
