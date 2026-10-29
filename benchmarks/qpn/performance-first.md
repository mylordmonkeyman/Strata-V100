# QPN performance-first widened gate/up experiment

**Branch:** `qpn/performance-first`; only on the isolated opt-in V100 code path.

The first matched-order experimental full-model A/B (run 40949838878)
showed QPN median prefill **14,074.25 ms** versus MMQ **13,993.5 ms**
(no speedup). A likely culprit is per-expert launch overhead: gather,
separate gate and up QPN projections, FP32 SwiGLU, FP16 staging, down.
The selected path executed **362** 17–32-row experts over three 768-row
prefill chunks, implying ~2,172 per-expert kernel launches, excluding
unchanged group overhead.

## Candidate and control

The extra `STRATA_QPN_PREFILL_FUSED_GU=1` requires both the existing
QPN compile gate and `STRATA_QPN_PREFILL_EXPERIMENTAL=1`. It reads the
same contiguous native GGUF gate+up weight blob, executes **one**
1280-output QPN projection instead of two 640-output projections,
then uses a fused FP32 SwiGLU/FP16 store kernel directly. This removes
**two launches per selected expert**, retaining original FP16 activation
staging, scales, and down projection. It does not rewrite model weights,
cache slots, routing ownership or MMQ fallback. The original unfused QPN
path remains available for direct A/B by leaving the extra flag unset.

Dedicated workflow compares original QPN, fused QPN, fused QPN, original
QPN (ABBA) with *the same pinned* model+prompt, matched expert order,
768-token staged prefill, native verifier `--spec 2`, NUMA, GPU and
weight-supply controls. It measures actual prefill, TTFT, decode,
first-logit KL/L2 and argmax, QPN groups, and hashes. This is intentionally
a **performance-first, not quality-qualified** experiment:
`--performance-only` records numerical divergence without failing
solely because KL exceeds 0.03; startup, nonfinite logits and absent
QPN execution remain failures.

### Decision gates

1. GPU build and host tests must pass. A successful run is not yet a
   kernel efficiency win; compare paired prefill times and their spread.
2. If fused QPN beats original QPN, repeat against the **matched MMQ**
   control (separate ABBA; do not compare cross-run absolute times).
3. Before production consider grouping/fusion beyond this optimization,
   actual model quality and the default stream-all async prefill path.
4. No `main` changes, production activation, pack changes or unrelated
   host/service changes.
