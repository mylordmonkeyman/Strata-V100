# Experimental complete 16-expert mixed GPU benchmark

The private fixture is pinned to successful V100 run **40908922475**,
metadata-only artifact **12913635075**, against the baseline M0 manifest
SHA-256 218ae1f7fa5bb03ee01aba98278848e13ce17478957ec2300a0e58272af8270d.
A separate fail-closed verifier checks the actual private file SHA-256 hashes,
all 16 distinct weight payloads, each routed row segment, sample bucket sizes
and the six-group contiguous fallback prefix before GPU execution.

The isolated CUDA/C++ tool runs the same complete 16-expert sample through
(A) the existing grouped NVFP4 W4A8/MMQ gate/up, SiLU and down path, and
(B) identical MMQ fallback for groups >32 with QPN kernels for groups <=32.
It includes MMQ gathering, quantization, FP16 staging, per-expert QPN launches
and output scaling in alternating end-to-end **sample-workload** CUDA-event
timings; all 16 real resident expert weight blobs are held in memory (>L2),
and neither backend gets synthetic repeated tokens or weight subsets.
Both arms use the same original captured activations.

Initial correctness gates cover whole-output MMQ/QPN L2 agreement, nonfinite
or unowned rows, 256-byte output tail guard, both full-workload graph replays,
changed-input replay and Compute Sanitizer memcheck. **Independent sampled
double-precision reference parity and wider sanitizer/racecheck sweeps remain
outstanding even if this first grouped run succeeds.** The benchmark excludes
host transfer, layer attention, routing/scheduling and full-model TTFT.
No production dispatch, cache slot, model pack, runner registration or
unrelated service changes are made.

## Second qualification: fewer QPN groups and independent numerical oracle

The first complete 16-expert V100 run [40912015593](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40912015593) passed whole-output parity (L2 0.00211041585), CUDA graph replay, changed-input replay, 256-byte canary and Compute Sanitizer memcheck with zero errors. Its six-repetition median was **1355.059195 μs** for all-MMQ versus **1823.590374 μs** for QPN on all 10 <=32-row groups (MMQ/QPN speedup 0.7430721x). That is **34.6% slower latency** for the tested hybrid, not a positive performance result. Artifact 12915270579 contains measurement and provenance.

A new isolated comparison preserves the exact same 16 private weight and captured-activation groups and includes two candidate policies: (1) QPN for all <=32 rows with independently qualified 16-row CTA tiles for 17–32, (2) QPN only at 17–32 with the other 12 groups on W4A8/MMQ. The second policy reorders the verified real groups and their exact rows *offline* into a contiguous MMQ fallback prefix, and applies that same order to the all-MMQ baseline, so no synthetic rows, omissions, duplicate assignments or private-file changes are introduced. The complete resident workload, prep and launches are still timed. Both candidates now have sampled first/last-row independent GGUF-dequantized double-precision reference checks against the output of each arm, in addition to full-output cross-arm parity. CUDA graph replay, changed-input checks and Compute Sanitizer are requested for both policies. The new policies are pending independent runner results.
