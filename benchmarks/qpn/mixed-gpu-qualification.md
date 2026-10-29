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
