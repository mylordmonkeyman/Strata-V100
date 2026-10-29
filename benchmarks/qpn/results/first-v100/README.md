# First V100 direct-layout result — scalar candidate rejected for dispatch

Source/binary revision: `7bba05df0f01f5481b5c88d94168843498b9d945`.
[Run 40799913497](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40799913497)
completed successfully on `strata-v100-qpn-docker`. Compiler/driver, device,
binary/fixture hashes and six alternating repetitions are in the raw records.

20 synthetic cases and 60 real-weight cases passed provisional numerical
checks, graph replay and output sentinels. Real fixtures cover layer/expert
0/7, 24/300, 47/511. Activations were seeded Gaussian, not model captures.
The nine sanitizer invocations (three tools at T=3/8/17, first real expert)
reported no memory/init errors or race hazards. Sanitizer timings are excluded
from performance analysis; the full logs remain attached to the Actions run.

For the FP32→FP16 input/hidden variant, medians across the three expert fixtures:

| Entries/expert | Legacy/QPN latency ratio | Interpretation |
|---|---:|---|
| 1 | 0.139 | QPN about 7.2× slower |
| 2 | 0.151 | QPN about 6.6× slower |
| 3 | 0.157 | QPN about 6.4× slower |
| 4 | 0.180 | QPN about 5.6× slower |
| 8 | 0.227 | QPN about 4.4× slower |
| 16 | 0.245 | QPN about 4.1× slower |
| 32 | 0.289 | QPN about 3.5× slower |

Worst whole-expert relative L2 over those FP32-staging real cases was about
0.040%, comfortably below 3%. This is much smaller than the activation-Q8
legacy error, but **accuracy does not justify a slower production backend**.

M1 performance gate: **fails for this scalar schedule/decoder**. No engine
integration or default change is warranted. This does not close other kernel
schedules or the independent prefill track. Next tests compare a register-LUT
decoder/vector activation loads against this retained scalar control, sweep
projection-specific schedules, and compare actual prefill NVFP4 MMQ and
dequantize→FP16/cuBLAS including scales and gather/conversion costs.

Limits: single-expert weights may fit L2; no real activation tails, mixed-group
ownership, multi-expert streaming working sets, model KL, MTP acceptance,
full-model M0 headroom profile or end-to-end TTFT has been measured. The
checkpoint revision was not included in this first artifact; fixture hashes
were. Subsequent jobs copy the source audit metadata to preserve that link.
