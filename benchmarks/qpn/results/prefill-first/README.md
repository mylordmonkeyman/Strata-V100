# First independent prefill crossover

Source: `be323b70f656cf4359e281c2e1540eb42dc896f6`.
[Run 40802984333](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40802984333).
33 real-weight cases (three fixtures, T=1..512) passed the 3% full-expert
numerical check. Actual type-40 W4A8 MMQ is executable on this V100 build:
`mmq::fits` returned true. No Q2_0/WMMA comparator was substituted.

Median completed-expert latency (microseconds across three fixtures):

| T | QPN register-LUT | NVFP4 W4A8 MMQ + gather | Dequant→FP16/cuBLAS | Cached FP16 control |
|---|---:|---:|---:|---:|
| 1 | 62.48 | 54.49 | 124.48 | 39.05 |
| 2 | 62.76 | 54.62 | 136.05 | 49.88 |
| 4 | 63.42 | 54.97 | 136.00 | 50.24 |
| 8 | 62.82 | 55.96 | 136.06 | 50.75 |
| 16 | 61.76 | 62.97 | 136.71 | 52.25 |
| 32 | 111.76 | 77.81 | 139.98 | 56.90 |
| 33 | unavailable | 83.72 | 143.74 | 61.57 |
| 64 | unavailable | 106.69 | 151.67 | 69.44 |
| 128 | unavailable | 168.35 | 176.12 | 94.41 |
| 256 | unavailable | 190.21 | 195.77 | 115.47 |
| 512 | unavailable | 250.44 | 287.03 | 206.24 |

QPN wins only narrowly at T=16 in this initial schedule; this is not enough
to claim a prompt-level improvement. The cached FP16 control excludes
conversion, consumes additional resident memory, and is not an eligible
production crossover winner. No engine prefill route or defaults change.
Inputs are Gaussian and the working set is one expert. CPU/PCIe/SSD supply,
cache capacity, real prompt histograms, TTFT and model-level quality remain
unmeasured; M4's end-to-end gate remains pending independently of decode.
