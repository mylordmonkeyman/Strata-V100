# V100 QPN 8/16/32-row tile sweep: synthetic single-expert diagnosis

Date: 2026-10-29 UTC. Branch `qpn/performance-first` only.
[8-vs-16 runner](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40964463037) at `3fa0f4da3aa3ec6e37ae6766fcfc467d441c120e`;
[8/16/32 runner](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40964553114) at `2a3578c1c816ea40a3f57ee2f9b560cc67ed1712`.
The latter is the primary comparison, with binary SHA-256 `10a93b15067e0e098666e9197e0f4bfa4bce635b2043be1c13a5fd0cd6b3ecf4`, source SHA-256 `17f31de175e7ec66e53867bafb4ee95cabdced65eb6b4556833c1b6954333207`. Synthetic zero weights and inputs only; all cases used the same single V100, same binary, same input dimensions, same split-K=4/fast decoder, 256 CUDA-event repetitions. The original 16-row schedule is unchanged in the full-model engine.

| Rows | Tile | GU CTA grid | Down CTA grid | GU timeline us | Down timeline us | GU idealized warp coverage | Down idealized warp coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 8 | 120 | 240 | 31.396 | 12.940 | 9.375% | 18.75% |
| 17 | **16** | 80 | 160 | **23.068** | **10.828** | 6.25% | 12.5% |
| 17 | 32 | 40 | 80 | 54.880 | 18.012 | 3.125% | 6.25% |
| 24 | 8 | 120 | 240 | 32.628 | 13.476 | 9.375% | 18.75% |
| 24 | **16** | 80 | 160 | **23.116** | **11.256** | 6.25% | 12.5% |
| 24 | 32 | 40 | 80 | 56.612 | 18.564 | 3.125% | 6.25% |
| 32 | 8 | 160 | 320 | 32.900 | 15.616 | 12.5% | 25% |
| 32 | **16** | 80 | 160 | **23.588** | **12.384** | 6.25% | 12.5% |
| 32 | 32 | 40 | 80 | 61.952 | 19.248 | 3.125% | 6.25% |

Per-specialization CUDA occupancy API limits: tile 8 = 9 CTAs/SM (56 regs/thread, 4096 B static shared); tile 16 = 4 CTAs/SM (113 regs/thread, 8192 B); tile 32 = 3 CTAs/SM (141 regs/thread, 16384 B). All use 128-thread CTAs and 80 SMs with 64-warps/SM capacity. The quoted warp coverage is **grid geometry**, not observed profiler occupancy. Nsight Compute remains installed but performance-counter access denied; no host permission changes were made.

**Result:** The existing **16-row tile is fastest for both widened GU and down** at every 17/24/32-row case. Tile 8 raises CTA coverage but makes GU roughly 36–41% slower; tile 32 makes GU roughly 2.38–2.63x slower. A tile-only change is therefore **not justified** for full-model qualification. Synthetic launch costs are about 1.9 us; the 16-row GU GPU timeline is 23.1–23.6 us and down 10.8–12.4 us, so launch overhead alone does not explain the observed compute timeline.

**Limitations:** Zero-weight cache-resident microbenchmarks cannot isolate the actual model's PCIe expert staging, GGUF layout penalties, host routing, or nonuniform 17..32-token group scheduling. The matched full-model [MMQ vs fused QPN](qpn-fused-vs-mmq-performance-first-2026-10-29.md) remains at essentially parity (median 0.99862x) and quality-unqualified (KL 0.03134775). Do not extrapolate a synthetic per-kernel improvement to end-to-end prefill.

**Next safe investigation:** Quantify repeated per-expert launch/synchronization and compressed-weight memory access under multi-expert working sets. A standalone uniform-batch diagnostic can establish an upper bound; **ragged grouped dispatch requires separate routing equivalence, canary/sanitizer, and model A/B gates** before any integration. Preserve original MMQ fallback, default-off QPN, and the 16-row full-model schedule.
