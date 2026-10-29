# V100 independent synthetic QPN launch/occupancy diagnosis

Date: 2026-10-29 UTC. Experimental `qpn/performance-first` only.
Source `d8ef048afebcd9d3fd43563f6102479faf629e24`;
[successful runner probe](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40964208251);
artifact `qpn-gpu-probe-40964208251-1` contains scalar JSON only.
Probe source SHA-256 `bf14404d33fc5d3b35062e97d252b7f1b6cdcaca9cc932681fd718598b20bad1`;
binary SHA-256 `4a6080132cded0b182807efe8a54a3307f5e2f2e5e5857d75d135d97313a3d2b`.

## What was independently measured

On the V100's 80 SMs, the full-model QPN schedule `{split_k=4,accumulators=1,decoder=1,token_tile=16}` has 128 threads (4 warps) per CTA, 113 registers/thread, 8192 B static shared memory/CTA, and CUDA occupancy API limit **4 CTAs/SM**. For one 17..32-row expert, widened GU (N=1280) launches 80 CTAs; down (N=2560) launches 160 CTAs. Therefore idealized grid/warp coverage is **6.25% GU** and **12.5% down** of 64 warps/SM, assuming perfect placement. These are **geometry-based estimates, not measured active warp occupancy**.

Synthetic zero-valued weights and activations (no private model) were used to time each independent projection via 256 repeated launches and CUDA events. CUDA-event timelines can include host enqueue gaps and cache-resident weights; they are not end-to-end expert costs.

| Routed rows | GU CUDA-event timeline (us/launch) | Down timeline (us/launch) | GU host enqueue (us) | Down host enqueue (us) |
| --- | ---: | ---: | ---: | ---: |
| 17 | 23.180 | 10.912 | 2.010 | 2.031 |
| 24 | 23.360 | 11.292 | 1.965 | 1.928 |
| 32 | 23.580 | 12.396 | 1.889 | 1.918 |

Independent no-op CUDA launch: 1.882 us GPU timeline, 1.880 us host enqueue. Thus **CPU enqueue overhead is real but much smaller than the synthetic GU kernel timeline**. One expert's small GEMM grids expose only 1 GU CTA/SM and 2 down CTAs/SM; testing smaller token tiles is justified before any invasive grouped/ragged dispatch.

Nsight Compute (`ncu`) is installed but **GPU hardware performance-counter access was denied** (`ERR_NVGPUCTRPERM`). No measured active-warps occupancy metric is available. Do not describe the grid estimates as profiler observations or change host permissions/services without explicit authorization.

## Interpretation and next test

The prior matched [MMQ vs fused-QPN full-model ABBA](qpn-fused-vs-mmq-performance-first-2026-10-29.md) showed **0.99862x median prefill speedup (parity/slightly slower)**, and first-logit KL `0.03134775` (quality unqualified). This synthetic diagnosis suggests low per-expert parallelism, but cannot apportion the full 14-second prefill to GEMM vs host routing, expert streaming, or PCIe traffic. Sweep token tiles 8 vs 16 with the same independent probe; only promote a tile variant to a full-model performance-only A/B if it shows a reproducible advantage and passes routing/safety checks. Preserve MMQ fallback and default-off behavior.
