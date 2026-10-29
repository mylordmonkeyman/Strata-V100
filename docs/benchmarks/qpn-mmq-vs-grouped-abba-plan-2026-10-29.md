# Original MMQ versus grouped QPN matched performance-only ABBA

On the isolated branch `qpn/performance-first`, compare **MMQ → QPN-batched → QPN-batched → MMQ**, using exactly the pinned prepared model/prompt and existing staged 768-row prefill controls.

Prior grouped-QPN vs fused single-expert QPN V100 run [40968226415](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40968226415) passed and measured 14,140.65 ms original fused versus 14,042.35 ms grouped (1.0070003x, +0.700%). First logits matched bit-for-bit, 144 selected layers, 362 selected groups and 4,863 experts in each arm. This does NOT yet establish parity or speedup against MMQ.

Current experiment uses `--batched-mmq-ab --performance-only` to keep the original MMQ fallback as the within-run reference. The unchanged plan-only control partitions routes identically at the host level, but actual routing may evolve differently across kernels due to model-level numerical differences; record group counts separately. Startup/TTFT differences are not kernel speedups.

Numerical quality is **not** broadly qualified: all four trials require finite first-logit dumps, runtime success, and expected QPN branch ownership; KL, relative L2, top-1 disagreement and hashes are retained in aggregate. No production or main changes.
