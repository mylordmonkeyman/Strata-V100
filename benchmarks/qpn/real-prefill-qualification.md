# Matched real-activation NVFP4 prefill crossover

This opt-in `qpn/prefill-real` branch runs the *independent* prefill qualification
track against unchanged Strata baseline captures: model revision
`b9ce18a6887e18c089fdea8509a33e63429a3300`,
baseline run [40815727472](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40815727472).

The workflow checks three model asset hashes, validates the baseline's capture
digests, extracts each expert's own actual routed activations, and runs
`nvfp4_volta_prefill_bench --activations` at only supported row counts
(1, 2, 4, 8, 16, 32 and actual count where available). No Gaussian padding or
repeated rows. Each case tests canonical QPN, genuine NVFP4 W4A8 MMQ, and
dequantization-to-FP16/cuBLAS, with a cached-FP16 control excluded from
automatic selection. Results record per-backend L2 numerical error and six
alternating completed-expert timings. MMQ eligibility is probed on-device.

**Scope:** Each expert is still tested separately. This establishes neither
mixed-group dispatch nor end-to-end TTFT/acceptance. Existing QPN split-K and
registration choices are unchanged. Cached-FP16 is a *diagnostic control*
because its greater VRAM residency cost is not free. Do not enable any
production QPN dispatch or packed cache on this evidence alone.

Weights and activation vectors remain on the isolated private runner volume.
Only fixture hashes, row indices, benchmark logs and summary statistics are
uploaded in the Actions artifact. The preserved synthetic prefill path remains
available separately for wider T=33..512 crossover tests.
