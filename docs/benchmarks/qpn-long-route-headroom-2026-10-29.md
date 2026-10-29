# Experimental QPN long-row headroom

Source: an isolated feature branch `qpn/large-route-probe`, branched from the full-model performance-first experiment. No production dispatch change. The full-model `qpn_candidate` predicate remains **17..32 rows**, and unchanged real-model ABBA run 40969669062 continues on `qpn/performance-first`.

The standalone `qpn::supported()` limit is extended from 32 to **128** rows only to investigate the much larger real-routed expert groups. The QPN CUDA kernel uses a grid.z tile index that owns disjoint output row ranges; no larger per-CTA shared memory is required with token_tile=16. This only authorizes *host-side geometry*, not numeric/memory safety, favorable speed or eligibility for model inference.

Required remaining gates before any engine integration:
1. Standalone V100 fast-decoder T33/40/48/64/96/128 projections checked against independent repeated 16/32-row segment reference; validate no stale outputs after changed real input, output guards, and finite numerical results.
2. CUDA Graph replay, Compute Sanitizer memcheck and racecheck for T33 and T97; no GPU permission or model pack changes.
3. Repeat with *real captured expert data* from the pinned private model and matched MMQ comparison, including host gather/packing and weight supply.
4. Only if a large-row speedup is established, design a new opt-in group policy; retain original MMQ fallback and all routing invariants.

Do not interpret a larger supported() limit or synthetic timing as a model-performance gain.
