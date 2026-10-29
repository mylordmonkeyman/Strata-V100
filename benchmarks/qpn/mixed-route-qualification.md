# QPN mixed-routing: host-only plan verification gate

This isolated branch adds a reproducible **host-only** exact ownership verifier for real private first-prefill captures from the pinned unchanged-engine baseline [40815727472](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40815727472).

- Verify capture SHA-256 and geometry against the baseline manifest for layers 0, 24 and 47, 1,024 tokens each, top-k 10.
- Reject duplicate expert IDs within a token, out-of-range IDs, malformed captures, nonfinite activations, changed expert counts and changed histogram.
- Build per-expert groups with exact token indices and flattened router slots; independently prove a one-to-one assignment of **every** token-expert slot, without missed or duplicate work.
- Keep full per-token plans in a private directory under /srv/strata-qpn/mixed-route-plans; upload only aggregate bucket counts and SHA-256 provenance.
- Unit-test missing/duplicate slots, incorrect expert ownership, duplicate routed experts, altered captures, and manifest pinning.
- No engine integration, production dispatch, model-pack change, or GPU timing claims.

**Not yet qualified:** realistic GPU grouped QPN/W4A8 comparison, multi-expert weight residency and transfers, graph replay and Compute Sanitizer for grouped launches, >32-token fallback, end-to-end TTFT. These are separate required gates. The existing single-expert 17..32-token advantage cannot establish any full-model speedup.
