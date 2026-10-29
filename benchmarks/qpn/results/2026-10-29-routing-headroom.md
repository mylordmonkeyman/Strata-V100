# Real prefill routing-headroom audit — October 29, 2026

## Provenance

- Unmodified Strata full-model baseline: [GitHub Actions run 40815727472](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40815727472), native expert pack, model revision b9ce18a6887e18c089fdea8509a33e63429a3300. The M0 metadata archive is artifact **12872852053**.
- Baseline manifest SHA-256: **218ae1f7fa5bb03ee01aba98278848e13ce17478957ec2300a0e58272af8270d**.
- Reproducible histogram analysis: [run 40899340662](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40899340662), **success**, source commit **0647ee32911f7e4ef61644027804d818c71a856b**, artifact **12908188539**, containing headroom.json. The audit checks expert ID ranges, positive group counts, agreement with top-k and captured histograms and provenance. Python host tests passed.
- Three first-prefill captures, one each for layers **0, 24 and 47**, each with 1,024 tokens and top-k **10**, 10,240 routed entries per layer. Samples do **not** represent the entire model, later chunks, all prompt lengths, or memory tiers. No weights/raw activations were downloaded for this analysis.

## Measured routing distribution

| Token rows routed to a given expert in this chunk | Expert-layer invocations | Share of invocations | Routed entries | Share of routed entries |
|---|---:|---:|---:|---:|
| 1–8 | 313 | 41.24% | 898 | 2.92% |
| 9–11 | 34 | 4.48% | 336 | 1.09% |
| 12–16 | 50 | 6.59% | 704 | 2.29% |
| **17–32** | **113** | **14.89%** | **2,841** | **9.25%** |
| 33–64 | 103 | 13.57% | 5,032 | 16.38% |
| 65+ | 146 | 19.24% | 20,909 | 68.06% |
| **Total** | **759** | **100%** | **30,720** | **100%** |

Layer 0 has 204 active experts, layer 24 has 336, and layer 47 has 219. Per-layer median routed rows were respectively 29, 12.5 and 6; the largest five layer-47 group sizes were 557–1,012 rows. This is a highly nonuniform routing workload.

## Implications

1. The verified 16-row QPN multi-CTA prefill optimization wins against W4A8/MMQ at **T=17..32** by about **1.13–1.24x** for individually resident experts, with graph replay and sampled Compute Sanitizer memcheck passing. That window accounts for **14.89% of expert invocations** but only **9.25% of routed entries** in these captures. The original two-tile QPN path also narrowly beats MMQ around T=12/16, another 2.29% of entries; small T=1..8 remains MMQ-favorable.
2. The current standalone QPN implementation supports only **T<=32**. Groups at **T>=33** collectively represent **249 expert invocations (32.81%) and 25,941 entries (84.44%)**. They cannot be accelerated by the current candidate without separate shape/schedule validation. The larger groups may also have very different weight supply and GPU parallelism behavior.
3. The percentages are counts, **not GPU time fractions**. They do not justify a claimed 9% model speedup, or any nonzero end-to-end TTFT improvement; they do not account for expert load transfers, overlap, decode, routing, batching or attention.
4. Next standalone gate: measure actual **nonuniform mixed-expert** grouping, >L2 distinct expert working sets, grouping/launch overhead, and the dominant larger-batch W4A8/MMQ headroom before proposing opt-in runtime dispatch. Before an end-to-end claim, run a fixed-seed full-model numerical/quality and TTFT A/B using identical prompt, residency and hardware state.

This is a metadata-only audit branch; **main, production dispatch, checkpoint format, cache slots and unrelated host services were not modified**.
