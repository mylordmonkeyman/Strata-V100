# Original MMQ vs grouped QPN staged prefill — matched V100 A/B/B/A (2026-10-29)

**Decision:** The new grouped/ragged 17–32 QPN path is functionally exercised but is **not measurably faster than original MMQ** in this pinned full-model test. Retain all experimental switches off by default and prioritize larger routed expert groups.

- Exact experiment: [successful V100 run 40969669062](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40969669062), candidate source `67d7df5bb458045cb7b29fe83fb69614335c3e7a`, aggregate artifact **12940555882**, SHA256 binary `af9b1c68974c4a2fb345af374296d1c10064c91af0e9a2d6e547ab797385a876`.
- Private pinned model pack digest `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`; pinned prompt digest `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`. These private inputs/raw logits are not uploaded.
- Scope: `--prefill 768` staged prefill, 2047 reported prefill tokens, same binary on one V100, common NUMA/24-GiB expert-cache/residency policies and both arms matched host-side QPN route partition controls. **Not** default 1024 stream-all or TTFT-derived throughput.
- Exact order: MMQ, grouped QPN, grouped QPN, MMQ; two paired comparisons. All four exits 0, all first-logit arrays finite. MMQ has zero executed QPN groups, grouped QPN has 144 plan layers, 362 selected groups and 4,863 QPN experts (same for both candidate trials). MMQ hosts had 368 selected planned groups and 4,855 plan experts due to propagated numerical-routing differences. Consequently model execution work is not exactly same route sequence, despite same prompt/model.

| Matched pair | MMQ prefill, ms | Grouped prefill, ms | MMQ/grouped speedup |
|---|---:|---:|---:|
| 0→1 | 14,038.4 | 14,134.6 | **0.993194×** (candidate slower) |
| 3→2 | 14,111.0 | 14,003.5 | **1.007677×** (candidate faster) |
| Median per mode | **14,074.7** | **14,069.05** | **1.000402×** (+0.04%, practical parity) |

Throughput from median prefill `2047/seconds` approximately 145.44 (MMQ) versus 145.50 (grouped). Opposite pair signs preclude a meaningful full-model speedup claim. Startup/TTFT includes model/expert loading; it must **not** be attributed to projection kernels.

Numerical result MMQ vs grouped QPN: first-logit KL **0.0313477502**, relative L2 **0.0737182921**, top-1 token matches (760), same SHA within each arm but distinct between arms. Performance-only acceptance reports `performance_only_quality_unqualified`, and no claim of quality qualification is made. The prior grouped-versus-fused-QPN matched full-model result (+0.70%, KL 0 between those variants) also does not establish MMQ speedup.

**Next test:** Independently validate and benchmark QPN on large routed groups **33–128** versus a within-run MMQ control using the pinned real 16-expert capture. The six >32 groups in this sample are exactly **40, 187, 96, 60, 48, 508** rows; only **40, 96, 60, 48** qualify for the new 33–128 test, while 187 and 508 must remain MMQ fallback. Do not silently truncate or fabricate rows. It must be a separate offline GPU microbenchmark with real weights/activations and numerical/graph/sanitizer verification, *not* a change to full-model routing. It should explicitly report the 2 >128 fallback groups and be labeled sample-specific.

Production `main`, default-off QPN policy, model pack, expert cache, runner privileges and unrelated services remain untouched.
