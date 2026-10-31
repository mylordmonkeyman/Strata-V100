# MMQ early CPU staging: matched-H2D positive diagnostic (2026-10-31)

**Decision: controlled H2D comparability passed and real CPU work omitted. Strong exploratory full-model latency signal; still NOT qualified for production or original asynchronous policy.** This is canonical-weight MMQ host-supply work, not a QPN-kernel acceleration result.

## Provenance
- [One-shot V100 run 41178870412](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41178870412), source `ba3f3200bf4db5d102071c97dbf5cfb16b0c6710`, [sanitized aggregate artifact 13034232629](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41178870412/artifacts/13034232629), ZIP SHA256 `17e6a99a6a666a3383c4404e27ab4b12160772d96c33c3f6436ac4c7bcc026b1`.
- Model SHA256 `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`; binary SHA256 `01af21f78534c4e58540096a5f638c42c7f2dcf882510ce407e08deb953b40b0`. No private prompts, logits, token IDs, weights, or routing records published.
- CUDA 12.8 model build, host and GPU jobs all succeeded. Four fresh-process ABBA runs with 2,047 prefill tokens and 16 generated tokens. A=canonical MMQ selective H2D, B=identical plus early CPU staging omission. Both use diagnostic `STRATA_QPN_MMQ_MATCHED_H2D=1`: wait for current-layer route publication before deciding H2D, deliberately giving up speculative transfer lookahead. This is not the original asynchronous baseline.

## Exact comparability and measurements

| Arm/order | First-chunk H2D copies | CPU jobs omitted | CPU bytes omitted | Prefill ms | TTFT ms | Process wall s | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 8,880 | 0 | 0 | 16,964.7 | 63,374.0 | 82.543 | 30.92 |
| B1 | 8,880 | 2,455 | 6,787,623,280 | 12,659.6 | 54,145.8 | 76.876 | 29.02 |
| B2 | 8,880 | 2,457 | 6,793,152,912 | 12,616.3 | 53,754.0 | 76.171 | 29.27 |
| A2 | 8,880 | 0 | 0 | 27,502.8 | 68,772.9 | 90.908 | 31.02 |
| **A median** | **8,880** | 0 | 0 | **22,233.75** | **66,073.45** | **86.726** | **30.970** |
| **B median** | **8,880** | 2,456 | ~6.79 GB | **12,637.95** | **53,949.90** | **76.523** | **29.145** |

- Exactly equal per-run first-chunk supply: **15,347 entries, 8,880 H2D copies, 6,467 skipped, 24,551,566,080 transferred bytes, 6,027 host jobs and 2,713 skip-eligible host jobs**. This repairs the 133-copy mismatch of the earlier four-run diagnostic without relaxing the gate.
- B omitted **2,455–2,457 real CPU staging jobs**, **6.321–6.327 GiB** of host preparation. The omission is substantial and directly measured.
- All **six** pairwise first-logit comparisons were bitwise equal across **248,320 FP32 logits**. No QPN groups executed. This does not qualify model quality, accepted speculative tokens, or all generated tokens.
- Exploratory median changes: prefill latency **−43.16%**, TTFT **−18.35%**, complete process wall **−11.76%**. Decode rate **−5.89%** (candidate worse), albeit only 16 generated tokens. This exceeds the permitted material-decode-regression concern until a longer sample resolves it.
- **High variance remains:** A2 prefill 27.50 s versus A1 16.96 s even though H2D is exactly matched. Fresh processes, page-cache/clock effects, host staging contention and startup cannot be disentangled. Forced route waits make this a controlled diagnostic, not a production-relevant gain estimate.

## Next bounded gate

A single source-controlled follow-up is justified by the matched-H2D positive evidence: two labeled process/page-cache warmups followed by **three balanced ABBA/BAAB/ABBA blocks (six timed processes per arm)**, 128 requested decode tokens, identical canonical weights and H2D policy, first-logit parity and exact H2D checks, per-block wall/TTFT/prefill reductions, p50/p95, and a predeclared **5% decode-regression stop**. Even a positive follow-up remains performance-only: no independent held-out prompt, warm-engine session, accepted-token/target-pass counters, or real asynchronous lookahead equivalence. Do not promote to production. Do not start QPN kernel/packed-residency sweeps. Next meaningful engineering task after this gate is an independently controlled realistic asynchronous supply comparison, and longer-context/accepted-token qualification.

**Safety:** one representation per expert in VRAM, canonical MMQ only, default-off diagnostic flags, shared V100 lock/concurrency, original pack/cache and production main unchanged.
