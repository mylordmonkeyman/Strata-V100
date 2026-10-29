# Off-by-default QPN staged full-model experiment

**Branch only:** `qpn/full-model-opt-in`. The previous standalone qualification
`qpn/mixed-route-bench` and production `main` are immutable controls.

## Gate and eligible execution

Both `-DSTRATA_ENABLE_VOLTA_QPN=ON` and
`-DSTRATA_QPN_PREFILL_EXPERIMENTAL=ON` are required at **build** time.
The candidate binary still defaults to unmodified MMQ unless environment
`STRATA_QPN_PREFILL_EXPERIMENTAL=1` is specified. Supported only on
the NVIDIA sm_70 CUDA backend with both routed expert matrices in native
GGUF NVFP4 and W4A8 activation scaling. The prototype deliberately
rejects/keeps MMQ for other types, stream-all, fused, peer, CPU-share,
WMMA, Hadamard, direct pointer-table, and other sorting modes.

In staged expert prefill (chunk sizes **below the stream-all threshold**),
the host routes and partitions genuine expert IDs so that all qualifying
17–32-row experts share complete 16-slot groups where possible. An
independent host verifier asserts exactly one owner for every actual
token/rank entry. Only groups in which **every** expert has 17–32
captured routed rows enter QPN; all remaining groups take unchanged W4A8/MMQ.
No model pack or expert cache is rewritten. QPN reads the *same native
compressed gate/up/down weights* after the existing gather, and uses
the existing combine's per-expert down scales. One per-stage <500 KiB
scratch bundle is opt-in only.

## First paired real model A/B

The [dedicated CI workflow](../../.github/workflows/qpn-full-model-opt-in.yml)
compiles the candidate separately, reuses the **already prepared private**
Qwen3.8-Flash-Next-FP4-Local checkpoint, native `experts.bin`, identical
2,048-token captured prompt IDs, and shared binary for both control/opt-in.
All four trials are separate engine invocations, ABBA to mitigate thermal,
cache, and trial-order effects. Greedy, native verifier window `--spec 2` (required for this pack), no MTP/lookup drafts (`--suffix-draft 0`), max-new 16,
24 GiB resident expert budget, 48 GiB runner memory cgroup, same GPU
and CPU affinity. **Prefill 768** is intentional to exercise staged
nonuniform groups; it is NOT a claim about the default 1,024-token
stream-all path.

Metrics: actual prefill timing, true reported time-to-first-token (includes
startup), decode tok/s, first-logit SHA-256 and relative L2/KL, greedy
argmax agreement, QPN actual selected group and expert counts, model/prompt/
binary hashes. Logit arrays, prompt IDs, model weights, and verbose run
logs stay private in /srv/strata-qpn; only aggregate JSON is uploaded.

Fail closed on startup failure, unexercised QPN selection, missing
logits/statistics, KL >=0.03 or first-logit argmax changes. A slower but
correct candidate is **not** treated as a test failure; it is a failed
performance hypothesis and must not be enabled in production.

This first experiment does not qualify full default streaming prefill,
broader quality suite, different prompts, kernel racecheck within the
full model, or deployment.

## Baseline-compatible verifier setting

The pinned successful M0 manifest from [run 40815727472](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40815727472) used `--spec 2`, not zero, for its native-expert pack. The first two candidate CI control runs (40946109728 and 40947374667) exited with code 2 in ~1.2 seconds, before inference, because the new A/B harness used `--spec 0`. The verifier-window requirement is enforced by the engine's native-pack startup gate. Keep `--spec 2` with `--suffix-draft 0` and no MTP model to match the baseline's non-drafting behavior. Those failed startups are **not performance results**.
