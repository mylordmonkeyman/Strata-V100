# Experimental full-model QPN staged-prefill A/B — October 29, 2026

## Decision: not qualified

**The first off-by-default full-model integration successfully executed QPN,
but did not demonstrate a full-model speedup and failed the matched-order
first-logit numerical acceptance gate.** Keep experimental dispatch disabled.
Neither this sample nor the previous standalone 16-expert speedup supports
production enablement.

Isolated branch `qpn/full-model-opt-in`, production `main` untouched.

## Provenance and experimental scope

- [First functional smoke run 40948757851](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40948757851):
  **success** under the original permissive KL < 0.03 and top-1 test,
  source [`8e4b1079e33a1f9a42bf2f3035a4e33cbed1603c`](https://github.com/mylordmonkeyman/Strata-V100/commit/8e4b1079e33a1f9a42bf2f3035a4e33cbed1603c),
  artifact [12930367928](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40948757851/artifacts/12930367928),
  ZIP SHA256 `8043cf8af4d81856598823fbfd355315e94ae15541b99f3930257671a420e03c`.
  **Not a matched-order comparison**: only QPN reorganized expert groups.
- [Definitive matched-order V100 A/B run 40949838878](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40949838878):
  **failed the first-logit KL quality gate** (not an execution, build or
  safety crash); host routing-policy test successful; source
  [`75a23e77a7a658f31fb2adc60955804b6d7e1f6d`](https://github.com/mylordmonkeyman/Strata-V100/commit/75a23e77a7a658f31fb2adc60955804b6d7e1f6d),
  artifact [12932305861](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40949838878/artifacts/12932305861),
  ZIP SHA256 `7d61711f2a633cdb4a2f26d839fa5fa54e14909757672e6d9e0121a7c21f6545`.
- Reused the **same private** NVIDIA Qwen3.8-Flash-Next-FP4-Local
  native NVFP4 expert pack, SHA256
  `1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5`,
  and the pinned 2,048-ID prompt, SHA256
  `df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414`.
  These match the [successful unchanged M0 baseline run 40815727472](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40815727472)
  in input/model identity.
- **768-token staged chunk** (actual 2,047 batched prefill tokens across
  three chunks); NOT the model's default 1,024-token stream-all mode.
  Same single V100 sm_70 GPU, same binary within each run, 24-GiB CPU
  resident expert budget, 48-GiB runner cgroup, pinned NUMA 0,
  no external GPU clients. Greedy first-token evaluation, no MTP/lookup
  drafting, native verifier window `--spec 2`, exactly as successful M0.
- Model weights, prompt IDs, logits, and verbose logs stayed on the
  private runner. CI artifacts contain only hashes, counts, timing
  aggregates, and numerical comparisons.

## Matched-order real-model measurements (run 40949838878)

Both arms set `STRATA_QPN_PREFILL_PLAN_CONTROL=1`: the MMQ control uses
the **same row-count sorting rule as QPN** but executes no QPN kernels.
It is an explicitly sorted MMQ control, not Strata's untouched default
MMQ expert order. Differences in downstream routed groups can emerge from
different numerical activations after earlier QPN layers.

| Trial | Mode | Prefill ms | TTFT ms | Decode tokens/s | Planned groups | Executed QPN groups |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Matched MMQ | 14,110.0 | 55,943.4 | 29.53 | 368 | 0 |
| 1 | Selective QPN | 14,078.5 | 54,422.7 | 29.28 | 362 | 362 |
| 2 | Selective QPN | 14,070.0 | 54,590.1 | 29.28 | 362 | 362 |
| 3 | Matched MMQ | 13,877.0 | 54,537.5 | 29.69 | 368 | 0 |

**Median prefill** MMQ 13,993.5 ms; QPN 14,074.25 ms,
**0.577% greater QPN latency** (MMQ/QPN throughput ratio 0.99426x).
Individual corresponding pair speedups are 1.00224x and 0.98628x;
these straddle unity. No claimed speedup. **TTFT includes substantial
model initialization and GPU expert-cache filling**: the first trial's
55.9-second TTFT versus later 54.4-54.6 seconds is startup/order noise,
not evidence that QPN accelerates first-token computation.
Decode rates are incidental and too close to be attributed to
prefill-only kernels.

**QPN execution actually exercised:** 362 qualifying pure 17–32-row
groups, 4,863 experts over the three prefill chunks (144 layer-chunk
diagnostic records). The matched MMQ control identified 368 qualifying
groups and 4,855 experts but executed all on MMQ; later routing can
diverge numerically.

## Numerical failure (run 40949838878)

- First-logit output dimension: **248,320** FP32 logits.
- The two MMQ first-logit byte hashes exactly agree:
  `ea59f668290eb6e98db07e7ab3f910636080b108e626ad9fed11fc470e14f178`.
- The two QPN hashes exactly agree:
  `be65c49bdea5749a305c928d971a91f187ed3d0e16cb5e75450fdbb84a2dcfa7`.
  This is a deterministic cross-mode difference, not repetition noise.
- Both first-token argmax IDs were 760, but **KL(MMQ || QPN)
  = 0.0313477502** and relative FP32 logit L2 = **0.0737182921**
  (7.37%). This **fails** the harness's KL < 0.03 gate.
  Matching one top token does NOT establish quality parity.
- The original **unmatched-order** smoke run gave KL 0.0169553559,
  relative L2 0.07278418. The change in KL reinforces why the
  matched-order control was required; neither result demonstrates
  broad quality equivalence.
- The earlier standalone real-expert graph/memcheck/racecheck passes
  only cover the prior standalone kernels/harness, **not integration-level
  memory safety for this staged full-model driver**.

## Findings and outstanding engineering gates

1. **Numerical alignment.** The QPN path currently stages the original
   FP16 `mixed_h` activations and FP32 hidden values to FP16, whereas
   MMQ uses Q8_1 activation quantization and Q8_1 hidden quantization.
   These are concrete potential sources of different rounding, but
   their fraction of the model-level discrepancy is **unmeasured**.
   The pinned llama.cpp source at commit
   `3cf03257f219afbe7334045ff7c6a06ac68c627d` confirms
   `block_q8_1_mmq` has **144 bytes**: `float d4[4]` (four
   FP32 scales for 32 values each) followed by `int8_t qs[128]`.
   `GGML_TYPE_NVFP4` selects `MMQ_Q8_1_DS_LAYOUT_D4`. Its quantizer
   stores a row's 128-value block at index
   `(column / 128) * total_rows + row`, not at contiguous 36-byte
   GGML `block_q8_1` offsets.
   Thus `qpn::stage_q8_1` **cannot** safely consume `Xq/Hq`: it
   assumes a linear sequence of 36-byte blocks with FP16 block scales.
   Implement a separately named **D4 layout-aware device staging
   kernel** with `float d4[(column % 128)/32] *
   qs[column % 128]` and FP16 output. Prove per-row byte/scale
   parity using both synthetic D4 buffers and actual
   `mmq::quantize_scatter`/group `mmq::quantize` output,
   including guards and memcheck, before enabling a Q8-aligned
   matched full-model A/B. No such Q8-alignment integration was
   enabled in the reported runs.
2. **Launch and memory overhead.** Current integration issues serial
   QPN gate/up/down projections and per-expert activation staging
   inside 16-expert groups. Test grouped/ragged launches with exact
   routed row ownership and matched MMQ weight-supply/cache costs;
   validate under CUDA Graph, memcheck and racecheck.
3. **Real production prefill shape.** Stream-all 1,024-token chunks
   use a different asynchronous copy/issuer and cache-lifetime
   protocol. They remain MMQ. Do not reroute their experts until
   replay/issuer/ring ownership has independently passed.
4. **Model-wide quality/performance.** Use multiple independently
   pinned prompts and longer continuations, first-token KL/top-k,
   real TTFT and per-layer/GPU timings. The present single-prompt
   ABBA has no demonstrated prefill benefit and fails numerical gate.

**Disposition:** experimental compiler flag and runtime switch remain
off by default; no production enablement, no main-branch changes,
no model-pack changes, no cache slot/runner/service changes.

## Pinned upstream layout evidence

Reviewed `ggml/src/ggml-cuda/mmq.cuh` (lines 18–77) and
`ggml/src/ggml-cuda/quantize.cu` (lines 456–552) at
[upstream commit `3cf03257`](https://github.com/ggml-org/llama.cpp/tree/3cf03257f219afbe7334045ff7c6a06ac68c627d).
For each `i0`, the source uses `k_block=i0/128`, `iqs=i0%128`,
`ib=k_block*ne1+i`, writes `y[ib].qs[iqs]` and the FP32 scale
`y[ib].d4[iqs/32]`. Any QPN conversion must preserve this indexing
and the caller's actual `ne1` (all routed rows for scattered X,
only group-relative rows for quantized H). These are verified source
facts, **not** yet measured numerical or performance improvements.
