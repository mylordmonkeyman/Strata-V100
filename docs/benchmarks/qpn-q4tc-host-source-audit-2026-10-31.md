# Q4TC host-only source/ABI feasibility audit (2026-10-31)

**Scope:** independent rev4.2 §3A Q4-0 audit using existing source only, on `qpn/prepacked-stream@b0fbc2e`. No Q4 pack generated, no kernel written, no GPU benchmark launched, no model or cache changed, and no second VRAM weight format. This audit does not constitute Q4-0 fixture qualification or a Q4 performance claim.

## Verified source interfaces and blockers

1. `src/kernels/cuda/iq_kernels.cu` lines 751–762 include type **12/Q4_K** in `STRATA_GU_FMTS` and type **20/IQ4_NL** in `STRATA_D_FMTS`. The latter has 32-value blocks; the former 256-value blocks. `iq_row_bytes` lines 4125–4151 uses `n/256 * sizeof(block_q4_K)` for Q4_K and `n/32 * sizeof(block_iq4_nl)` for IQ4_NL. The native grouped eligibility check at lines 4273–4277 requires both supported types and `n_embd % qg == 0`, `n_ff % qd == 0`. For the Flash-Next expert, gate/up K=2560 is divisible by 256 and down K=640 is divisible by 32; **Q4_K gate/up + IQ4_NL down is geometrically admissible**. Q4_K down at K=640 would not be.
2. `native_expert_layout` lines 4280–4297 computes contiguous gate/up/down offsets but **appends the 16-byte FP32 `s_gate/s_up/s_down` tail only if either type is NVFP4 (40)**. An all-Q4 surrogate therefore silently loses the QAD tail if the existing ABI is used unchanged. Any QAD-derived equal-byte Q4 control must explicitly define a versioned tail-bearing format and ensure both its legacy and TC arms apply the scales at the same FP32 points; simply appending 16 bytes to an all-Q4 pack is not sufficient.
3. `src/prefill/moe_mmq.cu` lines 270–295 supports IQ4_NL directly but Q4_K only under **`STRATA_MMQ_KQUANTS` and `kq_ok()`**; `fits()` lines 297–323 further checks available MMQ tiles/shared memory for the target device. The native grouped code's type support alone does **not** prove Q4_K prefill MMQ is enabled in the pinned build or actually fast on sm_70. This is a concrete build/runtime qualification prerequisite.
4. Rev4.2 §3A's equal-byte arithmetic uses Q4_K 144 bytes/256 and IQ4_NL 18 bytes/32, giving 921,600 bytes per projection and 2,764,800 bytes/expert body; adding an explicitly supported 16-byte scale tail would match canonical NVFP4's **2,764,816 bytes/expert**. The **ggml struct sizes, quantizer parity and source reconstruction are not independently verified in this audit**. Equal bytes do not mean equal weight values or quality.
5. QAD NVFP4 E2M1/UE4M3 is not a Q4_K/IQ4_NL bitstream. A QAD-derived Q4 surrogate requires an independently checked FP32 reconstruction and **lossy requantization**, separate from the fixed-bit Q4 legacy-vs-Q4TC kernel comparison. No same-checkpoint numerical fidelity, model accuracy, kernel performance, or VRAM/cache lifetime has been measured for Q4TC.

## Q4-0 decision

**PARTIAL, host-source geometry/ABI confirmed; fixture and numerical gates OPEN.** The next permissible work is host-only verification of pinned ggml block sizes and a small, private, source-locked quantization/dequantization fixture with independent oracle, no GPU residency. Before any Q4 GPU test, demonstrate exact tail handling, native Q4/MMQ build eligibility, immutable source identity, one representation per expert, and a bounded R1 performance hypothesis. Never introduce canonical+Q4TC sidecars or overlapping old/new same-expert GPU buffers. No branch beyond the three user-authorized experimental branches, no production change, and no GPU test is authorized by this audit.

## Cross-phase tail-consumer audit and synthetic host oracle (later 2026-10-31)

**New Q4-0 blocker:** Appending a 16-byte scale tail to all-Q4 blobs is
insufficient even if the loader and GPU grouped/MMQ layouts are extended.
The existing native CPU fast paths **return before applying the scales**:

- [native_expert.cpp](../../src/kernels/cpu/native_expert.cpp):
  when `STRATA_KQ256=1`, `nt>=2`, `gu_type=12`, the Q4_K GU path
  calls `kq256_gu_rows` and returns *before* `sg/su` are read.
  [kq_avx2.cpp](../../src/kernels/cpu/kq_avx2.cpp) takes no global scale
  arguments and computes SwiGLU itself.
- On the host's AVX2-capable CPU, the normally enabled `nt>=2`,
  `d_type=20` IQ4_NL down path calls `iq4nl256_down_rows` and
  returns *before* `s_down` is read. Its
  [iq_avx2.cpp](../../src/kernels/cpu/iq_avx2.cpp) implementation takes
  no global scale argument. The one-token fallback does apply a
  `tail_off` scale if present. A hypothetical tail-bearing Q4 pack
  would therefore risk verification-window-length-dependent results.
- CUDA grouped native reads `NativeExpertLayout.tail_off`. The prefill
  MMQ gather, `swiglu_scaled`, `down_row_scales`, and combine depend
  on `NativeFmt.tail_off` (see
  [prefill.cpp](../../src/prefill/prefill.cpp)). A versioned Q4-tail ABI
  must consistently cover loader/pack metadata, native CPU fast paths,
  native GPU, prefill MMQ, and verification. No such runtime change
  has been implemented.

**Host-only numerical fixture added:** `tools/qpn_q4tc_block_oracle.py`
and `tools/test_qpn_q4tc_block_oracle.py` independently exercise
synthetic Q4_K 256-element blocks (all 6-bit scale/min fields, both
nibble halves and subblock boundaries), IQ4_NL 32-element blocks (all
16 nonlinear codebook entries), actual K=2560/640 projection widths,
nonunit FP32 gate/up/down scale placement relative to SwiGLU, and
malformed/nonfinite input rejection. Four tests passed locally with
Python stdlib only. Existing hosted `qpn.yml` test discovery runs
`test_qpn*.py` on push without any new workflow privileges.

```bash
python3 -m unittest discover -s tools -p 'test_qpn_q4tc_block_oracle.py' -v
```

This is **synthetic dequantization/scale-placement reference only**:
not an actual quantizer, QAD model fixture, pinned ggml/native runtime
parity, GPU benchmark, quantization-quality test or performance result.
No private data, weight conversion, cache or GPU job was used.

**Decision:** Q4-0 remains PARTIAL and R1 exposed-time UNKNOWN. A
*tail-free, scale-folded* QAD-derived Q4 pack can use the current native
ABI but changes quantization error and must not be called exact
same-math QAD. A *tail-bearing* versioned Q4 ABI instead requires
scale-correct CPU fast paths, MMQ, GPU, loader and verification before
any model use. Both require a source-locked quantizer, independent
QAD-derived fixture, strict one-representation-per-expert lifetime,
and a new R1 critical-path justification before any GPU prototype.
The stopped early-staging and static-admission subtracks remain stopped.

## Source-locked pinned ggml quantizer qualification (later 2026-10-31)

**Host-only CI [41191656041](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41191656041) passed both hosted CPU and self-hosted compile-only jobs** at experimental commit 65b536e8b1944d46e704b957c6134231823efeb3. The new test executable (tests/host/qpn_q4tc_pinned_fixture.cpp) uses the actual pinned ggml-org/llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d through the existing Strata CMake dependency, with ggml_quantize_chunk, reference quantize_row functions and ggml dequantization. Its transient synthetic IPC is captured by the independent Python block oracle and is not published as raw bytes.

Six deterministic QAD-shaped **synthetic** projection rows were tested: Q4_K gate/up K=2560, IQ4_NL down K=640, each with tail-preserved and tail-folded nonunit FP32 scales. Source FP32 identity, ggml type/row/block sizes, finite dequantization and independent block decode all passed. Actual pinned ggml quantization (not a handwritten substitute) yielded the following synthetic tail-preserved / tail-folded RMSE against the scaled source:

| Role | Tail-preserved RMSE | Scale-folded RMSE | ggml_quantize_chunk bytes equal to quantize_row_*_ref |
|---|---:|---:|---|
| Gate Q4_K | 0.328902 | 0.328873 | yes / yes |
| Up Q4_K | 0.133257 | 0.133257 | yes / yes |
| Down IQ4_NL | 0.548816 | 0.548816 | **no / no** |

**Important ABI/quality finding:** the pinned ggml IQ4_NL reference quantizer calls quantize_row_iq4_nl_impl with ntry=-1, while the production chunk quantizer uses ntry=7, so they are **not byte-identical on these inputs**. A source-locked Q4 artifact must explicitly choose and record the actual quantizer entry point, not assume the _ref fixture is the same algorithm. The follow-up extends the host-only test to independently dequantize and report error for the reference quantizer as well. No NVIDIA checkpoint bytes, real experts, logits, prompts or routing were used.

This is genuine pinned-source **synthetic** Q4-0 progress, but not a QAD checkpoint fidelity test, model-quality qualification, Q4TC kernel performance result, measured R1 exposed-time opportunity, or authorization for a GPU prototype. The Q4 FP32 tail ABI and native CPU fastpath scale bypass remain unresolved. One resident representation per expert, unchanged production/default MMQ, and stopped QPN/early-staging tracks remain mandatory.
