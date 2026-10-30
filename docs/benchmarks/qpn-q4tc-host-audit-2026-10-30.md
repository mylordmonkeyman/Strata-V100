# Q4TC host-only source/geometry audit — 2026-10-30

**Scope:** Rev4.2 §3A Q4-0 source/byte/geometry/licensing screen, **source inspection only**. No private model, weights, activations, routes, prompts, logits, conversion, GPU runtime, Docker, cache, production code or host service changes. This document does not authorize Q4TC kernel work or a Q4 full-model pivot.

**Pinned sources:**
- Strata production reference `2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926`, [native expert geometry and tail](https://github.com/mylordmonkeyman/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/kernels/cuda/iq_kernels.cu#L4273-L4298), [type lists](https://github.com/mylordmonkeyman/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/kernels/cuda/iq_kernels.cu#L750-L765), [MMQ support](https://github.com/mylordmonkeyman/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/prefill/moe_mmq.cu#L270-L292).
- Superfast reference `403f06f580e3a2451061b444ad7e14fa897636f0`, [QPN eligibility](https://github.com/bitexploder/llama-superfast-v100/blob/403f06f580e3a2451061b444ad7e14fa897636f0/ggml/src/ggml-cuda/mmvq-qpn.cu#L2157-L2171), [repack geometry](https://github.com/bitexploder/llama-superfast-v100/blob/403f06f580e3a2451061b444ad7e14fa897636f0/ggml/src/ggml-cuda/mmvq-qpn.cu#L2257-L2283), [IQ4_NL 256-column fragment type](https://github.com/bitexploder/llama-superfast-v100/blob/403f06f580e3a2451061b444ad7e14fa897636f0/ggml/src/ggml-cuda/mmvq-qpn.cu#L108-L116), [root MIT license](https://github.com/bitexploder/llama-superfast-v100/blob/403f06f580e3a2451061b444ad7e14fa897636f0/LICENSE).

## Verified distinction: canonical IQ4_NL support != Superfast QPN eligibility

For the rev4.2 Flash-Next expert shape, gate/up each have **K=2560, N=640** and down has **K=640, N=2560**. Strata's `native_expert_supported` checks block divisibility separately for each type. Its grouped format lists contain type 12 (`Q4_K`) for gate/up and type 20 (`IQ4_NL`) for down. The canonical grouped path therefore has the necessary source-level shape/type dispatch for `Q4_K` gate/up plus `IQ4_NL` down: `2560 % 256 == 0` and `640 % 32 == 0`. This is **not** a successful end-to-end new-model test.

**New blocker:** Superfast's `ggml_cuda_qpn_takes` requires `t->ne[0] % QK_K == 0` with `QK_K=256`, for **all** QPN types, including `IQ4_NL`. Its IQ4_NL QPN fragment packs eight 32-value blocks into a 256-column unit, and the repacker calculates `nsb = ne[0]/256`. Consequently:
- `Q4_K` gate/up **K=2560** meets this particular QPN geometry requirement (other route/format gates still apply).
- `IQ4_NL` down **K=640** fails (`640 % 256 = 128`), even though its canonical 32-value GGUF blocks are valid. Merely enabling IQ4_NL in Superfast's QPN type mask cannot fix this.
- Padding down K from 640 to 768 would add **128 columns / 20% of down weight bytes and down arithmetic**. At the rev4.2 4.5-bpw accounting, the down body grows from 921,600 to 1,105,920 bytes, and the total three-projection body from 2,764,800 to **2,949,120 bytes (+6.67%)**, before metadata/scratch. Thus padded down is **not** the specified equal-byte comparison.
- A custom partial-128 fragment or mixed packed gate/up + **canonical** IQ4_NL down is a *different, unbenchmarked* design. Neither has an established speedup or model-quality result; neither is authorized for automatic GPU implementation.

## Separate scale-tail and MMQ conditions

Strata's `native_expert_layout` adds the 16-byte `{s_gate,s_up,s_down,0}` FP32 tail **only when either projection type is NVFP4 (40)**. An all-Q4 `Q4_K/IQ4_NL` surrogate does **not** automatically allocate or consume that tail. A same-math QAD-derived comparison therefore needs a separately versioned, tested tail ABI (or an explicitly different scale-folding experiment). Appending 16 bytes to an ordinary Q4 expert is insufficient.

Strata's prefill-MMQ path accepts `IQ4_NL` directly, but `Q4_K` support is conditional on compile-time `STRATA_MMQ_KQUANTS` and `kq_ok()` (which checks runtime `STRATA_MMQ_KQ=1` or the optional DSO availability). The production controls alone do not prove a new all-Q4 pack runs end to end. No such runtime test was performed.

## License / provenance screen

The pinned Superfast root `LICENSE` is MIT and attributes copyright to the ggml authors (2023–2026); copied or substantially adapted material would require retaining its notice. This is an initial root-license check, **not** a file-by-file provenance clearance. The upstream origin of any specific code fragment and applicable notices must be audited before copying. No Superfast source is copied into Strata by this report.

## Decision

**Q4-0 source audit: PARTIAL; equal-byte full Q4TC tensor-core path is NOT READY.** Canonical type/shape paths appear present, but the proposed unmodified Superfast packed IQ4_NL down path is ineligible at K=640; QAD FP32 tails are not handled by the existing all-Q4 ABI. No full-model Q4 conversion, GPU kernel, or automatic Q4 pivot is justified. Any future proposal must first resolve partial-K versus fallback with byte/scratch accounting, preserve numerical and licensing gates, and show a bounded exposed-critical-path opportunity. The negative static NVFP4 resident-admission screen and R1 timing UNKNOWN remain unchanged.

## Reproducible host-only Q4TC geometry and cross-phase memory bound

The pinned source facts can be calculated without accessing the original
model or creating a new quant: \`python3 tools/qpn_q4tc_geometry.py --report
/tmp/qpn-q4tc-geometry.json\` and \`python3 -m unittest discover -s tools -p
test_qpn_q4tc_geometry.py -v\`.

With \`Q4_K\` gate/up at 2,560 input columns and \`IQ4_NL\` down at
640, the three canonical 4.5-bpw projection bodies are 921,600 bytes each,
or 2,764,800 bytes per expert **excluding** the QAD 16-byte FP32 scale
tail. The unmodified Superfast repacker accepts gate/up's 256-column
geometry but rejects down's 640-column dimension. A *hypothetical*
mixed gate/up-packed, canonical-down design offers nominal 2/3
packed weight-body/MAC coverage, **not** measured acceleration and not a
complete executable pipeline. A custom 128-column remainder could in
principle cover 80% of the down input dimension with two full 256-column
fragments, leaving 20% as a new partial path; it requires unimplemented
code and has no speed or quality evidence.

Padding IQ4_NL down to 768 columns would increase down storage by 20%
(184,320 bytes per expert), total three-projection weight-body storage
by 6.67%, and would add 135 MiB across the previously selected 768
experts. It is **not equal-byte**, and valid zero padding/scale-tail
semantics would need proof.

**Cross-phase native-verification constraint:** preserving original
canonical gate/up *and* holding a separately packed gate/up body for
all 768 selected experts would require **1,350 MiB** of extra weight
bodies, before scratch, metadata or allocator fragmentation. The earlier
**737 MiB guarded spare-memory observation applies to the original NVFP4
configuration**, not a new Q4 model. As a *conditional comparator*,
737 MiB would hold at most 419 such gate/up-only sidecars (or 279
all-three-projection sidecars), ignoring all other allocations. This
does **not** cap replacement payload or certify Q4 allocator fit.
Replacing canonical bytes instead avoids duplicate storage but leaves
lossless restoration and native verification demands unmeasured.

**Decision remains Q4TC source/geometry PARTIAL and NOT READY.** This
is a constrained host-only alternatives analysis, with no raw private
model/route access, no conversion, no GPU kernel, no new model A/B, and
no relaxation of the rev4.2 R1/R3 gates. Existing static NVFP4 admission
is still STOPPED and full-request timing feasibility UNKNOWN.
