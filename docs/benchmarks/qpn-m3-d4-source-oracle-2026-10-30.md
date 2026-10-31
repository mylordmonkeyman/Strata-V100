# M3 D4-aware Q8_1 staging — isolated source/oracle implementation (2026-10-30)

**Scope:** rev4.2 M3 independent numerical-attribution preparation. D4 staging is separate from legacy 36-byte Q8_1 staging; experimental library only; not wired to the model. NO new QPN model run, cache, quantization, dual weight copies, private capture, or production dispatch.

## Pinned producer layout
- Upstream ggml-org/llama.cpp at `3cf03257f219afbe7334045ff7c6a06ac68c627d`, `ggml/src/ggml-cuda/mmq.cuh` defines 144-byte `block_q8_1_mmq` with four FP32 scales and 128 signed codes.
- Its `quantize.cu` D4 producer computes block index `(column/128)*ne1+row`, including scatter where `ne1=nrows_dst`, and non-scatter where the physical pitch is passed in.
- `GGML_TYPE_NVFP4` chooses `MMQ_Q8_1_DS_LAYOUT_D4`; do NOT reinterpret DS4/D2S6 bytes or apply the 36-byte `stage_q8_1`.

## Implemented, default-off
- `tools/qpn_d4_oracle.py`: independent, stdlib-only host reconstruction to binary16 from caller-supplied synthetic D4 bytes. Validates shape, physical row pitch, payload length, finite values and binary16 range.
- `tools/test_qpn_d4_oracle.py`: synthetic 128/256/640/2560 widths, transposed and padded rows, four distinct scales, corruption/overflow rejection. No model or private data used.
- `src/kernels/cuda/nvfp4_volta_qpn_decode.cu` and `include/strata/kernels/nvfp4_volta_qpn.hpp`: new named standalone `stage_mmq_d4_q8_1_to_fp16` with exact D4 pointer/pitch layout and explicit required caller-owned device invalid-count flag. The status counter MUST start at zero and MUST be checked after synchronization before output consumption; this is not integrated into any model dispatch. No weight repacking or additional same-expert residency is used.

Local host-only command:

```bash
python3 -m unittest discover -s tools -p 'test_qpn_d4_oracle.py' -v
```


## Pinned D4 producer-algorithm host reference (independent M3 increment)

`tools/test_qpn_d4_quantize_reference.py` emulates the pinned ggml
`quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>` arithmetic on synthetic FP32
source rows, including 32-value-group maximum magnitudes, scaled int8
rounding with CUDA `roundf` tie-away-from-zero semantics, four FP32 scale
words per 128-column physical block, and column-block-major row pitch.
The test also emulates the independently generated compact inverse-token
scatter mapping, including repeated source tokens in distinct compact rows.

The host-only reference checks the independent `decode_d4_to_f16` oracle
at widths 128/256/640/2560, with padded physical rows and precise
128-column block-boundary indexing. It bounds reconstructed error relative
to synthetic source values, tests ties-away-from-zero, and fail-closes on
synthetic all-zero and nonfinite subblocks rather than inventing CUDA's
behavior for such inputs. Combined with the preexisting four oracle tests,
**8 host tests passed**:

```bash
python3 -m unittest discover -s tools -p 'test_qpn_d4_*.py' -v
```

**Evidence classification:** this is a *source-algorithm host simulation*,
NOT a CUDA compilation, actual ggml GPU quantizer-output parity,
real-model quality test, or improved latency. All-zero block semantics
and exact float-operation/compiler behavior require direct device
verification; a matching host simulation alone cannot qualify M3.
Neither canonical nor QPN weight residency was changed.

## Verified producer padding correction (new M3 finding)

Source audit of `src/prefill/moe_mmq.cu` shows that both
`mmq::quantize` and `mmq::quantize_scatter` call ggml's MMQ Q8
producer with **`pad512(cols)` as its physical quantized width**.
`mmq::q8_bytes` allocates that physical region **plus 128 extra 144-byte
blocks of MMQ overread guard**. A logical 640-value hidden row therefore
uses **1024 physical columns**, not 640; 2560 stays 2560.

The original experimental CUDA D4 stage checked
`input_bytes == (logical_columns/128)*ne1*144`. That check cannot admit
the actual padded 640-column producer buffer and would also reject an
allocation that includes the existing MMQ guard. Corrected:
the required readable physical D4 extent is
`(pad512(logical_columns)/128)*ne1*144`, and `input_bytes` must cover
at least that extent. The kernel still emits **logical** `[rows,columns]`
FP16 values and never accesses padded columns or the trailing guard.
This remains conditional on verified W4A8 D4 provenance; W4A4/DS4/D2S6
sources must never be passed to this function.

The independent Python oracle accepts an explicit `physical_columns`
for padded real-producer fixtures; its default logical-width mode is
only for prior synthetic-format tests. The producer-algorithm test now
allocates `pad512(cols)` physical bytes and leaves unused columns
poisoned. A new 640→1024 test ensures it can read logical values and
never reads poisoned padding. **Nine host-only D4 tests passed.**

A separate **manual-only** CUDA target, `nvfp4_qpn_d4_stage_parity`,
was added to `cmake/volta_qpn.cmake`. It is behind
`STRATA_ENABLE_VOLTA_QPN=ON` and `STRATA_BUILD_TESTS=ON`, off in normal
builds, and deliberately **not registered with CTest** (no automatic
GPU run). It uses synthetic D4 activation buffers only: padded physical
widths, guard bytes, exact FP16 output checks, malformed-shape rejection,
device invalid-counter checks, and changed-input CUDA Graph replay.
No expert weights or quantization replacements are ever resident.

Build the new target using an already configured isolated CUDA-12.8
sm_70 build directory (no Docker rebuild):

```bash
cmake --build "$EXISTING_QPN_CUDA_BUILD" --target nvfp4_qpn_d4_stage_parity
```

This is **compile-only**. Any device execution must first independently
verify that the existing GPU is idle, obtain the **existing**
`/srv/strata-qpn/gpu.lock`, preserve concurrency
`strata-qpn-v100` / `cancel-in-progress:false`, and use the installed
CUDA 12.8 environment. Neither CUDA compilation nor device execution
has been performed by this change. Device testing against **actual**
ggml-generated D4 bytes is still necessary after synthetic-device
parity; a simulated host quantizer is not that proof.

## Not yet qualified / independent gates

Host oracle coverage is **not** real ggml quantize/scatter device-output parity, compiled-CUDA or device numerical verification. The experimental CUDA stage still requires a pinned CUDA-12.8 compile, actual producer D4 X/H reads with independently verified `ne1`, oracle parity, tail/invalid-count checks, changed-input Graph, guards and sanitizer checks. Full model requires earlier divergence attribution, FP32-versus-D4 controls, 100+ held-out quality prompts and accepted decode verification. Do not dispatch this kernel in a model or claim quality/performance until these tests pass.

Static packed admission remains **STOPPED** and timing-based R1 **UNKNOWN**. No QPN cache prototype or new Q4TC GPU kernel is authorized.

## Real ggml producer-to-stage parity target (manual only; pending execution)

A second, independent source-level qualification target
`nvfp4_qpn_mmq_d4_producer_parity` exercises **the pinned actual Strata
MMQ producer**, not only manually constructed D4 bytes:

- It prepares deterministic synthetic FP32 activation rows (3×640 and
  3×2560), calls `strata::prefill::mmq::quantize` for
  `GGML_TYPE_NVFP4` in W4A8, and then calls the new D4→FP16 stage.
- It downloads the resulting **synthetic device-produced D4 bytes** and
  independently reconstructs their logical FP16 output on the host,
  verifying bitwise agreement with the staging kernel, the invalid
  counter, and the untouched FP16 output guard.
- It separately calls `mmq::quantize_scatter` for 3×2 synthetic rows
  at 640 logical columns, checking compact inverse-row mapping and
  duplicate-source consistency. W4A4 mode and `STRATA_QUANT_GATHER=1`
  are explicitly rejected by this targeted test.
- It verifies `mmq::q8_bytes` has room for `pad512(cols)` plus the
  existing 128-block overread guard. No expert weight representations
  are loaded, converted or retained.

The test is an optional CMake target when
`STRATA_ENABLE_VOLTA_QPN=ON`, `STRATA_BUILD_TESTS=ON`, and
`strata_mmq` / `ggml-base` are built. It is **not** part of automatic
CTest and cannot run merely by building the target:

```bash
cmake --build "$EXISTING_QPN_CUDA_BUILD" --target nvfp4_qpn_mmq_d4_producer_parity
```

The actual device test may run only after independently checking existing
runner activity, taking the existing `gpu.lock`, and retaining all
CUDA12.8/cache/security constraints. No such run was performed here.
`M3 GPU parity` remains **UNVERIFIED** until successful device
execution and sanitizer review. Even a pass is not full-model
numerical-quality or performance qualification.
