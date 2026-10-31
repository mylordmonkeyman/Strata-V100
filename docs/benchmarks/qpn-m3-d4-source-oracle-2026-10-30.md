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

## Not yet qualified / independent gates

Host oracle coverage is **not** real ggml quantize/scatter device-output parity, compiled-CUDA or device numerical verification. The experimental CUDA stage still requires a pinned CUDA-12.8 compile, actual producer D4 X/H reads with independently verified `ne1`, oracle parity, tail/invalid-count checks, changed-input Graph, guards and sanitizer checks. Full model requires earlier divergence attribution, FP32-versus-D4 controls, 100+ held-out quality prompts and accepted decode verification. Do not dispatch this kernel in a model or claim quality/performance until these tests pass.

Static packed admission remains **STOPPED** and timing-based R1 **UNKNOWN**. No QPN cache prototype or new Q4TC GPU kernel is authorized.
