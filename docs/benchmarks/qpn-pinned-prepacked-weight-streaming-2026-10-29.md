# Prepacked QPN: from fast VRAM-resident kernels to measured pinned-host streaming

**Scope:** One V100 sm_70, isolated branch `qpn/prepacked-stream`. Native GGUF NVFP4 expert weights are compressed 36-byte blocks (four UE4M3 scales and 32 nibble-code bytes). The QPN kernel decodes the compressed blocks during MMA; it does **not** first dequantize the entire expert to FP16. The important layout variable is where these *compressed* blocks live relative to adjacent output rows and K-blocks.

## Verified prepacked/VRAM-resident precursor

[Successful V100 run 40976102848](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40976102848), source `ddbc14a9957779b31785e2e79aad68adf54f3d6b`, aggregate artifact **12942693497**.

- Six alternating paired rounds, ten complete 16-expert forwards each: all MMQ median **1450.185633 µs**; hybrid QPN with 14 MMQ + two prepacked T187/T508 QPN experts **1281.587219 µs**. Ratio **1.1315544×** (+13.16% throughput for this real mixed work), while Graph replay 1327.206421 versus **1175.398397 µs**, ratio **1.1291545×** (+12.92%). This is **not** full-model speedup.
- Exactly 1,100 original real routed rows; the two selected experts are 187 and 508 rows, totaling 695, other fourteen remain MMQ. Compressed expert bytes 44,237,056 > 6,291,456 V100 L2. Weights and activations from the private, SHA-verified fixture are never uploaded.
- Original precursor packed weights once on the host in `src/kernels/nvfp4_volta_mixed_bench.cpp`, uploaded an additional entire **44,237,056-byte** GPU buffer, and benchmarked both arms using already-resident weights. Reported repack **32199.767 µs** and *full buffer* upload **6415.755 µs**. Those timings are **outside the forward**; a model that repacks/uploads per request would lose this advantage.
- Numerical/safety: 32 sampled independent GGUF FP64 oracle rows, MMQ/oracle L2 **0.0112966272**, mixed/oracle **0.0101718218**, hybrid/MMQ **0.011639834**, finite; changed-input CUDA Graph replay bitwise parity; output guard intact; Compute Sanitizer memcheck **zero errors**, racecheck **zero hazards/errors/warnings**. Not broad model quality qualification.

## New real pinned-host stream comparison

Source commit [84f0b5f](https://github.com/mylordmonkeyman/Strata-V100/commit/84f0b5fdca6e5491508ff41fca2647b20086b60e) plus [stream CI e680c1b](https://github.com/mylordmonkeyman/Strata-V100/commit/e680c1bde114ebcf6f2b20885274f58b9791cf06). Benchmark workflow [QPN pinned prepacked NVFP4 streaming](https://github.com/mylordmonkeyman/Strata-V100/actions/workflows/qpn-prepacked-stream-real.yml).

One-time setup:
- Build the exact original canonical GGUF expert order and convert only the 1280-wide GU and 2560-wide down compressed block layout for selected experts T187/T508 from [row][K-block] into [K-block][row]. Each 36-byte block stays unchanged, retaining all quantization bytes and the expert's FP32 tail scalars. Original canonical GGUF values and MMQ remain available.
- Allocate one **44,237,056-byte pinned canonical host source** and one **5,529,632-byte pinned QPN-selected source**, together with independent GPU slots. No model weights or artifacts are changed. All host prepacking/allocation takes place **outside** ordinary/graph inference timing, and CPU repack time is reported explicitly.
- Baseline `all` issues pinned **44,237,056 bytes** host-to-GPU DMA from the canonical source, then all-MMQ 16 experts, on one stream.
- Hybrid `mix` issues pinned **38,707,424 bytes** of original canonical GGUF for the 14 MMQ fallback experts plus **5,529,632 bytes** of prepacked QPN-native compressed blocks for the two selected experts, totaling the exact **same 44,237,056 DMA bytes** as the baseline. No per-iteration GPU or CPU layout transposition. On-device gather/MMQ or QPN GU/SwiGLU/down follows stream copies.
- Both copies are on the compute stream, so paired CUDA-event and Graph times include HtoD and weight processing. This explicitly **does not model async copy/compute overlap**, real model layer-ahead lookahead, resident-cache hit rates, multi-layer requests or host-to-GPU DMA contention. Those require later model-integrated tests.

Qualification gate: complete CUDA/host compile, both all-MMQ and hybrid with matching bytes, ordinary and graph medians with at least 5 paired rounds, correct transferred bytes and branch selected counts, changed-activation CUDA Graph replay, 256-byte canaries, independently sampled FP64 numerical oracle, finite outputs, V100 Compute Sanitizer memcheck/racecheck zero hazards/errors. The original and packed host buffers must remain valid throughout CUDA Graph replay. Publish only scalar JSON and redacted sanitizer logs.

**Interpretation:** Comparing prepacked QPN-only compute against MMQ-only compute will overstate practical speedups whenever expert streaming is significant. Equal-byte pinned streaming measures a stricter cold-supply bound; no production integration is justified unless the speedup survives matched full-model measurements and model quality qualification. A persistent disk cache can eliminate repeated source conversion between launches, but does not itself reduce PCIe bytes or fix GPU scheduling. Cache validity must be keyed by model ID, original weight hash, layout version, shape and expert permutation and require correct original format for MMQ fallback.

No changes to `main`, runtime model formats, production cache/dispatch, default inference, runner permissions, or private weights.
