# V100 direct streaming from offline-verified QPN cache and async copy/compute experiment

**Successful independent cache-loaded result:** [GitHub Actions 40982469171](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40982469171), source `b33a5f98b5a319f96c379e6a6c7817d0d2b9e6d8`, aggregate artifact **12945497472**. This is a private real-activation 16-expert *sample*, not a complete model inference or verified model quality result.

- Both paths transfer **44,237,056 compressed NVFP4 bytes from persistent page-locked host RAM every forward**; all-MMQ gets 16 original GGUF blobs, hybrid gets 14 original MMQ blobs plus 2 chunk-major, preconverted QPN blobs for the *actual* 187- and 508-row experts (695 of 1,100 routed rows).
- An immutable offline compressed cache was generated from verified source, SHA-256 checked before use, then read and pinned. `disk_prepacked=true`, `cache_load_us=51,723.435` (outside repeated inference), `repack_cpu_us=0`, `repack_upload_us=0`. The original source remains intact for MMQ.
- Six alternating complete-workload pairs, ten repetitions each, timings include equal-byte pinned H2D and all kernels.

| Mode | All-MMQ µs median | Prepacked-QPN+MMQ µs median | MMQ/hybrid speedup |
|---|---:|---:|---:|
| Ordinary single-stream CUDA event | 5,218.815994 | **5,079.705620** | **1.0273855× (+2.7386%)** |
| CUDA Graph replay, captures H2D and compute | 5,183.334351 | **5,056.460762** | **1.0250914× (+2.5091%)** |

Ordinary baseline samples `[5324.387360,5227.519989,5226.496124,5208.780670,5211.135864,5198.950577]` µs; hybrid `[5176.316833,5141.603088,5080.985641,5078.425598,5071.974564,5071.155167]` µs. Graph MMQ `[5182.771301,5177.139282,5182.873535,5185.331345,5186.560059,5183.795166]` µs; hybrid `[5056.307220,5056.204987,5057.126236,5056.614304,5057.024002,5048.320007]` µs. All paired samples favor prepacked QPN. This does **not** isolate achieved PCIe bandwidth or model stage overlap.

**Correctness:** 32 independently dequantized GGUF FP64-oracle output rows; MMQ/oracle L2 0.0112966272, hybrid/oracle 0.0101718218, mixed-vs-MMQ L2 0.011639834. Finite outputs, original CUDA Graph changed-input replay parity and 256-byte output canary passed. Compute Sanitizer: memcheck **0 errors** and racecheck **0 hazards/errors/warnings**. Full model first-logit KL and quality remain unqualified.

## Next test: two-stream copy/compute overlap (not yet measured)

Experimental code `src/kernels/nvfp4_volta_mixed_bench.cpp` commit `c92859e4d8b85d4ad9f96ef136ab3af1591cb415`, opt-in `--qpn-async-pipeline`, test script commit `3b87fbb5b1a14da4cd07319ad79df04ef9bed7b7`, benchmark [Actions 40983676726](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40983676726).

- Each arm has identical 2-slot device staging and pinned host compressed-weight sources. Copy stream prepares iteration `i+1` while compute stream consumes `i` from the other slot.
- `cudaEventRecord(ready)` on copy stream, `cudaStreamWaitEvent(compute, ready)`; compute records `consumed`; copy stream waits for `consumed` before rewriting that slot. Initial staging, all real compressed H2D, and compute are included in a cross-stream GPU-event interval.
- All-MMQ streams exact original compressed 44.24MB for every forward; QPN hybrid streams 38.7MB canonical and 5.53MB cached chunk-major, also 44.24MB total. It computes the same real 16-expert outputs and verifies bitwise equality to the previously accepted serial reference, guards, independent oracle, prior ordinary and Graph checks, and memcheck/racecheck. Uses the same original pinned SHA-verification and immutable disk-cache manifest. No per-forward conversion or pack mutation.
- This is an **across-invocation throughput experiment**, not a model-layer pipeline: the model's next expert routing may not be known yet, transfers can conflict with other GPU work, and the real ring/resident cache needs ownership semantics. CUDA Graph validation continues to refer to the existing single-stream path; the two-stream path is timed and output-checked separately.
- Do not present an async speedup until V100 run completes and all gates pass. Even then, direct multi-layer full-model ABBA with quality and cache/residency handling is required before any experimental model routing change.

**Isolation:** `qpn/prepacked-stream` only; no modifications to `main`, production MMQ fallback, original model/weight pack, GPU expert cache, runner permissions or user host/services. Artifacts contain aggregate metadata, not private weights/activations/logits.
