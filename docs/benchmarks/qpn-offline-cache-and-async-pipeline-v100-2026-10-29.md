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

## Two-stream copy/compute overlap: completed real V100 measurement

**Qualified initial async benchmark:** [GitHub Actions 40983676726](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40983676726), aggregate artifact **12946332290**, experimental source `c92859e4d8b85d4ad9f96ef136ab3af1591cb415` and script `3b87fbb5b1a14da4cd07319ad79df04ef9bed7b7`. Same immutable 16-expert fixture, original source digest `ce605a8b34aa54c0dc266aa38be9560f9d8ecb2f4179111e8bf4575fa82801a7`; 14 MMQ fallback groups, 2 QPN large groups, 1,100 rows, **44,237,056 H2D bytes per forward in each arm**. The offline cache is loaded from disk (`disk_prepacked=true`, `cache_load_us=51,306.790`, `repack_cpu_us=0`, `repack_upload_us=0`); load time is outside the repeated forward.

| Within-run mode | All-MMQ median µs | Hybrid median µs | MMQ/hybrid ratio |
|---|---:|---:|---:|
| Single-stream ordinary, equal H2D | 5,209.548759 | 5,069.209480 | **1.0276846×** (+2.77%) |
| Single-stream CUDA Graph, equal H2D | 5,176.064110 | 5,051.289749 | **1.0247015×** (+2.47%) |
| **Two-stream overlapped copy/compute** | **3,992.753601** | **3,999.492836** | **0.9983150×** (hybrid 0.17% slower) |

Six paired async observations (µs): MMQ `[4003.260803,3996.060944,3990.800095,3991.567993,3993.939209,3990.406418]`; hybrid `[3999.119949,3999.430466,3999.555206,4004.892731,3996.953583,4001.980972]`. Five of six pairs favor MMQ; the ~0.17% difference is small and not established as statistically significant. **Do not claim a QPN throughput win under overlapped weight streaming.** Both arms improve substantially compared with their own single-stream measurements. This is repeated-forward throughput, **not** full-model TTFT, per-expert lookahead, cache residency, or actual multi-layer scheduling.

The two-slot pipeline uses a separate CUDA copy stream, copy-ready and compute-consumed events, and explicit slot-reuse dependencies. Each arm passes **bitwise equality** against its serial reference. Independent sampled GGUF FP64 oracle: MMQ L2 `0.0112966272`, hybrid L2 `0.0101718218`, hybrid-vs-MMQ L2 `0.011639834` over the complete output; 32 oracle rows, finite outputs, original single-stream changed-input Graph parity and 256-byte guard pass. Initial Compute Sanitizer logs show **memcheck 0 errors; racecheck 0 hazards/warnings**.

**Slot-reuse safety gate now verified:** [run 40984983931](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40984983931), source `d702ee7026a1c0f5c6cd48e1aa8db65d0ec0e12b`, aggregate artifact **12946746665**. Both sanitizer result JSONs explicitly report `iterations=3`, `async_pipeline=true`, `pipeline_output_equal=true`, finite/oracle and changed-input Graph gates passing. Published logs: memcheck **0 errors**; racecheck **0 hazards (0 errors, 0 warnings)**. This closes the two-slot reuse gap for the standalone repeated-forward sample, not engine integration. The rerun's async MMQ/hybrid ratio **1.0003735×** again indicates practical parity, not a material QPN speed gain.

**Next decision:** Prioritize a model-representative *per-expert* lookahead/residency diagnostic against Strata's existing prefill staging/ring protocol, preserving MMQ fallback and original packs. **Important existing behavior:** `src/prefill/prefill.cpp` already walks the routed expert `order`, skips resident experts in `stage_one`, issues individual pinned/unpinned H2D copies on `m.copy`, records `m.copied[slot]`, and guards slot reuse via `m.used`; grouped MMQ gathers can wait only for the last copy in an ordered group. Therefore do **not** assume production transfers all experts per layer or claim generic 'selective transfer' as a new optimization. First measure actual per-layer routed/transferred bytes, cache hits, warm/cold throughput, copy/compute stalls, and whether alternate packed-QPN supply would preserve the current ring ownership and grouping. Only advance to opt-in full-model matched ABBA after a material, repeatable improvement under realistic supply and the multi-iteration safety gates. Full-model first-logit KL/quality remain unqualified.

**Isolation:** `qpn/prepacked-stream` only; no modifications to `main`, production MMQ fallback, original model/weight pack, GPU expert cache, runner permissions or user host/services. Artifacts contain aggregate metadata, not private weights/activations/logits.
