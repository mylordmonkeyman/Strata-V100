# Performance-first opt-in: skip host staging for already-published unrouted MMQ experts

**Date:** 2026-10-30. **Scope:** rev4.2 §4.5 conditional *earlier host-side supply selection*, not the stopped static packed-resident QPN policy. No different weight encoding, no conversion, and no second VRAM representation of any expert. Original canonical NVFP4-QAD/MMQ is the only device format. Changes require the experimental `STRATA_QPN_PREFILL_EXPERIMENTAL` **build** gate; production/default builds have no new worker callback.

## New performance hypothesis grounded in existing evidence

The earlier MMQ-only selective-supply ABBA (source `d777b85efadb5a359ebbb7ebb68e88e475f20cc9`; aggregate `benchmarks/qpn/results/2026-10-30-selective-mmq-supply-abba.json`) showed that its first 1024-token streamed chunk scheduled **6,027 host Stager jobs** in both selective arms. **2,533 / 2,529** of those were later H2D-skipped (42.03% / 41.96%). At the pinned 2,764,816-byte expert size, this is **7,003,278,928 / 6,992,219,664 bytes** of host preparation that *might* be avoidable (6.52 / 6.51 GiB) if the route is known **before a worker begins that job**. These are opportunity counts, **not measurements of actual CPU work saved, exposed latency, memory bandwidth, or request speedup**.

The earlier selective implementation omitted only H2D: workers still called `memcpy` or `ExpertSource::copy_blobs`; the issuer waited for each staged job even when its device copy was skipped. The 32.1% H2D reduction corresponded to only ~0.71% change in median model prefill, with high variability. Removing host copy/assembly for *already-published unrouted* jobs is a different, separately falsifiable hypothesis, not another QPN kernel sweep.

## Implementation and invariants

- Default-off runtime `STRATA_QPN_MMQ_EARLY_STAGE=1` requires `STRATA_QPN_MMQ_SELECTIVE_SUPPLY=1`, single-device `sm_70`, native NVFP4 W4A8, full 1024 stream-all, disabled QPN dispatch, disabled CPU-share and peer, and the existing opt-in experimental build. Unsupported requests **fail before streaming**. The flag is inert in ordinary builds. Existing DMA-only selective baseline stays unchanged when early-stage is absent/0.
- The asynchronous `Stager` receives a **job-index→StreamEntry** mapping valid for the chunk lifetime. The worker first acquires `routing_ready`; it suppresses an eligible CPU `memcpy` or `copy_blobs` job only if that layer's route was published and the expert was unused. An unpublished layer always uses the original preparation, retaining lookahead and no speculative future-layer knowledge.
- Worker skip still waits for the previous DMA ring-slot completion, publishes `ready[j]`, and relies on the issuer's unchanged `issued_one`, `copied`, `used`, stage-live and ring counters. Batches of `copy_blobs` simply omit unused pairs; the source contract accepts arbitrary independent pairs.
- **Critical race avoidance:** in early-stage mode alone, both inline and threaded issuers wait for the worker's `ready[j]` **before** deciding `copy_blob`. This prevents an issuer's earlier unpublished-copy decision from conflicting with a worker's later published-skip decision. The default pipeline is unchanged. The route flags are ordinary bools, but only read after acquire of the layer's release publication.
- The `Stager` callback is cleared at `finish()` after active jobs retire, before the request-local routing table goes out of scope. For zero-job chunks its counters reset. No new disk/cache source and no GPU buffer/storage conversion is introduced.
- One new, aggregate-only diagnostic line per eligible candidate chunk: `strata qpn early-stage: chunk=... tokens=... cpu_jobs_omitted=... cpu_bytes_omitted=...`. The established selective-supply line still gives `jobs`, `skipped_jobs`, `skipped_bytes` and `copies`. A fail-closed check requires `cpu_jobs_omitted <= skipped_jobs` and `cpu_bytes_omitted <= skipped_bytes`. Do not publish private raw stderr, routes, prompts or logits.
- The new host-only C++ test covers unpublished/used/unused layer decisions, release/acquire publication across threads, and the unsafe interleaving in which the issuer initially sees an unpublished route while the worker later skips. This is **not** a complete model, CUDA, scheduling or sanitizer qualification.

## Performance gate — matched full-model tests only if runner idle

The two experimental arms must use the **same source/binary SHA, canonical checkpoint, prompt, routing policy, stream-all 1024 regime, adaptive cache, actual H2D skipping, CPU affinity/NUMA and workaround flags**. A is existing selective-supply H2D-only, B additionally omits host staging. Run a randomized or A/B/B/A block after warmup with ≥5 timed repeats per arm; include 2K and a representative longer context and accepted-token decode where available. Never turn first-chunk counters into full-request speedup. Record first-logit and generated-output parity, actual published-route CPU jobs/bytes saved, H2D copy counts/bytes (must match across arms, absent nondeterministic routes), host worker time, prefill wall time, full request latency, peak memory and accepted decode throughput.

Pinned baseline environment (both arms):

```bash
STRATA_MMQ_KQ=1 STRATA_PREFILL_MMQ=1 \
STRATA_ARENA_LOCK=0 STRATA_ARENA_PIN_GIB=1 \
STRATA_PREFILL_NVFP4=w4a8 STRATA_PREFILL_CPU_SHARE=0 \
STRATA_QPN_PREFILL_EXPERIMENTAL=0 STRATA_PREFILL_STREAM_MIN=1024 \
STRATA_QPN_MMQ_SELECTIVE_SUPPLY=1 \
STRATA_QPN_MMQ_EARLY_STAGE=0   # A: H2D-only selection
# STRATA_QPN_MMQ_EARLY_STAGE=1 # B: also skip eligible CPU staging
```

The exact CLI must use the existing runner's pinned source/model and private prompt, and must not launch or compete with another GPU job. Preserve `gpu.lock`, the existing noncanceling `strata-qpn-v100` concurrency policy and caches. No automated launch is added by this patch.

**Promotion/stop:** only retain for ongoing performance work if measured *full-request* speedup survives tracing controls, numerical output parity, unchanged memory and decode, and spec ≥5% materiality target. If `cpu_jobs_omitted` is near zero, no real CPU critical-path work is saved, H2D differs unexpectedly, quality fails, or request time regresses: **STOP**, do not run additional random sweeps. Rollback is simply `STRATA_QPN_MMQ_EARLY_STAGE=0` or not compiling the experimental gate. No change to production `main`.

**Status now:** source-only performance candidate with verified host predicate tests. **CUDA / full-model build, any actual CPU-copy omission, and speedup are unmeasured.** R1 packed single-representation feasibility remains UNKNOWN, static packed admission STOPPED, and no dual-format VRAM tests are permitted.


## Corrective first-run finding (2026-10-30, source 9bcc483)

[One-shot run 41167975170](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41167975170) compiled the complete CUDA12.8 binary and passed 17 Python tests plus the C++ publication test. The unchanged selective-H2D baseline process completed with original-MMQ prefill 2,047 tokens, prefill 19,320.7 ms, TTFT 67,104.6 ms, zero executed QPN groups, 15,347 expert stream entries, 6,027 host jobs, 4,939 H2D-skipped experts and 2,536 H2D-skipped staging jobs. **The early-staging candidate exited 1 before the prefill summary**; no speedup, parity or CPU bytes saved can be inferred. This is **execution failure**, not an observed performance regression.

Source review identified a concrete phase-guard defect: with pinned request geometry, stream-all first consumes 1,024 tokens but the 1,023-token trailing chunk uses staged prefill. The new request-wide `STRATA_QPN_MMQ_EARLY_STAGE=1` flag was erroneously rejected on any `!stream_all` chunk. A 72.4-second failing candidate process is consistent with that staged-tail rejection; raw private logs were deliberately not uploaded, so this is a source-derived causal diagnosis, **not direct confirmation from the raw error line**.

Corrective implementation:
- `qpn_mmq_early_stage_phase()` accepts the opt-in for genuinely eligible stream-all chunks, makes it **inactive** on a staged tail only when the same request previously had a stream-all chunk, and still fails closed for an unsupported staged-only request.
- An explicit host test covers 1,024 stream-all followed by a 1,023 staged tail, plus invalid standalone and control-off cases. The prior publication/issuer worker test remains.
- Future failed executions add three allowlisted boolean flags for known static error classes; they do not export private logs, paths, routes or logits.
- The **single corrective one-shot** is triggered by a new, distinct marker in the same existing lock-protected workflow; no generic recurring benchmark trigger or packed-weight alternative is introduced.

The first attempt's [sanitized aggregate](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41167975170/artifacts/13029538770) is preserved. Do not compare its baseline timing to its failed candidate as an A/B result. Only a fully completed, parity-passing corrected run can advance this hypothesis; a measured negative outcome should close it rather than prompt more broad sweeps.
