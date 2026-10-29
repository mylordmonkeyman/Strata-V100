# Strata-V100 fork: NVFP4 Volta Tensor-Core Acceleration — Implementation Specification (rev 3)

**Status:** Proposed; not implemented or benchmarked
**Revision date:** October 28, 2026
**Fork of:** `jmnargi/Strata-V100` (fork name TBD, e.g. `<you>/Strata-V100-QPN`)
**Target hardware:** NVIDIA Tesla V100 (`sm_70`, CUDA 12.8); primary rig is a single V100-PCIe-32GB, 128 GB RAM, dual Xeon E5-2697A v4 (AVX2, no AVX-512)
**Model:** `nvidia/Qwen3.8-Flash-Next-FP4-Local`, converted through Strata's existing `NVFP4-QAD` pipeline (unchanged)
**Baseline Strata commit:** `2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926` (`main`)
**Reference ninfer commit:** `b37d0dd3e1163b9d802d8bccfa89918bf68d793e` (`master`)
**Feature name:** `STRATA_NVFP4_VOLTA_QPN`

> **Revision notes (rev 3).** This revision accepts the stronger rev-2 scope (independent decode/prefill tracks), headroom audit, explicit device-side dispatch, cache DDR, fork management, and reversible milestones. Additional source-checked corrections:
> 1. **Correct NVFP4 prefill comparators:** `moe_fp16tc.cu` is Q2_0-only; the existing `gemm_iq_f16_grouped` path does not accept ggml NVFP4 type 40. Compare the **actual** NVFP4 W4A8/MMQ and dequantized-FP16 paths, not an ineligible WMMA baseline (§2.1, §5, M4).
> 2. Distinguish Strata's **interleaved GGUF 36-byte block** from ninfer's separate code/scale QPN planes; matrix-shape eligibility alone is not executable compatibility (§2.2, §3).
> 3. Require **mutually exclusive per-group GPU dispatch**; a general fallback launch must not recompute or overwrite QPN-completed entries (§3.2, §3.4).
> 4. Make activation sources (live FP32 vs. reconstructed Q8_1) a **measured numerical/performance choice** and mandate intermediate lifetime planning (§3.3).
> 5. Constrain split-K to each projection's actual K-domain (especially down K=640), and separately measure T=1 instead of presuming an outcome (§3.4).
> 6. Keep **independent decode and prefill stop/go decisions**. A failed kernel or Amdahl gate cannot cancel an unrelated track; no automatic whole-project cancellation from one test (§6, §8).
> 7. Strengthen packed-cache ownership, generation/event validation, CI matrix and benchmark consistency (§4, §7–§10).
>
> **Rev-2 changes retained for continuity:**
> 1. Goals now state **both decode and prefill** as first-class targets, with separate success metrics; prefill benchmarks are independently gated, though integration is sequenced after decode direct-mode (§1.2).
> 2. Added a **roofline/headroom audit in M0**; rev 3 makes stop/go decisions independent by track and requires representative workloads (§6, M0).
> 3. Added **fork management**: branch model, upstream sync, license/attribution checklist, CI and repo hygiene (§9).
> 4. Clarified "non-goals" wording (no *re-quantization*, but lossless *re-layout* is allowed), removed the ambiguous `sm_75 INT8` phrase, and fixed internal contradictions (default-off vs. `auto`, who owns the activation staging decision).
> 5. Turned the open "Activation-input rule" into a **concrete decision procedure with an M0 audit task**.
> 6. Turned the packed-slot "hard gate" into a **required design decision record** before any cache code is written.
> 7. Reorganised PRs so each is independently mergeable/revertable, and every PR has an explicit definition of done.
> 8. Identified facts that still require *artifact- or hardware-level* verification in M0; source assertions independently checked against pinned code are now identified as such.

---

## 1. Objective, scope, and decision rules

### 1.1 Objective

Add an **optional, `sm_70`-only** execution path for Strata's NVFP4 routed experts, using the software E2M1/UE4M3 decoder and FP16 tensor-core (`mma.m8n8k4`, FP32 accumulate) scheduling principles from `geoffwatts/ninfer-v100`, to raise **tokens/s in decode (incl. MTP verify windows) and prompt prefill** on GPU-resident experts, without:

- modifying the FP4-Local checkpoint or the `NVFP4-QAD` conversion output,
- altering expert routing, attention, GDN, PLE, or the MTP draft model,
- breaking Strata's CPU / PCIe / VRAM memory tiers or the adaptive expert cache,
- changing any other quantization path or non-Volta build.

### 1.2 Success criteria and honest expectations

| Track | Primary metric | Secondary metric |
|---|---|---|
| **Decode** (T=1, verify windows 2–8 entries, `--spec` modes) | End-to-end decode tok/s and per-request latency at fixed acceptance | Isolated resident-expert kernel time per T bucket |
| **Prefill** | Time-to-first-token at 2K/8K/32K prompts, separated into expert compute and supply/waits | Per-expert-batch time for 1–32 rows; crossover against **actual NVFP4 MMQ or dequantize→FP16** paths for larger batches |

A faster isolated kernel alone is **not sufficient** to enable anything by default.

Expectations to carry into planning (all to be confirmed by M0 measurement, not assumed):

- **Decode is likely partly memory/transfer-bound.** At small T each expert contains 2.76 MB of stored weights, and a full expert execution must access its three projections (subject to cache/hardware behavior). If the existing `dp4a` kernel already runs near V100 HBM2 bandwidth for resident experts, or if CPU-pool/PCIe waits dominate step time, tensor cores cannot help much (Amdahl). The M0 headroom gate exists to detect this early.
- **QPN is a skinny-GEMM design** (8 tokens × 32 rows per tile, T ≤ 32). It targets decode/verify and *small per-expert batches in prefill*. For **large per-expert prefill batches** (hundreds of rows), Strata's existing **NVFP4 W4A8 MMQ and dequantize→FP16 GEMM** paths are likely better-shaped; `moe_fp16tc.cu` is **Q2_0-specific and is not an NVFP4 baseline**. The QPN prefill track therefore focuses on the small-batch tail (where MMQ tiles are underfilled) and on measuring a crossover point, not on replacing the large-batch path.
- Prefill for NVFP4 on this stack is likely more constrained by H2D/CPU/SSD expert supply than by resident kernel math. Measure before optimizing.

### 1.3 Decision rule

Staged, measurement-gated, reversible:

1. Port a **checkpoint-layout-compatible (`qpn-direct`)** kernel that reads Strata's existing 36-byte GGUF blocks, so no cache mutation is needed.
2. Benchmark against **the existing NVFP4 `dp4a` grouped decode** and, for prefill, **NVFP4 W4A8 MMQ and a correctly scaled dequantize→FP16 GEMM reference**; do not treat Q2_0-specific FP16-TC paths as target-model baselines.
3. Add **load-time/fill-time repacking (`qpn-packed`)** only if direct mode shows a significant address/permute cost, measured amortization favors packed execution, and the cache-lifecycle design (§4.2) is approved. The work on prefill and decode is independently gated.
4. Runtime-dispatched, opt-in, with automatic fail-closed fallback. `off` is the default.

### 1.4 Non-goals

- No re-quantization, retraining, or change to NVIDIA's FP4-Local weights. (Lossless bit-level **re-layout** of codes/scales is permitted in packed mode.)
- No port of ninfer's dense-model graph, fixed MLP dimensions, scheduler, tokenizer, KV, speculation engine, or `.ninfer` artifact format.
- No replacement of the CPU NVFP4 expert implementation, GPU expert policy, adaptive cache algorithm, PCIe transport, routing, attention, GDN, PLE, or MTP draft model.
- No native FP4 tensor-core instructions (V100 has none); all accelerated MMA is **FP16 with FP32 accumulation**.
- No performance claim until hardware benchmarks establish it.

---

## 2. Verified existing interfaces and constraints

Items marked **[VERIFY]** require empirical confirmation using the *local converted artifact*, specific third-party ggml checkout, or hardware. The source interfaces below were cross-checked against the pinned repository revisions; verify the local tree has not drifted before implementing.

### 2.1 Strata's NVFP4 implementation

- Strata already supports this checkpoint under `NVFP4-QAD` via `tools/nvfp4_local_pack.py`, producing `experts.bin`, dense GGUF, PLE GGUF, and the draft model. **Do not change that pipeline's output.**
- A resident NVFP4 expert blob is `[gate | up | down | tail]`, tail = `{s_gate, s_up, s_down, 0}` (four FP32). Matrices use ggml `GGML_TYPE_NVFP4` blocks: **64 elements per 36 bytes** (32 E2M1-code bytes + four per-16 UE4M3 scale bytes). Nibble ordering is GGUF-block-specific and must be decoded explicitly, never assumed sequential. **[VERIFY the exact ggml split-half nibble mapping and type value against the local pinned ggml checkout and a golden real-expert fixture.]**
- Routed expert shape: `n_embd=2560`, `n_ff=640`; gate and up each `[640, 2560]`, down `[2560, 640]`. **[VERIFY the converted artifact and checkpoint config identify these shapes in every routed layer.]**
- `src/kernels/cuda/iq_kernels.cu` uses `vec_dot_nvfp4_q8_1`, `native_gu[_multi]_kernel`, `native_down[_multi]_kernel`, and `native_expert_grouped` with `NativeExpertLayout` (`include/strata/kernels/iq_kernels.hpp`). The current kernel decodes E2M1/UE4M3 in software and uses Q8_1 activations with `dp4a`. Gate and up are scaled **after** their dot products; down is scaled before MoE combination.
- The grouped interface takes device-side expert pointers, `grp_start`, `n_groups`, `ent_tok`, `ent_dst`, and `cap_groups`/`cap_entries` for a verify window. **Group and entry counts are GPU data**; launches must not require CPU readback in the hot path.
- Prompt path: `src/prefill/moe_mmq.cu` provides NVFP4 **W4A8** MMQ (actual eligibility depends on `mmq::fits` on sm_70) and the existing fallback dequantize→FP16 path. `src/prefill/moe_fp16tc.cu` and `STRATA_PREFILL_FP16TC` implement compact **Q2_0** experts, **not** NVFP4. In `src/kernels/cuda/iq_kernels.cu`, `gemm_iq_f16_grouped` selects Q2_0/IQ-specific decoders and rejects NVFP4 type 40. Accordingly, `STRATA_PF_WMMA=1` is **not a valid assumed NVFP4 comparator** and may select an unsupported dispatch in some paths. Preserve these other-format paths, and guard new NVFP4 prefill selection explicitly instead of changing them as a side effect.
- `ExpertCache` (`include/strata/core/expert_cache.hpp`, `src/core/expert_cache.cpp`) supports stable device slot pointers, async/queued H2D fills, segmented/elastic VRAM, replacements, per-layer admission, and lending/promotions. **Every write pathway must respect any QPN layout state.**

### 2.2 ninfer's transferable Volta implementation

- Generic QPN implementation: `src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh` plus launch `.cu`. Provides E2M1 → `half2` via shifts/rebias, E4M3 scale decode, FP16 MMA (`volta_mma_qp_n`), FP32 accumulators, split-K reduction, tiles of 8 tokens × 32 rows. Host-side support in the pinned generic launcher: `N % 128 == 0`, `K % 64 == 0`, `K/64 >= 8`, `1 <= T <= 32` (subject to per-kernel scheduling and scratch limitations).
- Geometric eligibility of Flash-Next projections (checked): gate/up N=640 (or fused N=1280), K=2560 → K/64 = 40; down N=2560, K=640 → K/64 = 10. All satisfy the **shape** guards, but the GGUF data layout and grouped launch protocol still need adaptation. The down projection has only ten 64-value K-chunks, so `SPLITK=16` must not be selected blindly.
- ninfer has **two layouts**: a checkpoint-native one (its own, *not* Strata's GGUF layout) that permutes activation lane order, and an in-place **prepacked** one in a Volta-fragment-compatible layout. Strata's 36-byte interleaved blocks match **neither**: a direct path must provide explicit GGUF address/nibble/scale extraction into the expected FP16 fragments; a packed path must losslessly rearrange codes *and* scales. The pretrained values and the FP32 `s_*` tail remain canonical. The reference's native/prepacked paths are **not drop-in kernels**.
- ninfer's fused SwiGLU kernel hardcodes a 17,408-wide dense intermediate: **do not copy it**. Its small-T dispatch uses two QPN projection passes into FP32 scratch plus an FP32 SwiGLU because the fused kernel was slower. Adopt that **split-pass pattern** first, with a 640-wide intermediate.
- ninfer's optimized single-token SIMT path is distinct from QPN. Do not assume QPN wins at T=1; benchmark and dispatch by group size.

### 2.3 Verified comparison limitation: NVFP4 prefill is not Q2_0 prefill

The pinned Strata `include/strata/prefill/moe_fp16tc.hpp` declares native **Q2_0** expert geometry and a `geom_ok` contract requiring ggml type 42; the optimized compact path is ineligible for FP4-Local type 40. Independently, `gemm_iq_f16_grouped` in `src/kernels/cuda/iq_kernels.cu` recognizes the special Q2_0 and IQ-family decoders, not NVFP4. Thus the rev-2 statements that compared against the "existing FP16-TC NVFP4" path are **incorrect**. The project may reuse those kernels' grouping and buffer-management ideas, but must introduce an NVFP4-specific decoder and backend to run QPN. M0 must record what **the actual FP4-Local configuration** executes on `sm_70` (`mmq::fits` may reject a tile), including each fallback.

### 2.4 Fixed per-expert size (memory planning)

| Component | Calculation | Bytes |
|---|---:|---:|
| Gate | `640*2560*36/64` | 921,600 |
| Up | `640*2560*36/64` | 921,600 |
| Down | `2560*640*36/64` | 921,600 |
| FP32 tail | `4*4` | 16 |
| **Entire expert** | | **2,764,816** |

Codes and scales can be rearranged losslessly at bit level with no growth: per matrix, codes `N*K/2` bytes + scales `N*K/16` bytes. Per-expert total stays 2,764,816 bytes before slot alignment/guards. In-place conversion is **not** assumed safe; temporary buffers are budgeted separately.

---

## 3. Proposed architecture

### 3.1 Backend and dispatch

A CUDA-only `sm_70` `Nvfp4VoltaBackend` with two optional layouts:

1. **`qpn-direct` (first):** read Strata's existing GGUF block format in a new kernel adapter, decode to FP16 fragments, run `mma.m8n8k4` with ninfer-style QPN scheduling. No slot rewrite; trivial rollback; activation-staging buffer allowed.
2. **`qpn-packed` (conditional, later):** rearrange codes/scales into a QPN layout when a slot is filled, and run a prepacked kernel variant. Original host/SSD representation is preserved. The runtime must track which layout each slot holds.

Environment interface (all default off/0):

```text
STRATA_NVFP4_VOLTA_QPN=off|direct|packed|auto   # default off
STRATA_NVFP4_VOLTA_QPN_MIN_T=2                  # provisional; T=1 measured but default fallback
STRATA_NVFP4_VOLTA_QPN_MAX_T=32                 # provisional upper bound for decode path
STRATA_NVFP4_VOLTA_QPN_PREFILL=0|1              # default 0; enables prefill small-batch route
STRATA_NVFP4_VOLTA_QPN_TRACE=0|1                # aggregate counters only
```

`auto` = device-/shape-/group-size-/layout-aware dispatch using only pre-qualified schedules from a checked-in, provenance-stamped tuning table; it is **never** unconditional QPN and is not a default until M6. Invalid values fail with a clear configuration error at startup. `STRATA_EXP_MODE`, `STRATA_SM70_TABLE`, `STRATA_PF_WMMA`, `STRATA_PREFILL_NVFP4` keep their current semantics; with the new feature off, behavior and performance must be unchanged.

### 3.2 Internal interfaces (illustrative)

New files:

```text
include/strata/kernels/nvfp4_volta_qpn.hpp
src/kernels/cuda/nvfp4_volta_qpn_decode.cu
src/kernels/cuda/nvfp4_volta_qpn_core.cuh
src/kernels/cuda/nvfp4_volta_qpn_pack.cu        # packed mode only
src/kernels/nvfp4_volta_qpn_parity.cpp
src/kernels/nvfp4_volta_qpn_bench.cpp
```

```cpp
enum class Nvfp4VoltaMode { Off, Direct, Packed, Auto };
enum class Nvfp4DeviceLayout { GgufInterleaved, VoltaQpn }; // packed only after DDR-1

struct Nvfp4ExpertView {
    const uint8_t* base;      // stable resident slot address
    Nvfp4DeviceLayout layout;
    int n_embd;               // 2560
    int n_ff;                 // 640
    size_t bytes;             // canonical slot blob size
};

bool nvfp4_volta_qpn_supported(int cc, int n_embd, int n_ff,
                               int ggml_type, int entries, Nvfp4DeviceLayout layout);

// Launches on caller-owned stream; no dynamic allocation or synchronization.
void nvfp4_volta_qpn_grouped(const NativeExpertLayout& L,
     const unsigned long long* group_ptr, const int32_t* group_start,
     const int32_t* n_groups, const int32_t* ent_dst,
     const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
     const float* x_f32, const void* x_q8_1, // negotiated activation source (not both assumed live)
     void* scratch, float* output, void* stream);
```

Adapt to Strata's actual buffers and scratch planner. **No allocation inside a captured CUDA graph.** Wrap at the `native_expert_grouped` dispatch boundary rather than rewriting generic IQ code.

**GPU-side selection and exclusive ownership requirement.** Group sizes (`grp_start[g+1]-grp_start[g]`) are device data. A tiny optional device classification kernel may build bounded `QPN_GROUP` / `LEGACY_GROUP` lists, or guarded grids may use one **shared deterministic predicate**. Every eligible group must be assigned to **exactly one** backend; every ineligible group to the legacy backend; empty groups to neither. Calling the unmodified `native_expert_grouped` across all groups **in addition** to QPN is prohibited (duplicate writes/races). If using masked legacy kernels, provide a modified internal variant consuming the same owner mask or a compact list. Preserve `grid_groups` handling and all `ent_tok`/`ent_dst` mappings. Group metadata, masks, or lists must live in graph-stable caller-owned scratch; populate them on GPU each replay, bounds-check `cap_groups`/`cap_entries`, and add no per-layer host synchronization. **PR 3 design record:** queue vs. guarded grids, exclusive-coverage proof, and launch/classification overhead.

### 3.3 Numerics and execution semantics

Contract per expert and token:

```text
g = s_gate * dot(W_gate, x)
u = s_up   * dot(W_up,   x)
h = silu(g) * u
y = s_down * dot(W_down, h)
```

- All `s_*` are FP32 scalars from the 16-byte tail. Keep gate/up in **FP32** until after scales and SwiGLU. Do not round gate/up to BF16 or fold `s_down` into `up` (Strata documents underflow/multi-percent error with that fold).
- FP16 weight decode must handle all 16 E2M1 values and UE4M3 zero/subnormal/normal/max scales over the checkpoint's actual scale range without overflow/underflow to non-finite results. Derive and validate any `rebias16384`, `*256`, or `weight_scale_divisor` algebra against **Strata's** `GGML_TYPE_NVFP4` + global-scale convention; ninfer's divisor is not substitutable for `s_gate/s_up/s_down`. Where the FP16 range is too tight, apply a documented power-of-two pre-scale and compensate in FP32.
- Strata's current GPU path uses Q8_1 activations; the new path uses FP16 activations with FP32 accumulate. Differences vs. `dp4a` are expected even with bit-identical weights. **Never assert bitwise parity with `dp4a`**; validate against an FP64/FP32 mathematical reference and measure model-level change separately.

**Activation-input decision procedure** (M0 audit task A-1):

1. Trace where normalized FP32 `x` is produced and when `x_q8_1` replaces it in the decode, verify, and prefill paths.
2. Build two standalone variants with the same reference weights: **(A) FP32 → FP16** if unquantized `x` can be retained (account for producer lifetime, conversion, and scratch) and **(B) Q8_1 → FP16** by explicitly decoding the existing q8_1 blocks (account for scales/sums and reconstruction error). No bit reinterpretation.
3. Measure *both* numerical error vs. FP64 and full fused-expert throughput for actual T histograms. The selected path must minimize **end-to-end** cost, not merely operand precision loss; do not require a new FP32 lifetime if it eliminates the speedup.
4. Repeat the same choice at the SwiGLU hidden (640-wide) going into down. Apply `s_gate`/`s_up` to FP32 gate/up accumulations before SwiGLU, then convert/reconstruct hidden once. Keep `s_down` on the down output.
5. Document per-stage activation source, stride, aliasing, device lifetime, alignment, and exact graph scratch/arena byte counts before PR 3. Do not assume `block_q8_1*` is FP16, or that FP32 buffers survive existing graph nodes.

### 3.4 Grouped-expert scheduling

A Flash-Next verify window routes up to ten experts per token **[VERIFY top-k]**; a resident expert usually gets far fewer entries than the window length. Microbenchmarks must use **measured per-expert group counts** (from M0's histogram), not T=8 for every expert. Buckets: `T=1, 2, 3, 4, 5–8, 9–16, 17–32`; `T>32` and unusual shapes fall back initially.

Staged optimization, split passes first:

- **P0:** `T=1` defaults to Strata SIMT/`dp4a`, but must be *benchmarked* with QPN (including staging) and may become a qualified opt-in only if it wins.
- **P1:** `T=2..8` QPN gate, QPN up, FP32 post-scale/SwiGLU, QPN down. Sweep split-K and accumulator chains *per projection*: gate/up K=2560 has 40 64-element chunks; down K=640 has only 10. Benchmark `SPLITK ∈ {4,8}` for down and `{4,8,16}` for gate/up **only where the kernel schedule assigns nonempty useful work to every participating warp**; `NACC ∈ {1,2}` subject to `ptxas` spills and shared-memory occupancy. Never inherit ninfer's dense-model launch settings without adapting them.
- **P2:** fused gate+up only if a register-pressure sweep beats the split path (ninfer found fusion can be slower).
- **P3:** `T=9..32` and small-batch prefill blocks; keep the existing prompt kernels for large batches.

Preserve `ent_dst` ordering and the weighted MoE combine. Handle duplicate expert IDs, empty groups, stale/invalid counts, groups spanning `gridDim.y` strides, and **exactly-once assignment** of each routed entry to a backend. Include duplicate-destination detection and verify the existing MoE combine observes the expected (unchanged) outputs.

---

## 4. Cache and layout plan: the critical correctness boundary

### 4.1 First deliverable: no repacking

The first prototype **must not mutate GPU cache slots**. Adapt the decoder to the interleaved 36-byte blocks and GGUF nibble order. If direct decode loses badly, test a bounded per-launch **deinterleave of only the active resident experts** into scratch, with that copy/convert time fully accounted for. Compare direct QPN against the **resident NVFP4 `dp4a` path**, plus a **correctly scaled dequantize→FP16 GEMM** reference. Do not include Q2_0-only FP16-TC as an NVFP4 competitor.

### 4.2 Conditional packed mode

**Prerequisite — Design Decision Record (DDR-1), required before any cache code:** choose and justify one consumer-compatibility strategy, with measured full cost:

- (a) QPN-aware implementations of **every relevant** packed-slot consumer (T=1 SIMT, `dp4a` grouped, NVFP4 MMQ or dequant→FP16 prefill, `verify_slot`, and any cache-maintenance/debug readers), or
- (b) reversible back-conversion ordered before any canonical fallback, or
- (c) a budgeted **packed sidecar** with the canonical slot untouched (preferred for correctness only if the near-double per-expert residency cost does not erase the throughput gain).

If none is cost-effective, **stay with `qpn-direct` and delete packed mode.**

Per-slot state machine:

```text
EMPTY -> COPYING_CANONICAL -> CANONICAL_READY
      -> PACKING_QPN -> QPN_READY -> IN_USE
EVICT / REPLACE / LEND / SHRINK -> invalidate layout readiness before reuse
       -> increment slot generation; wait for outstanding consumers before overwrite
```

Rules:

- A packed slot may **never** be passed to canonical `dp4a`/MMQ/dequantization kernels. In-place packed mode must restore canonical bytes **before** any canonical decode/prefill fallback or provide genuinely QPN-aware equivalents; otherwise prohibit mixed layouts. A canonical sidecar must maintain two explicit pointers and a bounded ownership map, not silently substitute the sidecar for `device_slot`.
- Every mutation path (queued copy, adaptive swap, same-layer replacement, peer transfer, D2D rotation, prompt lending/refill, `--vram-elastic` resize, new slot generation) must set or invalidate the layout descriptor. A memcpy of a packed slot is packed, not canonical; D2D copies propagate a **generation-tagged layout descriptor** only when both source and destination layouts/ordering are known, or re-pack from canonical. Do not rely on host-side descriptors alone for graph-replayed pointer tables.
- `verify_slot` currently compares canonical bytes and will fail on packed slots; add a **canonical-only pre-pack verification** or a decoded/reference comparison. Do not silently weaken validation.
- **Race rule:** a slot may not be exposed to a CUDA graph as QPN-ready before its fill/repack event is visible to the compute stream. Use existing stream/event ordering, per-slot monotonic generations and consumer-completion fences; no per-layer `cudaDeviceSynchronize()`. A graph that captures **stable pointers** must read refreshed, device-visible generation/layout metadata each replay (or the host must recapture safely). Do not overwrite, lend, rotate, or unmap a slot while a graph/work queue references its previous generation. In-place repacking preserves addresses but **changes semantics**; a sidecar changes both address and semantics.
- **Memory rule:** measure peak conversion scratch, staging, graph memory, and slot count. For in-place packed mode, preserve the 2,764,816-byte payload and guards/alignment unless proven otherwise; for sidecar mode, budget **additional approximately 2.76 MB per cached expert** and account for the resulting loss of resident slots; do not assume `open_sized` slots or elastic mappings tolerate an oversized blob. Report any reduced cache occupancy in end-to-end results.

### 4.3 CPU and non-resident experts

The host pool keeps canonical NVFP4 bytes and the AVX2 ggml-compatible implementation. **No AVX-512 requirement** may be introduced (target Xeons lack it). PCIe misses behave exactly as before. No CPU execution may touch a GPU-packed blob. NUMA placement stays under existing runtime settings.

---

## 5. Source-file work breakdown

| Change | Files | Scope |
|---|---|---|
| Kernel core | **new** `src/kernels/cuda/nvfp4_volta_qpn_core.cuh`, `..._decode.cu` | MMA wrapper, FP4/scale decode, direct GGUF adapter, split-K and output policies |
| Public/kernel dispatch | **new** `include/strata/kernels/nvfp4_volta_qpn.hpp`; **edit** `include/strata/kernels/iq_kernels.hpp`, `src/kernels/cuda/iq_kernels.cu` | Feature-gated grouped path; existing call remains fallback |
| Activation/staging | `src/program/generate.cpp`, `src/prefill/prefill.cpp`, scratch planner **as required** | Implements the §3.3 decision procedure; graph-compatible scratch |
| Cache integration (packed only) | `include/strata/core/expert_cache.hpp`, `src/core/expert_cache.cpp`; transfer call sites | Layout/epoch state, conversion on fill, verified invalidation |
| GPU conversion (packed only) | **new** `src/kernels/cuda/nvfp4_volta_qpn_pack.cu` | Lossless GGUF → QPN reordering |
| Prefill track | `src/prefill/moe_mmq.cu`, `src/prefill/prefill.cpp`, and a **new** NVFP4 QPN prefill launcher | Small-batch route + measured crossover. `moe_fp16tc.cu` remains unchanged/Q2_0-only; do not route NVFP4 through it (M4). |
| Build/config | `CMakeLists.txt`, CUDA build files, service env docs | `sm_70` only; CUDA 12.8; CMake option; explicit fallback; ensure no new kernels load/consume VRAM with feature off |
| Tests | **new** `src/kernels/nvfp4_volta_qpn_parity.cpp`, `..._bench.cpp`; extend `nvfp4_expert_gpu_parity.cpp` and grouped/verify parity suites | Decode, scale, MMA, cache, mixed residency, graph, end-to-end |
| Docs | `docs/NVFP4.md`, `docs/NVIDIA_V100.md`, `benchmarks/` report | Reproduction, numerics, results, rollback, limitations |

Do **not** copy dense-model sizes or model-specific dispatch from ninfer. Keep the backend small and reuse Strata's host orchestration. Preserve CUDA/HIP/SYCL builds via compile-time guards.

---

### 5.1 Kernel execution ABI and bounded scratch requirements

Before implementing the first CUDA kernel, fix the following contracts in a small design record:

- **Canonical expert pointer/format:** `NativeExpertLayout` and `grp_ptr[]` continue to point at canonical `[gate|up|down|tail]` experts in `direct` mode; type and dimensions are checked at startup. Output is the **same FP32 `out[ent_dst[e]*n_embd + row]` contract** as the existing grouped kernel, without multiplying routing weights twice.
- **Input ownership:** caller passes a checked FP32 or Q8_1 view (possibly neither available for QPN, in which case use legacy); no hidden allocation or host sync. QPN gate/up writes FP32 scratch for exactly the assigned entries, applies tail scales and SwiGLU, then converts the hidden once for down.
- **Scratch capacity:** plan for `cap_entries`, not an assumed runtime group count. Bound classification masks/list, FP16 activations, two FP32 640-wide gate/up planes (or an equally precise streaming fusion), hidden representation, and split-K reduction; provide checked byte-count functions with overflow guards. Graph capture must reuse stable addresses.
- **Dispatch:** QPN and fallback kernels receive identical immutable metadata and **disjoint group ownership**. Shared resident-pointer lifetime and completion events are unaffected when `off`.
- **Large prefill:** a prefill launcher uses the existing sorted/bounded host-plus-device grouping plan where available, not the decode `n_groups` contract by assumption. Do not replace the streamed expert supply path or its ring/promotion event ordering.

---

## 6. Milestones and go/no-go gates

### M0 — Baseline capture, headroom analysis, mechanical compatibility

1. Pin both commits, CUDA compiler, driver, and model checksum. Build an unmodified `sm_70` binary and run NVFP4-QAD end-to-end.
2. Record: resident expert counts, cache hit/miss rates, **per-group entry-count histogram** (decode, verify, prefill), GPU expert time, CPU pool time, PCIe wait, prompt/decode throughput.
3. **Headroom analysis:** measure resident-only, mixed-tier, and transfer-heavy workload fractions separately. Calculate `1/(1 - f*(1 - 1/s))` for speedup assumptions `s=1.25, 1.5, 2` using the *non-overlapped, end-to-end critical-path fraction* attributable to accelerated kernels, **not** total GPU activity. Separately report NVFP4 weight bytes, achieved HBM traffic/bandwidth vs. measured practical V100 bandwidth, instruction/launch cost, and PCIe/CPU waits. Account explicitly for overlap: isolated CUDA kernel savings do not add linearly to a window with CPU/transfer stalls.
4. Add a synthetic `GGML_TYPE_NVFP4` block fixture and a real FP4-Local expert blob (not committed) to the local harness; confirm size, scale convention, and nibble order. Complete audit task A-1 (activation input) and re-verify every **[VERIFY]** item above.
5. Confirm model identity and license terms for `nvidia/Qwen3.8-Flash-Next-FP4-Local` and that no model bytes enter the repo.

**Gate:** reproducible NVFP4-QAD baseline and no stale assumptions. **Track-specific decision:** where the measured critical-path ceiling for a 2× kernel improvement is <3%, deprioritize that *track* after documenting its scope and measurement uncertainty. Do **not** stop the independent prefill or decode track because the other fails, and do **not** stop the project on one synthetic or cache-state-specific estimate. Only terminate both tracks when representative workloads establish negligible headroom in both.

### M1 — Standalone direct-GGUF QPN proof of concept

1. Port only `volta_mma_qp_n` and the E2M1/E4M3 decode/scheduling primitives, with attribution (§9).
2. Adapt GGUF blocks to correct `half2` fragments; implement gate/up/down shapes, FP32 per-projection scale application, and FP32 `silu`. Support T = 1, 2, 3, 4, 8, 16, 32.
3. Benchmark against **NVFP4 `dp4a`**, a **dequantize→FP16 GEMM** reference (including `s_*` tails), and NVFP4 MMQ for prefill where executable. The existing `gemm_iq_f16_grouped` and compact `fp16tc` are not NVFP4 implementations; use them only as *architectural references*, not runnable NVFP4 A/B arms.
4. Profile: `ptxas -v` (registers/spills), Nsight Compute if available (occupancy, tensor-pipe use, HBM bytes), launch overhead, sync.

**Gate:** no wrong-scale paths or non-finite outputs. A realistic T≥2 bucket with ≥10% isolated completed-expert improvement is the preferred qualification target. If it fails, record why; still complete any already scoped, independent prefill microbenchmark and low-cost alternatives (e.g. activation staging or split-K) before closing the *kernel candidate*. Never treat a preliminary single-shape failure as evidence that every track lacks headroom.

### M2 — Standalone numerical qualification

Complete §7 unit-level checks (decoder, scales, expert-level parity at all T buckets, sanitizer runs) *before* touching the engine. **Gate:** all unit-level thresholds met.

### M3 — Opt-in runtime integration, direct mode (decode/verify)

1. Wire through `native_expert_grouped`, preserving device-side counts and caller scratch; keep T=1 and unsupported-size fallback.
2. Verify correctness across resident/CPU-miss splits, MTP verify windows of 1–8, multi-turn prefixes, and CUDA Graph capture/replay.
3. Measure the **combined** GPU-resident path (gate/up, SwiGLU, staging, down, scatter) against current grouped kernels.

**Gate:** no end-to-end correctness or memory regression, and a positive repeatable latency improvement on a representative workload. If the gain is local but invisible end-to-end, keep it as an experiment, not a default.

### M4 — Small-batch prefill track

Test QPN on per-expert batches of 1–32 rows against the **actual NVFP4 W4A8 MMQ, fallback dequantize→FP16 GEMM, and end-to-end Strata prefill route**; **do not** select `STRATA_PF_WMMA=1` or compact `STRATA_PREFILL_FP16TC` as comparators for ggml type 40; produce a **crossover table** (rows per expert → best backend). Keep the existing eligible NVFP4 MMQ or dequant→FP16 path for large batches. Enable only shapes with net prompt-time gain, excluding results dominated by gather, SSD/CPU reads, or repacking. **Gate:** measurable, repeatable TTFT improvement at ≥1 of 2K/8K/32K prompts without material regression elsewhere; otherwise leave `QPN_PREFILL=0`. Record whether each prompt's actual runtime is MMQ, fallback FP16, or new QPN, and include transfer/supply waits. Prefill does not depend on the decode M3 gate succeeding.

### M5 — Optional packed QPN slot layout

Requires approved **DDR-1** (§4.2). Implement lossless per-slot rearrangement, explicit layout state, and stream ordering; run every cache mutation path through the documented state machine. Compare direct vs. packed including amortized pack cost and any residency loss; test prompt lending, queued fills, adaptive replacement, VMM shrink/grow, and mixed layouts with sanitizers/guard tests. **Gate:** packed beats direct end-to-end **including pack amortization and any loss of GPU residency** on target workloads, with verified cross-stream generations and canonical fallback. Otherwise delete/disable packed mode.

### M6 — Final qualification and defaults

`off` remains default absent broad, controlled data. Consider `auto` as a Volta-only default only after the §7/§8 matrices pass and no regression appears on other GPU configurations or CPU-only/non-NVFP4 setups.

---

### 6.1 What to do if a gate fails

A red gate is a **recorded negative experimental result**, not a command for the engineering runner to terminate before producing the remaining reports. Every milestone must emit baseline and candidate timings, failures, and a recommendation; independent experiments continue unless a correctness or resource hazard makes them unsafe. Exclude a failing candidate from production dispatch, but finish the safe benchmark matrix necessary to distinguish a bad schedule, a bad layout, and a genuine no-headroom case. Only M6 may change the default from `off` after passing **all** correctness and end-to-end conditions.

---

## 7. Required numerical validation

Minimum coverage:

- All 16 E2M1 encodings (including signed-zero behavior), GGUF split-half code order, each UE4M3 scale class (zero/subnormal/normal/max; reject or explicitly handle unsupported encodings), positive and negative weights, and the real FP4-Local scale distribution.
- **Repack round-trip** (packed mode): all codes/scales decode bit-exactly vs. original for gate/up/down; tail FP32 bits unchanged; sentinels at every row and sub-block boundary; random real experts from low and high layers.
- Projection outputs vs. an FP64/FP32 reference on real activations, with error attributed separately to FP16 input conversion, weight conversion, FP32 accumulation, and scale placement. Include outliers, tiny activations, SwiGLU tails, long verify windows, and highly uneven expert counts.
- Full expert `gate/up → SwiGLU → down` vs. reference and current `dp4a` at 1, 2, 3, 4, 8, 16 entries. Guard scratch/output bounds; run Compute Sanitizer (memcheck, racecheck, initcheck) where feasible.
- Model-level: first-token KL vs. a fixed old-engine reference, top-1/top-k agreement, perplexity or held-out tasks, MTP acceptance. Deterministic seeds, fixed template, temperature 0 for token-level controls; equal output lengths for timing; log divergent outputs separately.
- No NaN/Inf; no lost, duplicated, reordered, or stale expert entries; CPU misses and GPU hits land in exactly the expected output slots; the `off` path is byte-for-byte unchanged in behavior.

**Provisional thresholds** (locked before full evaluation; revisions require an explicit numerical DDR): per-projection relative L2 vs. a correctly scaled FP64 reference ≤2% on non-pathological real activations; whole-expert relative L2 ≤3%; mean first-token KL against **the same FP4-Local checkpoint under the old Strata backend** ≤0.005 over ≥100 held-out prompts, plus tracked maxima, 95th/99th percentiles, and bootstrap confidence intervals. Report top-1 disagreement, MTP acceptance and held-out accuracy; substantial regressions veto enablement. This is not a claim of bitwise equivalence. Evaluate separate error budgets for `FP32→FP16`, `Q8_1→FP16`, and FP32 scale/SiLU; allow only explicitly documented finite edge cases and never ignore NaN/Inf. These are engineering proposals, not known guarantees. Compare with an FP16-reference control, and report maxima and percentile tails, not only means.

---

## 8. Performance matrix and instrumentation

### 8.1 Microbenchmarks

- **Operations:** gate, up, down individually; full expert; (prefill) batched expert GEMM. Confirm the reference computes `s_gate/s_up/s_down` at their native FP32 placement.
- **Entries per expert:** 1, 2, 3, 4, 5–8, 9–16, 17–32, plus mixed realistic histograms from M0, and (prefill) 33–512 for the crossover analysis.
- **Backends:** current NVFP4 `dp4a` grouped; QPN-direct; QPN-packed (if present); **NVFP4 W4A8 MMQ prefill**; dequantize→FP16/cuBLAS fallback/reference. The Q2_0-only `moe_fp16tc` and IQ/Q2-only `gemm_iq_f16_grouped` are excluded from NVFP4 timing comparisons unless separately extended and validated.
- **Working sets:** fully resident, mostly resident, mixed CPU/GPU, forced misses; cold vs. warm slots.
- **Measures:** latency per completed expert, kernel count, bytes read, registers/spills, occupancy, tensor utilization, pack time per slot, H2D/D2D bytes, temporary VRAM, numeric metrics.

Timings include conversion/staging, post-processing and synchronization wherever production pays them; bare-kernel timings are reported separately for diagnosis.

### 8.2 End-to-end (single V100-PCIe-32GB)

- Same packed weights and Strata execution modes; contexts 2K, 8K, 32K (+ longer if capacity permits).
- Prefill (fresh prompt) and continued-conversation decode measured separately; fixed 256-token generation cap or equal actual lengths.
- Speculation: disabled, `--spec 4`, and **only additional `--spec` widths accepted by the pinned binary** (do not assume `--spec 8` is valid). Always report **accepted draft tokens per target pass**; compare tok/s only at matched acceptance where possible.
- Controls: expert cache on/off where feasible, frozen profiles, adaptive mode, unchanged `--pcie-mode`; CPU and PCIe settings identical across arms.
- ≥5 timed repetitions after warmup, alternating A/B order, stabilized temperature/clocks, recorded power/thermal throttling, NUMA binding, binary/model SHA. Report medians, ranges, and a confidence interval if n permits.
- Time breakdown: GPU compute, cache fills/packs, CPU pool, PCIe waits, graph launches, prefill/decode, end-to-end response time. Report real session latency, not just peak tok/s.

**Reasonable, track-specific targets (not guarantees):** ≥10% isolated full-expert speedup at a meaningful T bucket; ≥5% GPU-resident decode-path improvement; ≥3% repeatable end-to-end decode gain; for prefill, ≥3% TTFT gain at one or more prompt lengths — with fully accounted peak and steady-state memory footprint, no substantial residency loss, and no material accuracy regression. A slower result triggers fallback, not special pleading about theoretical tensor-core throughput. If PCIe/CPU waits dominate one scenario, report the Amdahl limitation there and test other realistic residency distributions before deciding that track has no value. Do not optimize a path that failed correctness or repeatedly worsens end-to-end performance.

---

## 9. Build, compatibility, fork management, and safety

### 9.1 Build and compatibility

- CUDA **12.8**, `CMAKE_CUDA_ARCHITECTURES=70`; do not use CUDA 13 to compile `sm_70`. Guard device code with `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700` and add host-side device checks.
- No required CUTLASS upgrade, Blackwell ISA, native FP4 MMA, Turing-or-newer INT8 tensor-core instructions, AVX-512, or new Python conversion dependency.
- Add `STRATA_ENABLE_VOLTA_QPN` (**build-time default OFF**, explicit opt-in for sm_70). This is separate from the **runtime default `STRATA_NVFP4_VOLTA_QPN=off`**. Exclude kernels from non-Volta targets. Because Strata uses eager CUDA module loading, measure device module footprint even with the runtime feature off; if material, isolate the QPN kernels in a lazily loaded CUDA DSO following Strata’s K-quant DSO pattern. An `off` build without QPN must preserve original VRAM use.
- Keep HIP/SYCL portability by isolating V100-specific sources and not leaking CUDA-only headers into public interfaces.
- Validate alignment (16/32/128 B), row pitch, offsets, slot guards, and packed scale-plane bases; no reliance on inter-expert overreads.
- No dynamic allocation, host polling of device group counts, or global syncs in the decode graph. Check every launch; track per-slot conversion completion with existing stream/event semantics.

**Baseline build (validate options against the pinned tree and compiler before running):**

```bash
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build-v100 --target strata -j
```

**A/B operation (once implemented):**

```bash
STRATA_NVFP4_VOLTA_QPN=off     <identical Strata invocation>   # reference
STRATA_NVFP4_VOLTA_QPN=direct  <same invocation>               # GGUF-layout QPN
STRATA_NVFP4_VOLTA_QPN=packed  <same invocation>               # only if M5 passes
```

### 9.2 Fork management

- **Branches:** `main` tracks upstream `jmnargi/Strata-V100` main via a fast-forward-only mirror; feature work lives on `qpn/*` branches; `qpn/integration` collects merged PRs. Pin the baseline at `2c9b1dd…` until M0 completes, then rebase deliberately.
- **Upstream sync:** rebase or merge upstream at milestone boundaries only, re-running the M0 baseline smoke and the `off`-path unchanged check after each sync. Record the new upstream SHA in `benchmarks/`.
- **Upstreamability:** keep the feature isolated (new files, narrow hooks in `iq_kernels.cu` / `expert_cache.*`) so individual PRs can be offered upstream if the maintainer wants them.
- **CI/regression:** the `off` path must pass the existing parity suites unchanged on every PR; add a CPU-only/non-Volta compile check to prevent breakage of other builds.
- **Artifacts:** benchmark raw data, model/binary SHAs, and commands are committed under `benchmarks/qpn/`; no model weights or copyrighted blobs are committed.

### 9.3 Licensing and attribution

- ninfer-v100 is **Apache-2.0** and Strata is **MIT**, confirmed from the checked repository license files (recheck after rebasing). Reused or adapted code retains its Apache-2.0 header, a `NOTICE` entry, and a "modified" statement. State the resulting distribution arrangement explicitly in `README`/`LICENSES/`.
- Trace upstream provenance of the borrowed kernels (especially the `v100-skinny` geometry and TurboMind-derived E2M1 decoder) and preserve each applicable notice/license. Apache-2.0 code may be combined with MIT code if the Apache notice, license, modification disclosure and any third-party obligations are retained; do not simply relabel copied code MIT. Prefer independent implementation when provenance or license is unclear.
- Confirm that distributing derived performance results/docs is compatible with the model's license terms.

---

### 9.4 Minimum automated test matrix

| Target | Required check |
|---|---|
| CUDA `sm_70`, feature `off` | Build + existing tests, identical control output; no QPN work/scratch allocated or loaded unexpectedly |
| CUDA `sm_70`, `direct` | Synthetic and real NVFP4 parity, grouped mixed-T selection, graph replay, sanitizer, end-to-end FP4-Local |
| CUDA `sm_70`, `packed` (conditional) | All `direct` checks plus fill/promotion/lend/replacement/shrink/grow and versioned event races |
| CUDA non-Volta `sm_75`/`sm_80+` | Compile and normal NVFP4/other-format smoke; new path cannot launch |
| CPU-only / HIP / SYCL | Build/compile where supported, existing non-CUDA tests and no CUDA header leakage |
| Non-NVFP4 formats | Q2_0, IQ, K-quant regression controls; no changes to old FP16-TC/WMMA routes |

Exact GPU-backed CI availability may vary; record unrun hardware targets explicitly rather than treating compilation as runtime validation.

---

## 10. Failure and rollback conditions

The feature must fail closed to the existing implementation (or refuse to start before executing) on: unsupported ggml type, non-`sm_70` GPU, wrong geometry, insufficient scratch, unknown slot layout, unready packed slot, missing event ordering, failed scale/decoder validation, or packing failure. **No in-graph retry after a CUDA memory fault**; rely on preflight/dispatch fallback. A runtime failure that has already invalidated a stream or context terminates the affected request visibly rather than continuing with corrupt state.

Existing model formats, CPU-only experts, non-NVFP4 experts, non-Volta builds, and `STRATA_NVFP4_VOLTA_QPN=off` keep their prior paths and performance. When a gate fails, keep the measured evidence, move to any independently promising candidate, and never enable a path that failed correctness.

---

## 11. Deliverables and PR sequence

Each PR includes: reproducible commands, exact model/binary identifiers, raw benchmark data, old/new timing table, correctness coverage, a one-paragraph "what did **not** improve," and a definition of done.

1. **PR 1 — Audit, fixtures, baseline (M0).** Real dimensions, nibble format, scale derivation, activation audit A-1, group-size histograms, headroom/Amdahl analysis. *Done when:* the go/stop decision for M1 is documented.
2. **PR 2 — Standalone `qpn-direct` kernels + numerics (M1–M2).** Decode, shapes, FP32 scale + SwiGLU, parity and bench tools. No engine dispatch change. *Done when:* M1 gate and unit-level §7 checks pass.
3. **PR 3 — Opt-in resident decode/verify integration (M3).** Device-side group dispatch, scratch and graph integration, A/B throughput, T=1/unsupported fallback. *Done when:* M3 gate passes.
4. **PR 4 — Small-batch prefill route (M4).** Crossover table, `QPN_PREFILL` flag, TTFT results. *Done when:* M4 gate passes or the route is documented as not beneficial.
5. **PR 5 — Optional cache-aware packed layout (M5).** DDR-1, layout versioning, fills/replacements/lending, pointer stability, pack cost, end-to-end tests. **Only if direct-mode measurements, including prefill, show a worthwhile layout-conversion bottleneck and DDR-1 passes.**
6. **PR 6 — Qualification and defaults (M6).** Report, service flags, rollback docs; `auto` default only if the full matrix passes.

---

## 12. Source links / pinned references

**Target repo:** <https://github.com/jmnargi/Strata-V100/tree/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926>
- [NVFP4 formats, conversion, FP4-Local](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/docs/NVFP4.md)
- [Native grouped expert API](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/include/strata/kernels/iq_kernels.hpp)
- [CUDA native expert decoder and grouped implementation](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/kernels/cuda/iq_kernels.cu)
- [Expert cache state/copy interface](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/include/strata/core/expert_cache.hpp)
- [Volta optimized paths and measured bottlenecks](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/docs/NVIDIA_V100.md)
- [Prefill MMQ dispatch](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/prefill/moe_mmq.cu)
- [Volta prefill FP16-TC implementation — Q2_0 only](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/prefill/moe_fp16tc.cu)
- [Compact FP16-TC shape/type contract](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/include/strata/prefill/moe_fp16tc.hpp)
- [Grouped WMMA type gating (`gemm_iq_f16_grouped`)](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/kernels/cuda/iq_kernels.cu#L5885)

**Reference repo:** <https://github.com/geoffwatts/ninfer-v100/tree/b37d0dd3e1163b9d802d8bccfa89918bf68d793e>
- [Generic QPN NVFP4 decode / FP16 MMA](https://github.com/geoffwatts/ninfer-v100/blob/b37d0dd3e1163b9d802d8bccfa89918bf68d793e/src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh)
- [QPN shape and launch guards](https://github.com/geoffwatts/ninfer-v100/blob/b37d0dd3e1163b9d802d8bccfa89918bf68d793e/src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cu)
- [SwiGLU QPN scheduling and split rationale](https://github.com/geoffwatts/ninfer-v100/blob/b37d0dd3e1163b9d802d8bccfa89918bf68d793e/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp)
- [Volta V100 notes](https://github.com/geoffwatts/ninfer-v100/blob/b37d0dd3e1163b9d802d8bccfa89918bf68d793e/docs/v100.md)

**Model:** <https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local>

---

**Bottom line:** The likely useful transfer is **ninfer's software-NVFP4-to-FP16/QPN tensor-core primitive** applied to small per-expert batches, not a replacement of Strata's inference architecture. The current Strata compact FP16-TC/WMMA kernels must **not** be mistaken for existing NVFP4 alternatives. Decode (verify windows) is the strongest candidate; prefill gains are plausible only for the small-batch tail and must be shown by a crossover table. Each path may be declined after its own complete evidence-based gate; neither independent track is canceled by a failed preliminary result in the other.
