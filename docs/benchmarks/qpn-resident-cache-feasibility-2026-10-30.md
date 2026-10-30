# V100 resident prepacked QPN: feasibility gate (no model changes)

Source of the blob geometry: [completed full-model nonresident packed report](qpn-full-model-packed-negative-2026-10-30.md), 2026-10-30, source `c8d4d2fe4335e3feb7e9d39b96c675402dda79a9`, Actions `41075728954`, aggregate `12990608578`. This document does **not** supersede the implementation specification or claim a speedup.

## What is measured and what is not

The prior standalone V100 resident prepacked 16-expert fixture showed an arithmetic opportunity; actual full-model nonresident streamed QPN lost to MMQ for both tested row ranges. To evaluate packed residency, the existing native grouped DP4A decode path still needs **canonical expert weights**. A second packed format therefore consumes *additional* VRAM, or displaces canonical residency inside a fixed budget. The verified prepacked full-expert set contained 24,576 blobs and 67,948,118,016 bytes; one blob is **2,764,816 bytes**. Native compressed and losslessly transposed packed blobs have equal bytes here. The fixed starting 24-GiB native expert budget used in the earlier model workflow is a configuration **cap**, not a measured amount of idle VRAM, a proof that all slots fit, or a measured cache hit rate.

A complete QPN group needs **16 jointly eligible experts in one layer**. This provides a necessary, not sufficient, bound; actual candidate groups also depend on prompt-specific routing, row eligibility, partitioning, staging, MMQ fallback and kernel overhead.

| Hypothetical extra packed VRAM | Packed duplicate slots | Optimistic maximum layers containing one 16-expert group | Canonical slots displaced if budget is fixed |
|---:|---:|---:|---:|
| 256 MiB | 97 | 6/48 | 97 |
| 512 MiB | 194 | 12/48 | 194 |
| 1024 MiB | 388 | 24/48 | 388 |
| 2048 MiB | 776 | 48/48 | 776 |
| 3072 MiB | 1165 | 48/48 | 1165 |

The arithmetic minimum for 16 packed duplicates in **each** of 48 layers is **2,123,378,688 bytes = 1.978 GiB**; even that says nothing about whether the *routed* 16 are packed or whether the V100 has the spare memory. At a fixed 24-GiB total expert budget, 2,048 MiB of packed duplicates necessarily displace **776 native slots**, potentially increasing canonical decode/cache misses and H2D load.

## New host-only guard

`tools/qpn_resident_cache_feasibility.py` computes both additional-VRAM and fixed-total-budget scenarios, using only the committed verified blob geometry. It has strict geometry checks and a synthetic host test suite. An **optional** privately held `--dump-routing` file plus complete STRP expert profile can produce a same-trace *oracle* upper bound of packed-resident route-entry coverage and canonical route-entry losses; it never exports expert identities, raw trace, weights, prompts, activations, logits or routing IDs. The oracle uses the same trace for selection, so it is explicitly **not** a predictive hit rate or a measurable timing gain.

Example without private data:

```bash
python3 tools/qpn_resident_cache_feasibility.py \
  --native-budget-gib 24 --extra-mib 256 512 1024 2048 3072 \
  --report /tmp/qpn-resident-feasibility.json
python3 -m unittest discover -s tools -p test_qpn_resident_cache_feasibility.py -v
```

No model, cache, default dispatch, CUDA kernel, driver, runner privileges, service, or original expert pack was modified by this test. No GPU job or model conversion is required for this bound. This result does **not** qualify numerical correctness or model speed.

## Evidence needed before attempting a packed VRAM cache

1. **Measure live headroom**, after existing dense, KV, model, scratch, ring and canonical cache allocations on the *single pinned 32-GiB V100*. A pre-model `nvidia-smi` reading is insufficient. Do not infer free VRAM from `--resident-budget-gib 24`.
2. Privately collect per-layer actual routed group eligibility, existing canonical cache hits, and weight transfers for the fixed prompt. Estimate how many full eligible 16-expert QPN groups can be formed under 256/512/1024/2048 MiB, with canonical decode residency still retained. Never upload route identities.
3. For a **fixed-VRAM** comparison, charge every packed duplicate against the canonical cache, including H2D misses introduced by native evictions. For **extra-VRAM**, require measured headroom after safety reserve before allocation, not simply a declared budget.
4. Advance to a default-off resident packed proof-of-concept only if realistic eligibility and memory show a plausible *material* end-to-end advantage. Preserve original native fallback, slot identity/generation and guards. Then run matched model MMQ/QPN ABBA and phase-specific prefill, decode TPS/MTP, independently checked numerical and sanitizer gates.

The previously successful fixture resident ratios are not model-wide performance gains. The completed streamed full-model negative remains the last actual model result. The 1–16-row decode-shaped QPN tests lost against native grouped DP4A, so there is no evidenced packed decode dispatch to integrate.

**Gate:** Do not allocate new packed VRAM or launch another costly full-model prepacked cache build until the measured residency/eligible-group/VRAM opportunity is materially positive. Preserve all production and host boundaries.
