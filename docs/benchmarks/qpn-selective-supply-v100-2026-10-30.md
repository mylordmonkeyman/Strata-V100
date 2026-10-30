# Selective MMQ expert supply experiment

## Measured starting point

Pinned source `db7945c3e73186d47922d0b3146a172095692439`, [V100 run
41063269352](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41063269352),
aggregate artifact `12982875743`. Four fresh processes each completed 2047 prefill
tokens, exited zero, and executed zero QPN groups. Both forced-staged and stream-all
with transition fallback matched first logits **bytewise** with tracing off/on.
The expert-pack and prompt SHA256 are pinned by the harness; all four prior NaN
workaround settings are recorded in the artifact.

The **cross-supply** comparison did NOT match: relative L2 `0.4174716303`,
KL `0.4516201315`, argmax different. Thus forced-staged 11948.0 ms and stream-all
17302.6 ms (uninstrumented) do **not** establish a same-output speedup. These
are independently repeatable configurations, not numerically interchangeable.
This experiment does not investigate that difference or promote either default.

First-chunk sampled stream-all supply:

| Layer | Compressed H2D bytes | Copies | Copies beyond routed nonresident | H2D interval GB/s | Ready wait ms | Slot wait ms | H2D/product overlap ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 760324400 | 275 | 181 | 9.745 | 2.374 | 0.528 | 16.237 |
| 24 | 846033696 | 306 | 108 | 9.738 | 3.503 | 0.808 | 7.188 |
| 47 | 948331888 | 343 | 237 | 9.579 | 5.499 | 1.863 | 3.778 |

The three sampled layers transferred 2,554,689,984 bytes in stream-all versus
1,097,631,952 in forced-staged. This is not a whole-model byte total. Group
residency fractions were approximately 54%, 41%, 52%; routed-row residency about
59%, 34%, 73%. CUDA-event bandwidth includes submission/event overhead and is
not PCIe link saturation. Products are envelopes, not hardware occupancy. Excess
copy counts are an opportunity bound; routing isn't known before early DMA.
Host cache temperature is uncontrolled. No warm-engine or controlled-cold result.

## Next isolated experiment

`STRATA_QPN_MMQ_SELECTIVE_SUPPLY` is a new experimental **default-off** variable,
compiled only with the existing prefill experiment gate. Value `0` records aggregate
supply counts while retaining all original transfers. Value `1` also omits H2D for
an expert only after its current layer's exact routing has been published and that
expert has zero routed rows. Unknown future layers still transfer normally. No
routing prediction, persistent residency change, prepack conversion, or source/cache
mutation is introduced. The legacy WMMA skip flag retains its previous semantics.

Eligibility is single V100 sm70, native NVFP4 W4A8 throughout, MMQ, CPU share zero,
QPN execution off, no peer or split chain. Unsupported requested configurations
fail before stream issuance. The host staging job of every unpinned skipped expert
is still waited for and marked issued on the copy stream. Used waits, copied events,
live-slot state, entry order, consumed accounting and releases remain in place for
every entry. Only its H2D payload is omitted. This deliberately retains CPU read
cost to isolate DMA savings from host queue/lifetime changes.

The existing locked/cached V100 workflow now runs `--selective-abba`: baseline /
selective / selective / baseline, with transition fallback enabled in **both** arms,
all prior NaN flags, unchanged stream threshold, same pinned private model/prompt,
tracing off, full work/finite/output-size gates, zero QPN execution, strict bytewise
first-logit equality across all six comparisons. Aggregate counters must show actual
skips and equal total scheduled entries/bytes. No raw logits, expert IDs, activations,
weights or prompt contents are uploaded. Private temporary logs are removed on exit.
The older diagnostic workflow remains manually dispatchable; its automatic push
trigger is removed so this hook does not launch redundant GPU diagnostics. The
existing hourly optimization automation is unchanged.

A positive timing result would still need repeated controlled
measurements, changed-input/Graph and independent oracle/guards plus sanitizer
qualification of the new supply path before any integration or quality claim.
Existing standalone fixture sanitizer passes do not qualify this new model path.
If practical supply savings don't produce a reproducible material advantage, stop
QPN layout variants; do not infer a QPN gain from MMQ-only supply improvements.


## Completed ABBA: transfer reduction, no material prefill benefit

Source `d777b85efadb5a359ebbb7ebb68e88e475f20cc9`, [V100 run
41066208726](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/41066208726),
aggregate artifact `12982683441`. The host tests and GPU workflow succeeded.
The committed typed aggregate is
`benchmarks/qpn/results/2026-10-30-selective-mmq-supply-abba.json`.
Executable SHA256 `29130300738240816341c31de5b177eb7e59a5c205bd7b640ac5992f56c58a3d`.

All four processes completed 2047 prefill tokens, exited zero, and reported zero
QPN execution. All six within/cross-arm first-logit comparisons were bytewise
identical over 248,320 FP32 elements (L2/KL zero, argmax equal). Every run exercised
all 48 transition fallback layers. Every first chunk had the same 15,347 scheduled
entries and 42,431,631,152 scheduled compressed bytes, with 6,027 staging jobs.

| Measurement | Baseline A | Selective B |
|---|---:|---:|
| First-chunk transferred bytes | 42,431,631,152 (both runs) | 28,831,501,248 / 28,825,971,616 |
| First-chunk skipped experts | 0 | 4,919 / 4,921 |
| Skipped experts whose host jobs were still drained | 0 | 2,533 / 2,529 |
| Prefill times (ms) | 17077.8 / 17305.9 | 16849.2 / 17291.6 |
| Median prefill time (ms) | 17191.85 | 17070.40 |

The selective arm omitted about **13.60 GB / 32.1%** of first-chunk H2D, but
median prefill throughput was only **1.007115x** baseline (~0.71%). The 121.45-ms
median difference is smaller than either arm's observed timing range (228.1 /
442.4 ms). This single ABBA does not establish a reproducible material improvement.
TTFT medians (60,926.25 / 58,551.60 ms) reflect uncontrolled fresh-process startup;
particularly the first baseline's 63,238.1 ms. Do not claim a validated TTFT gain.
Transfer counters cover the first stream-all chunk, not the staged continuation.

**Decision:** retain this as a default-off, unqualified experimental negative result;
do not integrate it into production or spend further GPU time qualifying/promoting
it. Stop further QPN layout/arithmetic variants: resident-only wins, matched async
parity, and now large realistic DMA savings without a material prefill win do not
justify them. The experiment does not isolate whether unchanged CPU staging reads,
compute, or another component dominates the remaining critical path; it is not
proof that all selective-supply approaches can never help. Any new hypothesis needs
new evidence and its own matched control, not extrapolation from bytes saved.

The independent rev4.1 Q4 kernel/quantization track is the next eligible development
track, with separate artifact provenance, within-Q4 kernel-vs-MMQ gates and numerical
qualification. No below-4-bit artifact or reinterpretation of NVFP4 as Q4 is allowed.
The core QPN D4 numerical-alignment task and full-model quality remain unqualified;
neither is silently waived. No full-model sanitizer, changed-input Graph, intermediate
output-guard or independent GGUF FP64 qualification was performed for this discarded
supply prototype. Original model/cache, production defaults, main, host services and
runner privileges remain unchanged. The existing hourly automation is retained.
