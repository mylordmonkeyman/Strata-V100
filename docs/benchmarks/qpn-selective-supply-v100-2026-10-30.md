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

Results are pending. A positive timing result still needs repeated controlled
measurements, changed-input/Graph and independent oracle/guards plus sanitizer
qualification of the new supply path before any integration or quality claim.
Existing standalone fixture sanitizer passes do not qualify this new model path.
If practical supply savings don't produce a reproducible material advantage, stop
QPN layout variants; do not infer a QPN gain from MMQ-only supply improvements.
