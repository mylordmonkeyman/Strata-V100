# Balanced split-K V100 evidence

Source ca64e263c6a451de686d5cdeac78c409ee9f271d, GitHub run 40804676590.
Tesla V100 PCIe 32 GB, CUDA 12.8. All 348 configurations passed: 40 synthetic,
120 checkpoint-expert, 48 uniform streaming batches, 28 outlier-activation,
and 112 schedule-sweep cases. Nine compute-sanitizer invocations also passed.
Raw measurements, candidate settings, fixture hashes and provenance are retained here.

Best standalone completed-expert batch speedup was 2.407x versus the old expert
path. This is a measured candidate, not a production default or an end-to-end
model speedup. Uniform batches replicate one fixture's weights into separate
addresses; they do not reproduce mixed experts or actual routing distributions.
Default-schedule single-expert tests still lose to the old path. Model quality, representative
routing and activation qualification, mixed dispatch and TTFT remain open.

Best observed schedule per shape in the one-fixture FP32 sweep (six timing
repetitions; selecting the best observed value is exploratory):

| Groups | Entries/expert | GU split | Down split | Accumulators | Speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 16 | 8 | 1 | 0.423x |
| 1 | 4 | 16 | 8 | 1 | 0.521x |
| 1 | 8 | 16 | 8 | 1 | 0.715x |
| 1 | 16 | 8 | 8 | 2 | 0.945x |
| 1 | 32 | 8 | 8 | 2 | 1.082x |
| 8 | 2 | 16 | 8 | 1 | 0.711x |
| 8 | 4 | 16 | 8 | 1 | 0.898x |
| 8 | 8 | 16 | 8 | 1 | 1.319x |
| 8 | 16 | 8 | 4 | 1 | 1.953x |
| 8 | 32 | 4 | 4 | 2 | 2.407x |

The T=32 single-expert schedule's narrow win does not establish the decode
speed gate across shapes/experts. No runtime threshold is selected from this table.
