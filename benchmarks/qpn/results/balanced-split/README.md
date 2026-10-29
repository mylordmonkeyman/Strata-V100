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
Single-expert tests still lose to the old path. Model quality, representative
routing and activation qualification, mixed dispatch and TTFT remain open.
