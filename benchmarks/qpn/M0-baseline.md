# Full-model baseline runner

The `qpn/full-model-baseline` branch triggers `.github/workflows/qpn-m0.yml` on
the existing Docker V100 runner. No main-branch or engine changes are required.
The workflow builds the unchanged baseline at
`2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926` and prepares the NVIDIA checkpoint at
`b9ce18a6887e18c089fdea8509a33e63429a3300` with the existing conversion tools.
It requires at least 250 GiB free disk space. Assets and CPU Python dependencies
are cached under `/srv/strata-qpn/full-model`. An unused draft pack is not prepared.

Six separate engine invocations use the same 2048-token templated prompt and
32-token greedy continuation, with speculation disabled. The first is labeled
warmup. Existing instrumentation captures the first prefill chunk's MoE inputs
and routing at layers 0, 24 and 47 in the first three runs. These captures can
perturb timings; compare the final three uninstrumented-activation runs separately.
First-logit capture is enabled for every repetition. Raw logs include engine
statistics. Whole-process wall time includes startup and is not decode latency.
One repeated prompt and these three layers cannot establish representative
all-model routing or the specification's complete M0/headroom gate.

The container has a 48 GiB memory limit. This scenario explicitly reserves a
24 GiB CPU resident expert budget, uses the profile-filled GPU cache and mmap
fallback, and is not a qualification of the host's full 128 GiB RAM scenario.
No GPU clients are stopped, and the run refuses to overlap existing compute clients.
The shared GPU lock and Actions concurrency group serialize QPN jobs.

Only logs, configuration, counts and hashes are uploaded. Checkpoints, model
packs, activation vectors, token IDs and logits remain in the private runner
volume. Failed startup or timeout is retained and stops repeated allocations.
Successful execution is the start of M0 evidence gathering, not production QPN
qualification. The standalone kernel and prefill tracks remain independent.

`qpn/real-activations` triggers the matched-row follow-up workflow. Its artifact
input is pinned to baseline run 40807283254 and is serialized behind that run.
It verifies the prepared model hashes, verifies the captures against successful
baseline manifest entries, and extracts the top, median and least-routed active
experts per captured layer. At most 32 actual rows per expert are retained in
order, without duplication or synthetic padding. Each expert is tested against
its own rows at supported sizes, both FP32 and Q8_1 input/hidden staging paths.
The fixtures, checkpoint and activations stay in the runner volume. The index,
raw numerical/timing measurements and logs are uploaded. These samples qualify
part of M2; they do not establish end-to-end quality or mixed-group dispatch.
