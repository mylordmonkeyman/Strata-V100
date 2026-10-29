# NVFP4 Volta QPN prototype — hardware qualification pending

This branch contains a **standalone experimental direct-layout candidate**,
adapted from ninfer's QPN schedule for Strata's canonical NVFP4 expert bytes.
It does **not** integrate a new engine backend, change expert slots, implement
runtime QPN environment flags, or claim improved performance. `main` remains
at upstream baseline `2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926`.

The [rev-3 specification](specification-rev3.md) governs subsequent milestones.
M0 still needs real-model/hardware data. M1/M2 have a compilable candidate and
tools, **not passed gates**. M3, M4 runtime integration, M5 packed layout, and
M6 defaults are not implemented. Decode and prefill remain independent tracks.

## What is implemented

- QPN direct GGUF reads: four scale bytes followed by four split-half 16-value
  code sub-blocks; FP16 MMA and FP32 accumulation/output, no slot repack.
- Independent E2M1 bit construction and ggml-compatible UE4M3 decode. Global
  FP32 scales stay outside the weight decoder. Maximum canonical decoded
  weight magnitude is 2688; representable FP16 weights need no rebias/divisor.
- Separate gate/up projections, FP32 scales/SwiGLU, hidden staging, down
  projection, FP32 down scale. FP32→FP16 and Q8_1→FP16 staging are both exposed.
- T=1..32; split-K 4/8/16 and 1/2 accumulator chains. Down K=640 rejects
  split16; T>16 rejects split16's 64 KiB static shared-memory requirement.
- `STRATA_ENABLE_VOLTA_QPN=OFF` build default. When enabled the new library is
  linked only into a standalone benchmark. Even an enabled build's **engine
  executable does not load QPN**. No runtime module-memory claim is needed yet.
- Host format/schedule tests; CUDA device decoder probe; independent ggml
  dequantization plus FP64 projection/full-expert references; staging vs. MMA
  error attribution; output guards, graph replay, six alternating A/B rounds.
- Pack audit and fixture extraction, raw benchmark runner, provenance/licenses,
  CPU tests and CUDA **compile-only** CI.

All new GPU buffers are owned by the standalone harness; launch functions
allocate/synchronize nothing. The harness checks the active device is sm_70
before using the API. External callers must do the same preflight. There is
no claim of safe mixed-group production dispatch yet.

## Build on the V100 server

Use an existing CUDA **12.8** installation. From this repository's checkout:

```bash
cmake -S . -B build-qpn -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_VOLTA_QPN=ON
cmake --build build-qpn --target nvfp4_volta_qpn_bench nvfp4_qpn_host_test -j 8
ctest --test-dir build-qpn -R '^nvfp4_qpn_host$' --output-on-failure
python3 -m unittest discover -s tools -p test_qpn_audit.py -v
```

Strata fetches its pinned ggml checkout normally. To reuse a matching local
llama.cpp source, add `-DSTRATA_GGML_DIR=/absolute/path/to/llama.cpp` when
configuring. Match the pinned GGML NVFP4 contract before changing this source.

For a dependency-free host check only:

```bash
c++ -std=c++20 -O2 -Iinclude tests/nvfp4_qpn_host_test.cpp -o /tmp/nvfp4_qpn_host_test
/tmp/nvfp4_qpn_host_test
```

## Audit and benchmark a real expert

Use the **FP4-Local NVFP4-QAD pack**, not another GGUF quant. Replace the pack
path below with the directory containing `native_experts.txt` and `experts.bin`.
The audit refuses signed scale bytes, mixed quant formats, unexpected blob
sizes, noncontiguous layer offsets, length mismatch, or invalid global scales.
It checks payload geometry **by size only**; separately check checkpoint/GGUF
dimensions and routing top-k. It never modifies the input pack.

```bash
mkdir -p /tmp/strata-qpn-fixtures
python3 tools/qpn_audit.py /absolute/path/to/nvfp4-qad-pack \
  --layer 0 --expert 7 --extract /tmp/strata-qpn-fixtures/l0-e7.bin \
  > /tmp/strata-qpn-fixtures/l0-e7-audit.json
numactl --cpunodebind=0 --membind=0 python3 tools/qpn_run.py \
  build-qpn/nvfp4_volta_qpn_bench \
  --blob /tmp/strata-qpn-fixtures/l0-e7.bin \
  --out /tmp/strata-qpn-results --iterations 100
```

Extract more fixtures from low/high layers and pass `--blob` repeatedly.
`--sweep` additionally sweeps projection schedules and all four input/hidden
staging combinations; it can take substantial time due to FP64 controls.
`--tokens 1 2 3 4 5 8 9 16 17 32` includes bucket boundaries.
With no `--blob`, the benchmark uses synthetic weights and marks them synthetic.
With no `--activations`, activations are seeded Gaussian, **not model captures**.
Real `--activations` files contain little-endian contiguous FP32 token rows,
width 2560, at least `max(tokens)` rows. Do not commit proprietary inputs.

Run sanitizers independently of timings:

```bash
compute-sanitizer --tool memcheck build-qpn/nvfp4_volta_qpn_bench --tokens 3 --blob /tmp/strata-qpn-fixtures/l0-e7.bin --iterations 1
compute-sanitizer --tool racecheck build-qpn/nvfp4_volta_qpn_bench --tokens 8 --blob /tmp/strata-qpn-fixtures/l0-e7.bin --iterations 1
compute-sanitizer --tool initcheck build-qpn/nvfp4_volta_qpn_bench --tokens 17 --blob /tmp/strata-qpn-fixtures/l0-e7.bin --iterations 1
```

Bench exit codes: 0 = provisional numerical thresholds pass, 1 = finite
numerical failure, 2 = invalid configuration/CUDA/nonfinite/guard/replay failure.
The runner preserves failed candidates and continues independently launched
processes; it never retries within a damaged CUDA context. A result directory
must be new to avoid mixing runs. JSON contains binary/fixture/activation hashes,
commands, exit codes, stderr, repetitions, affinity, and GPU state. Commit only
reviewed measurement logs, never `.bin` fixtures/weights.

## Limits of these timings

Times include input staging, gate/up, FP32 SwiGLU, hidden staging, down and
FP32 down scale. The legacy arm includes input Q8_1 quantization; the Q8 input
candidate includes it too. Production may already have that Q8 image: these
are completed standalone-expert timings, not production launch-cost estimates.
CUDA events measure device work, not host submission latency. Graph replay is
checked separately; timed rounds use normal launches in both arms.

A single 2.76 MB expert may fit V100 L2. This first harness does not simulate
many-expert working sets or mixed group histograms, measure routing/cache/CPU/
PCIe waits, count accepted draft tokens, implement MMQ/FP16-GEMM prefill arms,
or establish model-level accuracy. It has output sentinels, not guards around
every scratch allocation. Sanitizers remain mandatory. No throughput or TTFT
claim follows from a green synthetic run.

Before M3: finish M0 profiles and activation A-1, M1/M2 real-activation and
multi-expert numerics/timing, then document exclusive device group ownership
and graph-stable scratch. Before M4: compare against actual NVFP4 W4A8 MMQ and
correctly scaled dequantize→FP16 GEMM, including supply waits and TTFT. Before
M5: measure direct-layout overhead and complete DDR-1. Do not promote `auto`
or add cache repacking on the basis of this prototype.

## Source audit A-1 (preliminary)

- `include/strata/kernels/iq_kernels.hpp`: grouped input contract is contiguous
  Q8_1 tokens indexed by device `ent_tok`; output is indexed by `ent_dst`.
- `src/core/verify.cpp` `pre`/`post`: ordinary paths quantize FP32 `mixed_` (or
  the Hadamard-rotated `nat_xr_`) to `nat_xq_`. `STRATA_QFUSE` and dedup paths
  may supply the Q8 image directly; **FP32 liveness is not established for all
  modes**, so engine integration must not assume it. Buffers are shared with
  attention/shared-expert work; inspect lifetime on both compute streams.
- `native_expert_grouped` in `iq_kernels.cu`: FP32 gate/up intermediates are
  scaled before SwiGLU, then hidden is quantized to Q8_1 for down. The new
  harness measures FP16 and reconstructed-Q8 staging independently, leaving
  the production scratch planner and lifetimes untouched.
- `tools/nvfp4_local_pack.py::nvfp4_blocks` and vendored
  `third_party/ggml/ggml-common.h` agree on 36-byte blocks. New fixtures are
  checked against the **linked ggml** dequantizer on the server as a second
  source. Sign-clear scales and 2560/640 actual checkpoint geometry remain
  artifact gates, not assumptions to silently apply to another checkpoint.
- `src/prefill/moe_mmq.cu`: type 40 selects NVFP4 W4A8 unless explicitly using
  another eligible mode; `mmq::fits` and prompt routing must be measured.
  Q2_0-only FP16-TC and IQ-only grouped WMMA are not NVFP4 baseline arms.

## What did not improve / pending evidence

No runtime speed, TTFT, accuracy, cache residency, or model acceptance result
was obtained in the development workspace because it has no NVIDIA device or
model pack. The candidate is intentionally absent from the engine link graph.
See [development validation](development-validation.md) for checks actually run.
