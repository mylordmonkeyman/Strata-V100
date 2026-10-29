# Development validation — October 28, 2026

Baseline: `2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926`.
Ninfer source: `b37d0dd3e1163b9d802d8bccfa89918bf68d793e`.
Environment: Linux x86-64, GCC 13.3, CMake 4.4.4, Python 3.12.
No NVIDIA GPU, driver, model pack, or captured model activations available.

## Checks run

| Check | Actual result |
|---|---|
| Dependency-free host oracle, `-O2 -Wall -Wextra -Werror` | Pass |
| CMake CPU configure with CUDA/native experts OFF | Pass |
| New CTest `nvfp4_qpn_host` | Pass |
| Python pack-audit tests | 5/5 pass |
| Runnable CPU CTests (28 existing + 1 new) | 29/29 pass |
| Full CPU build with native experts OFF | Blocked by baseline AVX1 unresolved symbols |
| Reproduce that link error in detached pinned-baseline worktree | Same three unresolved symbols |
| Reject QPN ON with CPU-only backend | Expected configure failure |
| CUDA 12.8.93 compile to `sm_70` object, all instantiated schedules | Pass, zero spills |
| C++ benchmark syntax against pinned ggml header, warnings as errors | Pass |
| Full CUDA CMake engine build/link | Not run locally; compile-only CI supplied |
| GPU device decoder, projection/full-expert parity, graph replay | Not run |
| Compute Sanitizer, mixed groups, CPU misses, model KL/accuracy | Not run |
| Prefill MMQ/fallback crossover or end-to-end TTFT | Not run |
| Completed-expert or engine speedup | No measurements |

The compiler was downloaded for offline compilation after confirming there
was no installed toolkit. NVIDIA's CUDA 12.8.93 compiler archive was used with
CUDA 12.8.90 runtime/CCCL headers. The benchmark was syntax-checked against
`ggml/include/ggml.h` at Strata's pinned llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Compilation is not runtime parity.

Compiler command, with local include paths abbreviated:

```bash
nvcc -std=c++17 -arch=sm_70 -I<cuda-runtime>/include -I<cccl>/include \
  -Iinclude --ptxas-options=-v \
  -c src/kernels/cuda/nvfp4_volta_qpn_decode.cu -o /tmp/qpn-decode.o
```

The [raw ptxas log](development/cuda12.8-sm70-ptxas.log) contains every
instantiation: QPN uses 64–143 registers/thread, 4–32 KiB static shared memory,
zero stack/spill bytes, and one barrier. No schedule is selected by this
report; occupancy/latency must be measured on the V100.

## Baseline CPU build limitation

The unmodified baseline with `STRATA_NATIVE_EXPERTS=OFF` fails linking
`pool_affinity_test` (also other CPU expert tests) on references to:

- `act_quant_q8_1_avx1`
- `q2_0_gguf_rows_multi_avx1`
- `q2_0_gguf_rows_multi_avx1_legacy`

The same error was reproduced by configuring and building only
`pool_affinity_test` in a separate detached baseline checkout. This prototype
does not alter the CPU sources. Nine CTests dependent on blocked/unbuilt
targets were excluded from the 29-test run; **the full suite is not green**.

Commands for the successfully runnable subset:

```bash
cmake -S . -B build-qpn-cpu -G Ninja -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_ENABLE_CUDA=OFF
cmake --build build-qpn-cpu --target nvfp4_qpn_host_test
# The other runnable targets had been built before the unrelated link failure.
ctest --test-dir build-qpn-cpu \
  -E 'pool|expert|q2_bitplane|direct_file|aux_cpus' --output-on-failure
python3 -m unittest discover -s tools -p test_qpn_audit.py -v
```

## Gate decisions

Decode: **pending**, not a pass or a negative performance result.
Prefill: **independently pending**; small-T candidate available, production
comparators and end-to-end measurements not available in this environment.
Packed layout: no implementation or DDR-1 approval; no cache code changed.
Defaults: engine behavior and link graph unchanged.

The next concrete step is to run the audit, synthetic smoke, real-expert and
real-activation harness on the V100, then complete M0's runtime profiles and
M1/M2's multi-expert/working-set coverage before production integration.
