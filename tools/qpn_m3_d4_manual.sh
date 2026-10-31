#!/usr/bin/env bash
# MANUAL ONLY — rev4.2 M3 D4 activation parity (no expert weights/model).
# Default: compile-only. Device execution requires --device, the existing
# gpu.lock, verified idle V100, and pinned CUDA 12.8. No Docker/CI automation.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash tools/qpn_m3_d4_manual.sh --build-dir EXISTING_QPN_CUDA_BUILD [--device]

Default: compile the two standalone synthetic D4 parity executables only.
--device: additionally run ONLY those executables, after acquiring the
  preexisting /srv/strata-qpn/gpu.lock and checking the V100 is idle.

No model, GGUF, expert weights, packed cache, benchmarks or downloads.
Does not configure/rebuild Docker, change CMake options, services or drivers.
Does not dispatch GitHub Actions or create an automation.
USAGE
}
fail() { printf 'M3 D4 qualification preflight failed: %s\n' "$*" >&2; exit 2; }

build=''
device=0
while (($#)); do
  case "$1" in
    --build-dir)
      (($# >= 2)) || fail '--build-dir needs a path'
      build=$2; shift 2 ;;
    --device) ((device == 0)) || fail 'duplicate --device'; device=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ -n "$build" ]] || fail 'an existing --build-dir is required'
[[ -f "$build/CMakeCache.txt" ]] || fail 'existing CMakeCache.txt not found; refusing to configure a new build'
for command in git cmake grep flock readlink sed head; do
  command -v "$command" >/dev/null || fail "missing tool: $command"
done
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ "$(git -C "$repo" rev-parse --show-toplevel)" == "$repo" ]] ||
  fail 'must be run from its own Git checkout'
[[ "$(git -C "$repo" branch --show-current)" == 'qpn/prepacked-stream' ]] ||
  fail 'only authorized qpn/prepacked-stream branch may qualify M3'
git -C "$repo" diff --quiet && git -C "$repo" diff --cached --quiet ||
  fail 'source checkout has uncommitted tracked modifications'
cache=$build/CMakeCache.txt
home=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache" | head -1)
[[ -n "$home" && "$(readlink -f "$home")" == "$repo" ]] ||
  fail 'existing CMake build belongs to a different source checkout'
grep -Eq '^STRATA_ENABLE_VOLTA_QPN:BOOL=ON$' "$cache" ||
  fail 'existing build must have STRATA_ENABLE_VOLTA_QPN=ON'
grep -Eq '^STRATA_BUILD_TESTS:BOOL=ON$' "$cache" ||
  fail 'existing build must have STRATA_BUILD_TESTS=ON'
grep -Eq '^STRATA_ENABLE_CUDA:BOOL=ON$' "$cache" ||
  fail 'existing build must have STRATA_ENABLE_CUDA=ON'
grep -Eq '^CMAKE_CUDA_ARCHITECTURES:(STRING|UNINITIALIZED)=70(-real)?$' "$cache" ||
  fail 'existing build must target only sm_70'

cuda_root=${QPN_CUDA_ROOT:-/usr/local/cuda-12.8}
nvcc=$cuda_root/bin/nvcc
[[ -x "$nvcc" ]] || fail 'CUDA compiler not found; no package installation permitted'
"$nvcc" --version | grep -qE 'release 12\.8,' || fail 'requires pinned CUDA 12.8'
build_cuda=$(sed -n 's/^CMAKE_CUDA_COMPILER:FILEPATH=//p' "$cache" | head -1)
[[ -n "$build_cuda" && "$(readlink -f "$build_cuda")" == "$(readlink -f "$nvcc")" ]] ||
  fail 'existing build uses a different CUDA compiler'
targets=(nvfp4_qpn_d4_stage_parity nvfp4_qpn_mmq_d4_producer_parity)

if ((device)); then
  command -v nvidia-smi >/dev/null || fail 'missing nvidia-smi'
  state=${QPN_STATE_DIR:-/srv/strata-qpn}
  [[ -f "$state/gpu.lock" ]] || fail 'existing shared gpu.lock is missing; will not create/replace'
  exec 9<>"$state/gpu.lock"
  flock -n 9 || fail 'another GPU qualification holds the shared lock'
  gpu=${QPN_GPU:-0}
  [[ "$gpu" =~ ^[0-9]+$ ]] || fail 'QPN_GPU must be a single numeric device'
  capability=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap --format=csv,noheader,nounits) ||
    fail 'nvidia-smi capability query failed'
  [[ "${capability//$'\r'/}" == '7.0' ]] || fail 'requires an idle sm_70 V100'
  clients=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits) ||
    fail 'cannot verify existing GPU compute clients'
  if grep -Eq '^[[:space:]]*[0-9]+' <<<"$clients"; then
    fail 'GPU has existing compute clients; do not stop or interfere with them'
  fi
else
  printf 'M3 compile-only (no gpu.lock required, no GPU execution)\n'
fi

# Build just these two targets with the existing CMake options and CUDA cache.
# No model artifacts, old benchmark binaries or full-model executables run.
cmake --build "$build" --target "${targets[@]}" --parallel "${QPN_BUILD_JOBS:-8}"

if ((device)); then
  clients=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits) ||
    fail 'cannot revalidate GPU clients'
  if grep -Eq '^[[:space:]]*[0-9]+' <<<"$clients"; then
    fail 'GPU acquired a compute client during build'
  fi
  for target in "${targets[@]}"; do
    binary=$build/$target
    [[ -x "$binary" ]] || fail "missing freshly built parity executable: $target"
    printf 'M3 synthetic-only parity: %s\n' "$target"
    env CUDA_VISIBLE_DEVICES="$gpu" \
      STRATA_QPN_PREFILL_EXPERIMENTAL=0 \
      STRATA_MMQ_KQ=1 STRATA_PREFILL_MMQ=1 \
      STRATA_ARENA_LOCK=0 STRATA_ARENA_PIN_GIB=1 \
      STRATA_PREFILL_NVFP4=w4a8 STRATA_QUANT_GATHER=0 \
      "$binary"
  done
else
  printf 'M3 compile-only completed. No device/model qualification claimed.\n'
fi
