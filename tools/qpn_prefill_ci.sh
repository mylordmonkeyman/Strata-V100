#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${QPN_STATE_DIR:=/srv/strata-qpn}"
: "${QPN_CUDA_ROOT:=/usr/local/cuda-12.8}"
: "${QPN_GPU:=0}"
: "${QPN_FIXTURE_DIR:=/srv/strata-qpn/fixtures}"
export PATH="$QPN_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$QPN_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$QPN_GPU"
export STRATA_PREFILL_NVFP4=w4a8
mkdir -p qpn-prefill-results
exec > >(tee qpn-prefill-results/job.log) 2>&1
exec 9>"$QPN_STATE_DIR/gpu.lock"
flock -n 9
source_key=$(printf '%s' "$(pwd)" | sha256sum | cut -c1-12)
build_dir="$QPN_STATE_DIR/builds/$source_key"
cmake --build "$build_dir" --target nvfp4_volta_prefill_bench -j 8
shopt -s nullglob
fixtures=("$QPN_FIXTURE_DIR"/*.bin)
args=()
for blob in "${fixtures[@]}"; do args+=(--blob "$blob"); done
numactl --cpunodebind=0 --membind=0 python3 tools/qpn_prefill_run.py \
  "$build_dir/nvfp4_volta_prefill_bench" "${args[@]}" --out qpn-prefill-results/crossover --iterations 50
