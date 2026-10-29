#!/usr/bin/env bash
# Independent grouped GPU qualification; never modifies engine, pack, caches.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${QPN_STATE_DIR:=/srv/strata-qpn}"
: "${QPN_CUDA_ROOT:=/usr/local/cuda-12.8}"
: "${QPN_GPU:=0}"
: "${QPN_MIXED_FIXTURE_DIR:=$QPN_STATE_DIR/mixed-gpu-fixtures/40908922475-1}"
: "${QPN_MIXED_PINNED_SUMMARY:=pinned-mixed-summary/summary.json}"
mkdir -p qpn-mixed-gpu-results
exec > >(tee qpn-mixed-gpu-results/job.log) 2>&1
export PATH="$QPN_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$QPN_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$QPN_GPU" CUDA_MODULE_LOADING=EAGER
export STRATA_PREFILL_NVFP4=w4a8 OMP_NUM_THREADS=8
# Prevent concurrent GPU work (including other QPN V100 jobs).
exec 9>"$QPN_STATE_DIR/gpu.lock"
flock -n 9
nvidia-smi --id="$QPN_GPU" --query-compute-apps=pid,process_name --format=csv,noheader \
  > qpn-mixed-gpu-results/processes-before.txt
if grep -Eq '^[[:space:]]*[0-9]+,' qpn-mixed-gpu-results/processes-before.txt; then
  echo 'V100 already has compute clients; refusing to benchmark' >&2; exit 2
fi
py="$QPN_STATE_DIR/full-model/venv/bin/python"
"$py" tools/qpn_mixed_gpu_verify.py \
  --private-dir "$QPN_MIXED_FIXTURE_DIR" --published-summary "$QPN_MIXED_PINNED_SUMMARY" \
  --out qpn-mixed-gpu-results/provenance.json
build="$QPN_STATE_DIR/builds/qpn-mixed-route"
cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER="$QPN_CUDA_ROOT/bin/nvcc" \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_VOLTA_QPN=ON
cmake --build "$build" --target nvfp4_volta_mixed_bench -j 8
args=(--weights "$QPN_MIXED_FIXTURE_DIR/weights.bin"
      --activations "$QPN_MIXED_FIXTURE_DIR/activations.f32"
      --counts "$QPN_MIXED_FIXTURE_DIR/counts.txt")
# One CUDA event interval covers full 16-expert work, staging, gathers & launches.
numactl --cpunodebind=0 --membind=0 "$build/nvfp4_volta_mixed_bench" \
  "${args[@]}" --rounds 6 --iterations 10 --graph-check \
  > qpn-mixed-gpu-results/measurement.json
"$py" -m json.tool qpn-mixed-gpu-results/measurement.json >/dev/null
if ! command -v compute-sanitizer >/dev/null; then
  echo 'Compute Sanitizer unavailable: fail closed' >&2; exit 2
fi
numactl --cpunodebind=0 --membind=0 compute-sanitizer \
  --tool memcheck --error-exitcode 98 --leak-check no \
  --log-file qpn-mixed-gpu-results/memcheck.log \
  "$build/nvfp4_volta_mixed_bench" "${args[@]}" \
  --rounds 1 --iterations 1 --graph-check \
  > qpn-mixed-gpu-results/memcheck-measurement.json
grep -q 'ERROR SUMMARY: 0 errors' qpn-mixed-gpu-results/memcheck.log
"$py" -m json.tool qpn-mixed-gpu-results/memcheck-measurement.json >/dev/null
echo 'GPU real mixed-group parity, graph and memcheck gates passed.'
