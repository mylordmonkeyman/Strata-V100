#!/usr/bin/env bash
# Independent real-activation NVFP4 prefill crossover. All model assets are
# private on the V100 runner; only metadata/timings leave the container.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${QPN_STATE_DIR:=/srv/strata-qpn}"
: "${QPN_CUDA_ROOT:=/usr/local/cuda-12.8}"
: "${QPN_GPU:=0}"
: "${QPN_BASELINE_MANIFEST:?Pass the pinned baseline manifest.json}"
report="$(pwd)/qpn-prefill-real-results"
mkdir -p "$report"
exec > >(tee "$report/job.log") 2>&1
export PATH="$QPN_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$QPN_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$QPN_GPU"
export CUDA_MODULE_LOADING=EAGER
export OMP_NUM_THREADS=8
while IFS= read -r variable; do unset "$variable"; done < <(compgen -v STRATA_ || true)
exec 9>"$QPN_STATE_DIR/gpu.lock"
flock -n 9
nvidia-smi --id="$QPN_GPU" --query-compute-apps=pid,process_name --format=csv,noheader > "$report/processes-before.txt"
if grep -Eq '^[[:space:]]*[0-9]+,' "$report/processes-before.txt"; then
  echo 'GPU has compute clients; refusing to overlap prefill qualification.'
  exit 2
fi
sha256sum --check "$(dirname "$QPN_BASELINE_MANIFEST")/model.sha256"
state="$QPN_STATE_DIR/full-model"
py="$state/venv/bin/python"
fixtures="$state/prefill-fixtures-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}"
"$py" tools/qpn_capture_fixtures.py --pack "$state/prepared/pack" --captures "$state/captures" \
  --manifest "$QPN_BASELINE_MANIFEST" --out "$fixtures"
source_key=$(printf '%s' "$(pwd)" | sha256sum | cut -c1-12)
build="$QPN_STATE_DIR/builds/$source_key"
cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER="$QPN_CUDA_ROOT/bin/nvcc" \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_VOLTA_QPN=ON
cmake --build "$build" --target nvfp4_volta_prefill_bench --parallel 32
numactl --cpunodebind=0 --membind=0 "$py" tools/qpn_prefill_real_run.py \
  --binary "$build/nvfp4_volta_prefill_bench" --fixtures "$fixtures" --out "$report/candidates"

# Check disjoint CTA tiles under Compute Sanitizer after ordinary benchmarks.
# We retain gpu.lock and never upload model/activation bytes to Actions.
if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo 'Compute Sanitizer unavailable: fail closed'
  exit 2
fi
for t in 17 31; do
  numactl --cpunodebind=0 --membind=0 compute-sanitizer \
    --tool memcheck --error-exitcode 98 --leak-check no \
    --log-file "$report/sanitizer-t${t}.log" \
    "$build/nvfp4_volta_prefill_bench" \
    --blob "$fixtures/l0-e43.bin" --activations "$fixtures/l0-e43.f32" \
    --tokens "$t" --iterations 1 --qpn-tile 16 --validate-graph \
    > "$report/sanitizer-t${t}.json"
  echo "compute-sanitizer T=${t}: exit=0"
done
