#!/usr/bin/env bash
# Requires a completed, pinned M0 artifact manifest. No model or activation upload.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${QPN_STATE_DIR:=/srv/strata-qpn}"
: "${QPN_CUDA_ROOT:=/usr/local/cuda-12.8}"
: "${QPN_GPU:=0}"
: "${QPN_BASELINE_MANIFEST:?Pass the downloaded baseline artifact manifest.json}"
report="$(pwd)/qpn-real-results"
mkdir "$report"
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
  echo 'GPU has compute clients; refusing to overlap tests.'
  exit 2
fi
# Verify that the prepared pack still matches the exact baseline's inputs.
sha256sum --check "$(dirname "$QPN_BASELINE_MANIFEST")/model.sha256"
state="$QPN_STATE_DIR/full-model"
py="$state/venv/bin/python"
fixtures="$state/real-fixtures-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}"
"$py" tools/qpn_capture_fixtures.py --pack "$state/prepared/pack" --captures "$state/captures" \
  --manifest "$QPN_BASELINE_MANIFEST" --out "$fixtures"
source_key=$(printf '%s' "$(pwd)" | sha256sum | cut -c1-12)
build="$QPN_STATE_DIR/builds/$source_key"
cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER="$QPN_CUDA_ROOT/bin/nvcc" \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_VOLTA_QPN=ON
cmake --build "$build" --target nvfp4_volta_qpn_bench --parallel 32
numactl --cpunodebind=0 --membind=0 "$py" tools/qpn_real_run.py \
  --binary "$build/nvfp4_volta_qpn_bench" --fixtures "$fixtures" --out "$report/candidates"
