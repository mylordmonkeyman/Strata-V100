#!/usr/bin/env bash
# Dedicated, one-shot baseline branch. Model/prepared data stay in the private
# runner volume; only logs/hashes/stats are uploaded. No host-service mutation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
feature_source=$(pwd)
mkdir -p qpn-m0-results
report="$feature_source/qpn-m0-results"
exec > >(tee "$report/job.log") 2>&1
: "${QPN_STATE_DIR:=/srv/strata-qpn}"
: "${QPN_CUDA_ROOT:=/usr/local/cuda-12.8}"
: "${QPN_GPU:=0}"
export PATH="$QPN_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$QPN_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$QPN_GPU"
export OMP_NUM_THREADS=8
export CUDA_MODULE_LOADING=EAGER
exec 9>"$QPN_STATE_DIR/gpu.lock"
flock -n 9
while IFS= read -r variable; do unset "$variable"; done < <(compgen -v STRATA_ || true)
nvidia-smi --id="$QPN_GPU" --query-gpu=name,uuid,compute_cap,memory.total --format=csv,noheader > "$report/gpu.txt"
python3 - "$report/gpu.txt" <<'PYGPU'
import csv,sys
with open(sys.argv[1]) as f: rows=list(csv.reader(f))
if len(rows)!=1 or rows[0][2].strip()!='7.0': raise SystemExit('Select exactly one sm_70 GPU')
PYGPU
nvidia-smi --id="$QPN_GPU" --query-compute-apps=pid,process_name --format=csv,noheader > "$report/processes-before.txt"
if grep -Eq '^[[:space:]]*[0-9]+,' "$report/processes-before.txt"; then
  echo 'GPU has compute clients; refusing to overlap the baseline.'
  exit 2
fi
numactl --cpunodebind=0 --membind=0 true
git rev-parse HEAD > "$report/harness-commit.txt"
df -h "$QPN_STATE_DIR" > "$report/filesystem.txt"
cat /sys/fs/cgroup/memory.max > "$report/cgroup-memory-max.txt"
python3 - "$QPN_STATE_DIR" <<'PY'
import shutil,sys
if shutil.disk_usage(sys.argv[1]).free < 250*1024**3:
    raise SystemExit('Need 250 GiB free before starting full checkpoint preparation')
PY
baseline=2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926
revision=b9ce18a6887e18c089fdea8509a33e63429a3300
state="$QPN_STATE_DIR/full-model"
src="$state/baseline-src"
build="$state/baseline-build"
mkdir -p "$state"
if [[ ! -d "$src/.git" ]]; then git clone --no-checkout https://github.com/mylordmonkeyman/Strata-V100.git "$src"; fi
git -C "$src" checkout --detach "$baseline"
test "$(git -C "$src" rev-parse HEAD)" = "$baseline"
test -z "$(git -C "$src" status --porcelain --untracked-files=no)"
cmake -S "$src" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER="$QPN_CUDA_ROOT/bin/nvcc" \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF
cmake --build "$build" --target strata --parallel 32
sha256sum "$build/strata" > "$report/binary.sha256"
# The baseline's converter looks here; CMake has already fetched its exact pin.
if [[ ! -e "$src/third_party/llama.cpp" ]]; then
  ln -s "$build/_deps/strata_llamacpp-src" "$src/third_party/llama.cpp"
fi
if [[ ! -x "$state/venv/bin/python" ]]; then python3 -m venv "$state/venv"; fi
py="$state/venv/bin/python"
"$py" -m pip install 'huggingface_hub>=1,<3' jinja2 regex pyyaml
"$py" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
"$py" -m pip install -r "$src/third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
"$py" - "$state/checkpoint" "$revision" <<'PY'
from huggingface_hub import snapshot_download
import sys
snapshot_download('nvidia/Qwen3.8-Flash-Next-FP4-Local',revision=sys.argv[2],local_dir=sys.argv[1],
                  allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','LICENSE*'])
PY
# Only the three baseline assets are needed: no unused draft quantization.
mkdir -p "$state/prepared"
prepare_step() {
  local name="$1"
  shift
  if [[ ! -e "$state/prepared/$name.done" ]]; then
    numactl --cpunodebind=0 --membind=0 "$py" "$src/tools/nvfp4_local_pack.py" "$@"
    touch "$state/prepared/$name.done"
  fi
}
prepare_step ple-nvfp4.gguf ple --model "$state/checkpoint" --out "$state/prepared/ple-nvfp4.gguf"
prepare_step dense.gguf convert --dense-only --model "$state/checkpoint" --outfile "$state/prepared/dense.gguf"
prepare_step pack pack --model "$state/checkpoint" --dense-gguf "$state/prepared/dense.gguf" --out "$state/prepared/pack"
"$py" -m pip freeze > "$report/python-packages.txt"
sha256sum "$state/prepared/pack/experts.bin" "$state/prepared/dense.gguf" "$state/prepared/ple-nvfp4.gguf" > "$report/model.sha256"
"$py" "$feature_source/tools/qpn_m0_bench.py" --source "$src" --binary "$build/strata" \
  --prepared "$state/prepared" --state "$state" --out "$report" --revision "$revision"
