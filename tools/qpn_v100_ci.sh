#!/usr/bin/env bash
# Native Ubuntu 24.04 V100 runner. No sudo, service changes, model downloads,
# fixture mutations, or automatic GPU process termination.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -e qpn-results ]]; then
  echo 'qpn-results already exists; archive it before a manual rerun.' >&2
  exit 2
fi
mkdir qpn-results
report_dir="$(pwd)/qpn-results"
exec > >(tee "$report_dir/job.log") 2>&1

: "${QPN_CUDA_ROOT:=/usr/local/cuda-12.8}"
: "${QPN_GPU:=0}"
: "${QPN_FIXTURE_DIR:=/srv/strata-qpn/fixtures}"
: "${QPN_STATE_DIR:=/srv/strata-qpn}"
: "${QPN_SWEEP:=0}"
: "${QPN_SANITIZERS:=1}"
[[ "$QPN_SWEEP" =~ ^[01]$ && "$QPN_SANITIZERS" =~ ^[01]$ ]] || { echo 'Flags must be 0 or 1'; exit 2; }
export PATH="$QPN_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$QPN_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$QPN_GPU"
export CUDA_MODULE_LOADING=EAGER
export OMP_NUM_THREADS=8
# Do not inherit undocumented engine tuning knobs from a login shell.
while IFS= read -r variable; do unset "$variable"; done < <(compgen -v STRATA_ || true)

for program in cmake ninja c++ python3 numactl flock nvidia-smi nvcc; do
  command -v "$program" >/dev/null || { echo "Missing prerequisite: $program"; exit 2; }
done
test -x "$QPN_CUDA_ROOT/bin/nvcc"
nvcc --version | tee "$report_dir/nvcc.txt"
grep -q 'release 12\.8,' "$report_dir/nvcc.txt" || { echo 'CUDA 12.8 required'; exit 2; }

# Shared host lock (not RUNNER_TEMP) serializes manual uses of this script too.
exec 9>"$QPN_STATE_DIR/gpu.lock"
flock -n 9 || { echo 'Another QPN run holds the GPU lock'; exit 2; }
nvidia-smi --id="$QPN_GPU" --query-gpu=name,uuid,compute_cap,memory.total --format=csv,noheader > "$report_dir/gpu.txt"
python3 - "$report_dir/gpu.txt" <<'PY'
import csv
import sys
with open(sys.argv[1]) as f:
    rows = list(csv.reader(f))
if len(rows) != 1 or len(rows[0]) != 4 or rows[0][2].strip() != '7.0':
    raise SystemExit('Select exactly one sm_70 GPU using QPN_GPU')
PY
nvidia-smi --id="$QPN_GPU" --query-compute-apps=pid,process_name --format=csv,noheader > "$report_dir/processes-before.txt"
if grep -Eq '^[[:space:]]*[0-9]+,' "$report_dir/processes-before.txt"; then
  cat "$report_dir/processes-before.txt"
  echo 'GPU already has compute clients. Stop the owning service/process yourself and rerun.'
  exit 2
fi
numactl --cpunodebind=0 --membind=0 true
git rev-parse HEAD > "$report_dir/commit.txt"
cmake --version > "$report_dir/cmake.txt"
c++ --version > "$report_dir/compiler.txt"
numactl --hardware > "$report_dir/numa.txt"
nvidia-smi --id="$QPN_GPU" -q > "$report_dir/gpu-before.txt"
monitor_pid=''
cleanup() {
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
nvidia-smi --id="$QPN_GPU" --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate --format=csv --loop=1 > "$report_dir/telemetry.csv" &
monitor_pid=$!

source_key=$(printf '%s' "$(pwd)" | sha256sum | cut -c1-12)
build_dir="$QPN_STATE_DIR/builds/$source_key"
mkdir -p "$build_dir"
printf '%s\n' "$build_dir" > "$report_dir/build-directory.txt"
cmake -S . -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER="$QPN_CUDA_ROOT/bin/nvcc" \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_VOLTA_QPN=ON 2>&1 | tee "$report_dir/configure.log"
cmake --build "$build_dir" --target nvfp4_volta_qpn_bench nvfp4_qpn_host_test -j 8 \
  2>&1 | tee "$report_dir/build.log"
ctest --test-dir "$build_dir" -R '^nvfp4_qpn_host$' --output-on-failure \
  2>&1 | tee "$report_dir/host-tests.log"
python3 -m unittest discover -s tools -p test_qpn_audit.py -v 2>&1 | tee "$report_dir/audit-tests.log"

failures=0
run_checked() {
  local name="$1"
  shift
  local rc=0
  "$@" > "$report_dir/$name.log" 2>&1 || rc=$?
  cat "$report_dir/$name.log"
  printf '%s\t%d\n' "$name" "$rc" >> "$report_dir/exit-codes.tsv"
  if (( rc != 0 )); then failures=$((failures + 1)); fi
}
bench="$build_dir/nvfp4_volta_qpn_bench"
binding=(numactl --cpunodebind=0 --membind=0)
run_checked synthetic "${binding[@]}" python3 tools/qpn_run.py "$bench" \
  --out "$report_dir/synthetic" --iterations 10 --tokens 1 2 3 4 5 8 9 16 17 32

# Fixtures are immutable canonical expert blobs prepared by qpn_audit.py.
# The whole pack is neither required nor read by the runner account.
shopt -s nullglob
fixtures=("$QPN_FIXTURE_DIR"/*.bin)
provenance=("$QPN_FIXTURE_DIR"/*-audit.json)
if (( ${#provenance[@]} > 0 )); then
  mkdir -p "$report_dir/fixture-provenance"
  cp -- "${provenance[@]}" "$report_dir/fixture-provenance/"
fi
real_args=()
for blob in "${fixtures[@]}"; do real_args+=(--blob "$blob"); done
if (( ${#fixtures[@]} > 0 )); then
  if [[ "$QPN_SWEEP" == 1 ]]; then real_args+=(--sweep); fi
  run_checked real "${binding[@]}" python3 tools/qpn_run.py "$bench" "${real_args[@]}" \
    --out "$report_dir/real" --iterations 100 --tokens 1 2 3 4 5 8 9 16 17 32
  # Optional model-captured FP32 rows. No file contents are uploaded.
  if [[ -r "$QPN_FIXTURE_DIR/activations.f32" ]]; then
    run_checked real-activations "${binding[@]}" python3 tools/qpn_run.py "$bench" "${real_args[@]}" \
      --activations "$QPN_FIXTURE_DIR/activations.f32" \
      --out "$report_dir/real-activations" --iterations 100 --tokens 1 2 3 4 5 8 9 16 17 32
  fi
else
  echo 'Real-expert tests SKIPPED: no *.bin fixtures in QPN_FIXTURE_DIR.'
fi
if [[ "$QPN_SANITIZERS" == 1 ]]; then
  if [[ -x "$QPN_CUDA_ROOT/bin/compute-sanitizer" ]]; then
    sanitizer_args=()
    if (( ${#fixtures[@]} > 0 )); then sanitizer_args=(--blob "${fixtures[0]}"); fi
    for tool in memcheck racecheck initcheck; do
      for tokens in 3 8 17; do
        run_checked "$tool-t$tokens" "${binding[@]}" "$QPN_CUDA_ROOT/bin/compute-sanitizer" --tool "$tool" --error-exitcode 99 \
          "$bench" "${sanitizer_args[@]}" --tokens "$tokens" --iterations 1
      done
    done
  else
    echo 'Requested Compute Sanitizer is not installed in QPN_CUDA_ROOT/bin.'
    failures=$((failures + 1))
  fi
fi
nvidia-smi --id="$QPN_GPU" -q > "$report_dir/gpu-after.txt"
{
  echo '### V100 QPN standalone run'
  printf '\nReal expert fixtures: %d\n\nFailed test commands: %d\n' "${#fixtures[@]}" "$failures"
  echo 'See the uploaded logs for skipped tests and per-candidate measurements.'
  echo 'This does not qualify mixed-group engine integration, model KL, or prefill TTFT.'
} | tee "$report_dir/summary.md"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then cat "$report_dir/summary.md" >> "$GITHUB_STEP_SUMMARY"; fi
(( failures == 0 ))
