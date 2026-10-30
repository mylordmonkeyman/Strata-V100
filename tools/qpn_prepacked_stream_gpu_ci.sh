#!/usr/bin/env bash
# Independent grouped GPU qualification; never modifies engine, pack, caches.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${QPN_STATE_DIR:=/srv/strata-qpn}"
: "${QPN_CUDA_ROOT:=/usr/local/cuda-12.8}"
: "${QPN_GPU:=0}"
: "${QPN_MIXED_FIXTURE_DIR:=$QPN_STATE_DIR/mixed-gpu-fixtures/40908922475-1}"
: "${QPN_MIXED_PINNED_SUMMARY:=pinned-mixed-summary/summary.json}"
mkdir -p qpn-prepacked-stream-results
exec > >(tee qpn-prepacked-stream-results/job.log) 2>&1
export PATH="$QPN_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$QPN_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$QPN_GPU" CUDA_MODULE_LOADING=EAGER
export STRATA_PREFILL_NVFP4=w4a8 OMP_NUM_THREADS=8
# Prevent concurrent GPU work (including other QPN V100 jobs).
exec 9>"$QPN_STATE_DIR/gpu.lock"
flock -n 9
nvidia-smi --id="$QPN_GPU" --query-compute-apps=pid,process_name --format=csv,noheader \
  > qpn-prepacked-stream-results/processes-before.txt
if grep -Eq '^[[:space:]]*[0-9]+,' qpn-prepacked-stream-results/processes-before.txt; then
  echo 'V100 already has compute clients; refusing to benchmark' >&2; exit 2
fi
py="$QPN_STATE_DIR/full-model/venv/bin/python"
"$py" tools/qpn_mixed_gpu_verify.py \
  --private-dir "$QPN_MIXED_FIXTURE_DIR" --published-summary "$QPN_MIXED_PINNED_SUMMARY" \
  --out qpn-prepacked-stream-results/provenance.json
# Use a PRIVATE, ephemeral cache: only the verified source weight file is read.
# Never upload cache.bin or its manifest and never mutate the model/fixture.
private_cache="$(mktemp -d "${RUNNER_TEMP:-/tmp}/qpn-prepacked-XXXXXXXX")"
chmod 700 "$private_cache"
cleanup() { rm -f "$private_cache/cache.bin" "$private_cache/manifest.json" "$private_cache/"*.log; rmdir "$private_cache" || true; }
trap cleanup EXIT
sha="$("$py" - <<'PY'
import json
print(json.load(open('qpn-prepacked-stream-results/provenance.json'))['weights_sha256'])
PY
)"
"$py" tools/qpn_prepack_expert_cache.py \
  --source "$QPN_MIXED_FIXTURE_DIR/weights.bin" \
  --output "$private_cache/cache.bin" --manifest "$private_cache/manifest.json" \
  --expected-sha256 "$sha" --experts 16 > /dev/null
"$py" - "$private_cache/manifest.json" "$private_cache/cache.bin" "$sha" <<'PY'
import hashlib,json,pathlib,sys
manifest=json.load(open(sys.argv[1]))
data=pathlib.Path(sys.argv[2])
assert manifest['layout']=='strata-qpn-nvfp4-chunkmajor-v1'
assert manifest['source_sha256']==sys.argv[3]
assert manifest['experts']==16
assert (manifest['rows_N'],manifest['hidden_FF'])==(2560,640)
assert data.stat().st_size==manifest['total_bytes']==44237056
assert hashlib.sha256(data.read_bytes()).hexdigest()==manifest['cache_sha256']
print('Offline cache SHA and format verified; cache stays private')
PY
build="$QPN_STATE_DIR/builds/qpn-mixed-route"
cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER="$QPN_CUDA_ROOT/bin/nvcc" \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_VOLTA_QPN=ON
cmake --build "$build" --target nvfp4_volta_mixed_bench --parallel 16
if [[ "${QPN_SWEEP:-0}" == 1 ]]; then
  sweep_extra=()
  if [[ "${QPN_SWEEP_NATIVE_ONLY:-0}" == 1 ]]; then sweep_extra+=(--native-only); fi
  "$py" tools/qpn_prepacked_sweep.py --binary "$build/nvfp4_volta_mixed_bench" \
    --fixture "$QPN_MIXED_FIXTURE_DIR" --cache "$private_cache/cache.bin" \
    --private "$private_cache" --report qpn-prepacked-stream-results "${sweep_extra[@]}"
  exit $?
fi
args=(--weights "$QPN_MIXED_FIXTURE_DIR/weights.bin"
      --activations "$QPN_MIXED_FIXTURE_DIR/activations.f32"
      --counts "$QPN_MIXED_FIXTURE_DIR/counts.txt" --qpn-max 512 --qpn-fused-gu --qpn-chunkmajor --qpn-pinned-stream
      --qpn-packed-file "$private_cache/cache.bin" --qpn-async-pipeline)
# One CUDA event interval covers full 16-expert work, staging, gathers & launches.
for min in 129; do
  numactl --cpunodebind=0 --membind=0 "$build/nvfp4_volta_mixed_bench" \
    "${args[@]}" --qpn-min "$min" --rounds 6 --iterations 10 --graph-check \
    > "qpn-prepacked-stream-results/measurement-min${min}.json"
  "$py" -m json.tool "qpn-prepacked-stream-results/measurement-min${min}.json" >/dev/null
  "$py" - <<'PY'
import json,math
p='qpn-prepacked-stream-results/measurement-min129.json'
j=json.load(open(p))
assert j['pinned_stream'] and j['chunkmajor'] and j['fused_gu']
assert j['disk_prepacked'] and j['repack_cpu_us']==0
assert j['cache_load_us']>0
assert j['async_pipeline'] and j['pipeline_output_equal']
assert len(j['pipeline_all_mmq_us'])==len(j['pipeline_mixed_us'])==6
assert math.isfinite(j['pipeline_speedup']) and j['pipeline_speedup']>0
assert j['pipeline_all_mmq_median_us']>0 and j['pipeline_mixed_median_us']>0
assert (j['qpn_groups'],j['fallback_groups'],j['routed_rows'])==(2,14,1100)
assert j['control_h2d_bytes']==j['hybrid_h2d_bytes']==j['weights_bytes']==44237056
assert j['graph_checked'] and j['graph_changed_input_equal'] and j['graph_replay_equal']
assert j['output_tail_guard_ok'] and j['numerics_pass'] and j['oracle_rows']>=32
assert math.isfinite(j['mixed_over_mmq_speedup']) and math.isfinite(j['mixed_graph_speedup'])
assert j['repack_upload_us']==0, 'pinned stream must not upload weights during one-time preparation'
print('Equal-byte pinned H2D streaming and real mixed-workload numerical/safety gates passed')
PY
done
if ! command -v compute-sanitizer >/dev/null; then
  echo 'Compute Sanitizer unavailable: fail closed' >&2; exit 2
fi
for min in 129; do
  numactl --cpunodebind=0 --membind=0 compute-sanitizer \
    --tool memcheck --error-exitcode 98 --leak-check no \
    --log-file "qpn-prepacked-stream-results/memcheck-min${min}.log" \
    "$build/nvfp4_volta_mixed_bench" "${args[@]}" \
    --qpn-min "$min" --rounds 1 --iterations 3 --graph-check \
    > "qpn-prepacked-stream-results/memcheck-measurement-min${min}.json"
  grep -q 'ERROR SUMMARY: 0 errors' "qpn-prepacked-stream-results/memcheck-min${min}.log"
  "$py" -m json.tool "qpn-prepacked-stream-results/memcheck-measurement-min${min}.json" >/dev/null
  "$py" -c 'import json,sys; j=json.load(open(sys.argv[1])); assert j["async_pipeline"] and j["pipeline_output_equal"] and j["iterations"]>=3, "two-slot reuse not covered by memcheck"' "qpn-prepacked-stream-results/memcheck-measurement-min${min}.json"
done
# Racecheck is separate from memcheck; fail closed on shared-memory hazards.
for min in 129; do
  numactl --cpunodebind=0 --membind=0 compute-sanitizer \
    --tool racecheck --error-exitcode 98 \
    --log-file "qpn-prepacked-stream-results/racecheck-min${min}.log" \
    "$build/nvfp4_volta_mixed_bench" "${args[@]}" \
    --qpn-min "$min" --rounds 1 --iterations 3 --graph-check \
    > "qpn-prepacked-stream-results/racecheck-measurement-min${min}.json"
  grep -Eq 'RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)' "qpn-prepacked-stream-results/racecheck-min${min}.log"
  "$py" -m json.tool "qpn-prepacked-stream-results/racecheck-measurement-min${min}.json" >/dev/null
  "$py" -c 'import json,sys; j=json.load(open(sys.argv[1])); assert j["async_pipeline"] and j["pipeline_output_equal"] and j["iterations"]>=3, "two-slot reuse not covered by racecheck"' "qpn-prepacked-stream-results/racecheck-measurement-min${min}.json"
done
echo 'Real-capture equal-byte 2-stream HtoD pipeline and original Graph/FP64/memcheck/racecheck gates passed.'
