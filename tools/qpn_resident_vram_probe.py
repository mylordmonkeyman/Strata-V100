#!/usr/bin/env python3
"""Read-only GPU-memory headroom probe during pinned MMQ full-model inference.

Private prompt, logs and logits stay in a mode-0700 temporary directory.
No expert cache, model data, production default, or persistent state is changed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

PINNED_MODEL = '1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5'
PINNED_PROMPT = 'df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414'


def parse_gpu_memory_csv(raw):
    # nvidia-smi: memory.free, memory.used, memory.total, in MiB.
    line = raw.strip()
    if not re.fullmatch(r'\d+\s*,\s*\d+\s*,\s*\d+', line):
        raise ValueError('unexpected GPU memory telemetry')
    free, used, total = (int(x) for x in line.split(','))
    if total < 16000 or total > 140000 or free < 0 or used < 0 or free + used > total + 512:
        raise ValueError('inconsistent GPU memory telemetry')
    return (free, used, total)


def summarize_samples(rows):
    if len(rows) < 10:
        raise ValueError('insufficient live process samples')
    if len(set(x[2] for x in rows)) != 1:
        raise ValueError('GPU total capacity changed across samples')
    return {'samples': len(rows), 'total_mib': rows[0][2],
            'min_free_mib': min(x[0] for x in rows),
            'max_used_mib': max(x[1] for x in rows),
            'latest_free_mib': rows[-1][0],
            'peak_headroom_after_512mib_guard_mib': max(0, min(x[0] for x in rows)-512),
            'measurement': 'NVML sampled across startup, prefill and decode; not exact allocator headroom'}


def run(args):
    from qpn_full_model_ab import sha, logits, read_summary, benchmark_env
    state, prepared, source = args.state, args.prepared, args.source
    tokenfile = state/'prompt-2048.txt'
    source_pack = prepared/'pack/experts.bin'
    for path in (tokenfile, source_pack, prepared/'dense.gguf', prepared/'ple-nvfp4.gguf', args.binary):
        if not path.is_file():
            raise ValueError('missing pinned private model input or compiled binary')
    if sha(source_pack) != PINNED_MODEL or sha(tokenfile) != PINNED_PROMPT:
        raise ValueError('pinned source pack or prompt digest mismatch')
    env = benchmark_env()
    env.update(STRATA_PREFILL_NVFP4='w4a8', STRATA_PREFILL_CPU_SHARE='0',
               STRATA_QPN_PREFILL_EXPERIMENTAL='0', STRATA_PREFILL_STREAM_MIN='4096',
               STRATA_PREFILL_GROUP_GATHER='1')
    cmd = [str(args.binary),'--pack',str(prepared/'pack'),'--native',str(prepared/'dense.gguf'),
           '--native-dense-gguf',str(prepared/'dense.gguf'),'--ple-gguf',str(prepared/'ple-nvfp4.gguf'),
           '--tokens-file',str(tokenfile),'--max-context','4096','--max-new','16','--greedy',
           '--expert-cache','auto','--expert-profile',str(source/'data/expert-profile.bin'),
           '--resident-budget-gib','24','--pool-workers','16','--prefill','1024','--spec','2',
           '--suffix-draft','0','--adapt-every','0','--vram-reserve-mib','1536','--stats']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='qpn-vram-private-', dir=state) as private_dir:
        os.chmod(private_dir, 0o700)
        private = Path(private_dir)
        log, logits_file = private/'model.log', private/'first_logits.f32'
        env['STRATA_DUMP_FIRST_LOGITS'] = str(logits_file)
        samples = []
        start = time.monotonic()
        with log.open('w') as f:
            process = subprocess.Popen(['numactl','--cpunodebind=0','--membind=0',*cmd],
                                       env=env, stdout=f, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    if time.monotonic()-start > args.timeout:
                        process.kill()
                        raise TimeoutError('pinned model timed out')
                    read = subprocess.run(['nvidia-smi','-i','0',
                        '--query-gpu=memory.free,memory.used,memory.total',
                        '--format=csv,noheader,nounits'], capture_output=True,text=True,timeout=10)
                    if read.returncode:
                        raise RuntimeError('GPU telemetry query failed')
                    samples.append(parse_gpu_memory_csv(read.stdout))
                    time.sleep(args.interval)
                code = process.wait(timeout=20)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        if code:
            raise RuntimeError('MMQ model returned nonzero (private log not uploaded)')
        private_log = log.read_text(errors='replace')
        if re.search(r'non[- ]finite|illegal memory access|CUDA error|device-side assert',
                     private_log, re.I):
            raise RuntimeError('MMQ model safety signal in private log')
        model = read_summary(log)
        if model['prefill_tokens'] != 2047 or model['qpn_groups'] != 0 or model['prefill_ms'] is None:
            raise RuntimeError('missing full model work or QPN executed unexpectedly')
        logits(logits_file)  # all values finite; raw arrays stay private
        mem = summarize_samples(samples)
        metadata = {'schema':1,
            'scope':'read-only pinned staged MMQ V100 memory-headroom observation',
            'model_sha256':PINNED_MODEL, 'prompt_sha256':PINNED_PROMPT,
            'binary_sha256':sha(args.binary),
            'source_commit':subprocess.check_output(
                ['git','-C',str(source),'rev-parse','HEAD'], text=True).strip(),
            'runtime_wall_s':round(time.monotonic()-start,3),
            'prefill_tokens':model['prefill_tokens'],
            'prefill_ms':model['prefill_ms'], 'qpn_groups':model['qpn_groups'],
            'headroom':mem, 'packed_cache_allocated':False,
            'control_flags':{k:env[k] for k in
                ('STRATA_MMQ_KQ','STRATA_PREFILL_MMQ','STRATA_ARENA_LOCK',
                 'STRATA_ARENA_PIN_GIB','STRATA_PREFILL_STREAM_MIN',
                 'STRATA_QPN_PREFILL_EXPERIMENTAL')},
            'performance_qualified':False, 'numerical_quality_qualified':False,
            'limitations':['sampled memory is not an allocator guarantee or warm-serving headroom',
                           'no QPN executed, no packed cache allocated, no throughput comparison',
                           'any packed cache must retain KV growth, ring/cache, and safety reserves']}
        args.report.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
        return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','state','prepared','binary','report'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--interval',type=float,default=0.35)
    p.add_argument('--timeout',type=int,default=900)
    args=p.parse_args()
    if not math.isfinite(args.interval) or not 0.1 <= args.interval <= 2:
        p.error('interval must be in [0.1,2.0] seconds')
    if args.timeout<60 or args.timeout>1800:
        p.error('invalid timeout')
    metadata=run(args)
    print('pinned MMQ memory sampling complete:',metadata['headroom'],flush=True)


if __name__=='__main__':
    main()
