#!/usr/bin/env python3
"""Pinned old-engine baseline, bounded by the runner's 48 GiB memory budget.

This is one mixed-tier scenario, not a 128 GiB all-RAM qualification claim.
Raw activation/logit dumps stay private in the runner state directory.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def capture_summary(path):
    # Existing STRATA_DUMP_MOE_INPUT format, not a new engine instrumentation.
    import numpy as np
    with path.open('rb') as f:
        header=f.read(24)
        if len(header)!=24: raise ValueError('truncated activation header')
        t,n,k=struct.unpack('<3q',header)
        if not 0<t<=32768 or n!=2560 or not 0<k<=512: raise ValueError('invalid activation dump geometry')
        x=np.frombuffer(f.read(t*n*4),dtype='<f4').reshape(t,n)
        ids=np.frombuffer(f.read(t*k*4),dtype='<i4').reshape(t,k)
        if f.read(1) or not np.isfinite(x).all() or (ids<0).any() or (ids>=512).any(): raise ValueError('invalid activation dump')
    counts=collections.Counter(int(v) for v in ids.reshape(-1))
    return {'tokens':t,'n_embd':n,'top_k':k,'expert_counts':dict(sorted(counts.items())),
            'entry_count_histogram':dict(sorted(collections.Counter(counts.values()).items())),
            'max_abs_activation':float(np.abs(x).max()),'rms_activation':float(np.sqrt(np.mean(x.astype(np.float64)**2))),
            'activation_dump_sha256':digest(path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','binary','prepared','state','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--revision',required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    for name in ('source','binary','prepared','state','out'): setattr(a,name,getattr(a,name).resolve())
    sys.path[:0]=[str(a.source/'tools'),str(a.source)]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    template=ChatTemplate(a.prepared/'pack/tokenizer/chat_template.jinja')
    tok=Tokenizer.from_gguf(a.prepared/'dense.gguf')
    text=('A compiler translates source code into machine instructions. A processor executes instructions and reads memory. '
          'A cache reduces repeated memory access. Measurements should compare equal workloads and include all required work. ')*500
    rendered=template.render([{'role':'user','content':text+' Summarize these ideas briefly.'}],enable_thinking=False)
    full=tok.encode(rendered,parse_special=True)
    # Preserve the actual model template's prefix and generation suffix while
    # removing repeated middle-body tokens to obtain exactly 2048 input tokens.
    if len(full)<2048: raise ValueError('benchmark source prompt too short')
    tokens=full[:2016]+full[-32:]
    token_path=a.state/'prompt-2048.txt'; token_path.write_text(','.join(map(str,tokens)))
    env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
    env['STRATA_PREFILL_NVFP4']='w4a8'
    args=[str(a.binary),'--pack',str(a.prepared/'pack'),'--native',str(a.prepared/'dense.gguf'),
          '--native-dense-gguf',str(a.prepared/'dense.gguf'),'--ple-gguf',str(a.prepared/'ple-nvfp4.gguf'),
          '--tokens-file',str(token_path),'--max-context','4096','--max-new','32','--greedy',
          '--expert-cache','auto','--expert-profile',str(a.source/'data/expert-profile.bin'),
          '--resident-budget-gib','24','--pool-workers','16','--prefill','1024','--spec','0',
          '--suffix-draft','0','--adapt-every','0','--vram-reserve-mib','1536','--stats']
    manifest={'baseline_source':'2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926','model_revision':a.revision,
              'binary_sha256':digest(a.binary),'template_sha256':hashlib.sha256(template.source.encode()).hexdigest(),
              'prompt_ids_sha256':digest(token_path),'argv':args,'memory_scope':'48 GiB cgroup; 24 GiB resident expert budget; not full 128 GiB host',
              'runs':[],'m0_gate':'pending representative headroom/verify/group histograms'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    dumps=a.state/'captures'; dumps.mkdir(exist_ok=True)
    failed=0
    for i in range(6):
        trial_env=env.copy()
        logits=dumps/f'baseline-first-logits-{i}.f32'
        trial_env['STRATA_DUMP_FIRST_LOGITS']=str(logits)
        layer=(0,24,47)[i%3]
        capture=dumps/f'layer-{layer}.moe'
        if i<3:
            trial_env['STRATA_DUMP_MOE_INPUT']=str(capture); trial_env['STRATA_DUMP_MOE_LAYER']=str(layer)
        start=time.monotonic()
        with (a.out/f'baseline-{i}.log').open('w') as f:
            try:
                r=subprocess.run(['numactl','--cpunodebind=0','--membind=0',*args],env=trial_env,stdout=f,stderr=subprocess.STDOUT,timeout=1800,check=False)
                returncode=r.returncode
            except subprocess.TimeoutExpired:
                returncode=124
                f.write('\nbaseline exceeded 1800 second timeout\n')
        row={'warmup':i==0,'returncode':returncode,'wall_seconds':time.monotonic()-start}
        if logits.exists(): row['first_logits_sha256']=digest(logits)
        if i<3 and capture.exists(): row['capture']=capture_summary(capture)
        manifest['runs'].append(row); failed+=int(returncode!=0)
        (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(f'baseline repetition {i}: exit={returncode}',flush=True)
        # A failed full-model startup is shared infrastructure, not evidence
        # against an independent kernel schedule. Retain diagnostics and stop
        # repeatedly allocating the same failed full-model configuration.
        if returncode!=0: break
    return int(failed>0)


if __name__=='__main__': raise SystemExit(main())
