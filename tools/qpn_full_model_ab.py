#!/usr/bin/env python3
"""Reversible staged-prefill full-model QPN A/B. Private prompt and logits never uploaded."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

PREFILL = re.compile(r"prefill\s+(\d+)\s+tokens in\s+([\d.]+)\s+ms.*time to first token\s+([\d.]+)\s+ms")
DECODE = re.compile(r"decode\s+(\d+)\s+tokens in\s+([\d.]+)\s+ms\s+->\s+([\d.]+)\s+tok/s")
ACTIVE = re.compile(r"strata qpn-experiment: layer\s+(\d+)\s+staged groups=(\d+)\s+experts=(\d+)")
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def logits(path):
    out=array.array('f')
    with open(path,"rb") as f:out.frombytes(f.read())
    if len(out)<1024 or len(out)>1000000 or not all(math.isfinite(v) for v in out):
        raise ValueError("first-logit dump size or finiteness invalid")
    return out
def diff(a,b):
    if len(a)!=len(b):raise ValueError("first-logit output dimension changed")
    aa=[float(x) for x in a];bb=[float(x) for x in b]
    mx_a=max(aa);mx_b=max(bb)
    pa=[math.exp(v-mx_a) for v in aa];pb=[math.exp(v-mx_b) for v in bb]
    za=sum(pa);zb=sum(pb)
    kl=0.
    for x,y in zip(pa,pb):
        if x>0:kl+=(x/za)*(math.log(x/za)-math.log(max(y/zb,1e-300)))
    diff2=sum((x-y)**2 for x,y in zip(aa,bb))
    base2=sum(x*x for x in aa)
    ia=max(range(len(aa)),key=lambda k:aa[k])
    ib=max(range(len(bb)),key=lambda k:bb[k])
    return {'n_logits':len(a),'relative_l2':math.sqrt(diff2/base2) if base2 else float("inf"),
            'kl_base_vs_candidate':max(0,kl),'argmax_match':ia==ib,
            'argmax_base':ia,'argmax_candidate':ib}
def safe_failure_diagnostics(log):
    """Keep raw model/CLI logs private; upload only typed startup failure hints."""
    raw=Path(log).read_text(errors="replace")
    lines=raw.splitlines()
    keys={
      "usage":r"\busage\b",
      "unknown_argument":r"unknown (?:argument|option)|unrecognized arguments|unrecognized option",
      "invalid_argument":r"invalid (?:argument|value|option|prefill|token|context)",
      "missing_required":r"(?:missing|required|must provide|must specify)",
      "model_load":r"(?i)model.*(?:load|open|fail)|(?:fail|cannot).*model",
      "cuda_allocation":r"out of memory|cudaMalloc|memory allocation|insufficient memory",
      "prefill_constraint":r"prefill.{0,60}(?:range|multiple|power|invalid|supported|between|must|requires)",
      "context_constraint":r"context.{0,60}(?:max|invalid|must|too|exceeds)",
      "pack_error":r"pack.{0,60}(?:missing|invalid|not found|failed)",
      "token_parse":r"tokens-file.{0,70}(?:invalid|failed|missing|not found)",
      "exception":r"(?:exception|traceback|terminate called|fatal)",
    }
    counts={k:len(re.findall(v,raw,re.IGNORECASE)) for k,v in keys.items()}
    # Log line templates can be printed without filenames, token values or
    # prompt text; expose only an allowlisted diagnostic phrase.
    safe_patterns=[
       ("unsupported_prefill","prefill",("unsupported","must be","multiple of","power of","invalid","range")),
       ("unknown_flag","unknown option",()),
       ("unrecognized_flag","unrecognized",("option","argument")),
       ("missing_pack","expert pack",("missing","not found","invalid")),
       ("out_of_memory","out of memory",()),
       ("device_failure","cuda",("failed","error")),
    ]
    categories=[]
    for line in lines:
        lower=line.lower()
        for name,term,quals in safe_patterns:
            if term in lower and (not quals or any(q in lower for q in quals)):
                if name not in categories:categories.append(name)
    return {"exit_log_sha256":sha(log),"log_lines":len(lines),
            "log_bytes":Path(log).stat().st_size,
            "pattern_counts":counts,"safe_categories":categories,
            "first_line_length":len(lines[0]) if lines else 0}

def read_summary(log):
    # Full raw logs, including prompt IDs, stay in private runner-only state.
    text=Path(log).read_text(errors="replace")
    p=PREFILL.findall(text);d=DECODE.findall(text)
    groups=ACTIVE.findall(text)
    return {'prefill_tokens':int(p[-1][0]) if p else None,
            'prefill_ms':float(p[-1][1]) if p else None,
            'ttft_ms':float(p[-1][2]) if p else None,
            'decode_toks_per_s':float(d[-1][2]) if d else None,
            'qpn_reported_layers':len(groups),
            'qpn_groups':sum(int(g[1]) for g in groups),
            'qpn_experts':sum(int(g[2]) for g in groups),
            'error_lines':sum("error" in line.lower() for line in text.splitlines())}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","binary","prepared","state","report"):p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--prefill",type=int,default=768)
    p.add_argument("--runs",type=int,default=4)
    a=p.parse_args()
    if a.prefill!=768 or a.runs!=4:raise ValueError("only pinned 768-token ABBA experiment qualified")
    a.report.mkdir(parents=True,exist_ok=True)
    a.state.mkdir(parents=True,exist_ok=True)
    private=a.state/"qpn-opt-in-ab";private.mkdir(parents=True,exist_ok=True)
    tokenfile=a.state/"prompt-2048.txt"
    if not tokenfile.is_file() or not (a.prepared/"pack/experts.bin").is_file():
        raise ValueError("the pinned private prepared model / prompt is missing")
    env={k:v for k,v in os.environ.items() if not k.startswith("STRATA_")}
    env.update({"STRATA_PREFILL_NVFP4":"w4a8",
                "STRATA_PREFILL_CPU_SHARE":"0",
                "STRATA_MMQ_DIRECT":"0",
                "STRATA_MMQ_RESIDENT_SORT_NE":"0",
                "STRATA_MMQ_GROUP_SORT":"0",
                "STRATA_PREFILL_GROUP_GATHER":"1",
                "STRATA_PREFILL_STREAM_MIN":"1024"})
    cmd=[str(a.binary),"--pack",str(a.prepared/"pack"),"--native",str(a.prepared/"dense.gguf"),
         "--native-dense-gguf",str(a.prepared/"dense.gguf"),"--ple-gguf",str(a.prepared/"ple-nvfp4.gguf"),
         "--tokens-file",str(tokenfile),"--max-context","4096","--max-new","16","--greedy",
         "--expert-cache","auto","--expert-profile",str(a.source/"data/expert-profile.bin"),
         "--resident-budget-gib","24","--pool-workers","16","--prefill","768","--spec","0",
         "--suffix-draft","0","--adapt-every","0","--vram-reserve-mib","1536","--stats"]
    summary={'schema':1,'scope':'staged 768-row full-model ABBA; not default 1024 stream-all path',
             'source_commit':subprocess.check_output(["git","-C",str(a.source),"rev-parse","HEAD"],text=True).strip(),
             'model_sha256':sha(a.prepared/"pack/experts.bin"),
             'binary_sha256':sha(a.binary),'prompt_sha256':sha(tokenfile),
             'control_flags':{k:v for k,v in env.items() if k.startswith("STRATA_")},
             'runs':[],'comparisons':[],'status':'pending'}
    outfile=a.report/"summary.json"
    def persist():outfile.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    persist()
    for idx,mode in enumerate(("mmq","qpn","qpn","mmq")):
        trialenv=env.copy()
        trialenv['STRATA_QPN_PREFILL_EXPERIMENTAL']='1' if mode=='qpn' else '0'
        raw_log=private/f"{idx}-{mode}.log"
        raw_logits=private/f"{idx}-{mode}.logits.f32"
        if raw_log.exists():raw_log.unlink()
        if raw_logits.exists():raw_logits.unlink()
        trialenv['STRATA_DUMP_FIRST_LOGITS']=str(raw_logits)
        started=time.monotonic()
        code=None
        with raw_log.open("w") as out:
            try:
                x=subprocess.run(["numactl","--cpunodebind=0","--membind=0",*cmd],
                                 env=trialenv,stdout=out,stderr=subprocess.STDOUT,
                                 check=False,timeout=1100)
                code=x.returncode
            except subprocess.TimeoutExpired:
                code=124;out.write("Full model timed out\n")
        row={'index':idx,'mode':mode,'returncode':code,'wall_s':time.monotonic()-started,
             'raw_log_private':True,'raw_logits_private':True}
        row.update(read_summary(raw_log))
        if code!=0:row['startup_diagnostics']=safe_failure_diagnostics(raw_log)
        if code==0 and raw_logits.is_file():
            row['first_logits_sha256']=sha(raw_logits)
            row['first_logits_bytes']=raw_logits.stat().st_size
        summary['runs'].append(row)
        persist()
        print(f"full-model ABBA {idx}: {mode} exit={code} TTFT={row['ttft_ms']} ms QPN groups={row['qpn_groups']}",flush=True)
        if(code!=0 or row['ttft_ms'] is None or row['decode_toks_per_s'] is None or
           not raw_logits.is_file() or (mode=="qpn" and row['qpn_groups']==0) or
           (mode=="mmq" and row['qpn_groups']!=0)):
            summary['status']='failed_execution_or_unexercised_kernel'
            persist()
            return 2
    for bi,qi in ((0,1),(3,2)):
        b=summary['runs'][bi];q=summary['runs'][qi]
        metrics=diff(logits(private/f"{bi}-mmq.logits.f32"),logits(private/f"{qi}-qpn.logits.f32"))
        metrics.update({'mmq_index':bi,'qpn_index':qi,
                        'mmq_ttft_ms':b['ttft_ms'],'qpn_ttft_ms':q['ttft_ms'],
                        'ttft_speedup':b['ttft_ms']/q['ttft_ms'],
                        'mmq_prefill_ms':b['prefill_ms'],'qpn_prefill_ms':q['prefill_ms'],
                        'prefill_speedup':b['prefill_ms']/q['prefill_ms'],
                        'mmq_decode_toks_per_s':b['decode_toks_per_s'],
                        'qpn_decode_toks_per_s':q['decode_toks_per_s']})
        summary['comparisons'].append(metrics)
    # Strict numerical guard: no quality claim on first logit mismatch. Distinguish
    # slower model from incorrect model; a regression is NOT a CI failure.
    summary['status']='pass' if all(x['kl_base_vs_candidate']<0.03 and x['argmax_match']
                                  for x in summary['comparisons']) else 'failed_first_logits_quality'
    persist()
    return 0 if summary['status']=='pass' else 3
if __name__=="__main__":raise SystemExit(main())
