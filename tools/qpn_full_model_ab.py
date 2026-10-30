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
import statistics
import subprocess
import time
import tempfile

# User's previously verified NaN benchmark recipe. Sanitization must preserve
# these explicitly; all compared arms receive the same controlled settings.
NAN_WORKAROUND = {
    "STRATA_MMQ_KQ": "1",
    "STRATA_PREFILL_MMQ": "1",
    "STRATA_ARENA_LOCK": "0",
    "STRATA_ARENA_PIN_GIB": "1",
}

def benchmark_env(environ=None):
    source = os.environ if environ is None else environ
    env = {k:v for k,v in source.items() if not k.startswith("STRATA_")}
    env.update(NAN_WORKAROUND)
    return env

PREFILL = re.compile(r"prefill\s+(\d+)\s+tokens in\s+([\d.]+)\s+ms.*time to first token\s+([\d.]+)\s+ms")
DECODE = re.compile(r"decode\s+(\d+)\s+tokens in\s+([\d.]+)\s+ms\s+->\s+([\d.]+)\s+tok/s")
PLAN = re.compile(r"strata qpn-plan: layer\s+(\d+)\s+staged groups=(\d+)\s+experts=(\d+)\s+execute=(0|1)")
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
    # The control exited before the model was loaded (four messages). Expose
    # only the final startup diagnostic template after replacing paths, URIs,
    # credentials and user-supplied token contents, not the raw log.
    last=""
    if lines and len(lines)<=12:
        rawlast=lines[-1].strip()[:1000]
        if rawlast.startswith(("strata ","strata:","strata generate:","FileExpertSource:")):
            rawlast=re.sub(r"https?://\S+","<URL>",rawlast)
            rawlast=re.sub(r"(?:(?:/|~[/\\])[^ \t,:;()\]\[{}]+)","<private-path>",rawlast)
            rawlast=re.sub(r"(?i)(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]+","<SECRET>",rawlast)
            rawlast=re.sub(r"[\w.+-]+@[\w.-]+","<email>",rawlast)
            rawlast=re.sub(r"\b[a-f0-9]{40,}\b","<hex-digest>",rawlast,flags=re.IGNORECASE)
            last=rawlast[:320]
    return {"exit_log_sha256":sha(log),"log_lines":len(lines),
            "log_bytes":Path(log).stat().st_size,
            "pattern_counts":counts,"safe_categories":categories,
            "redacted_last_startup_message":last,
            "first_line_length":len(lines[0]) if lines else 0}

def read_summary(log):
    # Full raw logs, including prompt IDs, stay in private runner-only state.
    text=Path(log).read_text(errors="replace")
    p=PREFILL.findall(text);d=DECODE.findall(text)
    groups=PLAN.findall(text)
    execution=[g for g in groups if g[3]=="1"]
    return {'prefill_tokens':int(p[-1][0]) if p else None,
            'prefill_ms':float(p[-1][1]) if p else None,
            'ttft_ms':float(p[-1][2]) if p else None,
            'decode_toks_per_s':float(d[-1][2]) if d else None,
            'qpn_plan_layers':len(groups),
            'qpn_plan_groups':sum(int(g[1]) for g in groups),
            'qpn_plan_experts':sum(int(g[2]) for g in groups),
            'qpn_reported_layers':len(execution),
            'qpn_groups':sum(int(g[1]) for g in execution),
            'qpn_experts':sum(int(g[2]) for g in execution),
            'error_lines':sum("error" in line.lower() for line in text.splitlines())}
def ab_modes(fusion_ab, batched_ab=False, batched_mmq_ab=False):
    if sum((bool(fusion_ab),bool(batched_ab),bool(batched_mmq_ab)))>1:
        raise ValueError("ABBA control modes are mutually exclusive")
    if batched_mmq_ab:
        return ("mmq", "qpn-batched", "qpn-batched", "mmq")
    if batched_ab:
        return ("qpn-fused", "qpn-batched", "qpn-batched", "qpn-fused")
    return ("qpn-base", "qpn-fused", "qpn-fused", "qpn-base") if fusion_ab else ("mmq", "qpn-fused", "qpn-fused", "mmq")


def classify_quality(comparisons, performance_only):
    # This boolean tests the first-logit threshold only, not model qualification.
    quality_ok=all(x["kl_base_vs_candidate"]<0.03 and x["argmax_match"] for x in comparisons)
    if quality_ok:
        return ("performance_only_quality_observed" if performance_only else "pass"), quality_ok
    return ("performance_only_quality_unqualified" if performance_only else "failed_first_logits_quality"), quality_ok


def record_quality(summary, comparisons, performance_only):
    status, first_logit_ok = classify_quality(comparisons, performance_only)
    summary.update(status=status, first_logit_acceptance_passed=first_logit_ok,
                   numerical_quality_qualified=False)
    # Two first-logit comparisons cannot certify full-model numerical quality.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","binary","prepared","state","report"):p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--prefill",type=int,default=768)
    p.add_argument("--runs",type=int,default=4)
    p.add_argument("--packed-cache",type=Path)
    p.add_argument("--qpn-range",default="17:32")
    choices=p.add_mutually_exclusive_group()
    choices.add_argument("--fusion-ab",action="store_true",
                   help="A=original QPN, B=fused per-expert QPN")
    choices.add_argument("--batched-ab",action="store_true",
                   help="A=fused per-expert QPN, B=grouped padded QPN")
    choices.add_argument("--batched-mmq-ab",action="store_true",
                   help="A=original MMQ fallback, B=grouped QPN (both using matched route plan)")
    p.add_argument("--performance-only",action="store_true",
                   help="record first-logit KL and top-1, but do not fail A/B solely on model-quality threshold")
    a=p.parse_args()
    low,high=map(int,a.qpn_range.split(':'))
    if a.runs!=4 or a.prefill not in ((768,1024) if a.packed_cache else (768,)):
        raise ValueError("unsupported pinned ABBA geometry")
    if low<1 or high<low or high>512 or (not a.packed_cache and (low,high)!=(17,32)):
        raise ValueError("unsupported QPN row range")
    if a.packed_cache and (a.fusion_ab or a.batched_ab or a.batched_mmq_ab):
        raise ValueError("packed model experiment uses per-expert fused QPN versus MMQ")
    a.report.mkdir(parents=True,exist_ok=True)
    a.state.mkdir(parents=True,exist_ok=True)
    private_context=tempfile.TemporaryDirectory(prefix="qpn-model-ab-",dir=a.state)
    private=Path(private_context.name);os.chmod(private,0o700)
    tokenfile=a.state/"prompt-2048.txt"
    if not tokenfile.is_file() or not (a.prepared/"pack/experts.bin").is_file():
        raise ValueError("the pinned private prepared model / prompt is missing")
    cache_meta=None
    if a.packed_cache:
        meta=json.loads((a.packed_cache/'manifest.json').read_text())
        source=a.prepared/'pack/experts.bin';packed=a.packed_cache/'experts.bin'
        source_hash=sha(source)
        if (source_hash!='1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5' or
            sha(tokenfile)!='df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414' or
            meta['layout']!='strata-qpn-nvfp4-chunkmajor-v1' or meta['version']!=1 or
            meta['source_sha256']!=source_hash or meta['experts']!=48*512 or
            (meta['rows_N'],meta['hidden_FF'],meta['bytes_per_expert'])!=(2560,640,2764816) or
            meta['total_bytes']!=source.stat().st_size or packed.is_symlink() or
            packed.stat().st_size!=meta['total_bytes'] or sha(packed)!=meta['cache_sha256']):
            raise ValueError('model packed cache/source SHA or geometry mismatch')
        cache_meta={k:meta[k] for k in ('layout','version','source_sha256','cache_sha256','experts','total_bytes')}
    env=benchmark_env()
    env.update({"STRATA_PREFILL_NVFP4":"w4a8",
                "STRATA_QPN_PREFILL_PLAN_CONTROL":"1",  # both arms get identical host expert order
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
         "--resident-budget-gib","24","--pool-workers","16","--prefill",str(a.prefill),"--spec","2",
         "--suffix-draft","0","--adapt-every","0","--vram-reserve-mib","1536","--stats"]
    if a.packed_cache:
        env.update(STRATA_QPN_PREFILL_PACKED_DIR=str(a.packed_cache),
                   STRATA_QPN_PREFILL_MIN_ROWS=str(low),STRATA_QPN_PREFILL_MAX_ROWS=str(high),
                   STRATA_PREFILL_STREAM_MIN='4096')
    modes=ab_modes(a.fusion_ab,a.batched_ab,a.batched_mmq_ab)
    summary={'schema':2,'scope':'performance-focused staged 768-row full-model ABBA; not default 1024 stream-all path',
             'control_mode':modes[0],'candidate_mode':modes[1],
             'performance_only':a.performance_only,'numerical_quality_qualified':False,
             'source_commit':subprocess.check_output(["git","-C",str(a.source),"rev-parse","HEAD"],text=True).strip(),
             'model_sha256':sha(a.prepared/"pack/experts.bin"),
             'binary_sha256':sha(a.binary),'prompt_sha256':sha(tokenfile),
             'control_flags':{k:('<private-cache>' if k=='STRATA_QPN_PREFILL_PACKED_DIR' else v) for k,v in env.items() if k.startswith("STRATA_")},
             'runs':[],'comparisons':[],'status':'pending'}
    if a.packed_cache:
        summary.update(scope='actual full-model staged prefill, nonresident prepacked QPN versus matched-order MMQ',
                       packed_cache=cache_meta,qpn_range=[low,high],prefill_chunk=a.prefill,
                       qpn_decode_executed=False,
                       limitations=['nonresident QPN only; canonical resident experts stay MMQ',
                                    'raw FP16 QPN input; D4 MMQ alignment and model quality unqualified',
                                    'decode uses original backend in both arms; decode TPS is diagnostic only',
                                    'fresh processes; uncontrolled page-cache temperature; no warm-engine claim',
                                    'full-model graph and sanitizer qualification not established'])
    outfile=a.report/"summary.json"
    def persist():outfile.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    persist()
    for idx,mode in enumerate(modes):
        trialenv=env.copy()
        trialenv['STRATA_QPN_PREFILL_EXPERIMENTAL']='0' if mode=='mmq' else '1'
        trialenv['STRATA_QPN_PREFILL_FUSED_GU']='1' if a.packed_cache or mode in ('qpn-fused','qpn-batched') else '0'
        trialenv['STRATA_QPN_PREFILL_BATCHED']='1' if mode=='qpn-batched' else '0'
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
        if re.search(r'non[- ]finite|illegal memory access|CUDA error|device-side assert',raw_log.read_text(errors='replace'),re.I):
            summary['status']='failed_safety';summary['runs'].append(row);persist();return 2
        if code!=0:row['startup_diagnostics']=safe_failure_diagnostics(raw_log)
        if code==0 and raw_logits.is_file():
            row['first_logits_sha256']=sha(raw_logits)
            row['first_logits_bytes']=raw_logits.stat().st_size
        summary['runs'].append(row)
        persist()
        print(f"full-model ABBA {idx}: {mode} exit={code} TTFT={row['ttft_ms']} ms QPN groups={row['qpn_groups']}",flush=True)
        if(code!=0 or row['prefill_tokens']!=2047 or row['ttft_ms'] is None or row['decode_toks_per_s'] is None or
           not raw_logits.is_file() or row['qpn_plan_groups']==0 or
           (mode!="mmq" and row['qpn_groups']==0) or
           (mode=="mmq" and row['qpn_groups']!=0)):
            summary['status']='failed_execution_or_unexercised_kernel'
            persist()
            return 2
    if a.packed_cache:
        summary['within_arm_comparisons']=[]
        for i,j in ((0,3),(1,2)):
            x=logits(private/f"{i}-{modes[i]}.logits.f32")
            y=logits(private/f"{j}-{modes[j]}.logits.f32")
            check={k:v for k,v in diff(x,y).items() if k not in ('argmax_base','argmax_candidate')}
            check.update(mode=modes[i],bitwise_equal=x.tobytes()==y.tobytes())
            summary['within_arm_comparisons'].append(check)
        if not all(r['bitwise_equal'] for r in summary['within_arm_comparisons']):
            summary['status']='failed_within_arm_repeatability';persist();return 3
    for bi,qi in ((0,1),(3,2)):
        b=summary["runs"][bi]
        q=summary["runs"][qi]
        metrics=diff(logits(private/f"{bi}-{modes[bi]}.logits.f32"),
                     logits(private/f"{qi}-{modes[qi]}.logits.f32"))
        metrics={k:v for k,v in metrics.items() if k not in ("argmax_base","argmax_candidate")}
        metrics.update({"control_index":bi,"candidate_index":qi,
                        "control_mode":modes[bi],"candidate_mode":modes[qi],
                        "control_ttft_ms":b["ttft_ms"],"candidate_ttft_ms":q["ttft_ms"],
                        "ttft_speedup":b["ttft_ms"]/q["ttft_ms"],
                        "control_prefill_ms":b["prefill_ms"],"candidate_prefill_ms":q["prefill_ms"],
                        "prefill_speedup":b["prefill_ms"]/q["prefill_ms"],
                        "control_decode_toks_per_s":b["decode_toks_per_s"],
                        "candidate_decode_toks_per_s":q["decode_toks_per_s"]})
        summary["comparisons"].append(metrics)
    summary["median_prefill_ms"]={
        "control":statistics.median(r["prefill_ms"] for r in summary["runs"] if r["mode"]==modes[0]),
        "candidate":statistics.median(r["prefill_ms"] for r in summary["runs"] if r["mode"]==modes[1])}
    c=summary["median_prefill_ms"]
    summary["median_prefill_speedup"]=c["control"]/c["candidate"]
    record_quality(summary,summary["comparisons"],a.performance_only)
    # Performance-only mode still fails closed on startup, missing QPN execution,
    # absent logits, or nonfinite numerical output. It simply records an
    # otherwise failing first-logit acceptance criterion as unqualified.
    persist()
    return 0 if a.performance_only or summary["status"]=="pass" else 3
if __name__=="__main__":raise SystemExit(main())
