#!/usr/bin/env python3
"""One pinned, private MMQ R1 prefill-capture parity probe; no QPN dispatch.

Produces ONLY allowlisted scalar metrics. Raw per-expert routing IDs, logits,
tokens and logs remain in an owner-only temporary directory and are deleted.
This is a sampling mechanism qualification, not R1 heldout or model performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from qpn_full_model_ab import benchmark_env, sha, logits, diff, read_summary
from qpn_r1_prefill_convert import decode_private_prefill

MODEL = '1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5'
PROMPT = 'df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414'


def summarize_capture(sample):
    chunks = sample["prefill"]
    if len(chunks) != 2 or [c["tokens"] for c in chunks] != [1024,1023] or sample["decode"]:
        raise ValueError("private probe must contain 1024/1023 prefill, no decode")
    values = dict(prefill_chunks=2, recorded_layers=0, routed_entries=0,
                  active_expert_uses=0, resident_expert_uses=0,
                  resident_routed_entries=0,
                  optimistic_complete_resident_16_groups_33_512=0,
                  optimistic_complete_resident_16_group_rows_33_512=0,
                  distinct_resident_expert_uses=0)
    residents = set()
    for chunk in chunks:
        if len(chunk["layers"]) != 48 or chunk["stream_all"]:
            raise ValueError("private probe requires forced-staged complete layers")
        for layer in chunk["layers"]:
            ids = {int(e):int(v) for e,v in layer["counts"]}
            resident = set(layer["resident_experts"])
            if sum(ids.values()) != chunk["tokens"] * 10:
                raise ValueError("private route work mismatch")
            values["recorded_layers"] += 1
            values["routed_entries"] += sum(ids.values())
            values["active_expert_uses"] += len(ids)
            values["resident_expert_uses"] += len(resident.intersection(ids))
            values["resident_routed_entries"] += sum(rows for e,rows in ids.items() if e in resident)
            residents.update((layer["layer"], e) for e in resident.intersection(ids))
            eligible = sorted((rows for e,rows in ids.items()
                               if e in resident and 33 <= rows <= 512),reverse=True)
            n = len(eligible)//16
            values["optimistic_complete_resident_16_groups_33_512"] += n
            values["optimistic_complete_resident_16_group_rows_33_512"] += sum(eligible[:16*n])
    values["distinct_resident_expert_uses"] = len(residents)
    return values


def run(a):
    state = a.state
    tok=state/"prompt-2048.txt"
    pack=a.prepared/"pack/experts.bin"
    for f in (tok,pack,a.binary,a.prepared/"dense.gguf",a.prepared/"ple-nvfp4.gguf"):
        if not f.is_file():raise ValueError("missing pinned private input or binary")
    if sha(tok)!=PROMPT or sha(pack)!=MODEL:
        raise ValueError("private prompt or model SHA mismatch")
    base=benchmark_env()
    base.update(STRATA_PREFILL_NVFP4="w4a8",STRATA_PREFILL_CPU_SHARE="0",
                STRATA_QPN_PREFILL_EXPERIMENTAL="0",
                STRATA_PREFILL_STREAM_MIN="4096",
                STRATA_PREFILL_GROUP_GATHER="1")
    command=[str(a.binary),"--pack",str(a.prepared/"pack"),
             "--native",str(a.prepared/"dense.gguf"),
             "--native-dense-gguf",str(a.prepared/"dense.gguf"),
             "--ple-gguf",str(a.prepared/"ple-nvfp4.gguf"),
             "--tokens-file",str(tok),"--max-context","4096",
             "--max-new","16","--greedy","--expert-cache","auto",
             "--expert-profile",str(a.source/"data/expert-profile.bin"),
             "--resident-budget-gib","24","--pool-workers","16",
             "--prefill","1024","--spec","2","--suffix-draft","0",
             "--adapt-every","0","--vram-reserve-mib","1536","--stats"]
    a.report.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qpn-private-r1-capture-",dir=state) as d:
        os.chmod(d,0o700)
        private=Path(d)
        records=[]
        for enabled in (False,True):
            index=len(records)
            logfile=private/f"arm-{index}.log"
            logit_path=private/f"arm-{index}.f32"
            env=base.copy()
            env["STRATA_DUMP_FIRST_LOGITS"]=str(logit_path)
            capture_path=private/"r1-prefill.qpr1"
            if enabled:env["STRATA_QPN_R1_PRIVATE_PREFILL_CAPTURE"]=str(capture_path)
            with logfile.open("w") as f:
                p=subprocess.run(["numactl","--cpunodebind=0","--membind=0",*command],
                                 env=env,stdout=f,stderr=subprocess.STDOUT,
                                 timeout=900,check=False)
            if p.returncode:raise RuntimeError("pinned MMQ arm failed; private log retained only until cleanup")
            raw=logfile.read_text(errors="replace")
            if re.search(r"non[- ]finite|illegal memory access|CUDA error|device-side assert",raw,re.I):
                raise RuntimeError("CUDA or nonfinite failure in private MMQ log")
            result=read_summary(logfile)
            if result["prefill_tokens"]!=2047 or result["qpn_groups"]!=0 or result["prefill_ms"] is None:
                raise RuntimeError("missing prompt work or QPN unexpectedly executed")
            vectors=logits(logit_path)
            records.append((sha(logit_path),vectors,result))
        if not capture_path.is_file():
            raise RuntimeError("default-off capture did not execute when enabled")
        sample=decode_private_prefill(capture_path)
        metrics=summarize_capture(sample)
        if records[0][0]!=records[1][0]:
            raise RuntimeError("trace enabled changed first logits: parity hard failure")
        cmp=diff(records[0][1],records[1][1])
        if cmp["relative_l2"]!=0 or cmp["kl_base_vs_candidate"]!=0:
            raise RuntimeError("trace changed numerical output")
        output=dict(schema=1,
            scope="single pinned prompt MMQ-only private prefill route capture validation; NOT heldout R1",
            source_commit=subprocess.check_output(["git","-C",str(a.source),"rev-parse","HEAD"],text=True).strip(),
            model_sha256=MODEL,prompt_sha256=PROMPT,binary_sha256=sha(a.binary),
            private_capture_sha256=sample["private_capture_sha256"],
            control_flags={k:base[k] for k in ("STRATA_MMQ_KQ","STRATA_PREFILL_MMQ",
                           "STRATA_ARENA_LOCK","STRATA_ARENA_PIN_GIB","STRATA_PREFILL_STREAM_MIN",
                           "STRATA_QPN_PREFILL_EXPERIMENTAL")},
            capture=metrics,first_logit_equal=True,
            first_logit_compared_elements=cmp["n_logits"],
            first_logit_relative_l2=cmp["relative_l2"],first_logit_kl=cmp["kl_base_vs_candidate"],
            model_prefill_tokens=2047,qpn_groups=0,
            performance_qualified=False,model_quality_qualified=False,
            actual_decode_routes_collected=False,heldout_prompts_collected=False,
            limitations=[
              "instrumented capture is private, intrusive host file IO; its runtime is NOT a performance comparison",
              "only one pinned prompt; not independent training/heldout prediction",
              "decode/verification routes absent; actual R1 eligibility is UNQUALIFIED",
              "optimistic group counts assume arbitrary regrouping of resident eligible experts",
              "GPU cache slots/layout and model behavior unchanged"])
        a.report.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+"\n")
        return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("source","prepared","state","binary","report"):
        ap.add_argument("--"+key,type=Path,required=True)
    a=ap.parse_args()
    run(a)
    print("Private prefill capture parity passed; aggregate-only report written")


if __name__=="__main__":main()
