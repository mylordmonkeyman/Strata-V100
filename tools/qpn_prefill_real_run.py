#!/usr/bin/env python3
"""Prefill crossover on each M0 expert's actual routed activation rows.

Every run uses the same model checkpoint + capture hashes and independently
compares QPN, actual NVFP4 MMQ and dequant->FP16/cuBLAS. The cached-FP16
arm remains an ineligible residency control, not an automatic winner.
"""
import argparse
import collections
import json
from pathlib import Path
import statistics
import subprocess
import sys

from qpn_m0_bench import digest
from qpn_prefill_run import captured_token_buckets


def qualified_ratio(row, competitor="nvfp4_w4a8_mmq"):
    if row.get("returncode") != 0: return None
    arms={a["backend"]: a for a in row.get("measurement", {}).get("arms", [])}
    qpn, other=arms.get("qpn_direct"), arms.get(competitor)
    if not qpn or not other: return None
    if any(not a.get("available") or not a.get("numerics_pass") or len(a.get("completed_expert_us", [])) < 5
           for a in (qpn,other)): return None
    q=statistics.median(qpn["completed_expert_us"])
    o=statistics.median(other["completed_expert_us"])
    return o/q if q>0 and o>0 else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "fixtures", "out"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--iterations",type=int,default=35)
    a=p.parse_args()
    index=json.loads((a.fixtures/"index.json").read_text())
    if not index.get("fixtures"): raise ValueError("empty captured expert fixture index")
    a.out.mkdir(parents=True,exist_ok=False)
    failures=0; counts=collections.Counter(); evidence=[]
    for item in index["fixtures"]:
        blob=a.fixtures/item["blob"]
        activation=a.fixtures/item["activations"]
        if digest(blob)!=item["expert_sha256"] or digest(activation)!=item["activations_sha256"]:
            raise ValueError("matched prefill fixture or activation SHA-256 changed")
        rows=item["rows"]
        if activation.stat().st_size!=rows*2560*4: raise ValueError("activation row count mismatch")
        stem=f"l{item['layer']}-e{item['expert']}"
        output=a.out/stem
        cmd=[sys.executable,str(Path(__file__).with_name("qpn_prefill_run.py")),
             str(a.binary),"--blob",str(blob),"--activations",str(activation),
             "--tokens",*map(str,captured_token_buckets(rows)),
             "--iterations",str(a.iterations),"--qpn-tiles","0","8","16",
             "--out",str(output)]
        res=subprocess.run(cmd,check=False)
        failures+=int(res.returncode!=0)
        raw=output/"raw.jsonl"
        if not raw.is_file(): raise ValueError(f"{stem}: no per-token measurements")
        data=[json.loads(line) for line in raw.read_text().splitlines() if line]
        records=[]
        for row in data:
            meas=row.get("measurement",{})
            winner=row.get("selected_backend")
            if winner: counts[winner]+=1
            records.append({"tokens":meas.get("tokens"),"qpn_tile":row.get("qpn_tile"),
                            "winner":winner,
                            "qpn_vs_mmq":qualified_ratio(row),
                            "qpn_vs_fp16":qualified_ratio(row,"dequant_fp16_cublas"),
                            "exit_code":row["returncode"]})
        evidence.append({"layer":item["layer"],"expert":item["expert"],
                         "rows":rows,"cases":records,"returncode":res.returncode})
        print(f"{stem}: {len(records)} cases, exit={res.returncode}",flush=True)
        (a.out/"summary.json").write_text(json.dumps({"fixtures_finished":len(evidence),
          "failed_fixtures":failures,"backend_wins":dict(counts),"cases":evidence,
          "scope":"real captured rows, individual experts; NOT mixed group, supply or TTFT"},indent=2))
    (a.out/"fixture-index.json").write_text(json.dumps(index,indent=2))
    return int(failures>0)


if __name__=="__main__": raise SystemExit(main())
