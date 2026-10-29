#!/usr/bin/env python3
"""Standalone prefill crossover; keeps unavailable/failing arms out of selection."""
import argparse
import json
from pathlib import Path
import statistics

from qpn_run import command, digest


def crossover(measurement):
    # Cached FP16 weights require much more residency; diagnostic control only.
    eligible = [a for a in measurement["arms"] if a["backend"] != "cached_fp16_cublas_control"
                and a["available"] and a["numerics_pass"] and len(a["completed_expert_us"]) >= 5]
    return min(eligible, key=lambda a: statistics.median(a["completed_expert_us"]))["backend"] if eligible else None


def captured_token_buckets(rows):
    """Use actual routed rows, never pad, duplicate or exceed the capture."""
    if not isinstance(rows, int) or rows < 1 or rows > 32:
        raise ValueError("expected 1..32 real routed activation rows")
    return sorted({t for t in (1, 2, 4, 8, 16, 32, rows) if t <= rows})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary",type=Path)
    p.add_argument("--blob",type=Path,action="append",default=[])
    p.add_argument("--activations",type=Path,help="raw FP32 actual routed rows; each row has 2560 floats")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--iterations",type=int,default=100)
    p.add_argument("--tokens",type=int,nargs="+",default=[1,2,4,8,16,32,33,64,128,256,512])
    a=p.parse_args()
    if a.iterations<1 or any(t<1 or t>512 for t in a.tokens): p.error("invalid iterations/tokens")
    binary=a.binary.resolve(strict=True); a.out.mkdir(parents=True,exist_ok=False)
    blobs=[b.resolve(strict=True) for b in a.blob]
    activation=a.activations.resolve(strict=True) if a.activations else None
    if activation and len(blobs)!=1: p.error("captured activations require exactly one matched --blob")
    if activation and activation.stat().st_size % (2560*4): p.error("captured activations need whole 2560-float rows")
    if activation and any(t*2560*4 > activation.stat().st_size for t in a.tokens): p.error("requested tokens exceed captured rows")
    metadata={"binary_sha256":digest(binary),"blobs":{str(b):digest(b) for b in blobs},
              "activations_sha256":digest(activation) if activation else None,
              "activation_source":"captured" if activation else "gaussian",
              "gpu":command(["nvidia-smi","-q"]),"commit":command(["git","rev-parse","HEAD"]),
              "limitation":"one expert at a time, up to 32 captured rows; no mixed-group routing/supply/TTFT qualification"}
    (a.out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    rows=[]; failures=0
    for blob in blobs or [None]:
        for t in a.tokens:
            argv=[str(binary),"--tokens",str(t),"--iterations",str(a.iterations)]
            if blob: argv += ["--blob",str(blob)]
            if activation: argv += ["--activations",str(activation)]
            r=command(argv)
            try:
                r["measurement"]=json.loads(r["stdout"])
                r["selected_backend"]=crossover(r["measurement"])
            except (ValueError,KeyError,TypeError) as exc: r["parse_error"]=str(exc)
            failures += int(r["returncode"]!=0 or "parse_error" in r)
            rows.append(r)
            with (a.out/"raw.jsonl").open("a") as f: f.write(json.dumps(r,allow_nan=False)+"\n")
            (a.out/"summary.json").write_text(json.dumps({"cases":len(rows),"failed":failures,"prefill_gate":"pending end-to-end TTFT", "crossover":[{"blob":str(blob) if blob else "synthetic","tokens":r.get("measurement",{}).get("tokens"),"backend":r.get("selected_backend")} for r in rows]},indent=2))
            print(f"prefill T={t}: {r.get('selected_backend')} exit={r['returncode']}",flush=True)
    return int(failures>0)


if __name__=="__main__": raise SystemExit(main())
