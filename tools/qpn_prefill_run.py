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
    # Exercise both CTA boundaries and a partial final tile, using only
    # genuine captured prefix rows (never duplicate or pad activations).
    return sorted({t for t in (1, 2, 4, 8, 16, 17, 24, 31, 32, rows) if t <= rows})


def tile_variants(tokens, choices):
    """Retain default as control; skip redundant and small-token alternatives."""
    base = 8 if tokens <= 8 else 16 if tokens <= 16 else 32
    return [t for t in choices if t == 0 or (tokens >= 12 and t != base)]


def graph_check_case(tokens, tile):
    """Run graph and canary checks only on new multi-CTA boundaries."""
    return tile == 16 and tokens in (17, 24, 31, 32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary",type=Path)
    p.add_argument("--blob",type=Path,action="append",default=[])
    p.add_argument("--activations",type=Path,help="raw FP32 actual routed rows; each row has 2560 floats")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--iterations",type=int,default=100)
    p.add_argument("--graph-check",action="store_true",
                   help="validate QPN graph/canary at T=17/24/31/32 tile=16")
    p.add_argument("--qpn-tiles",type=int,nargs="+",default=[0],
                   help="0 baseline schedule, plus optional explicit 8/16/32-row CTA tiles")
    p.add_argument("--tokens",type=int,nargs="+",default=[1,2,4,8,16,32,33,64,128,256,512])
    a=p.parse_args()
    if a.iterations<1 or any(t<1 or t>512 for t in a.tokens): p.error("invalid iterations/tokens")
    if not a.qpn_tiles or len(set(a.qpn_tiles)) != len(a.qpn_tiles) or 0 not in a.qpn_tiles or any(v not in (0,8,16,32) for v in a.qpn_tiles):
        p.error("QPN tile variants must be unique, contain 0, and use 0/8/16/32")
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
            for tile in tile_variants(t, a.qpn_tiles):
                argv=[str(binary),"--tokens",str(t),"--iterations",str(a.iterations),
                      "--qpn-tile",str(tile)]
                if blob: argv += ["--blob",str(blob)]
                if activation: argv += ["--activations",str(activation)]
                if a.graph_check and graph_check_case(t,tile): argv.append("--validate-graph")
                r=command(argv)
                r["blob"]=str(blob) if blob else "synthetic"
                r["qpn_tile"]=tile
                r["graph_check_requested"]="--validate-graph" in argv
                try:
                    r["measurement"]=json.loads(r["stdout"])
                    r["selected_backend"]=crossover(r["measurement"])
                except (ValueError,KeyError,TypeError) as exc: r["parse_error"]=str(exc)
                failures += int(r["returncode"]!=0 or "parse_error" in r)
                rows.append(r)
                with (a.out/"raw.jsonl").open("a") as f:
                    f.write(json.dumps(r,allow_nan=False)+"\n")
                (a.out/"summary.json").write_text(json.dumps({
                    "cases":len(rows),"failed":failures,"prefill_gate":"pending end-to-end TTFT",
                    "crossover":[{"blob":case["blob"],"tokens":case.get("measurement",{}).get("tokens"),
                                  "qpn_tile":case["qpn_tile"],"backend":case.get("selected_backend")}
                                 for case in rows]},indent=2))
                print(f"prefill T={t} tile={tile}: {r.get('selected_backend')} exit={r['returncode']}",flush=True)
    return int(failures>0)


if __name__=="__main__": raise SystemExit(main())
