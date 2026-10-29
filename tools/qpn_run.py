#!/usr/bin/env python3
"""Run independent standalone QPN candidates; keep failures and raw timings.

No default or engine-setting changes. Run on an idle single V100. Results are
microbenchmarks, not end-to-end decode/prefill qualification.
"""
import argparse
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    return {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def summarize(rows):
    valid = [r for r in rows if r.get("measurement", {}).get("numerics_pass") and r["returncode"] == 0]
    return {"candidates": len(rows), "passed": len(valid), "failed": len(rows)-len(valid),
            "best_completed_expert_speedup": max((r["completed_expert_speedup"] for r in valid), default=None),
            "decode_gate": "pending mixed-group/real-activation/model/hardware qualification",
            "prefill_gate": "pending NVFP4 MMQ and scaled FP16 crossover plus end-to-end TTFT",
            "engine_dispatch": "unchanged; standalone tools only"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary", type=Path)
    p.add_argument("--blob", type=Path, action="append", default=[])
    p.add_argument("--activations", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--sweep", action="store_true", help="all supported split/acc and input/hidden staging choices")
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 4, 8, 16, 32])
    args = p.parse_args()
    if args.iterations < 1 or any(t < 1 or t > 32 for t in args.tokens):
        p.error("iterations > 0, tokens 1..32 required")
    binary = args.binary.resolve(strict=True)
    blobs = [b.resolve(strict=True) for b in args.blob]
    args.out.mkdir(parents=True, exist_ok=False)
    gpu = command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate", "--format=csv"])
    if gpu["returncode"]:
        raise RuntimeError(gpu["stderr"])
    metadata = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "binary_sha256": digest(binary), "blobs": {str(b): digest(b) for b in blobs},
                "activations_sha256": digest(args.activations) if args.activations else None,
                "host": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "git": command(["git", "rev-parse", "HEAD"]), "gpu_before": gpu,
                "controls": {k: v for k, v in os.environ.items() if k.startswith(("STRATA_", "CUDA_", "OMP_"))},
                "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "blob"},
                "warning": "single expert working set may fit V100 L2; NOT a multi-expert residency benchmark"}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    for blob, t in itertools.product(blobs or [None], args.tokens):
        schedules = itertools.product([4, 8, 16], [4, 8], [1, 2], [False, True], [False, True]) if args.sweep else [
            (4, 4, 1, False, False), (4, 4, 1, True, True)]
        for gu, down, acc, input_q8, hidden_q8 in schedules:
            if t > 16 and gu == 16:
                continue  # exceeds 48 KiB static shared memory
            argv = [str(binary), "--tokens", str(t), "--iterations", str(args.iterations),
                    "--split-gu", str(gu), "--split-down", str(down), "--acc", str(acc)]
            if blob:
                argv += ["--blob", str(blob)]
            if args.activations:
                argv += ["--activations", str(args.activations.resolve())]
            if input_q8:
                argv.append("--input-q8")
            if hidden_q8:
                argv.append("--hidden-q8")
            print(f"candidate {len(rows)+1}: T={t} GU={gu} down={down} acc={acc} input_q8={input_q8} hidden_q8={hidden_q8}", flush=True)
            result = command(argv)
            try:
                measurement = json.loads(result["stdout"])
                result["measurement"] = measurement
                result["completed_expert_speedup"] = statistics.median(measurement["legacy_completed_expert_us"]) / statistics.median(measurement["qpn_completed_expert_us"])
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                result["parse_error"] = str(exc)
            rows.append(result)
            # Checkpoint every candidate; a red numerical gate does not cancel
            # unrelated schedules. Process/context failure terminates that child.
            with (args.out / "raw.jsonl").open("a") as f:
                f.write(json.dumps(result, allow_nan=False) + "\n")
            (args.out / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
    (args.out / "gpu_after.json").write_text(json.dumps(command(["nvidia-smi", "-q"]), indent=2))
    return 1 if summarize(rows)["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
