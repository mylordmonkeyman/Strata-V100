#!/usr/bin/env python3
"""Read-only M0 pack audit; optional extraction of a single canonical expert.

No dependencies beyond Python's standard library. Does not load the model or
claim a hardware go decision. Never writes back to the supplied pack.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import struct

BASELINE = "2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926"
REFERENCE = "b37d0dd3e1163b9d802d8bccfa89918bf68d793e"
MATRIX_BYTES = 921600
EXPERT_BYTES = 3 * MATRIX_BYTES + 16


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(pack, layer, expert, n_expert=None):
    table_bytes = (pack / "native_experts.txt").read_bytes()
    text = table_bytes.decode("utf-8")
    if n_expert is None:
        match = re.search(r"n_expert\s+(\d+)", text)
        if not match:
            raise ValueError("pass --n-expert; table header has no n_expert")
        n_expert = int(match[1])
    if n_expert < 1 or not 0 <= expert < n_expert:
        raise ValueError("invalid expert count/index")
    rows = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        l, gu, down, offset, size = map(int, fields[:5])
        if l in rows or l < 0 or offset < 0 or size < 1:
            raise ValueError("duplicate/invalid layer metadata")
        rows[l] = {"layer": l, "gu_type": gu, "down_type": down, "offset": offset, "bytes": size}
    if not rows or layer not in rows:
        raise ValueError("requested layer missing")
    end = 0
    for row in sorted(rows.values(), key=lambda r: r["offset"]):
        if row["gu_type"] != 40 or row["down_type"] != 40 or row["bytes"] != EXPERT_BYTES:
            raise ValueError(f"layer {row['layer']}: not canonical all-NVFP4 2560/640 payload size")
        if row["offset"] != end:
            raise ValueError("noncontiguous/overlapping layer table")
        end += row["bytes"] * n_expert
    path = pack / "experts.bin"
    if path.stat().st_size != end:
        raise ValueError("experts.bin length does not match table")
    with path.open("rb") as f:
        f.seek(rows[layer]["offset"] + expert * EXPERT_BYTES)
        blob = f.read(EXPERT_BYTES)
    if len(blob) != EXPERT_BYTES:
        raise ValueError("truncated expert")
    tail = struct.unpack_from("<4f", blob, 3 * MATRIX_BYTES)
    if any(not math.isfinite(v) or v <= 0 for v in tail[:3]) or tail[3] != 0:
        raise ValueError("noncanonical scale tail")
    scales = collections.Counter()
    for b in range(0, 3 * MATRIX_BYTES, 36):
        scales.update(blob[b:b+4])
    if any(b & 128 for b in scales):
        raise ValueError("signed UE4M3 byte: outside FP4-Local converter contract")
    report = {
        "baseline_sha": BASELINE, "ninfer_sha": REFERENCE,
        "native_table_sha256": sha(table_bytes), "expert_sha256": sha(blob),
        "experts_bin_bytes": end, "n_expert": n_expert, "layer": layer, "expert": expert,
        "layers": list(rows.values()), "scale_tail": tail,
        "scale_byte_histogram": dict(sorted(scales.items())),
        "payload_size_matches_target": True,
        "geometry_status": "payload size checked; separately verify 2560/640 in checkpoint config and GGUF",
        "decode_gate": "pending hardware profile and real activation parity",
        "prefill_gate": "pending actual NVFP4 MMQ/fallback profile",
        "model_top_k": None,
        "routing_histogram": None,
        "headroom": None,
    }
    return report, blob


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pack", type=Path)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--n-expert", type=int)
    p.add_argument("--extract", type=Path, help="write exactly one canonical expert; refuses overwrite")
    p.add_argument("--routing", type=Path, help="JSON array of per-expert entry counts from a measured trace")
    p.add_argument("--gpu-fraction", type=float, help="measured resident expert compute fraction of decode time")
    p.add_argument("--candidate-speedup", type=float, help="measured isolated full-expert speedup")
    args = p.parse_args()
    report, blob = audit(args.pack, args.layer, args.expert, args.n_expert)
    if args.routing:
        counts = json.loads(args.routing.read_text())
        if not isinstance(counts, list) or any(type(n) is not int or n < 0 for n in counts):
            p.error("routing must be a JSON array of nonnegative integer group sizes")
        report["routing_histogram"] = dict(sorted(collections.Counter(counts).items()))
    if args.gpu_fraction is not None:
        f = args.gpu_fraction
        if not math.isfinite(f) or not 0 <= f <= 1:
            p.error("gpu fraction must be 0..1")
        s = args.candidate_speedup
        if s is not None and (not math.isfinite(s) or s <= 0):
            p.error("candidate speedup must be positive and finite")
        report["headroom"] = {"measured_gpu_fraction": f,
                              "infinite_compute_speedup_limit": 1/(1-f) if f < 1 else None,
                              "predicted_end_to_end_speedup": 1/(1-f+f/s) if s else None}
    if args.extract:
        with args.extract.open("xb") as f:
            f.write(blob)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
