#!/usr/bin/env python3
"""Host-only exact routing-plan audit from private pinned M0 captures.

Full token/router-slot plans stay private on the V100 runner; only aggregate
counts and hashes are uploaded. This does NOT benchmark grouped GPU throughput.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
from qpn_routing_headroom import BASELINE_SOURCE, bucketize

N_EMBD = 2560
N_EXPERTS = 512
LAYERS = (0, 24, 47)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_captures(manifest):
    if manifest.get("baseline_source") != BASELINE_SOURCE or not isinstance(manifest.get("model_revision"), str):
        raise ValueError("unpinned baseline source or model revision")
    rows = [(i, r["capture"]) for i, r in enumerate(manifest.get("runs", []))
            if r.get("returncode") == 0 and "capture" in r]
    if [i for i, _ in rows] != [0, 1, 2]:
        raise ValueError("expected exactly three pinned first-prefill captures")
    for _, c in rows:
        if type(c.get("tokens")) is not int or not 1 <= c["tokens"] <= 32768:
            raise ValueError("invalid pinned token count")
        if type(c.get("top_k")) is not int or not 1 <= c["top_k"] <= N_EXPERTS:
            raise ValueError("invalid pinned top-k")
        if c.get("n_embd") != N_EMBD:
            raise ValueError("unexpected embedding width")
        if not isinstance(c.get("activation_dump_sha256"), str) or len(c["activation_dump_sha256"]) != 64:
            raise ValueError("missing capture SHA-256")
    return dict(zip(LAYERS, (c for _, c in rows)))


def validate_plan(ids, plan):
    """Prove complete bijection between all routed slots and group-owned slots."""
    if ids.ndim != 2 or ids.dtype != np.int32:
        raise ValueError("router shape/dtype mismatch")
    tokens, top_k = ids.shape
    total = tokens * top_k
    visited = np.zeros(total, dtype=np.uint8)
    groups = plan["groups"]
    if [g["expert"] for g in groups] != sorted(set(ids.ravel().tolist())):
        raise ValueError("missing, extra, duplicate or unordered expert group")
    flat_ids = ids.ravel()
    for group in groups:
        expert = group["expert"]
        slots, token_rows = group["slots"], group["token_rows"]
        if not slots or len(slots) != len(token_rows):
            raise ValueError("empty group or mismatched token/slot lengths")
        if slots != sorted(set(slots)):
            raise ValueError("duplicate or out-of-order group slots")
        for flat, token in zip(slots, token_rows):
            if type(flat) is not int or not 0 <= flat < total:
                raise ValueError("out-of-bounds routed slot")
            if type(token) is not int or token != flat // top_k:
                raise ValueError("group row/slot disagreement")
            if visited[flat]:
                raise ValueError("router slot assigned more than once")
            if flat_ids[flat] != expert:
                raise ValueError("group owns a slot for the wrong expert")
            visited[flat] = 1
    if not np.all(visited == 1):
        raise ValueError("missing routed slot(s)")
    return True


def audit_capture(path, expected, layer):
    if digest(path) != expected["activation_dump_sha256"]:
        raise ValueError(f"layer {layer}: private capture SHA-256 differs from pinned baseline")
    with path.open("rb") as f:
        header = f.read(24)
        if len(header) != 24:
            raise ValueError("truncated capture header")
        tokens, n_embd, top_k = struct.unpack("<3q", header)
        if (tokens, n_embd, top_k) != (expected["tokens"], N_EMBD, expected["top_k"]):
            raise ValueError("capture geometry differs from baseline")
        raw_x = f.read(tokens * n_embd * 4)
        raw_ids = f.read(tokens * top_k * 4)
        if len(raw_x) != tokens * n_embd * 4 or len(raw_ids) != tokens * top_k * 4 or f.read(1):
            raise ValueError("truncated or trailing capture bytes")
    if not np.isfinite(np.frombuffer(raw_x, dtype="<f4")).all():
        raise ValueError("nonfinite captured activations")
    ids = np.frombuffer(raw_ids, dtype="<i4").reshape(tokens, top_k)
    if np.any(ids < 0) or np.any(ids >= N_EXPERTS):
        raise ValueError("out-of-range expert ID")
    if top_k > 1 and np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError("duplicate expert within a routed token")
    counts = collections.Counter(int(v) for v in ids.ravel())
    recorded = {int(e): v for e, v in expected["expert_counts"].items()}
    if counts != recorded:
        raise ValueError("captured expert counts disagree with pinned manifest")
    hist = collections.Counter(counts.values())
    if hist != {int(size): n for size, n in expected["entry_count_histogram"].items()}:
        raise ValueError("expert count histogram disagrees with pinned manifest")
    groups = []
    for expert in sorted(counts):
        slots = np.flatnonzero(ids.ravel() == expert).tolist()
        groups.append({"expert": expert, "slots": slots,
                       "token_rows": [slot // top_k for slot in slots]})
    plan = {"layer": layer, "tokens": tokens, "top_k": top_k,
            "capture_sha256": expected["activation_dump_sha256"], "groups": groups}
    validate_plan(ids, plan)
    buckets = bucketize(list(counts.values()))
    if sum(b["expert_invocations"] for b in buckets) != len(groups) or sum(b["routed_entries"] for b in buckets) != tokens * top_k:
        raise ValueError("routing bucket coverage mismatch")
    plan_bytes = (json.dumps(plan, separators=(",", ":"), sort_keys=True) + "\n").encode()
    summary = {"layer": layer, "tokens": tokens, "top_k": top_k,
               "expert_invocations": len(groups), "routed_entries": tokens * top_k,
               "capture_sha256": expected["activation_dump_sha256"],
               "private_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
               "bucket_counts": buckets, "no_duplicate_or_missing_slots": True,
               "per_token_unique_experts": True, "pinned_expert_counts_equal": True}
    return plan_bytes, summary


def audit(manifest_path, captures, private_dir=None):
    manifest = json.loads(manifest_path.read_text())
    expected = expected_captures(manifest)
    if private_dir is not None:
        if private_dir.resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError("private plan output must be outside checkout")
        private_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    reports = []
    for layer in LAYERS:
        plan, summary = audit_capture(captures / f"layer-{layer}.moe", expected[layer], layer)
        if private_dir is not None:
            target = private_dir / f"layer-{layer}-plan.json"
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(plan)
        reports.append(summary)
    return {"model_revision": manifest["model_revision"], "baseline_source": BASELINE_SOURCE,
            "baseline_manifest_sha256": digest(manifest_path), "layers": reports,
            "total_expert_invocations": sum(r["expert_invocations"] for r in reports),
            "total_routed_entries": sum(r["routed_entries"] for r in reports),
            "scope": "host-only exact routing ownership; GPU grouped timing not yet measured",
            "private_plan_saved": private_dir is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--private-dir", type=Path)
    args = p.parse_args()
    if args.summary.resolve().is_relative_to(args.captures.resolve()):
        p.error("summary cannot overwrite private captures")
    result = audit(args.manifest, args.captures, args.private_dir)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Verified {result['total_expert_invocations']} exact groups and "
          f"{result['total_routed_entries']} unique routed slots across three private captures")


if __name__ == "__main__":
    main()
