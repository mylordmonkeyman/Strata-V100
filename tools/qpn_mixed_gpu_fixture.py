#!/usr/bin/env python3
"""Prepare *private* nonuniform multi-expert V100 GPU benchmark inputs.

Select real routed groups from pinned first-prefill captures, pack actual
expert weights and their exact FP32 token rows, and publish only hashes and
aggregate sizes. The pack is never modified. This is NOT a GPU benchmark.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
from qpn_mixed_route_plan import LAYERS, N_EMBD, audit_capture, digest, expected_captures
from qpn_audit import EXPERT_BYTES, audit as audit_expert

MANIFEST_SHA = "218ae1f7fa5bb03ee01aba98278848e13ce17478957ec2300a0e58272af8270d"
# Four shape regimes including small groups, the QPN crossover and large fallback.
TARGETS = ((4, 4), (10, 10), (14, 14), (20, 24, 29, 32),
           (40, 48, 60), (96, 192, 512))
BOUNDS = ((1, 8), (9, 11), (12, 16), (17, 32), (33, 64), (65, 1024))


def choose_groups(plans):
    """Pick exactly 16 distinct (layer, expert) groups with deterministic ties."""
    candidates = []
    for layer in LAYERS:
        plan = plans[layer]
        for group in plan["groups"]:
            count = len(group["token_rows"])
            if count < 1 or count > plan["tokens"]:
                raise ValueError("invalid group length")
            if len(group["slots"]) != count:
                raise ValueError("slot/row mismatch")
            candidates.append({"layer": layer, "expert": group["expert"],
                               "count": count, "token_rows": group["token_rows"]})
    picked = []
    used = set()
    per_layer = collections.Counter()
    for (low, high), targets in zip(BOUNDS, TARGETS):
        available = [g for g in candidates if low <= g["count"] <= high]
        if len(available) < len(targets):
            raise ValueError(f"not enough real routed groups for bucket {low}..{high}")
        for target in targets:
            remaining = [g for g in available if (g["layer"], g["expert"]) not in used]
            if not remaining:
                raise ValueError("selection exhausted bucket")
            # Shape fidelity first, then prefer coverage across layers.
            selected = min(remaining, key=lambda g: (abs(g["count"]-target),
                           per_layer[g["layer"]], g["layer"], g["expert"]))
            used.add((selected["layer"], selected["expert"]))
            per_layer[selected["layer"]] += 1
            picked.append(selected)
    if len(picked) != 16 or len(used) != 16:
        raise ValueError("selection does not contain 16 distinct expert-layer groups")
    # Exact nearest-target ties can concentrate all 16 picks in one layer.
    # Repair layer coverage *within* the original count buckets, preserving
    # the shape mix and 16 distinct experts. Never fabricate or reuse a group.
    for absent_layer in (layer for layer in LAYERS if not per_layer[layer]):
        repairs = []
        for index, incumbent in enumerate(picked):
            if per_layer[incumbent["layer"]] <= 1:
                continue
            bucket = next((lo, hi) for lo, hi in BOUNDS
                          if lo <= incumbent["count"] <= hi)
            for candidate in candidates:
                if (candidate["layer"] == absent_layer
                        and bucket[0] <= candidate["count"] <= bucket[1]
                        and (candidate["layer"], candidate["expert"]) not in used):
                    repairs.append((abs(candidate["count"]-incumbent["count"]),
                                    candidate["count"], index, candidate["expert"],
                                    candidate))
        if not repairs:
            raise ValueError("missing model layer in selected experts")
        _, _, index, _, replacement = min(repairs, key=lambda r: r[:4])
        incumbent = picked[index]
        used.remove((incumbent["layer"], incumbent["expert"]))
        per_layer[incumbent["layer"]] -= 1
        picked[index] = replacement
        used.add((replacement["layer"], replacement["expert"]))
        per_layer[absent_layer] += 1
    if set(LAYERS) != set(per_layer) or len(used) != 16:
        raise ValueError("missing model layer or duplicate group after repair")
    # Fallback groups first so native_grouped can operate on a contiguous prefix.
    picked.sort(key=lambda g: (g["count"] <= 32, g["layer"], g["expert"]))
    if not any(g["count"] > 32 for g in picked) or not any(g["count"] <= 32 for g in picked):
        raise ValueError("must exercise both MMQ fallback and QPN eligible groups")
    return picked


def private_write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def prepare(manifest_path, captures, pack, private_dir):
    if digest(manifest_path) != MANIFEST_SHA:
        raise ValueError("baseline manifest SHA-256 differs from pinned M0")
    manifest = json.loads(manifest_path.read_text())
    expected = expected_captures(manifest)
    plans = {}
    activation = {}
    for layer in LAYERS:
        path = captures / f"layer-{layer}.moe"
        raw, _ = audit_capture(path, expected[layer], layer)
        plans[layer] = json.loads(raw)
        with path.open("rb") as f:
            tokens, width, top_k = struct.unpack("<3q", f.read(24))
            if (tokens, width, top_k) != (expected[layer]["tokens"], N_EMBD, expected[layer]["top_k"]):
                raise ValueError("capture dimensions changed")
            x = np.frombuffer(f.read(tokens*N_EMBD*4), dtype="<f4").reshape(tokens, N_EMBD)
            activation[layer] = x.copy()
    selected = choose_groups(plans)
    if private_dir.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("private fixtures must be outside the checkout")
    if private_dir.exists():
        raise FileExistsError("refuse to overwrite existing private fixture directory")
    # Perform all selection/validation before creating a private destination.
    blobs, rows, groups = [], [], []
    for group in selected:
        layer, expert, count = group["layer"], group["expert"], group["count"]
        report, blob = audit_expert(pack, layer, expert, 512)
        if len(blob) != EXPERT_BYTES:
            raise ValueError("canonical NVFP4 expert payload length changed")
        x = np.ascontiguousarray(activation[layer][group["token_rows"]], dtype="<f4")
        if x.shape != (count, N_EMBD) or not np.isfinite(x).all() or np.max(np.abs(x)) > 65504:
            raise ValueError("routed activations invalid or outside FP16 representable range")
        if len(set(group["token_rows"])) != count:
            raise ValueError("duplicate token row within selected group")
        blobs.append(blob)
        rows.append(x.tobytes())
        groups.append({"layer": layer, "expert": expert, "tokens": count,
                       "expert_sha256": report["expert_sha256"],
                       "activation_rows_sha256": hashlib.sha256(rows[-1]).hexdigest()})
    weight_bytes = b"".join(blobs)
    activation_bytes = b"".join(rows)
    if len({g["expert_sha256"] for g in groups}) != len(groups):
        raise ValueError("duplicate expert weight content: cannot claim distinct weights")
    fallback = sum(g["tokens"] > 32 for g in groups)
    if any(g["tokens"] <= 32 for g in groups[:fallback]) or any(g["tokens"] > 32 for g in groups[fallback:]):
        raise ValueError("fallback group prefix not contiguous")
    private_dir.mkdir(parents=True, mode=0o700)
    private_write(private_dir/"weights.bin", weight_bytes)
    private_write(private_dir/"activations.f32", activation_bytes)
    private_write(private_dir/"counts.txt", (",".join(str(g["tokens"]) for g in groups)+"\n").encode())
    private_write(private_dir/"selected.json", (json.dumps(groups, indent=2)+"\n").encode())
    summary = {"baseline_manifest_sha256": MANIFEST_SHA, "model_revision": manifest["model_revision"],
               "capture_sha256": {str(layer): expected[layer]["activation_dump_sha256"] for layer in LAYERS},
               "groups": len(groups), "fallback_groups": fallback, "qpn_eligible_groups": len(groups)-fallback,
               "tokens_per_group": [g["tokens"] for g in groups],
               "distinct_expert_weights": len(groups), "expert_payload_bytes": EXPERT_BYTES,
               "total_weight_bytes": len(weight_bytes), "total_activation_bytes": len(activation_bytes),
               "total_routed_entries_in_sample": sum(g["tokens"] for g in groups),
               "weight_payload_sha256": hashlib.sha256(weight_bytes).hexdigest(),
               "activation_payload_sha256": hashlib.sha256(activation_bytes).hexdigest(),
               "private_group_metadata_sha256": digest(private_dir/"selected.json"),
               "scope": "16 selected nonuniform expert-layer groups from 3 first-prefill captures; private raw rows and weights; GPU timing not yet measured"}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("manifest", "captures", "pack", "private-dir", "summary"):
        p.add_argument("--"+field, type=Path, required=True)
    a = p.parse_args()
    if a.summary.resolve().is_relative_to(a.private_dir.resolve()):
        p.error("summary must be outside the private directory")
    result = prepare(a.manifest, a.captures, a.pack, a.private_dir)
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(f"Prepared {result['groups']} real distinct experts, "
          f"{result['total_routed_entries_in_sample']} routed entries, "
          f"{result['total_weight_bytes']} bytes of weights; all raw bytes private.")


if __name__ == "__main__":
    main()
