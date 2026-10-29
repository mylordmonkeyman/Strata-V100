#!/usr/bin/env python3
"""Check immutable private 16-expert pack before a V100 GPU qualification."""
import argparse
import hashlib
import json
from pathlib import Path

BASELINE_SHA = "218ae1f7fa5bb03ee01aba98278848e13ce17478957ec2300a0e58272af8270d"
EXPERT_BYTES = 2764816
N = 2560
BUCKETS = ((1, 8, 2), (9, 11, 2), (12, 16, 2),
           (17, 32, 4), (33, 64, 3), (65, 1024, 3))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(directory, summary):
    meta = json.loads(summary.read_text())
    if meta.get("baseline_manifest_sha256") != BASELINE_SHA or meta.get("groups") != 16:
        raise ValueError("pinned model or selected group count changed")
    groups_path = directory / "selected.json"
    groups = json.loads(groups_path.read_text())
    counts = [int(n) for n in (directory / "counts.txt").read_text().strip().split(",")]
    if len(groups) != 16 or len(counts) != 16 or any(
        type(g.get("tokens")) is not int or g["tokens"] != n
        for g, n in zip(groups, counts)
    ):
        raise ValueError("private selected group counts disagree")
    if len({(g["layer"], g["expert"]) for g in groups}) != 16 or \
            {g["layer"] for g in groups} != {0, 24, 47}:
        raise ValueError("selected model layer or group uniqueness failure")
    for low, high, required in BUCKETS:
        if sum(low <= n <= high for n in counts) != required:
            raise ValueError("private fixture stratification changed")
    if not all(n > 32 for n in counts[:6]) or not all(n <= 32 for n in counts[6:]):
        raise ValueError("fallback prefix not preserved")
    if meta.get("tokens_per_group") != counts or meta.get("fallback_groups") != 6 or \
            meta.get("qpn_eligible_groups") != 10:
        raise ValueError("pinned selected expert population disagrees")
    for name, size, expected in (
        ("weights.bin", 16 * EXPERT_BYTES, meta["weight_payload_sha256"]),
        ("activations.f32", sum(counts) * N * 4, meta["activation_payload_sha256"]),
        ("selected.json", None, meta["private_group_metadata_sha256"]),
    ):
        path = directory / name
        if path.is_symlink() or not path.is_file() or (size is not None and path.stat().st_size != size):
            raise ValueError("missing/symlinked or wrong-sized private " + name)
        if digest(path) != expected:
            raise ValueError("pinned SHA-256 mismatch for " + name)
    with (directory / "weights.bin").open("rb") as f:
        for g in groups:
            if hashlib.sha256(f.read(EXPERT_BYTES)).hexdigest() != g["expert_sha256"]:
                raise ValueError("wrong per-expert weight bytes")
    with (directory / "activations.f32").open("rb") as f:
        for g in groups:
            if hashlib.sha256(f.read(g["tokens"] * N * 4)).hexdigest() != g["activation_rows_sha256"]:
                raise ValueError("wrong per-group activation rows")
    return {"pinned_baseline_sha256": BASELINE_SHA, "groups": 16,
            "fallback_groups": 6, "qpn_eligible_groups": 10,
            "selected_routed_rows": sum(counts),
            "weights_sha256": meta["weight_payload_sha256"],
            "activations_sha256": meta["activation_payload_sha256"],
            "selected_sha256": meta["private_group_metadata_sha256"],
            "scope": "real private fixture digest and topology only; not a GPU result"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-dir", required=True, type=Path)
    p.add_argument("--published-summary", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    result = verify(args.private_dir, args.published_summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("Pinned private mixed fixture verified (16 unique expert groups).")


if __name__ == "__main__":
    main()
