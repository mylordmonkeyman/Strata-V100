#!/usr/bin/env python3
"""Host-only M0 routing histogram; no weights or raw activations required."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

BASELINE_SOURCE = "2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926"
BUCKETS = (("1-8", 1, 8), ("9-11", 9, 11), ("12-16", 12, 16),
           ("17-32", 17, 32), ("33-64", 33, 64), ("65+", 65, None))


def bucketize(sizes):
    return [{"bucket": label, "min": low, "max": high,
             "expert_invocations": sum(low <= x and (high is None or x <= high) for x in sizes),
             "routed_entries": sum(x for x in sizes if low <= x and (high is None or x <= high))}
            for label, low, high in BUCKETS]


def summarize(manifest):
    if manifest.get("baseline_source") != BASELINE_SOURCE:
        raise ValueError("unrecognized unchanged-engine baseline")
    if not isinstance(manifest.get("model_revision"), str):
        raise ValueError("missing pinned model revision")
    captures, combined = [], []
    for i, run in enumerate(manifest.get("runs", [])):
        if run.get("returncode") != 0 or "capture" not in run:
            continue
        c = run["capture"]
        if type(c.get("tokens")) is not int or not 1 <= c["tokens"] <= 32768:
            raise ValueError("invalid routed token count")
        if type(c.get("top_k")) is not int or not 1 <= c["top_k"] <= 512:
            raise ValueError("invalid top-k")
        if not isinstance(c.get("activation_dump_sha256"), str) or len(c["activation_dump_sha256"]) != 64:
            raise ValueError("missing capture SHA-256")
        raw = c.get("expert_counts")
        if not isinstance(raw, dict) or not (1 <= len(raw) <= 512):
            raise ValueError("missing or invalid expert counts")
        sizes = []
        for expert, count in raw.items():
            if not str(expert).isdigit() or not 0 <= int(expert) < 512:
                raise ValueError("invalid expert ID")
            if type(count) is not int or not 1 <= count <= c["tokens"]:
                raise ValueError("invalid expert routed-token count")
            sizes.append(count)
        if sum(sizes) != c["tokens"] * c["top_k"]:
            raise ValueError("routed entries do not equal tokens * top_k")
        hist = c.get("entry_count_histogram")
        if hist is not None:
            actual = {str(k): v for k, v in collections.Counter(sizes).items()}
            if {str(k): v for k, v in hist.items()} != actual:
                raise ValueError("captured expert-count histogram disagreement")
        combined.extend(sizes)
        captures.append({"run_index": i, "tokens": c["tokens"], "top_k": c["top_k"],
                         "expert_invocations": len(sizes), "routed_entries": sum(sizes),
                         "activation_dump_sha256": c["activation_dump_sha256"],
                         "buckets": bucketize(sizes)})
    if not captures:
        raise ValueError("no successful M0 capture")
    total_groups, total_entries = len(combined), sum(combined)
    rows = bucketize(combined)
    for row in rows:
        row["expert_invocation_fraction"] = row["expert_invocations"] / total_groups
        row["routed_entry_fraction"] = row["routed_entries"] / total_entries
    return {"baseline_source": manifest["baseline_source"],
            "model_revision": manifest["model_revision"],
            "captured_chunks": len(captures),
            "expert_invocations": total_groups,
            "routed_entries": total_entries,
            "captures": captures,
            "aggregate_buckets": rows,
            "scope": "sampled first-prefill chunks only, not a full-model routing distribution",
            "warning": "Entry proportions do not directly imply GPU-time or TTFT savings."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = a.manifest.read_bytes()
    report = summarize(json.loads(data))
    report["manifest_sha256"] = hashlib.sha256(data).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for b in report["aggregate_buckets"]:
        print(f"{b['bucket']}: {b['expert_invocations']} expert invocations "
              f"({100*b['expert_invocation_fraction']:.2f}%), "
              f"{b['routed_entries']} entries ({100*b['routed_entry_fraction']:.2f}%)")
    print(f"Validated {report['captured_chunks']} chunks, {report['expert_invocations']} "
          f"expert invocations and {report['routed_entries']} entries")


if __name__ == "__main__":
    main()
