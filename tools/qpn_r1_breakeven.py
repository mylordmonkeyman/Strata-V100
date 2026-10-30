#!/usr/bin/env python3
"""Host-only R1 break-even gate. Aggregate inputs only; never reads private routes.

All durations are seconds for the SAME complete workload. This calculator does
not infer timings from routed-entry shares or optimistic grouping counts.
"""
import argparse
import json
import math
from pathlib import Path

NUMERIC_FIELDS = ("mmq_prefill_s", "mmq_decode_s", "qpn_prefill_exposed_s",
                  "qpn_decode_s", "regroup_s", "transition_s", "restore_s",
                  "dispatch_s", "packed_bytes", "safe_headroom_bytes")
TIMING_FIELDS = NUMERIC_FIELDS[:-2]
QUALIFICATIONS = ("independent_heldout_qualified", "actual_decode_qualified",
                  "numerical_quality_qualified", "critical_path_measured_qualified",
                  "memory_fit_measured_qualified")


def assess(x):
    """Three-valued gate: known infeasibility FAIL, incomplete evidence UNKNOWN."""
    if not isinstance(x, dict):
        raise ValueError("expected aggregate object")
    for k in NUMERIC_FIELDS:
        v = x.get(k)
        if v is not None and (type(v) not in (int, float) or
                              not math.isfinite(v) or v < 0):
            raise ValueError("invalid nonnegative finite aggregate: " + k)
    for k in QUALIFICATIONS:
        if k in x and type(x[k]) is not bool:
            raise ValueError("qualification must be boolean: " + k)
    missing = [k for k in NUMERIC_FIELDS if x.get(k) is None]
    absent_gates = [k for k in QUALIFICATIONS if x.get(k) is not True]
    out = {
        "schema": 2,
        "gate": "R1 full-workload >=5% improvement including native decode",
        "decision": "UNKNOWN",
        "missing_measurements": missing,
        "unqualified_gates": absent_gates,
        "notes": [
            "All durations must describe the SAME full workload and nonoverlapping exposed critical-path costs.",
            "packed_bytes is peak ADDITIONAL live VRAM including scratch; not total packed payload size.",
            "Measured guarded headroom is an optimistic upper bound, NOT proof of allocator-safe fit.",
            "Verifier windows do not establish accepted-token autoregressive decode.",
            "Qualification flags are input assertions, not independently verified by this calculator.",
            "No model numerical-quality qualification is implied by a projected speedup."
        ]
    }
    if x.get("packed_bytes") is not None and x.get("safe_headroom_bytes") is not None:
        fits_bound = x["packed_bytes"] <= x["safe_headroom_bytes"]
        out["memory_upper_bound_respected"] = fits_bound
        if not fits_bound:
            out["decision"] = "FAIL"
            out["failure_reason"] = "additional live memory exceeds guarded headroom bound"
    if all(x.get(k) is not None for k in TIMING_FIELDS):
        baseline = x["mmq_prefill_s"] + x["mmq_decode_s"]
        if baseline <= 0:
            raise ValueError("zero full-workload MMQ baseline")
        overhead = sum(x[k] for k in ("regroup_s", "transition_s", "restore_s", "dispatch_s"))
        candidate = x["qpn_prefill_exposed_s"] + x["qpn_decode_s"] + overhead
        threshold = 0.95 * baseline
        out.update(
            baseline_s=baseline, candidate_s=candidate,
            speedup_fraction=1-candidate/baseline, overhead_s=overhead,
            maximum_total_overhead_for_5pct_s=threshold-
                x["qpn_prefill_exposed_s"]-x["qpn_decode_s"],
            projected_5pct_threshold_met=candidate <= threshold
        )
        if candidate > threshold:
            out["decision"] = "FAIL"
            out["failure_reason"] = "full-workload time misses 5% improvement"
    if (out["decision"] != "FAIL" and not missing and not absent_gates
            and out.get("memory_upper_bound_respected") is True):
        out["decision"] = "PASS"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate-input", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    x = json.loads(a.aggregate_input.read_text())
    if not isinstance(x, dict):
        p.error("expected aggregate JSON object")
    report = assess(x)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("R1 decision:", report["decision"])

if __name__ == "__main__":
    main()
