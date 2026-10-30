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
        "gate": "R3 final full-workload >=5% latency reduction including native decode",
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


def assess_feasibility(x):
    """R1 bounds authorize investigation only; final qualification uses assess().

    Savings are exposed seconds removed from the SAME baseline request.
    Optimistic/conservative bounds must include all eligible resident work,
    without interpreting route-entry shares as time shares.
    """
    fields = ("baseline_request_s", "optimistic_saved_s", "conservative_saved_s",
              "minimum_overhead_s", "maximum_overhead_s",
              "additional_peak_bytes", "available_incremental_bytes")
    gates = ("baseline_measured", "heldout_routes_measured",
             "exposed_compute_measured", "transition_cost_bounded",
             "memory_budget_bounded")
    if not isinstance(x, dict):
        raise ValueError("expected feasibility object")
    for k in fields:
        v = x.get(k)
        if v is not None and (type(v) not in (int, float) or
                              not math.isfinite(v) or v < 0):
            raise ValueError("invalid feasibility value: " + k)
    for k in gates:
        if k in x and type(x[k]) is not bool:
            raise ValueError("invalid evidence flag: " + k)
    missing = [k for k in fields if x.get(k) is None]
    unverified = [k for k in gates if x.get(k) is not True]
    out = dict(schema=1, gate="R1 preimplementation feasibility",
               decision="UNKNOWN", missing_measurements=missing,
               unqualified_gates=unverified, production_qualified=False,
               notes=["Bounds are supplied evidence, not authenticated by this calculator.",
                      "Candidate status requires DDR-1 and safety review before implementation.",
                      "Overhead includes regrouping, dispatch, lost overlap, fallback and canonical restoration.",
                      "5% means request latency reduction, not throughput increase."])
    base = x.get("baseline_request_s")
    if base == 0:
        raise ValueError("baseline request must be positive")
    lo, hi = x.get("conservative_saved_s"), x.get("optimistic_saved_s")
    omin, omax = x.get("minimum_overhead_s"), x.get("maximum_overhead_s")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("reversed savings bounds")
    if omin is not None and omax is not None and omin > omax:
        raise ValueError("reversed overhead bounds")
    if base is not None and any(v is not None and v > base for v in (lo, hi)):
        raise ValueError("removable time exceeds complete baseline")
    peak, available = x.get("additional_peak_bytes"), x.get("available_incremental_bytes")
    if peak is not None and available is not None and peak > available:
        out.update(decision="STOP", reason="additional memory exceeds budget")
        return out
    if base is not None and hi is not None and omin is not None:
        out["optimistic_latency_reduction"] = (hi-omin)/base
        if hi-omin < .05*base:
            out.update(decision="STOP", reason="even optimistic net savings miss target")
            return out
    if missing or unverified:
        return out
    out["conservative_latency_reduction"] = (lo-omax)/base
    out["decision"] = "PROTOTYPE_CANDIDATE" if lo-omax >= .05*base else "MEASURE_UNCERTAINTY"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("qualification", "feasibility"), default="qualification")
    p.add_argument("--aggregate-input", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    x = json.loads(a.aggregate_input.read_text())
    if not isinstance(x, dict):
        p.error("expected aggregate JSON object")
    report = assess_feasibility(x) if a.stage == "feasibility" else assess(x)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(report["gate"] + ":", report["decision"])

if __name__ == "__main__":
    main()
