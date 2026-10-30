#!/usr/bin/env python3
"""Host-only R1 break-even gate. Aggregate inputs only; never reads private routes.

All durations are seconds for the SAME complete workload. This calculator does
not infer timings from routed-entry shares or optimistic grouping counts.
"""
import argparse
import json
from pathlib import Path

def assess(x):
    required = ("mmq_prefill_s", "mmq_decode_s", "qpn_prefill_exposed_s",
                "qpn_decode_s", "regroup_s", "transition_s", "restore_s",
                "dispatch_s", "packed_bytes", "safe_headroom_bytes")
    missing = [k for k in required if x.get(k) is None]
    out = {"schema": 1, "gate": "R1 full-workload >=5% improvement",
           "decision": "UNKNOWN", "missing_measurements": missing,
           "numerical_quality_qualified": False,
           "independent_heldout_qualified": bool(x.get("independent_heldout_qualified", False)),
           "actual_decode_qualified": bool(x.get("actual_decode_qualified", False)),
           "notes": ["Times must cover the same full workload and expose the actual critical path.",
                     "Verification-window routing is not accepted autoregressive decode.",
                     "No measured QPN numerical-quality qualification is implied."]}
    if "packed_bytes" in x and "safe_headroom_bytes" in x:
        if x["packed_bytes"] < 0 or x["safe_headroom_bytes"] < 0:
            raise ValueError("negative memory")
        out["resident_memory_feasible"] = x["packed_bytes"] <= x["safe_headroom_bytes"]
    if missing:
        return out
    if any(type(x[k]) not in (int, float) or x[k] < 0 for k in required):
        raise ValueError("invalid timing or memory")
    baseline = x["mmq_prefill_s"] + x["mmq_decode_s"]
    if baseline <= 0:
        raise ValueError("zero baseline")
    overhead = sum(x[k] for k in ("regroup_s", "transition_s", "restore_s", "dispatch_s"))
    candidate = x["qpn_prefill_exposed_s"] + x["qpn_decode_s"] + overhead
    out.update(baseline_s=baseline, candidate_s=candidate,
               speedup_fraction=1-candidate/baseline,
               overhead_s=overhead,
               maximum_total_overhead_for_5pct_s=0.95*baseline-
                   x["qpn_prefill_exposed_s"]-x["qpn_decode_s"])
    if not out["resident_memory_feasible"] or candidate > 0.95*baseline:
        out["decision"] = "FAIL"
    elif (out["independent_heldout_qualified"] and out["actual_decode_qualified"]
          and bool(x.get("numerical_quality_qualified", False))):
        out["decision"] = "PASS"
        out["numerical_quality_qualified"] = True
    else:
        out["decision"] = "UNKNOWN"
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
