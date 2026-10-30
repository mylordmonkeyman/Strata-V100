"""Synthetic host-only tests for the R1 break-even calculator."""
import unittest
from qpn_r1_breakeven import assess, assess_feasibility

class BreakEvenTests(unittest.TestCase):
    def base(self):
        return dict(mmq_prefill_s=80, mmq_decode_s=20,
                    qpn_prefill_exposed_s=65, qpn_decode_s=20,
                    regroup_s=2, transition_s=2, restore_s=2, dispatch_s=1,
                    packed_bytes=100, safe_headroom_bytes=200)

    def test_unqualified_fast_is_unknown(self):
        r=assess(self.base())
        self.assertEqual(r["decision"], "UNKNOWN")
        self.assertAlmostEqual(r["speedup_fraction"], .08)

    def test_qualified_fast_passes(self):
        x=self.base()
        x.update(independent_heldout_qualified=True,
                 actual_decode_qualified=True, numerical_quality_qualified=True,
                 critical_path_measured_qualified=True, memory_fit_measured_qualified=True,
                 single_representation_per_expert=True)
        self.assertEqual(assess(x)["decision"], "PASS")

    def test_slow_fails(self):
        x=self.base()
        x["regroup_s"]=20
        self.assertEqual(assess(x)["decision"], "FAIL")

    def test_memory_fails(self):
        x=self.base()
        x["packed_bytes"]=201
        self.assertEqual(assess(x)["decision"], "FAIL")

    def test_known_memory_failure_even_with_missing_timings(self):
        self.assertEqual(assess({"packed_bytes": 201, "safe_headroom_bytes": 200})["decision"], "FAIL")

    def test_missing_measured_qualification_is_unknown(self):
        x = self.base()
        x.update(independent_heldout_qualified=True,
                 actual_decode_qualified=True, numerical_quality_qualified=True)
        self.assertEqual(assess(x)["decision"], "UNKNOWN")
        self.assertIn("memory_fit_measured_qualified", assess(x)["unqualified_gates"])

    def test_reject_nonfinite_or_bool_timings(self):
        for bad in (float("nan"), float("inf"), True, -1):
            x = self.base()
            x["mmq_prefill_s"] = bad
            with self.assertRaises(ValueError):
                assess(x)

    def test_reject_string_qualification(self):
        x = self.base()
        x["actual_decode_qualified"] = "true"
        with self.assertRaises(ValueError):
            assess(x)

    def test_737_mib_aggregate_evidence_stays_unknown(self):
        r = assess({"safe_headroom_bytes": 737 * 1048576,
                    "independent_heldout_qualified": False,
                    "actual_decode_qualified": False})
        self.assertEqual(r["decision"], "UNKNOWN")
        self.assertIn("qpn_prefill_exposed_s", r["missing_measurements"])

    def test_reject_two_resident_formats_even_if_other_gates_pass(self):
        x=self.base()
        x.update(independent_heldout_qualified=True, actual_decode_qualified=True,
                 numerical_quality_qualified=True, critical_path_measured_qualified=True,
                 memory_fit_measured_qualified=True, single_representation_per_expert=False)
        r=assess(x)
        self.assertEqual(r['decision'], 'FAIL')
        self.assertIn('dual-format', r['failure_reason'])

    def test_single_representation_missing_cannot_pass(self):
        x=self.base()
        x.update(independent_heldout_qualified=True, actual_decode_qualified=True,
                 numerical_quality_qualified=True, critical_path_measured_qualified=True,
                 memory_fit_measured_qualified=True)
        r=assess(x)
        self.assertEqual(r['decision'], 'UNKNOWN')
        self.assertIn('single_representation_per_expert', r['unqualified_gates'])

    def test_missing_is_unknown(self):
        x=self.base()
        del x["restore_s"]
        self.assertEqual(assess(x)["decision"], "UNKNOWN")
        self.assertIn("restore_s",assess(x)["missing_measurements"])


class FeasibilityTests(unittest.TestCase):
    def data(self):
        return dict(baseline_request_s=100, optimistic_saved_s=15,
                    conservative_saved_s=10, minimum_overhead_s=2,
                    maximum_overhead_s=4, additional_peak_bytes=100,
                    available_incremental_bytes=200, baseline_measured=True,
                    heldout_routes_measured=True, exposed_compute_measured=True,
                    transition_cost_bounded=True, memory_budget_bounded=True,
                    single_representation_per_expert=True)

    def test_candidate_does_not_require_unbuilt_candidate_quality(self):
        r=assess_feasibility(self.data())
        self.assertEqual(r['decision'], 'PROTOTYPE_CANDIDATE')
        self.assertFalse(r['production_qualified'])

    def test_optimistic_failure_can_stop_before_full_evidence(self):
        r=assess_feasibility(dict(baseline_request_s=100, optimistic_saved_s=4,
                                  minimum_overhead_s=0))
        self.assertEqual(r['decision'], 'STOP')

    def test_uncertain_bounds_need_measurement(self):
        x=self.data(); x['maximum_overhead_s']=9
        self.assertEqual(assess_feasibility(x)['decision'], 'MEASURE_UNCERTAINTY')

    def test_unverified_heldout_cannot_authorize(self):
        x=self.data(); x['heldout_routes_measured']=False
        self.assertEqual(assess_feasibility(x)['decision'], 'UNKNOWN')

    def test_memory_failure_stops(self):
        self.assertEqual(assess_feasibility(dict(additional_peak_bytes=201,
                         available_incremental_bytes=200))['decision'], 'STOP')

    def test_invalid_bounds_rejected(self):
        for change in ({'conservative_saved_s':16}, {'minimum_overhead_s':5},
                       {'optimistic_saved_s':101}, {'baseline_request_s':0},
                       {'additional_peak_bytes':True}, {'maximum_overhead_s':float('nan')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                assess_feasibility(dict(self.data(), **change))

    def test_dual_residency_stops_even_when_time_unknown(self):
        self.assertEqual(assess_feasibility(
            dict(single_representation_per_expert=False))['decision'], 'STOP')

    def test_single_representation_missing_is_not_a_candidate(self):
        x=self.data()
        del x['single_representation_per_expert']
        r=assess_feasibility(x)
        self.assertEqual(r['decision'], 'UNKNOWN')
        self.assertIn('single_representation_per_expert', r['unqualified_gates'])

    def test_invalid_single_representation_flag_rejected(self):
        for bad in ('true', 1, 0, None):
            x=self.data()
            x['single_representation_per_expert']=bad
            with self.assertRaises(ValueError): assess_feasibility(x)

    def test_headroom_alone_unknown(self):
        self.assertEqual(assess_feasibility(dict(available_incremental_bytes=737*1048576))['decision'], 'UNKNOWN')

if __name__ == "__main__":
    unittest.main()
