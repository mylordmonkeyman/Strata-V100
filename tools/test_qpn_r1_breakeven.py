"""Synthetic host-only tests for the R1 break-even calculator."""
import unittest
from qpn_r1_breakeven import assess

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
                 critical_path_measured_qualified=True, memory_fit_measured_qualified=True)
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

    def test_missing_is_unknown(self):
        x=self.base()
        del x["restore_s"]
        self.assertEqual(assess(x)["decision"], "UNKNOWN")
        self.assertIn("restore_s",assess(x)["missing_measurements"])

if __name__ == "__main__":
    unittest.main()
