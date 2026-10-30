import unittest
from qpn_full_model_ab import ab_modes, classify_quality, benchmark_env, NAN_WORKAROUND


class PerformanceFirstABTest(unittest.TestCase):
    def test_nan_workaround_survives_environment_sanitization(self):
        original = {"PATH":"/example", "STRATA_MMQ_KQ":"0", "STRATA_UNRELATED":"1"}
        result = benchmark_env(original)
        self.assertEqual({k:result[k] for k in NAN_WORKAROUND}, {
            "STRATA_MMQ_KQ":"1", "STRATA_PREFILL_MMQ":"1",
            "STRATA_ARENA_LOCK":"0", "STRATA_ARENA_PIN_GIB":"1"})
        self.assertEqual(result["PATH"],"/example")
        self.assertNotIn("STRATA_UNRELATED",result)
        self.assertEqual(original["STRATA_MMQ_KQ"],"0")

    def test_modes(self):
        self.assertEqual(ab_modes(True), ("qpn-base","qpn-fused","qpn-fused","qpn-base"))
        self.assertEqual(ab_modes(False), ("mmq","qpn-fused","qpn-fused","mmq"))

    def test_batched_ab_controls(self):
        self.assertEqual(ab_modes(False,True),
                         ("qpn-fused","qpn-batched","qpn-batched","qpn-fused"))
        with self.assertRaises(ValueError):
            ab_modes(True,True)

    def test_batched_vs_mmq_ab(self):
        self.assertEqual(ab_modes(False,False,True),
                         ("mmq","qpn-batched","qpn-batched","mmq"))
        with self.assertRaises(ValueError):
            ab_modes(False,True,True)

    def test_strict_gate_remains_default(self):
        a=[{"kl_base_vs_candidate":0.0314,"argmax_match":True}]
        self.assertEqual(classify_quality(a,False),("failed_first_logits_quality",False))
        self.assertEqual(classify_quality(a,True),("performance_only_quality_unqualified",False))

    def test_performance_only_never_claims_qualification(self):
        a=[{"kl_base_vs_candidate":0.002,"argmax_match":True}]
        self.assertEqual(classify_quality(a,True),("performance_only_quality_observed",True))
        self.assertEqual(classify_quality(a,False),("pass",True))

    def test_top1_disagreement_remains_visible(self):
        a=[{"kl_base_vs_candidate":0.001,"argmax_match":False}]
        self.assertEqual(classify_quality(a,True),("performance_only_quality_unqualified",False))


if __name__=="__main__": unittest.main()
