import unittest
from qpn_full_model_ab import ab_modes, classify_quality


class PerformanceFirstABTest(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(ab_modes(True), ("qpn-base","qpn-fused","qpn-fused","qpn-base"))
        self.assertEqual(ab_modes(False), ("mmq","qpn-fused","qpn-fused","mmq"))

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
