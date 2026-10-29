import unittest
from qpn_prefill_run import crossover


def arm(name, available=True, valid=True, times=None):
    return {"backend":name,"available":available,"numerics_pass":valid,
            "completed_expert_us":[10]*6 if times is None else times}


class CrossoverTest(unittest.TestCase):
    def test_excludes_residency_control_and_failing_backends(self):
        self.assertEqual(crossover({"arms":[arm("cached_fp16_cublas_control",times=[1]*6),
                                            arm("qpn_direct",valid=False,times=[2]*6),
                                            arm("nvfp4_w4a8_mmq",available=False,times=[3]*6),
                                            arm("dequant_fp16_cublas",times=[20]*6)]}),"dequant_fp16_cublas")

    def test_requires_real_repetitions_and_handles_all_unavailable(self):
        self.assertIsNone(crossover({"arms":[arm("qpn_direct",times=[1]),arm("nvfp4_w4a8_mmq",available=False)]}))

    def test_uses_median(self):
        self.assertEqual(crossover({"arms":[arm("qpn_direct",times=[9,9,9,9,9,1000]),
                                            arm("dequant_fp16_cublas",times=[10]*6)]}),"qpn_direct")


if __name__=="__main__": unittest.main()
