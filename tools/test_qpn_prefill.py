import unittest
from qpn_prefill_run import crossover, captured_token_buckets, tile_variants


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


class CapturedActivationTest(unittest.TestCase):
    def test_buckets_use_only_actual_rows(self):
        self.assertEqual(captured_token_buckets(1), [1])
        self.assertEqual(captured_token_buckets(3), [1, 2, 3])
        self.assertEqual(captured_token_buckets(9), [1, 2, 4, 8, 9])
        self.assertEqual(captured_token_buckets(29), [1, 2, 4, 8, 16, 17, 24, 29])
        self.assertEqual(captured_token_buckets(31), [1, 2, 4, 8, 16, 17, 24, 31])
        self.assertEqual(captured_token_buckets(32), [1, 2, 4, 8, 16, 17, 24, 31, 32])

    def test_rejects_invalid_counts(self):
        for rows in (0, -1, 33, 1.5, None):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                captured_token_buckets(rows)


class TokenTileSweepTest(unittest.TestCase):
    def test_distinct_schedules_and_default_control(self):
        vs=[0,8,16]
        self.assertEqual(tile_variants(1,vs),[0])
        self.assertEqual(tile_variants(8,vs),[0])
        self.assertEqual(tile_variants(12,vs),[0,8])
        self.assertEqual(tile_variants(16,vs),[0,8])
        self.assertEqual(tile_variants(29,vs),[0,8,16])
        self.assertEqual(tile_variants(32,vs),[0,8,16])


if __name__=="__main__": unittest.main()
