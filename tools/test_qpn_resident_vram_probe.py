"""CPU-only telemetry parser tests; no private pack, GPU or prompt required."""
import unittest
from qpn_resident_vram_probe import parse_gpu_memory_csv, summarize_samples


class TestVramProbe(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_gpu_memory_csv('2048, 30520, 32568\n'),
                         (2048,30520,32568))
        for x in ('','hello','-1,5,32568','5,40000,32568','5,5,100'):
            with self.assertRaises(ValueError):
                parse_gpu_memory_csv(x)

    def test_minimum_observed_free_is_guarded(self):
        x=summarize_samples([(2050,30518,32568)]*10+[(1800,30768,32568)])
        self.assertEqual((x['samples'],x['min_free_mib'],
                          x['peak_headroom_after_512mib_guard_mib']),
                         (11,1800,1288))

    def test_reject_short_or_inconsistent(self):
        with self.assertRaises(ValueError):
            summarize_samples([(1,32567,32568)]*9)
        with self.assertRaises(ValueError):
            summarize_samples([(1,32567,32568)]*10+[(1,1,32400)])


if __name__=='__main__':
    unittest.main()
