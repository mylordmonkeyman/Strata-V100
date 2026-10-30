import array
import unittest
from qpn_transition_abba import ORDER, mode_flags, compare, evaluate


class TransitionABTest(unittest.TestCase):
    def summary(self):
        return {'runs': [dict(mode=m, prefill_ms=100+i*10, ttft_ms=200+i*10)
                         for i, m in enumerate(ORDER)], 'comparisons': [], 'status': 'pending'}

    def test_abba_modes_and_opt_in(self):
        self.assertEqual(ORDER, ('both-off', 'transition-fallback', 'transition-fallback', 'both-off'))
        self.assertEqual(mode_flags('both-off'),
                         dict(STRATA_QPN_DIAG_GG_DISABLE_STREAM='1',
                              STRATA_QPN_DIAG_GG_DISABLE_STAGED='1',
                              STRATA_QPN_DIAG_GG_TRANSITION_FALLBACK='0'))
        self.assertEqual(mode_flags('transition-fallback'),
                         dict(STRATA_QPN_DIAG_GG_DISABLE_STREAM='0',
                              STRATA_QPN_DIAG_GG_DISABLE_STAGED='0',
                              STRATA_QPN_DIAG_GG_TRANSITION_FALLBACK='1'))
        with self.assertRaises(ValueError):
            mode_flags('both-on')

    def test_all_bitwise_pass_and_aggregate_only(self):
        x = array.array('f', [0.0, 2.0, -1.0])
        s = self.summary()
        self.assertTrue(evaluate(s, [x, x, x, x]))
        self.assertEqual(s['status'], 'mmq_abba_bitwise_parity_passed_quality_unqualified')
        self.assertEqual(len(s['comparisons']), 6)
        self.assertTrue(all(d['bitwise_equal'] for d in s['comparisons']))
        self.assertFalse(any('argmax_base' in d or 'argmax_candidate' in d for d in s['comparisons']))
        self.assertEqual(s['median_prefill_ms']['both-off'], 115)
        self.assertEqual(s['median_prefill_ms']['transition-fallback'], 115)
        self.assertEqual(s['median_prefill_speedup'], 1.0)

    def test_cross_arm_and_within_arm_mismatch_fail_closed(self):
        x = array.array('f', [0.0, 2.0, -1.0])
        y = array.array('f', [0.0, 2.01, -1.0])
        for outputs in ([x, y, y, x], [x, x, y, x]):
            s = self.summary()
            self.assertFalse(evaluate(s, outputs))
            self.assertEqual(s['status'], 'failed_mmq_abba_bitwise_parity')

    def test_incomplete_abba_fails_closed(self):
        with self.assertRaises(ValueError):
            evaluate(self.summary(), [array.array('f', [1.0])])

    def test_no_argmax_ids_in_comparison(self):
        x = array.array('f', [1.0, 2.0])
        d = compare(x, x, 'both-off', 'transition-fallback')
        self.assertTrue(d['bitwise_equal'])
        self.assertNotIn('argmax_base', d)
        self.assertNotIn('argmax_candidate', d)


if __name__ == '__main__':
    unittest.main()
