import array
import unittest
from qpn_transition_abba import ORDER, mode_flags, compare, evaluate, supply_plan, supply_mode_flags, validate_supply_traces, evaluate_supply


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


class SupplyControlTest(unittest.TestCase):
    def traces(self, mode):
        return [dict(chunk=c, tokens=t, stream_all=(i == 0 and mode == 'transition-fallback'),
                     layers=[dict(layer=l, h2d_count=3, products=[[0, 1, 0]],
                                  routed=dict(groups=4, resident_groups=2, rows=10, resident_rows=6))
                             for l in (0, 24, 47)])
                for i, (c, t) in enumerate(((0, 1024), (1024, 1023)))]

    def test_forced_staged_does_not_emit_transition_counter(self):
        staged = supply_mode_flags('forced-staged')
        transition = supply_mode_flags('transition-fallback')
        self.assertEqual(staged['STRATA_QPN_DIAG_GG_TRANSITION_FALLBACK'], '0')
        self.assertEqual(transition['STRATA_QPN_DIAG_GG_TRANSITION_FALLBACK'], '1')
        self.assertEqual(staged['STRATA_QPN_DIAG_GG_DISABLE_STAGED'], '0')
        self.assertEqual(staged['STRATA_QPN_DIAG_GG_DISABLE_STREAM'], '0')
        with self.assertRaises(ValueError):
            supply_mode_flags('both-off')

    def test_complete_accounting_both_supply_paths(self):
        self.assertEqual(supply_plan(), [('forced-staged', False), ('forced-staged', True),
                                         ('transition-fallback', False), ('transition-fallback', True)])
        for mode in ('forced-staged', 'transition-fallback'):
            l = validate_supply_traces(self.traces(mode), mode)[0]['layers'][0]
            self.assertEqual(l['resident_group_fraction'], .5)
            self.assertEqual(l['resident_row_fraction'], .6)
            self.assertEqual(l['copies_beyond_routed_nonresident'], 1)

    def test_partial_or_wrong_regime_rejected(self):
        for mutation in ('chunk', 'layer', 'route', 'products', 'regime'):
            t = self.traces('transition-fallback')
            if mutation == 'chunk': t.pop()
            if mutation == 'layer': t[0]['layers'].pop()
            if mutation == 'route': t[0]['layers'][0]['routed'] = None
            if mutation == 'products': t[0]['layers'][0]['products'] = []
            if mutation == 'regime': t[0]['stream_all'] = False
            with self.assertRaises(ValueError): validate_supply_traces(t, 'transition-fallback')

    def test_trace_parity_gate_and_no_instrumented_speed_claim(self):
        x, y = (array.array('f', a) for a in ([1., 2.], [1., 2.01]))
        s = dict(runs=[{}]*4)
        self.assertTrue(evaluate_supply(s, [x, x, x, x]))
        self.assertNotIn('median_prefill_speedup', s)
        self.assertFalse(evaluate_supply(s, [x, y, x, x]))
        with self.assertRaises(ValueError): evaluate_supply(s, [x])

    def test_signed_zero_is_not_bitwise_equal(self):
        self.assertFalse(compare(array.array('f', [0., 1.]), array.array('f', [-0., 1.]),
                                 'off', 'trace')['bitwise_equal'])


if __name__ == '__main__':
    unittest.main()
