import array
import unittest

from qpn_transition_abba import (
    EARLY_STAGE_ORDER, selective_flags, selective_accounting,
    early_stage_accounting, evaluate_early_stage,
)


def selective_line(copies=10428, skipped=4919, skipped_jobs=2533):
    entry = 15347
    return (f'strata qpn selective-supply: chunk=0 tokens=1024 entries={entry} '
            f'copies={copies} skipped={skipped} bytes={copies * 2764816} '
            f'skipped_bytes={skipped * 2764816} jobs=6027 '
            f'skipped_jobs={skipped_jobs}\n')


def early_line(jobs=100):
    return (f'strata qpn early-stage: chunk=0 tokens=1024 '
            f'cpu_jobs_omitted={jobs} cpu_bytes_omitted={jobs * 2764816}\n')


class EarlyStageAbaTests(unittest.TestCase):
    def test_optin_keeps_exact_mmq_supply_controls(self):
        self.assertEqual(EARLY_STAGE_ORDER,
                         ('selective-supply', 'early-stage', 'early-stage', 'selective-supply'))
        a = selective_flags('selective-supply')
        b = selective_flags('early-stage')
        self.assertEqual(a['STRATA_QPN_MMQ_SELECTIVE_SUPPLY'], '1')
        self.assertEqual(b['STRATA_QPN_MMQ_SELECTIVE_SUPPLY'], '1')
        self.assertEqual(a['STRATA_QPN_MMQ_EARLY_STAGE'], '0')
        self.assertEqual(b['STRATA_QPN_MMQ_EARLY_STAGE'], '1')
        self.assertEqual({k: v for k, v in a.items() if k != 'STRATA_QPN_MMQ_EARLY_STAGE'},
                         {k: v for k, v in b.items() if k != 'STRATA_QPN_MMQ_EARLY_STAGE'})
        self.assertEqual(selective_flags('transition-fallback')['STRATA_QPN_MMQ_SELECTIVE_SUPPLY'], '0')

    def test_strict_cpu_omission_accounting(self):
        raw = selective_line() + early_line(90)
        supply = selective_accounting(raw, 'early-stage')
        self.assertEqual(early_stage_accounting(raw, 'early-stage', supply),
                         dict(cpu_jobs_omitted=90, cpu_bytes_omitted=90 * 2764816))
        self.assertEqual(early_stage_accounting(selective_line(), 'selective-supply', supply),
                         dict(cpu_jobs_omitted=0, cpu_bytes_omitted=0))
        for malformed in (early_line(2534), early_line(90) * 2,
                          early_line(0) + early_line(90), early_line(90).replace('tokens=1024', 'tokens=768'),
                          early_line(90).replace('chunk=0', 'chunk=1'),
                          early_line(90).replace('cpu_bytes_omitted=', 'raw_weights=')):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                early_stage_accounting(malformed, 'early-stage', supply)
        with self.assertRaises(ValueError):
            early_stage_accounting(early_line(), 'selective-supply', supply)
        with self.assertRaises(ValueError):
            early_stage_accounting(early_line(), 'unsupported', supply)

    def test_early_stage_rejects_same_arm_supply_difference(self):
        # Even two candidate runs must have comparable H2D workloads.
        x = array.array('f', [1., 2., 3.])
        s = self.summary(100., 100., 90, 90)
        self.assertTrue(evaluate_early_stage(s, [x, x, x, x]))
        self.assertEqual(s['status'], 'early_stage_initial_screen_no_material_benefit')
        self.assertFalse(s['full_request_performance_qualified'])
        self.assertFalse(s['exploratory_prefill_5pct_gate'])
        s = self.summary(100., 94., 90, 90)
        self.assertTrue(evaluate_early_stage(s, [x, x, x, x]))
        self.assertTrue(s['exploratory_prefill_5pct_gate'])
        self.assertFalse(s['full_request_performance_qualified'])
        s = self.summary(100., 94., 90, 90)
        s['runs'][1]['selective_supply']['copies'] += 30
        self.assertFalse(evaluate_early_stage(s, [x, x, x, x]))
        self.assertEqual(s['status'], 'failed_mismatched_H2D')

    def test_changed_logits_fails_parity(self):
        s = self.summary(100., 94., 90, 90)
        x = array.array('f', [1., 2., 3.])
        y = array.array('f', [1., 2., 3.01])
        self.assertFalse(evaluate_early_stage(s, [x, y, y, x]))
        self.assertEqual(s['status'], 'failed_mmq_abba_bitwise_parity')

    def summary(self, a_ms, b_ms, omitted1, omitted2):
        modes = EARLY_STAGE_ORDER
        ms = (a_ms, b_ms, b_ms, a_ms)
        runs = []
        for i, mode in enumerate(modes):
            supply = selective_accounting(selective_line(), mode)
            jobs = (omitted1 if i == 1 else omitted2 if i == 2 else 0)
            runs.append(dict(mode=mode, prefill_ms=ms[i], ttft_ms=ms[i] + 200.,
                             selective_supply=supply,
                             early_stage=dict(cpu_jobs_omitted=jobs,
                                              cpu_bytes_omitted=jobs * 2764816)))
        return dict(runs=runs, comparisons=[], status='pending')


if __name__ == '__main__':
    unittest.main()
