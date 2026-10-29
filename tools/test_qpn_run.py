import unittest

from qpn_run import summarize


class SummaryTest(unittest.TestCase):
    def test_missing_or_invalid_timings_cannot_pass(self):
        rows = [{'measurement': {'numerics_pass': True}, 'returncode': 0},
                {'measurement': {'numerics_pass': True}, 'returncode': 0,
                 'completed_expert_speedup': float('nan')},
                {'measurement': {'numerics_pass': True}, 'returncode': 0,
                 'parse_error': 'timing repetitions missing', 'completed_expert_speedup': 2},
                {'measurement': {'numerics_pass': True}, 'returncode': 0,
                 'completed_expert_speedup': 1.2}]
        report = summarize(rows)
        self.assertEqual((report['passed'], report['failed']), (1, 3))
        self.assertEqual(report['best_completed_expert_speedup'], 1.2)


if __name__ == '__main__': unittest.main()
