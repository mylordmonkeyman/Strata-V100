import array
import unittest
from qpn_transition_abba import (
    REPEAT_MEASURED_ORDER, REPEAT_WARMUP_ORDER, evaluate_early_stage_repeat
)

class RepeatGateTests(unittest.TestCase):
    def sample(self, wall_a=100., wall_b=88., decode_a=30., decode_b=30.):
        rows=[]
        for i,mode in enumerate(REPEAT_MEASURED_ORDER):
            candidate=mode=='early-stage'
            rows.append(dict(mode=mode, prefill_ms=50. if candidate else 70.,
                ttft_ms=70. if candidate else 90., wall_s=wall_b if candidate else wall_a,
                decode_toks_per_s=decode_b if candidate else decode_a, decode_tokens=128,
                selective_supply=dict(entries=15347,copies=8880,skipped=6467,
                    bytes=24551566080,skipped_bytes=17880065072,jobs=6027,skipped_jobs=2713),
                early_stage=dict(cpu_jobs_omitted=2455 if candidate else 0,
                    cpu_bytes_omitted=2455*2764816 if candidate else 0)))
        return dict(runs=rows,warmups=[{},{}],status='pending',
                    full_request_performance_qualified=False)
    def test_balanced_order_and_positive(self):
        self.assertEqual(len(REPEAT_WARMUP_ORDER),2)
        self.assertEqual(len(REPEAT_MEASURED_ORDER),12)
        self.assertEqual(REPEAT_MEASURED_ORDER.count('early-stage'),6)
        self.assertEqual(REPEAT_MEASURED_ORDER.count('selective-supply'),6)
        x=array.array('f',[1.,2.,3.])
        s=self.sample()
        self.assertTrue(evaluate_early_stage_repeat(s,[x]*12))
        self.assertEqual(s['status'],'repeat_screen_positive_unqualified')
        self.assertTrue(s['all_blocks_full_process_5pct'])
        self.assertFalse(s['full_request_performance_qualified'])
        self.assertEqual(len(s['four_run_blocks']),3)
    def test_decode_regression_is_not_promoted(self):
        x=array.array('f',[1.,2.,3.])
        s=self.sample(decode_b=28.)
        self.assertTrue(evaluate_early_stage_repeat(s,[x]*12))
        self.assertTrue(s['decode_regression_over_5pct'])
        self.assertEqual(s['status'],'repeat_screen_no_material_gain_or_decode_regression')
    def test_exact_h2d_and_bitwise_gates(self):
        x=array.array('f',[1.,2.,3.])
        y=array.array('f',[1.,2.,3.1])
        s=self.sample()
        self.assertFalse(evaluate_early_stage_repeat(s,[x]*11+[y]))
        self.assertEqual(s['status'],'failed_mmq_repeat_bitwise_parity')
        s=self.sample()
        s['runs'][1]['selective_supply']['copies']+=1
        self.assertFalse(evaluate_early_stage_repeat(s,[x]*12))
        self.assertEqual(s['status'],'failed_repeat_mismatched_H2D_or_host_work')
    def test_short_decode_fails(self):
        x=array.array('f',[1.,2.,3.])
        s=self.sample()
        s['runs'][0]['decode_tokens']=16
        self.assertFalse(evaluate_early_stage_repeat(s,[x]*12))
        self.assertEqual(s['status'],'failed_repeat_insufficient_decode_tokens')

if __name__=='__main__':
    unittest.main()
