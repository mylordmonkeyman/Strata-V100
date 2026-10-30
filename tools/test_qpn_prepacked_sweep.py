import copy
from pathlib import Path
import unittest
from qpn_prepacked_sweep import cases, command, validate

class PrepackedSweepTest(unittest.TestCase):
    def counts(self):
        return [33,48,64,96,187,508,1,8,9,11,12,16,17,20,24,32]

    def test_natural_buckets_and_decode_have_matched_supply_controls(self):
        plan=cases(self.counts())
        self.assertEqual(len({c['name'] for c in plan}),len(plan))
        decode=[c for c in plan if c['row_cap']==1]
        self.assertEqual(len(decode),3)
        self.assertTrue(all(c['expected_groups']==16 and c['expected_rows']==16 for c in decode))
        for c in plan:
            if c['row_cap']: self.assertLessEqual(c['expected_rows'],sum(self.counts()))
            cmd=command(Path('/binary'),Path('/private'),Path('/cache'),c)
            self.assertIn('--qpn-packed-file',cmd)
            self.assertIn('--qpn-chunkmajor',cmd)
            self.assertEqual('--qpn-pinned-stream' in cmd,c['regime']!='resident')
            self.assertEqual('--qpn-async-pipeline' in cmd,c['regime']=='overlap-h2d')

    def test_invalid_counts_refused(self):
        for counts in ([],[1]*15,[0]*16,[True]*16,[1025]*16):
            with self.assertRaises(ValueError): cases(counts)

    def test_gates_reject_missing_work_and_graph(self):
        c=cases(self.counts())[0]
        j=dict(fused_gu=True,chunkmajor=True,disk_prepacked=True,explicit_range=True,numerics_pass=True,
               graph_checked=True,graph_replay_equal=True,graph_changed_input_equal=True,output_tail_guard_ok=True,
               qpn_groups=c['expected_groups'],fallback_groups=16-c['expected_groups'],routed_rows=c['expected_rows'],
               row_cap=c['row_cap'],qpn_min=c['low'],qpn_max=c['high'],iterations=10,rounds=6,
               repack_cpu_us=0,oracle_rows=32,original_routed_rows=1100,mixed_vs_mmq_output_l2=.01,
               mmq_reference_l2=.01,mixed_reference_l2=.01,pinned_stream=False,async_pipeline=False,
               control_h2d_bytes=0,hybrid_h2d_bytes=0)
        for key in ('all_mmq_us','mixed_qpn_mmq_us','all_mmq_graph_us','mixed_graph_us'): j[key]=[1.]*6
        validate(j,c)
        for key,value in (('routed_rows',0),('graph_changed_input_equal',False),('repack_cpu_us',1),
                          ('mixed_reference_l2',float('nan')),('all_mmq_us',[1.])):
            bad=copy.deepcopy(j);bad[key]=value
            with self.assertRaises(ValueError): validate(bad,c)

if __name__=='__main__': unittest.main()
