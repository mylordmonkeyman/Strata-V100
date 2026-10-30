"""Host-only constructed synthetic DAG tests: never GPU/private route data."""
import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).parent))
from qpn_r1_dag_screen import MODEL_SHA, CONTROLS, EVIDENCE_FLAGS, screen, ensure_distinct_paths


def fixture():
    # CPU staging overlaps the first GPU expert operation, but GPU operations
    # serialize. Even making the eligible work FREE leaves the other lane busy.
    return dict(schema=1, model_sha256=MODEL_SHA,
                full_request_latency_ms=10.0,
                control_flags=dict(CONTROLS, STRATA_QPN_PREFILL_EXPERIMENTAL='0'),
                evidence_flags={k:True for k in EVIDENCE_FLAGS},
                nodes=[
                    dict(id=0,lane=0,order=0,phase='prefill',kind='cpu_stage',duration_ms=8,eligible=False),
                    dict(id=1,lane=1,order=0,phase='prefill',kind='expert_compute',duration_ms=2,eligible=True),
                    dict(id=2,lane=1,order=1,phase='verification',kind='other_compute',duration_ms=8,eligible=False),
                ], edges=[])


class R1DagScreenTests(unittest.TestCase):
    def test_parallel_operation_not_exposed_after_other_work(self):
        x=fixture()
        r=screen(x)
        self.assertEqual(r['graph_screen'], 'CONDITIONAL_FIXED_DAG_POSSIBLE')
        self.assertAlmostEqual(r['conditional_max_latency_reduction_fraction'], .2)
        self.assertEqual(r['r1_feasibility'], 'UNKNOWN')
        self.assertFalse(r['qualifies_actual_qpn_kernel_speed'])

    def test_critical_path_switch_and_limited_savings(self):
        x=fixture()
        x['nodes'][0]['duration_ms']=9.9
        r=screen(x)
        self.assertAlmostEqual(r['conditional_max_latency_reduction_fraction'], .01)
        self.assertEqual(r['graph_screen'], 'CONDITIONAL_FIXED_DAG_BELOW_FIVE_PERCENT')
        self.assertEqual(r['r1_feasibility'], 'UNKNOWN')

    def test_missing_causal_evidence_fails_closed(self):
        x=fixture()
        x['evidence_flags']['all_cpu_gpu_work_and_wait_edges']=False
        r=screen(x)
        self.assertEqual(r['graph_screen'],'UNKNOWN_INCOMPLETE_CAUSAL_EVIDENCE')
        self.assertIsNone(r['conditional_max_latency_reduction_fraction'])
        self.assertIsNone(r['conditional_assumed_factor_latency_reduction_fraction'])
        x=fixture()
        x['full_request_latency_ms']=40
        r=screen(x)
        self.assertEqual(r['graph_screen'],'UNKNOWN_INCOMPLETE_CAUSAL_EVIDENCE')
        self.assertIsNone(r['conditional_max_latency_reduction_fraction'])

    def test_cross_resource_dependency_reduces_overlap(self):
        x=fixture()
        x['edges']=[[0,1]]
        x['full_request_latency_ms']=18
        r=screen(x)
        self.assertEqual(r['graph_screen'], 'CONDITIONAL_FIXED_DAG_POSSIBLE')
        self.assertAlmostEqual(r['conditional_max_latency_reduction_fraction'], 2/18)

    def test_cycle_fails_even_when_lane_edges_implicit(self):
        x=fixture()
        x['edges']=[[2,1]]
        with self.assertRaises(ValueError):screen(x)

    def test_reject_unqualified_decode_and_duplicate_order(self):
        x=fixture(); x['nodes'][1]['phase']='verification'
        with self.assertRaises(ValueError):screen(x)
        x=fixture(); x['nodes'][2]['order']=0
        with self.assertRaises(ValueError):screen(x)

    def test_no_unrecognized_fields_or_private_data_in_report(self):
        x=fixture()
        x['raw_prompt']='DO_NOT_UPLOAD'
        x['nodes'][0]['route_ids']='DO_NOT_UPLOAD'
        with self.assertRaises(ValueError):screen(x)
        x=fixture()
        x['raw_prompt']='DO_NOT_UPLOAD'
        report=screen(x)
        self.assertNotIn('DO_NOT_UPLOAD',json.dumps(report))
        self.assertNotIn('nodes',report)
        self.assertNotIn('edges',report)

    def test_reject_nonfinite_bad_flags_and_bad_model(self):
        for field,bad in [('duration_ms',float('nan')),('duration_ms',True),('duration_ms',-2)]:
            x=fixture(); x['nodes'][1][field]=bad
            with self.assertRaises(ValueError):screen(x)
        x=fixture(); x['evidence_flags']['representative_heldout']='true'
        with self.assertRaises(ValueError):screen(x)
        x=fixture(); x['model_sha256']='0'*64
        with self.assertRaises(ValueError):screen(x)

    def test_huge_sparse_lane_rank_fails_without_expensive_range(self):
        x=fixture()
        x['nodes'][2]['order']=10**1000
        with self.assertRaises(ValueError): screen(x)

    def test_finite_individual_durations_cannot_overflow_path(self):
        x=fixture()
        x['nodes'][1]['duration_ms']=1e308
        x['nodes'][2]['duration_ms']=1e308
        x['full_request_latency_ms']=1e308
        with self.assertRaises(ValueError): screen(x)

    def test_huge_integer_and_unhashable_node_fields_rejected(self):
        x=fixture()
        x['nodes'][1]['duration_ms']=10**1000
        with self.assertRaises(ValueError): screen(x)
        x=fixture()
        x['nodes'][1]['phase']=['prefill']
        with self.assertRaises(ValueError): screen(x)

    def test_nonfinite_wall_reconciliation_rejected(self):
        x=fixture()
        x['full_request_latency_ms']=1e-308
        with self.assertRaises(ValueError): screen(x)

    def test_report_alias_cannot_overwrite_private_graph(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'private.json'
            source.write_text('PRIVATE TEST SENTINEL')
            alias=Path(d)/'same.json'
            alias.symlink_to(source)
            with self.assertRaises(ValueError): ensure_distinct_paths(source,source)
            with self.assertRaises(ValueError): ensure_distinct_paths(source,alias)
            self.assertEqual(source.read_text(),'PRIVATE TEST SENTINEL')
            ensure_distinct_paths(source,Path(d)/'safe-report.json')

    def test_no_eligible_compute_produces_zero_ceiling(self):
        x=fixture(); x['nodes'][1]['eligible']=False
        r=screen(x)
        self.assertEqual(r['eligible_prefill_operation_count'],0)
        self.assertAlmostEqual(r['conditional_max_latency_reduction_fraction'],0)
        self.assertEqual(r['graph_screen'],'CONDITIONAL_FIXED_DAG_BELOW_FIVE_PERCENT')


if __name__=='__main__':unittest.main()