"""Host-only tests. No private samples or GPU runtime."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from qpn_r1_aggregate_audit import audit


class AggregateAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Always validate the immutable committed benchmark evidence. Unit tests
        # may mutate private in-memory copies, but a missing evidence file MUST
        # fail CI rather than silently substituting synthetic data.
        repo = Path(__file__).resolve().parent.parent
        h = repo / 'benchmarks/qpn/results/2026-10-30-joint-heldout-routing.json'
        s = repo / 'benchmarks/qpn/results/2026-10-30-selective-mmq-supply-abba.json'
        if not h.is_file() or not s.is_file():
            raise RuntimeError('required pinned aggregate evidence files unavailable')
        cls.heldout = json.loads(h.read_text())
        cls.supply = json.loads(s.read_text())

    def test_real_safe_aggregate_mmq_only_unknown(self):
        result = audit(self.heldout, self.supply)
        self.assertEqual(result['r1_full_request_feasibility'], 'UNKNOWN')
        self.assertEqual(result['static_single_representation_admission'], 'STOPPED')
        self.assertFalse(result['evidence_flags']['eligible_exposed_full_request_time_measured'])
        self.assertAlmostEqual(result['routing_scenarios'][-1]['optimistic_entry_share_fraction'],
                               30774 / 982560)
        self.assertAlmostEqual(result['supply_observation']['median_prefill_control_ms'], 17191.85)
        self.assertAlmostEqual(result['supply_observation']['median_prefill_selective_ms'], 17070.4)
        self.assertGreater(result['supply_observation']['first_chunk_selective_h2d_reduction_fraction'], .32)
        self.assertLess(result['supply_observation']['median_prefill_reduction_fraction'], .008)

    def test_reject_false_eligibility_or_missing_routing(self):
        h = copy.deepcopy(self.heldout)
        h['scenarios'][-1]['optimistic_qpn_entry_share'] = .90
        with self.assertRaises(ValueError): audit(h, self.supply)
        h = copy.deepcopy(self.heldout)
        h['requests'][1]['prompt_sha256'] = h['requests'][0]['prompt_sha256']
        with self.assertRaises(ValueError): audit(h, self.supply)
        h = copy.deepcopy(self.heldout)
        h['actual_qpn_decode_executed'] = True
        with self.assertRaises(ValueError): audit(h, self.supply)

    def test_reject_nonparity_or_missing_abba(self):
        s = copy.deepcopy(self.supply)
        s['comparisons'][2]['bitwise_equal'] = False
        with self.assertRaises(ValueError): audit(self.heldout, s)
        s = copy.deepcopy(self.supply)
        s['runs'][1]['prefill_ms'] = float('nan')
        with self.assertRaises(ValueError): audit(self.heldout, s)
        s = copy.deepcopy(self.supply)
        s['runs'][1]['selective_supply']['skipped_bytes'] = 0
        with self.assertRaises(ValueError): audit(self.heldout, s)

    def test_raw_unrecognized_fields_never_exported(self):
        h = copy.deepcopy(self.heldout)
        s = copy.deepcopy(self.supply)
        h['private_route_ids'] = ['DO_NOT_LEAK']
        h['scenarios'][0]['expert_id'] = 'DO_NOT_LEAK'
        s['raw_log'] = 'DO_NOT_LEAK'
        result = audit(h, s)
        self.assertNotIn('DO_NOT_LEAK', json.dumps(result))
        self.assertNotIn('private_route_ids', json.dumps(result))

    def test_reject_changed_source_or_unbacked_budget(self):
        h = copy.deepcopy(self.heldout)
        h['source_commit'] = '0' * 40
        with self.assertRaises(ValueError): audit(h, self.supply)
        s = copy.deepcopy(self.supply)
        s['source_commit'] = '0' * 40
        with self.assertRaises(ValueError): audit(self.heldout, s)
        h = copy.deepcopy(self.heldout)
        h['scenarios'][1]['selected_packed_experts'] = 4096
        with self.assertRaises(ValueError): audit(h, self.supply)

    def test_reject_mismatched_transfer_counts_and_published_median(self):
        s = copy.deepcopy(self.supply)
        s['runs'][1]['selective_supply']['copies'] += 1
        with self.assertRaises(ValueError): audit(self.heldout, s)
        s = copy.deepcopy(self.supply)
        s['median_prefill_ms']['selective-supply'] += 1
        with self.assertRaises(ValueError): audit(self.heldout, s)

    def test_env_controls_immutable(self):
        h = copy.deepcopy(self.heldout)
        h['control_flags']['STRATA_PREFILL_MMQ'] = '0'
        with self.assertRaises(ValueError): audit(h, self.supply)


if __name__ == '__main__':
    unittest.main()