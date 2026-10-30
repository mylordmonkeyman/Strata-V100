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
        root = Path(__file__).parent
        repo = root.parent
        h = repo / 'benchmarks/qpn/results/2026-10-30-joint-heldout-routing.json'
        s = repo / 'benchmarks/qpn/results/2026-10-30-selective-mmq-supply-abba.json'
        if h.is_file() and s.is_file():
            cls.heldout = json.loads(h.read_text())
            cls.supply = json.loads(s.read_text())
        else:
            # Minimal synthetic fixtures for standalone local tests; GitHub CI
            # exercises the actual immutable aggregate files from this repo.
            from qpn_r1_aggregate_audit import MODEL_SHA, PROMPT_SHA, REQUIRED_ENV, BUDGETS
            flags = dict(REQUIRED_ENV, STRATA_QPN_PREFILL_EXPERIMENTAL='0')
            scenarios = []
            rows = [(48, 0, 0, 78, 9), (96, 2, 6442, 138, 14),
                    (192, 6, 16006, 245, 39), (272, 7, 18381, 374, 67),
                    (384, 10, 25990, 525, 93), (768, 12, 30774, 1198, 203)]
            for budget, (selected, n, entries, fallbacks, verified) in zip(BUDGETS, rows):
                scenarios.append(dict(replacement_payload_budget_mib=budget,
                                      selected_packed_experts=selected,
                                      upper_bound_complete_qpn_groups=n,
                                      upper_bound_qpn_entries=entries,
                                      prefill_routed_entries=982560,
                                      optimistic_qpn_entry_share=entries/982560,
                                      prefill_packed_canonical_fallback_expert_uses=fallbacks,
                                      verification_distinct_packed_experts_to_restore=verified))
            cls.heldout = dict(schema=1, model_sha256=MODEL_SHA,
                              control_flags=flags, performance_qualified=False,
                              model_quality_qualified=False, actual_qpn_decode_executed=False,
                              actual_accepted_token_decode_qualified=False,
                              common_observed_resident_experts=9293,
                              requests=[dict(input_tokens=n, first_logit_equal=True,
                                             verification_windows=16, prompt_sha256=c*64)
                                        for n,c in ((1024, 'a'), (2048, 'b'))],
                              scenarios=scenarios)
            variants = ['transition-fallback', 'selective-supply',
                        'selective-supply', 'transition-fallback']
            timings = [17077.8, 16849.2, 17291.6, 17305.9]
            skippeds = [0, 13600129904, 13605659536, 0]
            runs = []
            for i, (mode, t, skipped) in enumerate(zip(variants, timings, skippeds)):
                runs.append(dict(index=i, mode=mode, returncode=0,
                                 prefill_tokens=2047, prefill_ms=t, qpn_groups=0,
                                 trace=False, control_flags=flags,
                                 selective_supply=dict(chunk=0, tokens=1024,
                                  entries=15347, bytes=42431631152-skipped,
                                  skipped_bytes=skipped)))
            cls.supply = dict(schema=1, model_sha256=MODEL_SHA,
                              prompt_sha256=PROMPT_SHA, control_flags=flags,
                              order=variants, performance_only=True,
                              numerical_quality_qualified=False,
                              status='mmq_abba_bitwise_parity_passed_quality_unqualified',
                              runs=runs, comparisons=[dict(bitwise_equal=True,
                                    n_logits=248320, relative_l2=0.0) for _ in range(6)])

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

    def test_env_controls_immutable(self):
        h = copy.deepcopy(self.heldout)
        h['control_flags']['STRATA_PREFILL_MMQ'] = '0'
        with self.assertRaises(ValueError): audit(h, self.supply)


if __name__ == '__main__':
    unittest.main()