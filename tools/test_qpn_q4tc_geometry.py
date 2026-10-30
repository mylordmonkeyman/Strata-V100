"""Host-only exact Q4TC geometry tests; no GPU, GGUF or private model data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from qpn_q4tc_geometry import (
    geometry_screen, row_major_quant_bytes, MIB, QAD_SCALE_TAIL_BYTES
)


class Q4TCGeometryTests(unittest.TestCase):
    def test_exact_three_projection_bytes_and_scale_tail(self):
        x = geometry_screen()
        c = x['canonical_q4']
        self.assertEqual((c['gate_body_bytes'], c['up_body_bytes'], c['down_body_bytes']),
                         (921600, 921600, 921600))
        self.assertEqual(c['three_projection_body_bytes'], 2764800)
        self.assertEqual(c['original_qad_native_blob_body_plus_scale_tail_bytes'],
                         2764800 + QAD_SCALE_TAIL_BYTES)

    def test_unmodified_superfast_rejects_640_down(self):
        x = geometry_screen()['source_qpn_geometry_only']
        self.assertTrue(x['unmodified_superfast_gate_eligible'])
        self.assertTrue(x['unmodified_superfast_up_eligible'])
        self.assertFalse(x['unmodified_superfast_down_eligible'])
        self.assertEqual((x['down_full_256_column_tiles'], x['down_remainder_columns']), (2, 128))
        self.assertAlmostEqual(x['down_complete_tile_fraction'], 0.8)

    def test_partial_and_mixed_are_unimplemented_arithmetic_only(self):
        x = geometry_screen()['source_qpn_geometry_only']
        self.assertAlmostEqual(x['mixed_packed_gate_up_canonical_down_nominal_mac_fraction'], 2 / 3)
        self.assertAlmostEqual(x['custom_partial_128_down_nominal_mac_fraction_if_implemented'], 14 / 15)

    def test_padding_breaks_equal_byte_comparison(self):
        x = geometry_screen()['padded_down_hypothesis_not_equal_byte']
        self.assertEqual((x['padded_k'], x['padding_columns']), (768, 128))
        self.assertEqual(x['padded_down_body_bytes'], 1105920)
        self.assertEqual(x['extra_per_expert_bytes'], 184320)
        self.assertAlmostEqual(x['extra_down_body_fraction'], .2)
        self.assertAlmostEqual(x['extra_total_body_fraction'], 1 / 15)
        self.assertAlmostEqual(x['extra_selected_experts_mib'], 135)

    def test_extra_memory_versus_replacement_budget_not_conflated(self):
        x = geometry_screen()['cross_phase_sidecar_optimistic_bounds']
        self.assertEqual(x['observed_resident_experts'], 9293)
        self.assertEqual(x['selected_experts_2gib_replacement_screen'], 768)
        self.assertEqual(x['original_topology_guarded_spare_upper_bound_mib'], 737)
        self.assertAlmostEqual(x['selected_gate_up_sidecar_body_mib'], 1350)
        self.assertEqual(x['ideal_gate_up_sidecar_expert_count_upper_bound'], 419)
        self.assertEqual(x['ideal_all_three_sidecar_expert_count_upper_bound'], 279)
        self.assertEqual(x['ideal_full_body_replacement_expert_count_in_2gib'], 776)
        self.assertTrue(x['selected_gate_up_sidecar_exceeds_original_guarded_spare'])

    def test_quantized_geometry_rejects_unaligned_or_invalid_shapes(self):
        for shape in ((640, 2560, 256, 144), (0, 640, 256, 144),
                      (True, 640, 256, 144), (-1, 640, 256, 144),
                      (640, 2560, 0, 18)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                row_major_quant_bytes(*shape)
        self.assertEqual(row_major_quant_bytes(640, 2560, 32, 18), 921600)

    def test_no_performance_or_private_metadata_exported(self):
        x = geometry_screen()
        self.assertEqual(x['decision'], 'Q4TC_SOURCE_GEOMETRY_PARTIAL_NO_PROTOTYPE')
        self.assertEqual(x['r1_feasibility'], 'UNKNOWN')
        self.assertFalse(x['production_qualified'])
        s = json.dumps(x, allow_nan=False)
        for token in ('prompt_ids', 'expert_ids', 'raw_logits', 'routed_ids',
                      'qpn_speedup_percent', 'production_speedup'):
            self.assertNotIn(token, s)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'geometry.json'
            p.write_text(json.dumps(x, indent=2, allow_nan=False))
            self.assertEqual(json.loads(p.read_text())['decision'], x['decision'])


if __name__ == '__main__':
    unittest.main()
