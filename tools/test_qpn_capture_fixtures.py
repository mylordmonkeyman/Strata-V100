import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qpn_capture_fixtures import export, routed_samples
from qpn_m0_bench import digest


class RoutedFixtureTest(unittest.TestCase):
    def test_actual_rows_are_matched_without_repeating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / 'layer-24.moe'
            x = np.arange(4*2560, dtype='<f4').reshape(4, 2560)
            ids = np.array([[3, 4], [4, 5], [3, 4], [4, 7]], dtype='<i4')
            capture.write_bytes(struct.pack('<3q', 4, 2560, 2) + x.tobytes() + ids.tobytes())
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'baseline_source': '2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926',
                'model_revision': 'test-revision', 'runs': [{'returncode': 0,
                'capture': {'activation_dump_sha256': digest(capture)}}]}))
            def fake_audit(pack, layer, expert, n_expert):
                self.assertEqual((layer, n_expert), (24, 512))
                return {'expert_sha256': str(expert)}, bytes([expert])
            with patch('qpn_capture_fixtures.audit', side_effect=fake_audit):
                index = export(root, root, manifest, root / 'fixtures')
            for row in index['fixtures']:
                expected = np.flatnonzero((ids == row['expert']).any(axis=1))
                np.testing.assert_array_equal(row['source_rows'], expected)
                got = np.fromfile(root / 'fixtures' / row['activations'], dtype='<f4').reshape(-1, 2560)
                np.testing.assert_array_equal(got, x[expected])
            self.assertEqual(index['fixtures'][0]['expert'], 4)
            self.assertEqual(index['fixtures'][0]['rows'], 4)
            capture.write_bytes(capture.read_bytes()[:-1])
            with self.assertRaises(ValueError): export(root, root, manifest, root / 'other')

    def test_duplicate_route_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'layer-0.moe'
            path.write_bytes(struct.pack('<3q', 1, 2560, 2) + np.ones(2560, dtype='<f4').tobytes()
                             + np.array([1, 1], dtype='<i4').tobytes())
            with self.assertRaises(ValueError): routed_samples(path)


if __name__ == '__main__': unittest.main()
