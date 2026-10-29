import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qpn_m0_bench import capture_summary


class CaptureSummaryTest(unittest.TestCase):
    def write_capture(self, directory, x, ids):
        path = Path(directory) / 'capture.moe'
        path.write_bytes(struct.pack('<3q', len(x), 2560, ids.shape[1])
                         + x.astype('<f4').tobytes() + ids.astype('<i4').tobytes())
        return path

    def test_counts_real_routed_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_capture(directory, np.ones((3, 2560)), np.array([[1, 2], [1, 3], [2, 4]]))
            got = capture_summary(path)
            self.assertEqual(got['expert_counts'], {1: 2, 2: 2, 3: 1, 4: 1})
            self.assertEqual(got['entry_count_histogram'], {1: 2, 2: 2})
            self.assertEqual(got['rms_activation'], 1)

    def test_rejects_corruption_and_nonfinite_activations(self):
        with tempfile.TemporaryDirectory() as directory:
            for change in ('truncate', 'append', 'bad_id', 'nan'):
                x = np.ones((2, 2560))
                ids = np.array([[1], [512 if change == 'bad_id' else 2]])
                if change == 'nan': x[0, 0] = np.nan
                path = self.write_capture(directory, x, ids)
                if change == 'truncate': path.write_bytes(path.read_bytes()[:-1])
                if change == 'append': path.write_bytes(path.read_bytes() + b'x')
                with self.subTest(change=change), self.assertRaises(ValueError):
                    capture_summary(path)


if __name__ == '__main__': unittest.main()
