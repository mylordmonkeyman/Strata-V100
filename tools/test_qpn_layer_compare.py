import tempfile
from pathlib import Path
import struct
import unittest
import numpy as np
from qpn_layer_compare import records, stats, N, K

class LayerCompareTest(unittest.TestCase):
    def test_stats_reject_nonfinite_and_detect_change(self):
        a=np.array([1.,2.],dtype=np.float16)
        self.assertTrue(stats(a,a)['bitwise_equal'])
        self.assertGreater(stats(a,a+1)['relative_l2'],0)
        with self.assertRaises(ValueError):stats(a,np.array([float('nan'),2]))
    def test_zero_reference_and_signed_zero_remain_visible(self):
        self.assertIsNone(stats(np.array([0.]),np.array([1.]))['relative_l2'])
        self.assertFalse(stats(np.array([0.]),np.array([-0.]))['bitwise_equal'])
    def test_private_record_parser_and_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'features.bin'
            record=struct.pack('<4i',0,0,1,1)+b'\0'*(2*K*2+2*N)
            p.write_bytes(record)
            rows=list(records(p));self.assertEqual(rows[0][0],(0,0));self.assertEqual(rows[0][1],1)
            p.write_bytes(record[:-1])
            with self.assertRaises(ValueError):list(records(p))
            p.write_bytes(record+record)
            with self.assertRaises(ValueError):list(records(p))
    def test_geometry_rejected_before_payload_allocation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'features.bin';p.write_bytes(struct.pack('<4i',0,0,2**30,1))
            with self.assertRaises(ValueError):list(records(p))

if __name__ == '__main__':unittest.main()
