"""Public synthetic block fixture; no weights, model, routes or GPU."""
import math
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from qpn_q4tc_block_oracle import (decode_q4_k, decode_iq4_nl, decode_row,
                                    scaled_dot, swiglu_from_unscaled)


def q4_block():
    scales = [1, 2, 3, 4, 63, 17, 31, 48]
    mins = [0, 1, 2, 3, 49, 31, 32, 63]
    packed = bytes(
        [scales[i] | ((scales[4 + i] & 48) << 2) for i in range(4)] +
        [mins[i] | ((mins[4 + i] & 48) << 2) for i in range(4)] +
        [(scales[4 + i] & 15) | ((mins[4 + i] & 15) << 4) for i in range(4)])
    codes = bytes(((i % 16) << 4) | (15 - i % 16) for i in range(128))
    return struct.pack('<ee', 0.5, 0.25) + packed + codes


def iq4_block():
    return struct.pack('<e', 0.25) + bytes(((i % 16) << 4) | (15 - i % 16)
                                             for i in range(16))


class Q4TCBlockOracleTests(unittest.TestCase):
    def test_q4k_all_256_positions_and_subblock_boundaries(self):
        got = decode_q4_k(q4_block())
        self.assertEqual(len(got), 256)
        expected = []
        scales = [1, 2, 3, 4, 63, 17, 31, 48]
        mins = [0, 1, 2, 3, 49, 31, 32, 63]
        for j in range(8):
            for i in range(32):
                q = (15 - i % 16) if j % 2 == 0 else (i % 16)
                expected.append(0.5 * scales[j] * q - 0.25 * mins[j])
        self.assertEqual(got, tuple(expected))
        self.assertLess(min(got), 0)
        self.assertGreater(max(got), 0)

    def test_iq4_nl_all_codes_and_half_block_order(self):
        codebook = (-127, -104, -83, -65, -49, -35, -22, -10,
                     1, 13, 25, 38, 53, 69, 89, 113)
        got = decode_iq4_nl(iq4_block())
        self.assertEqual(got, tuple(0.25 * codebook[15 - i] for i in range(16)) +
                         tuple(0.25 * codebook[i] for i in range(16)))

    def test_actual_k_shapes_and_nonunit_projection_tail_math(self):
        gu = q4_block() * 10  # K=2560, one synthetic gate or up row
        down = iq4_block() * 20  # K=640, one synthetic down row
        x = [((i % 23) - 11) / 32.0 for i in range(2560)]
        h = [((i % 19) - 9) / 16.0 for i in range(640)]
        sg, su, sd, zero = struct.unpack('<4f', struct.pack('<4f', 1.25, 0.5, 2.0, 0.0))
        self.assertEqual(zero, 0.0)
        base_g = math.fsum(w * a for w, a in zip(decode_row(gu, 'Q4_K', 2560), x))
        base_d = math.fsum(w * a for w, a in zip(decode_row(down, 'IQ4_NL', 640), h))
        self.assertAlmostEqual(scaled_dot(gu, 'Q4_K', 2560, x, sg), base_g * sg)
        self.assertAlmostEqual(scaled_dot(down, 'IQ4_NL', 640, h, sd), base_d * sd)
        self.assertAlmostEqual(swiglu_from_unscaled(-1.5, 2.0, sg, su),
                               (-1.5 * sg) / (1 + math.exp(1.5 * sg)) * (2.0 * su))
        self.assertNotAlmostEqual(swiglu_from_unscaled(-1.5, 2.0, sg, su),
                                  ((-1.5) / (1 + math.exp(1.5))) * 2.0 * sg * su)

    def test_malformed_and_nonfinite_fail_closed(self):
        for b in (b'', q4_block()[:-1], q4_block() + b'\0'):
            with self.assertRaises(ValueError): decode_q4_k(b)
        for b in (b'', iq4_block()[:-1], iq4_block() + b'\0'):
            with self.assertRaises(ValueError): decode_iq4_nl(b)
        with self.assertRaises(ValueError): decode_q4_k(struct.pack('<e', math.inf) + q4_block()[2:])
        with self.assertRaises(ValueError): decode_iq4_nl(struct.pack('<e', math.nan) + iq4_block()[2:])
        for kind, data, k in [('Q4_K', q4_block(), 640), ('IQ4_NL', iq4_block(), 31),
                              ('NVFP4', q4_block(), 256)]:
            with self.assertRaises(ValueError): decode_row(data, kind, k)
        with self.assertRaises(ValueError): scaled_dot(q4_block(), 'Q4_K', 256, [math.nan] * 256, 1.0)
        with self.assertRaises(ValueError): scaled_dot(q4_block(), 'Q4_K', 256, [0.0] * 255, 1.0)


if __name__ == '__main__':
    unittest.main()
