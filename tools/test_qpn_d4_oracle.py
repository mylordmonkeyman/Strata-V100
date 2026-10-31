"""Independent synthetic-only coverage for D4 MMQ activation reconstruction."""
import struct
import unittest

from qpn_d4_oracle import D4_BLOCK_BYTES, decode_d4_to_f16


def sample(rows, columns, pitch):
    data = bytearray(columns // 128 * pitch * D4_BLOCK_BYTES)
    expected = []
    for block in range(columns // 128):
        for row in range(pitch):
            off = (block * pitch + row) * D4_BLOCK_BYTES
            scales = [0.125 * (j + 1) * (row + 1) + block / 32 for j in range(4)]
            struct.pack_into('<4f', data, off, *scales)
            for i in range(128):
                code = ((i * 7 + row * 11 + block * 3) % 255) - 127
                struct.pack_into('<b', data, off + 16 + i, code)
    for row in range(rows):
        for col in range(columns):
            block, i = divmod(col, 128)
            scale = 0.125 * ((i // 32) + 1) * (row + 1) + block / 32
            code = ((i * 7 + row * 11 + block * 3) % 255) - 127
            expected.append(struct.unpack('<e', struct.pack('<e', scale * code))[0])
    return bytes(data), expected


class D4StageOracle(unittest.TestCase):
    def test_transposed_rows_and_four_scales(self):
        for rows, cols, pitch in [(1, 128, 1), (3, 256, 5),
                                  (2, 640, 4), (3, 2560, 3)]:
            with self.subTest(rows=rows, cols=cols, pitch=pitch):
                payload, expected = sample(rows, cols, pitch)
                result = decode_d4_to_f16(payload, rows, cols, ne1=pitch)
                actual = [v[0] for v in struct.iter_unpack('<e', result)]
                self.assertEqual(actual, expected)
                self.assertEqual(len(result), 2 * rows * cols)

    def test_does_not_read_padded_rows(self):
        payload, expected = sample(2, 256, 4)
        damaged = bytearray(payload)
        for colblock in range(2):
            for row in (2, 3):
                struct.pack_into('<f', damaged,
                                 (colblock * 4 + row) * D4_BLOCK_BYTES, float('nan'))
        self.assertEqual(decode_d4_to_f16(payload, 2, 256, ne1=4),
                         decode_d4_to_f16(damaged, 2, 256, ne1=4))

    def test_reject_bad_layout_or_dimensions(self):
        payload, _ = sample(2, 128, 3)
        for kw in ({'rows': 0, 'columns': 128, 'ne1': 3},
                   {'rows': 2, 'columns': 129, 'ne1': 3},
                   {'rows': 2, 'columns': 128, 'ne1': 1},
                   {'rows': 2, 'columns': 128, 'ne1': True}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                decode_d4_to_f16(payload, **kw)
        with self.assertRaises(ValueError):
            decode_d4_to_f16(payload[:36], 2, 128, ne1=3)

    def test_reject_nonfinite_and_half_overflow(self):
        payload, _ = sample(1, 128, 1)
        for scalar in (float('nan'), float('inf'), 1.0e10):
            broken = bytearray(payload)
            struct.pack_into('<f', broken, 0, scalar)
            with self.subTest(scalar=scalar), self.assertRaises(ValueError):
                decode_d4_to_f16(broken, 1, 128)


if __name__ == '__main__':
    unittest.main()
