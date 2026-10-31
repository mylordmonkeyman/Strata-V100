"""Pinned ggml MMQ D4 producer-algorithm host simulation, synthetic inputs only.

This is independent of the existing byte-level D4 decoding oracle. It mirrors
quantize_mmq_q8_1<D4> block arithmetic and the non-scatter / scatter destination
index formulas in ggml-org/llama.cpp@3cf03257, but is NOT a GPU producer test.
"""
import math
import struct
import unittest

from qpn_d4_oracle import decode_d4_to_f16


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def round_cuda(value):
    """roundf(x): nearest integer, ties away from zero (finite input only)."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def producer_d4(rows, physical_pitch=None):
    """Reference physical row transposition of ggml's 128-code MMQ D4 blocks.

    The input `rows` is already in compact physical output-row order; caller
    constructs the scattered source-to-row map independently. Physical
    quantizer width follows Strata's pad512(cols), including 640 -> 1024.
    """
    if not rows or len(rows[0]) % 128:
        raise ValueError('requires nonempty 128-aligned rows')
    columns = len(rows[0])
    pitch = len(rows) if physical_pitch is None else physical_pitch
    if pitch < len(rows) or any(len(row) != columns for row in rows):
        raise ValueError('bad row pitch / row lengths')
    # Intentional opaque sentinels for unused physical rows.
    physical_columns = ((columns + 511) // 512) * 512
    data = bytearray(b'\x5a' * ((physical_columns // 128) * pitch * 144))
    for colblock in range(columns // 128):
        for rowidx, row in enumerate(rows):
            base = (colblock * pitch + rowidx) * 144
            for quarter in range(4):
                values = [f32(x) for x in row[colblock * 128 + quarter * 32:
                                                   colblock * 128 + quarter * 32 + 32]]
                amax = max(map(abs, values))
                if not amax or not math.isfinite(amax):
                    raise ValueError('synthetic producer requires finite nonzero quarters')
                d_inv = f32(f32(127.0) / f32(amax))
                d = f32(1.0 / d_inv)
                struct.pack_into('<f', data, base + 4 * quarter, d)
                for j, value in enumerate(values):
                    q = round_cuda(f32(value * d_inv))
                    if not -127 <= q <= 127:
                        raise ValueError('out of range reference q8 code')
                    struct.pack_into('<b', data, base + 16 + quarter * 32 + j, q)
    return bytes(data), pitch


def synthetic_rows(count, columns):
    return [[f32((math.sin((i + 1) * (r + 2) * 0.173) * 7 +
                  math.cos((i + 11) * 0.13) * 3 + (r - 2) * 0.1))
             for i in range(columns)] for r in range(count)]


class PinnedD4ProducerReference(unittest.TestCase):
    def test_non_scatter_shapes_and_padded_pitch(self):
        for columns in (128, 256, 640, 2560):
            with self.subTest(columns=columns):
                source = synthetic_rows(3, columns)
                packed, pitch = producer_d4(source, physical_pitch=5)
                actual = [v[0] for v in struct.iter_unpack(
                    '<e', decode_d4_to_f16(packed, 3, columns, ne1=pitch,
                                           physical_columns=((columns + 511) // 512) * 512))]
                self.assertEqual(len(actual), 3 * columns)
                for rowidx, row in enumerate(source):
                    for quarter_start in range(0, columns, 32):
                        amax = max(abs(v) for v in row[quarter_start:quarter_start + 32])
                        for i in range(quarter_start, quarter_start + 32):
                            # Q8 rounding <= amax / 254 plus FP16 reconstruction
                            # rounding of at most 0.5 ulp for this input magnitude.
                            self.assertLessEqual(abs(actual[rowidx * columns + i] - row[i]),
                                                 amax / 254 + 0.01)

    def test_scatter_compact_rows_duplicate_sources(self):
        src = synthetic_rows(4, 640)
        # Emulate inverse slot mapping: token 0 -> slots 0,3;
        # token 1 -> 2; token 2 -> 1,4; token 3 -> 5.
        inverse = [[0, 3], [2], [1, 4], [5]]
        compact = [None] * 6
        for token, indices in enumerate(inverse):
            for index in indices:
                compact[index] = src[token]
        payload, pitch = producer_d4(compact)
        result = decode_d4_to_f16(payload, len(compact), 640, ne1=pitch,
                                   physical_columns=1024)
        unpacked = [v[0] for v in struct.iter_unpack('<e', result)]
        self.assertEqual(unpacked[0:640], unpacked[3*640:4*640])
        self.assertEqual(unpacked[1*640:2*640], unpacked[4*640:5*640])
        self.assertNotEqual(unpacked[0:640], unpacked[1*640:2*640])

    def test_cuda_round_ties_and_nonfinite_source_rejected(self):
        self.assertEqual(round_cuda(2.5), 3)
        self.assertEqual(round_cuda(-2.5), -3)
        self.assertEqual(round_cuda(2.49), 2)
        for value in (0.0, float('nan'), float('inf')):
            # All-zero and invalid quarters cannot be given a reliable
            # reference quantization value by this standalone simulation.
            with self.subTest(value=value), self.assertRaises(ValueError):
                producer_d4([[value] * 128])

    def test_physical_block_index_at_128_boundary(self):
        rows = synthetic_rows(2, 256)
        packed, pitch = producer_d4(rows, physical_pitch=4)
        # Column-block-major means row 0, second 128-column block is
        # at offset 4*144 (not 2*144), and row 1 first block at 144.
        r0 = struct.unpack_from('<4f', packed, 0)
        r1 = struct.unpack_from('<4f', packed, 144)
        r0_second = struct.unpack_from('<4f', packed, 4*144)
        self.assertNotEqual(r0, r1)
        self.assertNotEqual(r0, r0_second)
        self.assertEqual(packed[2*144:2*144+144], b'\x5a' * 144)


if __name__ == '__main__':
    unittest.main()
