"""Source-locked NVFP4 QAD synthetic-byte decoder boundary tests (host only)."""
import math
import unittest
from qpn_q4tc_nvfp4_oracle import decode_nvfp4_row, ue4m3_half, E2M1_DOUBLED


class QadNvfp4DecoderTests(unittest.TestCase):
    def test_all_ue4m3_bytes_and_e2m1_nibbles(self):
        codes = bytes((j & 15) | ((15 - j) << 4) for j in range(16)) * 2
        for scale_code in range(256):
            row = bytes((scale_code,)*4) + codes
            got = decode_nvfp4_row(row, 64)
            scale = ue4m3_half(scale_code)
            expected = tuple(v for sub in range(4) for v in (
                *(E2M1_DOUBLED[j & 15] * scale for j in range(sub*8,sub*8+8)),
                *(E2M1_DOUBLED[15-(j & 15)] * scale for j in range(sub*8,sub*8+8))))
            self.assertEqual(got, expected)
            self.assertTrue(all(map(math.isfinite, got)))
        self.assertEqual(ue4m3_half(0x01), 2**-10)
        self.assertEqual(ue4m3_half(0x08), 2**-7)
        self.assertEqual(ue4m3_half(0x7F), 0)
        self.assertEqual(ue4m3_half(0xFF), 240)

    def test_projection_widths_and_subblock_boundaries(self):
        for k in (640, 2560):
            blocks = bytearray()
            for b in range(k // 64):
                scales = bytes((0, 0x7F, 0x28 + (b % 8), 0x7E))
                codes = bytes(((j+b)&15) | (((15-j-b)&15)<<4) for j in range(32))
                blocks.extend(scales + codes)
            out = decode_nvfp4_row(bytes(blocks), k)
            self.assertEqual(len(out), k)
            self.assertTrue(all(map(math.isfinite, out)))
            self.assertEqual(out[:16], (0.0,) * 16)
            self.assertEqual(out[16:32], (0.0,) * 16)
            self.assertNotEqual(out[32:48], (0.0,) * 16)

    def test_invalid_lengths_and_codes_fail_closed(self):
        row = bytes(36)
        for data,k in ((b'',64),(row[:-1],64),(row+b'\0',64),(row,640),(row,63),(row,0),(row,True)):
            with self.assertRaises(ValueError):
                decode_nvfp4_row(data,k)
        for code in (-1,256,1.0,True):
            with self.assertRaises(ValueError):
                ue4m3_half(code)


if __name__ == '__main__':
    unittest.main()
