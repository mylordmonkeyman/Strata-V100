"""Host-only tests for PRIVATE bounded QDV1 verification route parsing."""
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from qpn_r1_verify_convert import HEADER, COUNTS, LAYERS, decode_private_verify, write_private_json


def frame(pos, tokens, layer):
    counts = [0] * 512
    for e in range(10):
        counts[e] = tokens
    return HEADER.pack(b"QDV1", 1, pos, tokens, layer, 512, 10, 0) + COUNTS.pack(*counts)


def window(pos=2047, tokens=2):
    return b"".join(frame(pos, tokens, l) for l in range(LAYERS))


class PrivateVerifyCaptureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "private.qdv1"

    def write(self, raw):
        self.path.write_bytes(raw)
        os.chmod(self.path, 0o600)

    def test_valid_windows_are_unclassified(self):
        self.write(window() + window(2049, 1))
        obj = decode_private_verify(self.path)
        self.assertEqual(len(obj["verify"]), 2)
        self.assertEqual(obj["decode"], [])
        self.assertEqual(obj["verify"][0]["position"], 2047)
        self.assertEqual(len(obj["verify"][0]["layers"]), 48)
        self.assertEqual(sum(x[1] for x in obj["verify"][0]["layers"][0]["counts"]), 20)
        out = Path(self.temp.name) / "private.json"
        write_private_json(out, obj)
        self.assertEqual(out.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            write_private_json(out, obj)

    def test_incomplete_and_bad_magic_rejected(self):
        self.write(window()[:-1])
        with self.assertRaises(ValueError):
            decode_private_verify(self.path)
        raw = bytearray(window())
        raw[0:4] = b"QPR1"
        self.write(raw)
        with self.assertRaises(ValueError):
            decode_private_verify(self.path)

    def test_bad_counts_and_permissions_rejected(self):
        raw = bytearray(window())
        at = HEADER.size
        raw[at:at + 4] = struct.pack("<i", 3)  # > two tokens
        self.write(raw)
        with self.assertRaises(ValueError):
            decode_private_verify(self.path)
        self.write(window())
        os.chmod(self.path, 0o644)
        with self.assertRaises(ValueError):
            decode_private_verify(self.path)

    def test_window_limit(self):
        self.write(window() * 33)
        with self.assertRaises(ValueError):
            decode_private_verify(self.path)


if __name__ == "__main__":
    unittest.main()
