import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("qpn_audit", Path(__file__).with_name("qpn_audit.py"))
qpn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qpn)


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pack = Path(self.temp.name)
        self.blob = bytearray(qpn.EXPERT_BYTES)
        struct.pack_into("<4f", self.blob, 3*qpn.MATRIX_BYTES, .1, .2, .3, 0)
        (self.pack / "experts.bin").write_bytes(self.blob)
        self.table("0 40 40 0 2764816")

    def table(self, rows):
        (self.pack / "native_experts.txt").write_text("# native experts (n_expert 1)\n" + rows + "\n")

    def test_canonical_read_only(self):
        report, blob = qpn.audit(self.pack, 0, 0)
        self.assertEqual(blob, self.blob)
        self.assertEqual(sum(report["scale_byte_histogram"].values()), 3*qpn.MATRIX_BYTES//36*4)
        self.assertEqual((self.pack / "experts.bin").read_bytes(), self.blob)
        self.assertIn("pending", report["decode_gate"])

    def test_reject_bad_table(self):
        for row in ["0 42 40 0 2764816", "0 40 40 1 2764816", "0 40 40 0 2764815",
                    "0 40 40 0 2764816\n0 40 40 0 2764816"]:
            with self.subTest(row=row):
                self.table(row)
                with self.assertRaises(ValueError): qpn.audit(self.pack, 0, 0)

    def test_reject_truncation(self):
        (self.pack / "experts.bin").write_bytes(self.blob[:-1])
        with self.assertRaises(ValueError): qpn.audit(self.pack, 0, 0)

    def test_reject_signed_scale(self):
        self.blob[0] = 128
        (self.pack / "experts.bin").write_bytes(self.blob)
        with self.assertRaises(ValueError): qpn.audit(self.pack, 0, 0)

    def test_reject_invalid_tail(self):
        for value in [float("nan"), float("inf"), -1, 0]:
            struct.pack_into("<f", self.blob, 3*qpn.MATRIX_BYTES, value)
            (self.pack / "experts.bin").write_bytes(self.blob)
            with self.assertRaises(ValueError): qpn.audit(self.pack, 0, 0)


if __name__ == "__main__":
    unittest.main()
