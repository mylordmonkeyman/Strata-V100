import json
import tempfile
import unittest
from pathlib import Path
from qpn_mixed_gpu_verify import verify, BASELINE_SHA


class MixedGpuVerifierTests(unittest.TestCase):
    def test_rejects_wrong_manifest_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);m=d/"summary.json"
            m.write_text(json.dumps({"baseline_manifest_sha256":"wrong","groups":16}))
            with self.assertRaisesRegex(ValueError,"pinned model"):
                verify(d,m)

    def test_rejects_duplicate_or_missing_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);m=d/"summary.json"
            m.write_text(json.dumps({"baseline_manifest_sha256":BASELINE_SHA,"groups":16}))
            (d/"counts.txt").write_text(",".join(["4"]*16)+"\n")
            (d/"selected.json").write_text(json.dumps([
                {"layer":0,"expert":i,"tokens":4} for i in range(16)
            ]))
            with self.assertRaisesRegex(ValueError,"selected model layer"):
                verify(d,m)

    def test_rejects_bad_stratification(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);m=d/"summary.json"
            m.write_text(json.dumps({"baseline_manifest_sha256":BASELINE_SHA,"groups":16}))
            (d/"counts.txt").write_text(",".join(["4"]*16)+"\n")
            (d/"selected.json").write_text(json.dumps([
                {"layer":(0,24,47)[i%3],"expert":i,"tokens":4} for i in range(16)
            ]))
            with self.assertRaisesRegex(ValueError,"stratification"):
                verify(d,m)


if __name__=="__main__":
    unittest.main()
