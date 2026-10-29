import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from qpn_mixed_route_plan import audit_capture, validate_plan, expected_captures, N_EMBD, BASELINE_SOURCE


def capture(tmp, ids, bad_x=False):
    path = Path(tmp) / "layer-0.moe"
    t, k = ids.shape
    x = np.zeros((t, N_EMBD), dtype="<f4")
    if bad_x:
        x[0, 0] = np.nan
    path.write_bytes(struct.pack("<3q", t, N_EMBD, k) + x.tobytes() + ids.astype("<i4").tobytes())
    sizes = {str(int(e)): int((ids == e).sum()) for e in np.unique(ids)}
    hist = {}
    for size in sizes.values():
        hist[str(size)] = hist.get(str(size), 0) + 1
    return path, {"tokens": t, "top_k": k, "n_embd": N_EMBD,
                  "activation_dump_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "expert_counts": sizes, "entry_count_histogram": hist}


class MixedRoutePlanTest(unittest.TestCase):
    def test_complete_bijection_nonuniform_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = np.array([[3, 2, 1], [3, 2, 4], [3, 5, 4], [3, 6, 4]], dtype=np.int32)
            path, meta = capture(tmp, ids)
            plan_bytes, report = audit_capture(path, meta, 0)
            plan = json.loads(plan_bytes)
            self.assertEqual([g["expert"] for g in plan["groups"]], [1, 2, 3, 4, 5, 6])
            self.assertEqual(plan["groups"][2]["token_rows"], [0, 1, 2, 3])
            self.assertEqual(report["routed_entries"], 12)
            self.assertTrue(validate_plan(ids, plan))
            plan["groups"][2]["slots"].pop()
            plan["groups"][2]["token_rows"].pop()
            with self.assertRaisesRegex(ValueError, "missing routed slot"):
                validate_plan(ids, plan)

    def test_duplicate_router_expert_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, meta = capture(tmp, np.array([[3, 3], [4, 5]], dtype=np.int32))
            with self.assertRaisesRegex(ValueError, "duplicate expert"):
                audit_capture(path, meta, 0)

    def test_corruption_and_nonfinite_capture_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = np.array([[3, 4], [5, 6]], dtype=np.int32)
            path, meta = capture(tmp, ids)
            path.write_bytes(path.read_bytes()[:-4])
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                audit_capture(path, meta, 0)
            path, meta = capture(tmp, ids, bad_x=True)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                audit_capture(path, meta, 0)

    def test_wrong_group_owner_and_duplicate_slot_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = np.array([[3, 4], [3, 5]], dtype=np.int32)
            path, meta = capture(tmp, ids)
            plan = json.loads(audit_capture(path, meta, 0)[0])
            plan["groups"][0]["slots"][0] = 1
            plan["groups"][0]["token_rows"][0] = 0
            with self.assertRaisesRegex(ValueError, "wrong expert"):
                validate_plan(ids, plan)
            plan = json.loads(audit_capture(path, meta, 0)[0])
            plan["groups"][0]["slots"].append(2)
            plan["groups"][0]["token_rows"].append(1)
            with self.assertRaisesRegex(ValueError, "duplicate or out-of-order"):
                validate_plan(ids, plan)

    def test_exact_three_pinned_captures(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, meta = capture(tmp, np.array([[3, 4], [3, 5]], dtype=np.int32))
            manifest = {"baseline_source": BASELINE_SOURCE, "model_revision": "pinned",
                        "runs": [{"returncode": 0, "capture": meta}]}
            with self.assertRaisesRegex(ValueError, "exactly three"):
                expected_captures(manifest)
            manifest["runs"] = [{"returncode": 0, "capture": meta} for _ in range(3)]
            self.assertEqual(set(expected_captures(manifest)), {0, 24, 47})


if __name__ == "__main__":
    unittest.main()
