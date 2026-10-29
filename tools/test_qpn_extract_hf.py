import struct
import unittest

import numpy as np

import qpn_extract_hf as q


class FakeCheckpoint:
    def __init__(self):
        self.tensors = {}
        for i, (role, rows, cols) in enumerate(q.ROLES):
            base = q.prefix(0, 7, role)
            self.tensors[base + "weight"] = (np.arange(rows * cols // 2, dtype=np.uint32) % 256).astype(np.uint8).reshape(rows, cols//2)
            fixed = np.zeros((rows//16, 16*(1+cols//32)), np.uint8)
            fixed[:, :16] = 56  # E4M3 1.0, code deltas zero
            self.tensors[base + "weight_scale" + q.CSF_FIXED] = fixed
            self.tensors[base + "weight_scale" + q.CSF_EXC] = np.zeros(0, np.uint32)
            self.tensors[base + "weight_scale_2"] = np.array([.1*(i+1)], np.float32)

    def dtype(self, name):
        return {np.dtype("uint8"): "U8", np.dtype("uint32"): "U32", np.dtype("float32"): "F32"}[self.tensors[name].dtype]

    def array(self, name, dtype):
        return self.tensors[name]


class ExtractTest(unittest.TestCase):
    def test_required_tensor_shards(self):
        index = {q.prefix(0, 7, role)+s: f"tensors/{role}.safetensors" for role, _, _ in q.ROLES for s in q.SUFFIXES}
        self.assertEqual(q.needed_shards(index, [(0, 7)]), ["tensors/down.safetensors", "tensors/gate.safetensors", "tensors/up.safetensors"])
        del index[q.prefix(0, 7, "gate") + "weight_scale_2"]
        with self.assertRaises(ValueError): q.needed_shards(index, [(0, 7)])

    def test_blob_matches_independent_element_order_and_tail(self):
        ck = FakeCheckpoint()
        blob, report = q.expert_blob(ck, 0, 7)
        self.assertEqual(len(blob), 2764816)
        self.assertEqual(report["ggml_type"], 40)
        for matrix, (role, rows, cols) in enumerate(q.ROLES):
            weights = ck.tensors[q.prefix(0, 7, role) + "weight"]
            pitch = cols//64*36
            # Check all positions in boundary rows for all three projections.
            for row in [0, 1, rows-1]:
                for col in range(cols):
                    b = matrix*921600 + row*pitch + col//64*36
                    self.assertEqual(blob[b+col%64//16], 56)
                    encoded = blob[b+4+col%64//16*8+col%8]
                    code = (encoded >> (0 if col%16 < 8 else 4)) & 15
                    original = (int(weights[row, col//2]) >> (4*(col%2))) & 15
                    self.assertEqual(code, original)
        np.testing.assert_array_equal(struct.unpack_from("<4f", blob, 2764800), np.array([.1,.2,.3,0], np.float32))

    def test_reject_shape_scale_and_nonfinite(self):
        for kind in ("shape", "signed", "nonfinite", "dtype"):
            ck = FakeCheckpoint()
            base = q.prefix(0, 7, "gate")
            if kind == "shape": ck.tensors[base+"weight"] = ck.tensors[base+"weight"][:-1]
            if kind == "signed": ck.tensors[base+"weight_scale"+q.CSF_FIXED][0, 0] = 128
            if kind == "nonfinite": ck.tensors[base+"weight_scale_2"][0] = np.nan
            if kind == "dtype": ck.tensors[base+"weight"] = ck.tensors[base+"weight"].astype(np.float32)
            with self.subTest(kind=kind), self.assertRaises(ValueError): q.expert_blob(ck, 0, 7)


if __name__ == "__main__":
    unittest.main()
