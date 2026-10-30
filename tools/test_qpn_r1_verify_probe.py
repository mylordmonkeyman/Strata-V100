"""Host-only aggregate tests; never use real private routes."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from qpn_r1_verify_probe import aggregate_verify


def sample(position=2047, tokens=2):
    layers = [{"layer": l, "counts": [[e, tokens] for e in range(10)]}
              for l in range(48)]
    return {"verify": [{"position": position, "tokens": tokens, "layers": layers}], "decode": []}


class AggregateVerifyTest(unittest.TestCase):
    def test_post_prefill(self):
        got = aggregate_verify(sample())
        self.assertEqual(got["post_prefill_verification_windows"], 1)
        self.assertEqual(got["verification_layer_records"], 48)
        self.assertEqual(got["verification_routed_entries"], 960)
        self.assertEqual(got["verification_distinct_layer_experts"], 480)
        self.assertEqual(got["qpn_33_512_row_groups"], 0)

    def test_reject_prompt_only_and_missing(self):
        with self.assertRaises(ValueError):
            aggregate_verify(sample(position=100))
        with self.assertRaises(ValueError):
            aggregate_verify({"verify": [], "decode": []})

    def test_reject_wrong_counts(self):
        s = sample()
        s["verify"][0]["layers"][0]["counts"][0][1] = 1
        with self.assertRaises(ValueError):
            aggregate_verify(s)


if __name__ == "__main__":
    unittest.main()
