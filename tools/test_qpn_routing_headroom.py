import copy
import unittest
from qpn_routing_headroom import BASELINE_SOURCE, summarize


def fixture(counts=None, tokens=8, top_k=2):
    if counts is None:
        counts = {"1": 8, "2": 8}
    histogram = {}
    for size in counts.values():
        histogram[str(size)] = histogram.get(str(size), 0) + 1
    return {"baseline_source": BASELINE_SOURCE, "model_revision": "pinned-checkpoint",
            "runs": [{"returncode": 0, "capture": {"tokens": tokens, "top_k": top_k,
                       "activation_dump_sha256": "a"*64,
                       "expert_counts": counts, "entry_count_histogram": histogram}}]}


class RoutingHeadroomTest(unittest.TestCase):
    def test_exact_routing_and_boundaries(self):
        counts = {"0": 1, "1": 8, "2": 9, "3": 16, "4": 17,
                  "5": 32, "6": 33, "7": 64, "8": 65}
        data = summarize(fixture(counts, tokens=245, top_k=1))
        self.assertEqual(data["expert_invocations"], 9)
        self.assertEqual(data["routed_entries"], 245)
        self.assertEqual([v["expert_invocations"] for v in data["aggregate_buckets"]], [2,1,1,2,2,1])
        self.assertEqual(sum(v["routed_entries"] for v in data["aggregate_buckets"]), 245)
        self.assertAlmostEqual(sum(v["routed_entry_fraction"] for v in data["aggregate_buckets"]), 1.)

    def test_rejects_undercount_or_changed_histogram(self):
        a = fixture()
        a["runs"][0]["capture"]["expert_counts"]["2"] = 7
        with self.assertRaisesRegex(ValueError, "tokens . top_k"):
            summarize(a)
        b = fixture()
        b["runs"][0]["capture"]["entry_count_histogram"]["8"] = 3
        with self.assertRaisesRegex(ValueError, "histogram disagreement"):
            summarize(b)

    def test_rejects_bad_experts(self):
        for expert, size in (("512", 8), ("1", 0), ("1", True)):
            a = fixture()
            if expert == "512":
                del a["runs"][0]["capture"]["expert_counts"]["1"]
            a["runs"][0]["capture"]["expert_counts"][expert] = size
            with self.subTest(expert=expert,size=size), self.assertRaises(ValueError):
                summarize(a)

    def test_ignores_failed_capture(self):
        a = fixture()
        failed = copy.deepcopy(a["runs"][0])
        failed["returncode"] = 1
        a["runs"].append(failed)
        self.assertEqual(summarize(a)["captured_chunks"], 1)
        self.assertEqual(summarize(a)["routed_entries"], 16)

    def test_rejects_wrong_baseline(self):
        a = fixture()
        a["baseline_source"] = "wrong"
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            summarize(a)


if __name__ == "__main__":
    unittest.main()
