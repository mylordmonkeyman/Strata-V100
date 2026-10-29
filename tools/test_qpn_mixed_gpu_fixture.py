import unittest
from qpn_mixed_gpu_fixture import choose_groups, LAYERS, TARGETS, BOUNDS


def synthetic():
    plans = {}
    for layer in LAYERS:
        groups = []
        e = 0
        for (low, high), targets in zip(BOUNDS, TARGETS):
            for t in targets:
                count = max(low, min(high, t + (layer // 24)))
                groups.append({"expert": e, "token_rows": list(range(count)),
                               "slots": list(range(count))})
                e += 1
        plans[layer] = {"tokens": 1024, "groups": groups}
    return plans


class FixtureSelectionTests(unittest.TestCase):
    def test_nonuniform_selection_and_fallback_prefix(self):
        chosen = choose_groups(synthetic())
        self.assertEqual(len(chosen), 16)
        self.assertEqual(len({(g["layer"], g["expert"]) for g in chosen}), 16)
        self.assertEqual({g["layer"] for g in chosen}, set(LAYERS))
        fallback = sum(g["count"] > 32 for g in chosen)
        self.assertEqual(fallback, 6)
        self.assertTrue(all(g["count"] > 32 for g in chosen[:fallback]))
        self.assertTrue(all(g["count"] <= 32 for g in chosen[fallback:]))
        self.assertEqual(chosen, choose_groups(synthetic()))

    def test_missing_bucket_fails_closed(self):
        plans = synthetic()
        for layer in LAYERS:
            plans[layer]["groups"] = [g for g in plans[layer]["groups"] if len(g["token_rows"]) <= 64]
        with self.assertRaisesRegex(ValueError, "not enough real routed groups"):
            choose_groups(plans)

    def test_invalid_row_count_fails_closed(self):
        plans = synthetic()
        plans[0]["groups"][0]["slots"] = []
        with self.assertRaisesRegex(ValueError, "slot/row mismatch"):
            choose_groups(plans)


if __name__ == "__main__":
    unittest.main()
