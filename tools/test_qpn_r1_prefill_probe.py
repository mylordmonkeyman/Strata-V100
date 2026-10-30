"""Pure host tests for aggregate-only pinned R1 prefill capture qualification."""
import unittest
from qpn_r1_prefill_probe import summarize_capture

class ProbeMath(unittest.TestCase):
    @staticmethod
    def sample():
        chunks=[]
        for tokens in (1024,1023):
            remaining=tokens*10-16*40
            counts=[[e,40] for e in range(16)]
            for e in range(16,496):
                take=min(20,remaining)
                if take:counts.append([e,take])
                remaining-=take
            assert remaining==0
            chunks.append({"tokens":tokens,"stream_all":False,
                "layers":[{"layer":l,"counts":counts,
                           "resident_experts":list(range(16))} for l in range(48)]})
        return {"prefill":chunks,"decode":[]}

    def test_complete_groups_and_native_hits(self):
        r=summarize_capture(self.sample())
        self.assertEqual(r["recorded_layers"],96)
        self.assertEqual(r["routed_entries"],2047*10*48)
        self.assertEqual(r["resident_routed_entries"],96*16*40)
        self.assertEqual(r["optimistic_complete_resident_16_groups_33_512"],96)
        self.assertEqual(r["optimistic_complete_resident_16_group_rows_33_512"],96*16*40)
        self.assertEqual(r["distinct_resident_expert_uses"],48*16)

    def test_reject_missing_layers(self):
        r=self.sample()
        r["prefill"][0]["layers"].pop()
        with self.assertRaises(ValueError):summarize_capture(r)

    def test_reject_stream_all(self):
        r=self.sample()
        r["prefill"][1]["stream_all"]=True
        with self.assertRaises(ValueError):summarize_capture(r)

    def test_reject_decode_without_explicit_collector(self):
        r=self.sample()
        r["decode"]=[{"layers":[{"layer":0,"counts":[[0,1]]}]}]
        with self.assertRaises(ValueError):summarize_capture(r)

if __name__=="__main__":unittest.main()
