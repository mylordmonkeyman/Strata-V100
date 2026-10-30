"""Host-only synthetic tests; never use private route IDs."""
import json
from pathlib import Path
import tempfile
import unittest
from qpn_resident_r1 import read_private_sample, evaluate, score_holdout
from qpn_resident_cache_feasibility import PACKED_BLOB, MIB


class TestResidentR1(unittest.TestCase):
    def setUp(self):
        self.ranking=[(l,e) for l in range(48) for e in range(512)]
        self.train={'prefill':[{0:{e:40 for e in range(32)}}], 'decode':[]}
        self.test={'prefill':[{0:{e:40 for e in range(32)}}],
                   'decode':[{0:{0:1,4:1,20:1}}]}

    def test_heldout_complete_groups_and_decode_restoration(self):
        scores=evaluate(self.train,self.test,self.ranking,[0,128],24)
        self.assertEqual(scores['scenarios'][0]['upper_bound_complete_qpn_groups'],0)
        sc=scores['scenarios'][1]
        self.assertEqual(sc['predicted_packed_groups'],2)
        self.assertEqual(sc['upper_bound_complete_qpn_groups'],2)
        self.assertEqual(sc['upper_bound_qpn_entries'],1280)
        self.assertEqual(sc['decode_distinct_packed_experts_to_restore'],3)
        self.assertEqual(sc['optimistic_decode_restore_bytes'],3*PACKED_BLOB)
        self.assertEqual(sc['canonical_ids_evicted'],sc['selected_packed_experts'])
        self.assertEqual(sc['resident_canonical_ids_retained']+
                         sc['canonical_ids_evicted'],scores['native_slots'])
        self.assertEqual(sc['max_concurrent_resident_weight_copies_per_expert'],1)

    def test_no_selected_subset_keeps_two_resident_formats(self):
        scores=evaluate(self.train,self.test,self.ranking,[0,128,256],24)
        for sc in scores['scenarios']:
            self.assertEqual(sc['max_concurrent_resident_weight_copies_per_expert'],1)
            self.assertEqual(sc['canonical_ids_evicted'],sc['selected_packed_experts'])
            self.assertEqual(sc['resident_canonical_ids_retained']+
                             sc['selected_packed_experts'], scores['native_slots'])
            self.assertLessEqual(sc['selected_packed_experts'], scores['native_slots'])

    def test_incomplete_group_requires_canonical(self):
        selected={(0,e) for e in range(16)}
        test={'prefill':[{0:{e:40 for e in range(15)}}], 'decode':[]}
        r=score_holdout(test,selected,selected)
        self.assertEqual(r['upper_bound_complete_qpn_groups'],0)
        self.assertEqual(r['prefill_packed_canonical_fallback_expert_uses'],15)

    def test_training_misprediction_charged(self):
        test={'prefill':[{0:{e:10 for e in range(16)}}], 'decode':[{0:{e:1 for e in range(16)}}]}
        r=score_holdout(test,{(0,e) for e in range(16)},{(0,e) for e in range(16)})
        self.assertEqual(r['upper_bound_complete_qpn_groups'],0)
        self.assertEqual(r['prefill_packed_canonical_fallback_expert_uses'],16)
        self.assertEqual(r['decode_distinct_packed_experts_to_restore'],16)

    def test_eligibility_cannot_be_inferred_from_aggregate_rows(self):
        test={'prefill':[{0:{e:(600 if e<16 else 40) for e in range(32)}}], 'decode':[]}
        r=score_holdout(test,{(0,e) for e in range(32)},{(0,e) for e in range(32)})
        self.assertEqual(r['upper_bound_complete_qpn_groups'],1)
        self.assertEqual(r['upper_bound_qpn_entries'],640)

    def test_private_input_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.json'
            p.write_text(json.dumps({'schema':1,'prefill':[{'layers':[{'layer':0,'counts':[[4,20],[4,30]]}]}],'decode':[]}))
            with self.assertRaises(ValueError):read_private_sample(p)

    def test_private_input_no_ids_in_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.json'
            p.write_text(json.dumps({'schema':1,'prefill':[{'layers':[{'layer':0,'counts':[[123,40]]}]}],'decode':[]}))
            raw=read_private_sample(p)
            r=evaluate(raw,raw,self.ranking,[128],24)
            self.assertNotIn('123',json.dumps(r))
            self.assertIn('scenarios',r)

    def test_caps_and_bad_budget(self):
        with self.assertRaises(ValueError):evaluate(self.train,self.test,self.ranking,[4000])
        with self.assertRaises(ValueError):evaluate(self.train,self.test,self.ranking,[128],low=513)
        self.assertEqual(128*MIB//PACKED_BLOB//16,3)

if __name__=='__main__':unittest.main()

