import copy
import json
import unittest
from qpn_r1_joint_probe import normalize_request, score_requests
import test_qpn_r1_prefill_probe as prefill_fixture
from test_qpn_r1_verify_probe import sample

class JointTest(unittest.TestCase):
    def test_join_retains_native_consumers_without_exporting_ids(self):
        request,resident=normalize_request(prefill_fixture.ProbeMath.sample(),sample(),2047)
        scores=score_requests(request,request,resident)
        self.assertEqual(scores[0]['verification_routed_entries'],960)
        self.assertGreater(scores[0]['verification_distinct_packed_experts_to_restore'],0)
        self.assertGreater(scores[0]['upper_bound_qpn_entries'],0)
        self.assertNotIn('decode_routed_entries',scores[0])
        self.assertNotIn('counts',json.dumps(scores))
        self.assertNotIn('resident_experts',json.dumps(scores))

    def test_future_heldout_rows_do_not_train_selection(self):
        request,resident=normalize_request(prefill_fixture.ProbeMath.sample(),sample(),2047)
        train=copy.deepcopy(request)
        for chunk in train['prefill']:
            for counts in chunk.values():
                for expert in counts: counts[expert]=1
        scores=score_requests(train,request,resident)
        self.assertTrue(all(x['selected_packed_experts']==0 for x in scores))

    def test_prompt_tail_and_cap_rejected(self):
        for verify in (sample(position=2046), {'verify': sample()['verify']*32}):
            with self.assertRaises(ValueError):
                normalize_request(prefill_fixture.ProbeMath.sample(),verify,2047)

    def test_partial_prefill_and_changing_residents_rejected(self):
        with self.assertRaises(ValueError):
            normalize_request(prefill_fixture.ProbeMath.sample(),sample(),1023)
        pf=prefill_fixture.ProbeMath.sample();pf['prefill'][1]['layers'][0]['resident_experts']=[]
        with self.assertRaises(ValueError):normalize_request(pf,sample(),2047)

if __name__=='__main__':unittest.main()
