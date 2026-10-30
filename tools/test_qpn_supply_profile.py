import json
import unittest
from qpn_supply_profile import parse_traces, duration, overlap, PREFIX, KINDS

class SupplyProfileTest(unittest.TestCase):
    def trace(self):
        layer = dict(layer=24, routed=dict(groups=2, resident_groups=1, rows=10, resident_rows=3))
        layer.update({k:[] for k in KINDS})
        layer['h2d'] = [[0,2,2764816], [3,5,2764816]]
        layer['products'] = [[1,4,0]]
        return dict(schema=1, instrumented=True, chunk=0, tokens=1024, ring=48, stream_all=True, layers=[layer])
    def parse(self, obj):
        return parse_traces('private model output\n'+PREFIX+json.dumps(obj))
    def test_accounting_and_private_field_allowlist(self):
        t = self.trace(); t['secret'] = 'private prompt'; t['layers'][0]['weights'] = 'private'
        r = self.parse(t)[0]
        self.assertNotIn('secret', r)
        self.assertNotIn('weights', r['layers'][0])
        self.assertEqual(r['layers'][0]['h2d_bytes'], 5529632)
        self.assertEqual(r['layers'][0]['h2d_products_overlap_ms'], 2)
        self.assertEqual(r['layers'][0]['h2d_interval_ms'], 4)
    def test_union_avoids_nested_double_count(self):
        self.assertEqual(duration([[0,4], [1,2], [3,5]]), 5)
        self.assertEqual(overlap([[0,4],[1,2]], [[2,6],[3,4]]), 2)
    def test_reject_corrupt_events_and_residency(self):
        for p in ([2,1,10], [0,float('nan'),10], [0,1,-1], [False,1,2]):
            t=self.trace(); t['layers'][0]['h2d']=[p]
            with self.assertRaises(ValueError): self.parse(t)
        t=self.trace(); t['layers'][0]['routed']['resident_groups']=3
        with self.assertRaises(ValueError): self.parse(t)
    def test_unavailable_routes_remain_null(self):
        t=self.trace(); t['layers'][0]['routed']=None
        self.assertIsNone(self.parse(t)[0]['layers'][0]['routed'])
    def test_duplicate_layers_rejected(self):
        t=self.trace(); t['layers']*=2
        with self.assertRaises(ValueError): self.parse(t)

if __name__ == '__main__': unittest.main()
