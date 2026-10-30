"""Pure host tests: no private model weights, routes, or GPU needed."""
from collections import Counter
import struct
from pathlib import Path
import tempfile
import unittest

from qpn_resident_cache_feasibility import (
    PACKED_BLOB, GIB, MIB, capacity_sweep, read_private_routing_trace, read_ranked_profile,
)


class ResidentCapacityTest(unittest.TestCase):
    def test_exact_cache_geometry(self):
        self.assertEqual(PACKED_BLOB*24576, 67948118016)
        self.assertEqual(48*16*PACKED_BLOB, 2123378688)

    def test_additional_vs_fixed_vram(self):
        x, y = capacity_sweep(24*GIB, 512*MIB)
        self.assertEqual(x['canonical_slots_displaced'], 0)
        self.assertEqual(x['packed_duplicate_slots'], 194)
        self.assertEqual(y['canonical_slots_displaced'], y['packed_duplicate_slots'])
        self.assertEqual((x['layers_with_one_group_ceiling'],
                          y['layers_with_one_group_ceiling']), (12, 12))
        self.assertIsNone(x['qpn_groups_executed'])
        self.assertIsNone(y['model_speedup'])

    def test_small_budgets_cannot_cover_all_layers(self):
        x, _ = capacity_sweep(24*GIB, 1024*MIB)
        self.assertLess(x['layers_with_one_group_ceiling'], 48)
        x, _ = capacity_sweep(24*GIB, 3072*MIB)
        self.assertEqual(x['layers_with_one_group_ceiling'], 48)

    def test_private_trace_oracle_is_aggregate_only(self):
        rank = [(0,i) for i in range(512)]
        freq = Counter({(0,0):40,(0,1):30,(0,2):15,(0,3):5,(0,4):15})
        rows = capacity_sweep(6*PACKED_BLOB, 2*PACKED_BLOB, layers=1,
                              trace_counts=freq, ranking=rank)
        self.assertEqual(rows[0]['oracle_packed_routed_entries'], 70)
        self.assertEqual(rows[1]['canonical_route_entries_lost'], 15)
        self.assertAlmostEqual(rows[1]['oracle_packed_routing_share'], 70/105)
        self.assertNotIn('ranking', rows[0])
        self.assertNotIn('expert_id', str(rows))

    def test_profile_and_trace_reject_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'profile'
            p.write_bytes(b'STRP'+struct.pack('<5I',1,48,512,2,2))
            with self.assertRaises(ValueError): read_ranked_profile(p)
            t = Path(tmp)/'trace'
            t.write_bytes(struct.pack('<ii',0,2)+struct.pack('<i',3))
            with self.assertRaises(ValueError): read_private_routing_trace(t)
            t.write_bytes(struct.pack('<ii',0,2)+struct.pack('<ii',3,3)+b'\\0'*8)
            with self.assertRaises(ValueError): read_private_routing_trace(t)
            t.write_bytes(struct.pack('<ii',0,2)+struct.pack('<ii',3,4)+b'\\0'*8)
            count, n, digest = read_private_routing_trace(t)
            self.assertEqual((count[(0,3)],n,len(digest)), (1,1,64))

    def test_complete_profile_roundtrip(self):
        pairs = [(l,e) for l in range(48) for e in range(512)]
        data = b'STRP' + struct.pack('<5I',1,48,512,len(pairs),len(pairs))
        data += b''.join(struct.pack('<HH',*pair) for pair in pairs)
        data += struct.pack('<%di' % len(pairs),*range(len(pairs)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'profile'
            path.write_bytes(data)
            ranked,digest = read_ranked_profile(path)
            self.assertEqual((len(ranked),ranked[-1],len(digest)), (24576,(47,511),64))

    def test_invalid_budgets(self):
        for native,extra in ((0,0),(24*GIB,-1)):
            with self.assertRaises(ValueError):
                capacity_sweep(native,extra)


if __name__ == '__main__':
    unittest.main()
