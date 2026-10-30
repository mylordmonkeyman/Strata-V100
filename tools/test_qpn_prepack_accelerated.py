import importlib.util
import random
import struct
import unittest
from qpn_prepack_expert_cache import transform_expert, projection_bytes

@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy fast encoder not installed')
class FastPrepackTest(unittest.TestCase):
    def test_fast_encoder_is_identical_to_block_reference_in_both_directions(self):
        rng=random.Random(20261030)
        for n,ff in ((128,64),(256,128)):
            raw=rng.randbytes(3*projection_bytes(ff,n))+struct.pack('<4f',.5,1.25,2.5,0.)
            ref=transform_expert(raw,n,ff)
            fast=transform_expert(raw,n,ff,fast=True)
            self.assertEqual(ref,fast)
            self.assertEqual(transform_expert(fast,n,ff,True,fast=True),raw)
            self.assertEqual(fast[-16:],raw[-16:])

if __name__=='__main__': unittest.main()
