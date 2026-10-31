"""Host-only pinned ggml quantizer vs independent Q4_K/IQ4_NL block oracle.

No real model, GPU or private data. Executable synthetic IPC is captured
in memory; only aggregate errors appear in CI logs.
"""
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
from qpn_q4tc_block_oracle import decode_row

TAILS={"gate":1.25,"up":0.5,"down":2.0}
WIDTHS={"gate":2560,"up":2560,"down":640}
E2M1=(0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.)
def f32(x):return struct.unpack("<f",struct.pack("<f",x))[0]
def synthetic(k,seed):
    out=[]
    for i in range(k):
        code=(i*7+(i//16)*3+seed*5)&15
        s=0x28+((i//16+seed*2)%9)*3
        scale=(1+(s&7)/8)*2**(((s>>3)&15)-7)
        out.append(f32(E2M1[code]*scale))
    return out
def floats(h,k):
    raw=bytes.fromhex(h)
    if len(raw)!=k*4:raise ValueError("bad synthetic IPC length")
    return struct.unpack("<"+str(k)+"f",raw)
def rmse(a,b):
    return math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b))/len(a))

class PinnedGgmlFixture(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("QPN_Q4TC_GGML_FIXTURE"),
                         "pinned ggml fixture executable not built")
    def test_pinned_ggml_quantizer(self):
        r=subprocess.run([os.environ["QPN_Q4TC_GGML_FIXTURE"]],
                         capture_output=True,text=True,check=True,timeout=45)
        self.assertFalse(r.stderr.strip())
        rows={}
        for line in r.stdout.splitlines():
            f=line.split()
            self.assertEqual(len(f),7)
            role,mode,kt,ref,src_h,q_h,dq_h=f
            self.assertIn(role,TAILS)
            self.assertIn(mode,("tail","folded"))
            k=int(kt)
            self.assertEqual(k,WIDTHS[role])
            self.assertNotIn((role,mode),rows)
            src=floats(src_h,k)
            expected=synthetic(k,("gate","up","down").index(role))
            if mode=="folded":expected=[f32(v*TAILS[role]) for v in expected]
            self.assertEqual(src,tuple(expected))
            q=bytes.fromhex(q_h)
            kind="IQ4_NL" if role=="down" else "Q4_K"
            self.assertEqual(len(q),k//(32 if role=="down" else 256)*
                             (18 if role=="down" else 144))
            dq=floats(dq_h,k)
            oracle=decode_row(q,kind,k)
            worst=max(abs(a-b) for a,b in zip(dq,oracle))
            self.assertLessEqual(worst,1e-5*max(1,max(map(abs,dq))),
                                 "independent decoder disagrees with pinned ggml")
            self.assertTrue(all(map(math.isfinite,dq)))
            rows[(role,mode)]=(src,dq,int(ref),worst)
        self.assertEqual(len(rows),6)
        for role in TAILS:
            src,dq,ref,_=rows[(role,"tail")]
            _,folded,ref_folded,_=rows[(role,"folded")]
            if role!="down":self.assertEqual((ref,ref_folded),(1,1))
            target=[v*TAILS[role] for v in src]
            preserved=rmse(target,[v*TAILS[role] for v in dq])
            folded_err=rmse(target,folded)
            self.assertGreater(preserved,0)
            self.assertGreater(folded_err,0)
            self.assertTrue(math.isfinite(preserved) and math.isfinite(folded_err))
            print(f"Q4TC synthetic pinned ggml {role}: tail_RMSE={preserved:.6g} "
                  f"folded_RMSE={folded_err:.6g} ref_match={ref}/{ref_folded}")
if __name__=="__main__":unittest.main()
