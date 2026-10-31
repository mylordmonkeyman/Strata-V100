"""Host-only pinned ggml quantizer vs independent Q4_K/IQ4_NL block oracle.

No real model, GPU or private data. Executable synthetic IPC is captured
in memory; only aggregate errors appear in CI logs.
"""
import math
import statistics
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
from qpn_q4tc_block_oracle import decode_row
from qpn_q4tc_nvfp4_oracle import decode_nvfp4_row

TAILS={"gate":1.25,"up":0.5,"down":2.0}
TAIL_MULTIPLIERS=(1.0,0.25,2.5,-0.75)
WIDE_SCALES=(0x08,0x18,0x28,0x38,0x58,0x78)
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
def encoded_qad(k,seed):
    pattern,role_seed=divmod(seed,3)
    if pattern not in range(4):raise ValueError("invalid synthetic pattern")
    out=bytearray()
    for b in range(k//64):
        for s in range(4):
            i=b*64+s*16
            scale=0x28+((i//16+role_seed*2)%9)*3
            if pattern==1:
                scale=0x7F if s==1 and b%3==0 else 0x20+((i//16+role_seed)%6)*5
            if pattern==2:scale=WIDE_SCALES[(b*4+s+role_seed)%6]
            if pattern==3:scale=0x18+((i//16+role_seed*3)%11)*5
            out.append(scale)
        for s in range(4):
            i=b*64+s*16
            for j in range(8):
                lo=((i+j)*7+((i+j)//16)*3+role_seed*5)&15
                hi=((i+j+8)*7+((i+j+8)//16)*3+role_seed*5)&15
                if pattern==1 and (b+s)%3==0:lo=hi=0
                if pattern==3:
                    lo=15 if j%4==0 else 8+((j+s+role_seed)%8)
                    hi=7 if j%3==0 else (j*3+s+role_seed)%8
                out.append(lo|(hi<<4))
    return bytes(out)
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
            self.assertEqual(len(f),11)
            role,seed_t,mode,kt,ref,src_h,q_h,qref_h,dq_h,qad_h,qad_dq_h=f
            seed=int(seed_t)
            pattern,role_seed=divmod(seed,3)
            self.assertIn(pattern,range(4))
            self.assertEqual(role_seed,("gate","up","down").index(role))
            tail=f32(TAILS[role]*TAIL_MULTIPLIERS[pattern])
            self.assertIn(role,TAILS)
            self.assertIn(mode,("tail","folded"))
            k=int(kt)
            self.assertEqual(k,WIDTHS[role])
            self.assertNotIn((role,pattern,mode),rows)
            qad=bytes.fromhex(qad_h)
            self.assertEqual(len(qad),k//64*36)
            self.assertEqual(qad,encoded_qad(k,seed))
            original=floats(qad_dq_h,k)
            self.assertEqual(original,decode_nvfp4_row(qad,k),
                             "independent NVFP4 decoder disagrees with pinned ggml")
            if pattern==0:
                self.assertEqual(original,tuple(synthetic(k,seed)))
            src=floats(src_h,k)
            expected=original if mode=="tail" else tuple(f32(v*tail) for v in original)
            self.assertEqual(src,expected)
            q=bytes.fromhex(q_h)
            qref=bytes.fromhex(qref_h)
            self.assertEqual(int(q==qref),int(ref))
            kind="IQ4_NL" if role=="down" else "Q4_K"
            self.assertEqual(len(q),k//(32 if role=="down" else 256)*
                             (18 if role=="down" else 144))
            self.assertEqual(len(qref),len(q))
            ref_dq=decode_row(qref,kind,k)
            dq=floats(dq_h,k)
            oracle=decode_row(q,kind,k)
            worst=max(abs(a-b) for a,b in zip(dq,oracle))
            self.assertLessEqual(worst,1e-5*max(1,max(map(abs,dq))),
                                 "independent decoder disagrees with pinned ggml")
            self.assertTrue(all(map(math.isfinite,dq)))
            rows[(role,pattern,mode)]=(src,dq,int(ref),worst,ref_dq)
        self.assertEqual(len(rows),24)
        mismatched_iq4=0
        for role in TAILS:
            preserved_errors=[]
            folded_errors=[]
            reference_errors=[]
            for pattern in range(4):
                tail=f32(TAILS[role]*TAIL_MULTIPLIERS[pattern])
                src,dq,ref,_,ref_dq=rows[(role,pattern,"tail")]
                _,folded,ref_folded,_,_=rows[(role,pattern,"folded")]
                if role!="down":self.assertEqual((ref,ref_folded),(1,1))
                if role=="down" and not ref:mismatched_iq4+=1
                target=[v*tail for v in src]
                preserved=rmse(target,[v*tail for v in dq])
                folded_err=rmse(target,folded)
                ref_err=rmse(target,[v*tail for v in ref_dq])
                self.assertTrue(all(math.isfinite(e) for e in (preserved,folded_err,ref_err)))
                # Source-dependent error is allowed; never silently treat
                # quantizer disagreement or scale folding as bitwise parity.
                denom=math.sqrt(math.fsum(v*v for v in target)/len(target))
                self.assertGreater(denom,0)
                preserved_errors.append(preserved/denom)
                folded_errors.append(folded_err/denom)
                reference_errors.append(ref_err/denom)
            print(f"Q4TC synthetic {role} 4 patterns: "
                  f"tail_relative_RMSE_median={statistics.median(preserved_errors):.6g} "
                  f"folded_relative_RMSE_median={statistics.median(folded_errors):.6g} "
                  f"ref_relative_RMSE_median={statistics.median(reference_errors):.6g} "
                  f"worst_tail_relative_RMSE={max(preserved_errors):.6g}")
        self.assertGreater(mismatched_iq4,0,
                           "pinned IQ4_NL production/reference distinction disappeared")
if __name__=="__main__":unittest.main()
