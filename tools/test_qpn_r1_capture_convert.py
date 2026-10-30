"""Private QPR1 prefill capture regression checks: host-only; synthetic IDs."""
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import tempfile
import unittest

from qpn_r1_prefill_convert import (HEADER, COUNTS, N_EXPERTS, N_LAYERS,
                                     decode_private_prefill, write_private_json)


class QpnR1PrefillTest(unittest.TestCase):
    def build_frames(self, path, chunks=(4,3), omit=None, corrupt=False):
        with open(path,"wb") as f:
            off=0
            for chunk,t in enumerate(chunks):
                for layer in range(N_LAYERS):
                    if (chunk,layer)==omit:
                        continue
                    counts=[t if e<10 else 0 for e in range(N_EXPERTS)]
                    if corrupt and chunk==0 and layer==0:counts[0]-=1
                    f.write(HEADER.pack(b"QPR1",1,off,t,layer,512,10,0))
                    f.write(COUNTS.pack(*counts))
                    f.write(bytes([1 if e < 96 else 0 for e in range(N_EXPERTS)]))
                off+=t
        os.chmod(path,0o600)

    def test_complete_prefill_and_private_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"capture.bin"
            self.build_frames(p)
            data=decode_private_prefill(p)
            self.assertEqual(sum(c["tokens"] for c in data["prefill"]),7)
            self.assertEqual(len(data["prefill"]),2)
            self.assertEqual(len(data["prefill"][0]["layers"]),48)
            self.assertEqual(len(data["prefill"][0]["layers"][0]["counts"]),10)
            self.assertEqual(data["decode"],[])
            target=Path(d)/"private.json"
            write_private_json(target,data)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode),0o600)
            self.assertEqual(json.loads(target.read_text())["schema"],1)
            with self.assertRaises(FileExistsError):write_private_json(target,data)

    def test_missing_layer_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"capture"
            self.build_frames(p,omit=(0,47))
            with self.assertRaises(ValueError):decode_private_prefill(p)

    def test_inconsistent_counts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"capture"
            self.build_frames(p,corrupt=True)
            with self.assertRaises(ValueError):decode_private_prefill(p)

    def test_reject_world_readable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"capture"
            self.build_frames(p)
            p.chmod(0o644)
            with self.assertRaises(ValueError):decode_private_prefill(p)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"capture"
            self.build_frames(p)
            q=Path(d)/"link"
            q.symlink_to(p)
            with self.assertRaises(ValueError):decode_private_prefill(q)

    def test_header_compiles_and_private_writes(self):
        root=Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as d:
            code=Path(d)/"capture.cpp"
            exe=Path(d)/"capture"
            code.write_text("""
#include "src/prefill/qpn_r1_prefill_capture.hpp"
#include <vector>
int main() {
  std::vector<int32_t> counts(512,0), resident(48*512,-1);
  for(int i=0;i<10;i++){counts[i]=4;resident[i]=i;}
  strata::prefill::QpnR1PrefillCapture c(true);
  if(!c.good())return 1;
  for(int i=0;i<48;i++)if(!c.record(0,4,i,10,counts,resident.data(),512,false))return 2;
  return 0;
}
""")
            cp=subprocess.run(["g++","-std=c++17","-Wall","-Wextra",
                "-DSTRATA_QPN_PREFILL_EXPERIMENTAL=1","-I",str(root),
                str(code),"-o",str(exe)],text=True,capture_output=True)
            self.assertEqual(cp.returncode,0,cp.stderr)
            output=Path(d)/"generated.bin"
            run=subprocess.run([str(exe)],env={**os.environ,
                "STRATA_QPN_R1_PRIVATE_PREFILL_CAPTURE":str(output)},
                capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            data=decode_private_prefill(output)
            self.assertEqual(data["prefill"][0]["layers"][0]["counts"][0],[0,4])
            self.assertEqual(stat.S_IMODE(output.stat().st_mode),0o600)


if __name__=="__main__":unittest.main()
