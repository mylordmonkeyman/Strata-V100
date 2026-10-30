#!/usr/bin/env python3
"""PRIVATE rev4.2 R1 staged/stream-all prefill route-count capture decoder.

Input QPR1 frames are written by the opt-in host-routed prefill hook. Their
expert IDs and residency bits are sensitive. Never upload either this file or
the converted JSON. Actual decode/verification routing is NOT captured here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

HEADER = struct.Struct("<4sIqqiiii")
N_EXPERTS = 512
N_LAYERS = 48
MAX_FILE = 64 * 1024 * 1024
COUNTS = struct.Struct("<512i")
FRAME_LEN = HEADER.size + COUNTS.size + N_EXPERTS


def decode_private_prefill(path):
    path = Path(path)
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_FILE or st.st_size <= 0:
        raise ValueError("private capture must be a bounded regular file")
    if st.st_mode & 0o077:
        raise ValueError("private capture must not be group/world-accessible")
    raw = path.read_bytes()
    if len(raw) != st.st_size or len(raw) % FRAME_LEN:
        raise ValueError("truncated or inconsistent private QPR1 frames")
    chunks = []
    n_records = len(raw) // FRAME_LEN
    if n_records % N_LAYERS or n_records == 0:
        raise ValueError("incomplete prefill layer coverage")
    offset_expected = 0
    for base_index in range(0, n_records, N_LAYERS):
        layers = []
        token_offset = token_count = stream_all = None
        for index in range(base_index, base_index + N_LAYERS):
            at = index * FRAME_LEN
            magic, version, p0, t, layer, experts, top_k, flags = HEADER.unpack_from(raw, at)
            if magic != b"QPR1" or version != 1 or experts != N_EXPERTS or top_k < 1 or top_k > N_EXPERTS or flags not in (0,1):
                raise ValueError("unexpected private prefill capture geometry")
            if layer != index - base_index or t < 1 or t > 32768 or p0 < 0:
                raise ValueError("missing/duplicated/out-of-order prefill layer")
            if token_offset is None:
                token_offset, token_count, stream_all = p0, t, bool(flags)
                if token_offset != offset_expected:
                    raise ValueError("noncontiguous private prefill chunk offsets")
            elif (p0, t, bool(flags)) != (token_offset, token_count, stream_all):
                raise ValueError("inconsistent chunk geometry across layers")
            counts = COUNTS.unpack_from(raw, at + HEADER.size)
            presence = raw[at + HEADER.size + COUNTS.size:at + FRAME_LEN]
            if sum(counts) != t * top_k or any(c < 0 or c > t for c in counts):
                raise ValueError("incorrect routed count totals")
            if any(bit not in (0,1) for bit in presence):
                raise ValueError("invalid private residency bit")
            # Never print these IDs or the converted per-expert vectors.
            layers.append({"layer":layer,
                           "counts":[[e,int(count)] for e,count in enumerate(counts) if count],
                           "resident_experts":[e for e,bit in enumerate(presence) if bit]})
        chunks.append({"token_offset":token_offset, "tokens":token_count,
                       "stream_all":stream_all, "layers":layers})
        offset_expected += token_count
    return {"schema":1,"prefill":chunks,"decode":[],
            "private_capture_sha256":hashlib.sha256(raw).hexdigest(),
            "scope":"complete model prefill host routing only; NO actual decode routes"}


def write_private_json(path, payload):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("private output must be an absolute path")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os,"O_NOFOLLOW",0)
    flags |= getattr(os,"O_CLOEXEC",0)
    fd = os.open(path,flags,0o600)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            fd = -1
            json.dump(payload,f,separators=(",",":"),allow_nan=False)
            f.write("\n")
    finally:
        if fd != -1:os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-capture",required=True,type=Path)
    p.add_argument("--private-json",required=True,type=Path)
    a=p.parse_args()
    try:
        report=decode_private_prefill(a.private_capture)
        write_private_json(a.private_json,report)
    except (ValueError,OSError,struct.error) as e:
        p.error("private R1 capture invalid or unsafe; details suppressed")
    print("Private prefill capture converted; decode unavailable; no raw IDs printed")


if __name__=="__main__":main()
