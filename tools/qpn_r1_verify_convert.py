#!/usr/bin/env python3
"""Strict PRIVATE QDV1 verification-route decoder. Never upload raw output.

QDV1 is a read-only, default-off GPU-router snapshot from Verifier::run.
It is NOT automatically an autoregressive decode sample: prompt-tail and
speculative verification windows may also use Verifier::run. The caller must
classify the phase using private request context before any R1 evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

HEADER = struct.Struct("<4sIqqiiii")
COUNTS = struct.Struct("<512i")
LAYERS = 48
MAX_WINDOWS = 32
FRAME_BYTES = HEADER.size + COUNTS.size


def decode_private_verify(path):
    path = Path(path)
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077 or not 0 < st.st_size <= MAX_WINDOWS * LAYERS * FRAME_BYTES:
        raise ValueError("private verification capture must be bounded, regular and owner-only")
    raw = path.read_bytes()
    if len(raw) != st.st_size or len(raw) % (LAYERS * FRAME_BYTES):
        raise ValueError("truncated private verification window")
    windows = []
    for start in range(0, len(raw), LAYERS * FRAME_BYTES):
        layers = []
        window_pos = window_tokens = None
        for layer in range(LAYERS):
            off = start + layer * FRAME_BYTES
            magic, version, pos, tokens, l, experts, k, flags = HEADER.unpack_from(raw, off)
            if (magic != b"QDV1" or version != 1 or pos < 0 or not 1 <= tokens <= 8 or
                    l != layer or experts != 512 or k != 10 or flags != 0):
                raise ValueError("invalid private verification geometry")
            if window_pos is None:
                window_pos, window_tokens = pos, tokens
            elif (pos, tokens) != (window_pos, window_tokens):
                raise ValueError("mixed positions or sizes within a verification window")
            counts = COUNTS.unpack_from(raw, off + HEADER.size)
            if any(x < 0 or x > tokens for x in counts) or sum(counts) != tokens * k:
                raise ValueError("invalid private verification route counts")
            layers.append({"layer": layer, "counts": [[e, x] for e, x in enumerate(counts) if x]})
        windows.append({"position": window_pos, "tokens": window_tokens, "layers": layers})
    return {
        "schema": 1, "verify": windows, "decode": [],
        "private_capture_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "genuine GPU verification router snapshots; phase UNCLASSIFIED; not heldout R1",
        "limitations": ["Only first 32 Verifier::run windows captured",
                        "Prompt-tail, speculative verification and autoregressive decode not distinguished",
                        "No residency snapshots, training/heldout split, QPN dispatch or performance evidence"],
    }


def write_private_json(path, payload):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("private output path must be absolute")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = -1
            json.dump(payload, f, separators=(",", ":"), allow_nan=False)
            f.write("\n")
    finally:
        if fd != -1:
            os.close(fd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-capture", type=Path, required=True)
    p.add_argument("--private-json", type=Path, required=True)
    a = p.parse_args()
    try:
        write_private_json(a.private_json, decode_private_verify(a.private_capture))
    except (ValueError, OSError, struct.error):
        p.error("private verification capture invalid or unsafe; details suppressed")
    print("Private verification routes converted; phase unclassified; no raw IDs printed")


if __name__ == "__main__":
    main()
