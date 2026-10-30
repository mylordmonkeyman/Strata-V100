#!/usr/bin/env python3
"""Build an immutable compressed NVFP4 QPN cache from a canonical native expert blob.

Only the *order* of 36-byte NVFP4 blocks changes. Quantization scales,
codes and the 16-byte expert tail are preserved bit-for-bit. This is an
offline tool, NOT a Strata loader, CUDA code, or a modification to GGUF.

The cache must be validated against model/source SHA and layout-version
metadata before a future host-pinned stream source consumes it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile

BLOCK = 36
CODEC = "strata-qpn-nvfp4-chunkmajor-v1"
TAIL = 16
COPY_CHUNK = 1 << 20


def projection_bytes(rows: int, cols: int) -> int:
    if rows < 1 or cols < 1 or cols % 64:
        raise ValueError("positive rows and K divisible by 64 are required")
    return rows * (cols // 64) * BLOCK


def rearrange(blocks: bytes, rows: int, cols: int, inverse: bool = False) -> bytes:
    stride = cols // 64
    if len(blocks) != projection_bytes(rows, cols):
        raise ValueError("projection has invalid native compressed byte length")
    src = memoryview(blocks)
    result = bytearray(len(blocks))
    # Canonical native GGUF: [row][K/64][36].
    # QPN layout: [K/64][row][36].
    for kblock in range(stride):
        for row in range(rows):
            canonical = (row * stride + kblock) * BLOCK
            packed = (kblock * rows + row) * BLOCK
            if inverse:
                result[canonical:canonical + BLOCK] = src[packed:packed + BLOCK]
            else:
                result[packed:packed + BLOCK] = src[canonical:canonical + BLOCK]
    return bytes(result)


def transform_expert(data: bytes, n: int, ff: int, inverse: bool = False) -> bytes:
    matrix = projection_bytes(ff, n)
    if matrix != projection_bytes(n, ff):
        raise ValueError("inconsistent weight matrix sizes")
    if len(data) != 3 * matrix + TAIL:
        raise ValueError("expert blob length mismatch")
    tail = data[-TAIL:]
    scales = struct.unpack("<4f", tail)
    if not all(math.isfinite(s) and s > 0 for s in scales[:3]) or scales[3] != 0:
        raise ValueError("noncanonical expert tail")
    # A contiguous, combined GU projection (all gate rows, then all up
    # rows), then an independent down projection and unmodified tail.
    return (
        rearrange(data[:2 * matrix], 2 * ff, n, inverse)
        + rearrange(data[2 * matrix:3 * matrix], n, ff, inverse)
        + tail
    )


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(COPY_CHUNK):
            h.update(block)
    return h.hexdigest()


def _temp_path(parent: Path, name: str) -> Path:
    fd, raw = tempfile.mkstemp(prefix="." + name + ".", dir=parent)
    os.fchmod(fd, 0o600)
    os.close(fd)
    return Path(raw)


def convert(source: Path, output: Path, manifest: Path,
            n: int, ff: int, expected_hash: str | None,
            expert_count: int | None, overwrite: bool) -> dict:
    source, output, manifest = source.resolve(), output.resolve(), manifest.resolve()
    if len({source, output, manifest}) != 3:
        raise ValueError("source/output/manifest must be distinct")
    if not source.is_file() or not output.parent.is_dir() or not manifest.parent.is_dir():
        raise ValueError("input must exist and output directories must already exist")
    if (output.exists() or manifest.exists()) and not overwrite:
        raise FileExistsError("output or manifest already exists; use --overwrite explicitly")
    matrix = projection_bytes(ff, n)
    if matrix != projection_bytes(n, ff):
        raise ValueError("expert dimensions produce inconsistent matrices")
    size = 3 * matrix + TAIL
    raw_bytes = source.stat().st_size
    if raw_bytes <= 0 or raw_bytes % size:
        raise ValueError("expert source is not an exact sequence of native blobs")
    experts = raw_bytes // size
    if expert_count is not None and experts != expert_count:
        raise ValueError("expert count differs from expected")
    orig_sha = digest_file(source)
    if expected_hash and orig_sha.lower() != expected_hash.lower():
        raise ValueError("source SHA256 differs from requested pinned fixture")
    tmp: Path | None = None
    meta_tmp: Path | None = None
    try:
        tmp = _temp_path(output.parent, output.name)
        h = hashlib.sha256()
        with source.open("rb") as src, tmp.open("wb") as dst:
            for i in range(experts):
                original = src.read(size)
                if len(original) != size:
                    raise ValueError("truncated expert blob")
                packed = transform_expert(original, n, ff)
                if transform_expert(packed, n, ff, True) != original:
                    raise ValueError(f"block-level round-trip failed for expert {i}")
                dst.write(packed)
                h.update(packed)
            if src.read(1):
                raise ValueError("unexpected trailing source bytes")
            dst.flush()
            os.fsync(dst.fileno())
        if source.stat().st_size != raw_bytes or digest_file(source) != orig_sha:
            raise ValueError("source changed while the cache was prepared")
        meta = {
            "layout": CODEC,
            "dtype": "GGUF_NVFP4_compressed_blocks_reordered",
            "matrix_format": "GU_then_down_then_unchanged_tail",
            "rows_N": n, "hidden_FF": ff,
            "experts": experts, "bytes_per_expert": size,
            "total_bytes": raw_bytes,
            "source_sha256": orig_sha, "cache_sha256": h.hexdigest(),
            "version": 1,
            "mmq_requires_original_canonical_source": True,
            "quantization_values_bitwise_preserved": True,
        }
        meta_tmp = _temp_path(manifest.parent, manifest.name)
        with meta_tmp.open("w", encoding="utf-8") as out:
            json.dump(meta, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, output)
        tmp = None
        os.replace(meta_tmp, manifest)  # manifest is written last
        meta_tmp = None
        return meta
    finally:
        for item in (tmp, meta_tmp):
            if item is not None:
                item.unlink(missing_ok=True)


def self_test() -> None:
    import random
    from tempfile import TemporaryDirectory
    rng = random.Random(20261029)
    n, ff = 128, 64
    size = 3 * projection_bytes(ff, n) + TAIL
    tails = struct.pack("<4f", 0.5, 1.25, 2.5, 0.0)
    originals = []
    for _ in range(3):
        data = rng.randbytes(size - TAIL) + tails
        packed = transform_expert(data, n, ff)
        assert len(packed) == len(data) and packed != data
        assert packed[-TAIL:] == tails
        assert transform_expert(packed, n, ff, True) == data
        originals.append(data)
    with TemporaryDirectory() as dirname:
        path = Path(dirname)
        source, output, manifest = path / "source.bin", path / "cache.bin", path / "cache.json"
        source.write_bytes(b"".join(originals))
        sha = digest_file(source)
        meta = convert(source, output, manifest, n, ff, sha, 3, False)
        assert output.stat().st_size == source.stat().st_size
        assert meta["source_sha256"] == sha
        assert meta["cache_sha256"] == digest_file(output)
        assert json.loads(manifest.read_text()) == meta
        raw = output.read_bytes()
        for i in range(3):
            assert transform_expert(raw[i * size:(i + 1) * size], n, ff, True) == originals[i]
        try:
            convert(source, output, manifest, n, ff, "0" * 64, 3, True)
        except ValueError:
            assert digest_file(output) == meta["cache_sha256"]
        else:
            raise AssertionError("mismatched input SHA unexpectedly accepted")
        try:
            convert(source, output, manifest, n, ff, sha, 3, False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing cache overwritten without permission")
    print("QPN offline compressed-block cache roundtrip, SHA and immutability tests PASSED")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--n", type=int, default=2560)
    p.add_argument("--ff", type=int, default=640)
    p.add_argument("--expected-sha256")
    p.add_argument("--experts", type=int)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.source, args.output, args.manifest)):
        p.error("--source, --output and --manifest required unless --self-test")
    meta = convert(args.source, args.output, args.manifest, args.n, args.ff,
                   args.expected_sha256, args.experts, args.overwrite)
    # Print metadata only. Never output weight values or private file contents.
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
