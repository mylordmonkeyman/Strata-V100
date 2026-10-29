#!/usr/bin/env python3
"""Create canonical QPN expert fixtures without converting a full model pack.

Download just the checkpoint index and shards containing selected experts.
Reuses Strata's existing CSF decoder and lossless GGUF block rearrangement.
No PyTorch, GPU, llama.cpp conversion, PLE, dense weights or MTP preparation.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct

import numpy as np

from nvfp4_local_pack import Checkpoint, CSF_FIXED, CSF_EXC, csf_decode, nvfp4_blocks

REPO = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
ROLES = (("gate", 640, 2560), ("up", 640, 2560), ("down", 2560, 640))
SUFFIXES = ("weight", "weight_scale" + CSF_FIXED, "weight_scale" + CSF_EXC, "weight_scale_2")


def selection(value):
    try:
        layer, expert = map(int, value.split(":"))
        if layer < 0 or expert < 0:
            raise ValueError()
        return layer, expert
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected nonnegative LAYER:EXPERT") from exc


def prefix(layer, expert, role):
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}_proj."


def needed_shards(weight_map, picks):
    names = [prefix(l, e, role) + suffix for l, e in picks for role, _, _ in ROLES for suffix in SUFFIXES]
    missing = [n for n in names if n not in weight_map]
    if missing:
        raise ValueError("checkpoint index lacks required tensor: " + missing[0])
    files = sorted({weight_map[n] for n in names})
    for filename in files:
        path = PurePosixPath(filename)
        if path.is_absolute() or ".." in path.parts or not filename.endswith(".safetensors"):
            raise ValueError("unexpected checkpoint shard path")
    return files


def expert_blob(ck, layer, expert):
    parts, globals_, histograms = [], [], {}
    for role, rows, cols in ROLES:
        base = prefix(layer, expert, role)
        for suffix, dtype in zip(SUFFIXES, ("U8", "U8", "U32", "F32")):
            if ck.dtype(base + suffix) != dtype:
                raise ValueError(f"{base + suffix}: expected dtype {dtype}")
        weights = ck.array(base + "weight", np.uint8)
        if weights.shape != (rows, cols // 2):
            raise ValueError(f"{base}: unexpected weight shape {weights.shape}")
        scales = csf_decode(ck.array(base + "weight_scale" + CSF_FIXED, np.uint8),
                            ck.array(base + "weight_scale" + CSF_EXC, np.uint32))
        if scales.shape != (rows, cols // 16) or (scales & 128).any():
            raise ValueError(f"{base}: invalid scale shape/sign bit")
        global_scale = ck.array(base + "weight_scale_2", np.float32).reshape(-1)
        if global_scale.size != 1 or not np.isfinite(global_scale[0]) or global_scale[0] <= 0:
            raise ValueError(f"{base}: invalid FP32 tensor-global scale")
        globals_.append(float(global_scale[0]))
        parts.append(nvfp4_blocks(weights, scales).tobytes())
        codes, counts = np.unique(scales, return_counts=True)
        histograms[role] = {int(k): int(v) for k, v in zip(codes, counts)}
    blob = b"".join(parts) + struct.pack("<4f", *globals_, 0)
    if len(blob) != 2764816:
        raise ValueError("unexpected expert size")
    return blob, {"layer": layer, "expert": expert, "n_embd": 2560, "n_ff": 640,
                  "ggml_type": 40, "bytes": len(blob), "scale_tail": globals_ + [0],
                  "scale_byte_histograms": histograms, "expert_sha256": hashlib.sha256(blob).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True, help="partial checkpoint download directory")
    p.add_argument("--expert", type=selection, action="append", help="repeatable LAYER:EXPERT; default low/middle/high")
    p.add_argument("--revision", default="main", help="HF revision; resolved to an exact SHA before download")
    p.add_argument("--local-only", action="store_true", help="use already downloaded index/shards without network")
    args = p.parse_args()
    picks = list(dict.fromkeys(args.expert or [(0, 7), (24, 300), (47, 511)]))
    args.out.mkdir(parents=True, exist_ok=True)
    # Refuse to replace fixtures silently, including a partially completed run.
    for l, e in picks:
        for suffix in (".bin", "-audit.json"):
            if (args.out / f"l{l}-e{e}{suffix}").exists():
                p.error("output exists; use a new --out directory")
    index_name = "model.safetensors.index.json"
    if args.local_only:
        revision = None
        index_path = args.source_dir / index_name
    else:
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi().model_info(REPO, revision=args.revision).sha
        if not revision:
            raise ValueError("could not resolve checkpoint revision")
        # Keep each downloaded revision separate to avoid stale cross-revision files.
        args.source_dir = args.source_dir / revision
        index_path = Path(hf_hub_download(REPO, index_name, revision=revision, local_dir=args.source_dir))
    index = json.loads(index_path.read_text())
    files = needed_shards(index["weight_map"], picks)
    print(f"Selected {len(picks)} experts; need {len(files)} checkpoint shards. Shards may be large.", flush=True)
    for filename in files:
        if not args.local_only:
            hf_hub_download(REPO, filename, revision=revision, local_dir=args.source_dir)
        elif not (args.source_dir / filename).is_file():
            raise FileNotFoundError(args.source_dir / filename)
    ck = Checkpoint(args.source_dir, partial_ok=True)
    for layer, expert in picks:
        blob, report = expert_blob(ck, layer, expert)
        report.update(repo_id=REPO, revision=revision,
                      index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                      source_shards=files, method="existing Strata CSF decode and nvfp4_blocks; no requantization")
        target = args.out / f"l{layer}-e{expert}.bin"
        with target.open("xb") as f:
            f.write(blob)
        target.with_name(f"l{layer}-e{expert}-audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {target} ({len(blob):,} bytes, sha256 {report['expert_sha256']})", flush=True)


if __name__ == "__main__":
    main()
