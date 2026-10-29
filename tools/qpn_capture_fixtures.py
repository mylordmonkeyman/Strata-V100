#!/usr/bin/env python3
"""Match unchanged-engine MoE rows to their actual expert's canonical weights.

Raw fixtures stay on the runner. Selected routed experts are coverage samples,
not an all-model routing distribution or a mixed-group engine benchmark.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from qpn_audit import audit
from qpn_m0_bench import capture_summary, digest


def routed_samples(path):
    report = capture_summary(path)
    with path.open('rb') as f:
        t, n, k = struct.unpack('<3q', f.read(24))
        x = np.frombuffer(f.read(t*n*4), dtype='<f4').reshape(t, n)
        ids = np.frombuffer(f.read(t*k*4), dtype='<i4').reshape(t, k)
    # A row belongs to an expert once even if a corrupt router repeats an ID.
    if any(len(set(row)) != k for row in ids.tolist()):
        raise ValueError('duplicate expert ID within one routed token')
    counts = report['expert_counts']
    ranked = sorted(counts, key=lambda e: (-counts[e], e))
    selected = dict.fromkeys((ranked[0], ranked[len(ranked)//2], ranked[-1]))
    return report, [(e, np.flatnonzero((ids == e).any(axis=1))[:32], x) for e in selected]


def export(pack, captures, manifest_path, out):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('baseline_source') != '2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926':
        raise ValueError('unexpected activation baseline source')
    expected = {r['capture']['activation_dump_sha256'] for r in manifest['runs']
                if r['returncode'] == 0 and 'capture' in r}
    if not expected:
        raise ValueError('no successful baseline captures')
    candidates = sorted(captures.glob('layer-*.moe'))
    validated = []
    for path in candidates:
        layer = int(path.stem.removeprefix('layer-'))
        report, samples = routed_samples(path)
        if report['activation_dump_sha256'] in expected:
            validated.append((layer, report, samples))
    if {r['activation_dump_sha256'] for _, r, _ in validated} != expected:
        raise ValueError('missing or changed baseline activation captures')
    out.mkdir(parents=True, exist_ok=False)
    index = {'baseline_source': manifest['baseline_source'], 'model_revision': manifest['model_revision'],
             'baseline_manifest_sha256': digest(manifest_path), 'fixtures': [],
             'scope': 'first prefill chunks; top/median/least-routed sampled experts; actual rows without repetition'}
    for layer, report, samples in validated:
        for expert, indices, x in samples:
            weights, blob = audit(pack, layer, expert, 512)
            stem = f'l{layer}-e{expert}'
            weight_path = out / (stem + '.bin')
            activation_path = out / (stem + '.f32')
            weight_path.write_bytes(blob)
            activation_path.write_bytes(x[indices].astype('<f4', copy=False).tobytes())
            row = {'layer': layer, 'expert': expert, 'blob': weight_path.name,
                   'activations': activation_path.name, 'rows': len(indices), 'source_rows': indices.tolist(),
                   'routed_entries_in_chunk': report['expert_counts'][expert],
                   'activation_dump_sha256': report['activation_dump_sha256'],
                   'activations_sha256': digest(activation_path), 'expert_sha256': weights['expert_sha256']}
            index['fixtures'].append(row)
    (out / 'index.json').write_text(json.dumps(index, indent=2))
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pack', 'captures', 'manifest', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    result = export(a.pack, a.captures, a.manifest, a.out)
    print(f"Exported {len(result['fixtures'])} matched expert/activation fixtures")


if __name__ == '__main__': main()
