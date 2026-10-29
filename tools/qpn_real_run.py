#!/usr/bin/env python3
"""Run standalone schedules with each captured expert's own routed rows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from qpn_m0_bench import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    index = json.loads((a.fixtures / 'index.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    failures = 0
    for row in index['fixtures']:
        blob, activations = (a.fixtures / row[k] for k in ('blob', 'activations'))
        if digest(blob) != row['expert_sha256'] or digest(activations) != row['activations_sha256']:
            raise ValueError('matched fixture changed after export')
        tokens = sorted({t for t in (1, 2, 3, 4, 8, 16, 32, row['rows']) if t <= row['rows']})
        cmd = [sys.executable, str(Path(__file__).with_name('qpn_run.py')), str(a.binary),
               '--blob', str(blob), '--activations', str(activations), '--decoders', '1',
               '--tokens', *map(str, tokens), '--iterations', '50', '--out',
               str(a.out / f"l{row['layer']}-e{row['expert']}")]
        result = subprocess.run(cmd, check=False)
        failures += int(result.returncode != 0)
    # This index contains provenance/row indices, not activation vectors.
    (a.out / 'fixture-index.json').write_text(json.dumps(index, indent=2))
    return int(failures > 0)


if __name__ == '__main__': raise SystemExit(main())
