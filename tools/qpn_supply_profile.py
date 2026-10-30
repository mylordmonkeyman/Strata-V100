#!/usr/bin/env python3
"""Profile existing MMQ supply at 768 staged / 1024 stream-all; no QPN dispatch.

Raw CLI logs, prompt IDs and logits remain in a private runner directory. Only
strictly typed aggregate event intervals and mathematical comparisons leave it.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from qpn_full_model_ab import sha, logits, diff, read_summary, safe_failure_diagnostics

PREFIX = 'strata qpn-supply: '
KINDS = ('h2d', 'slot_wait', 'ready_wait', 'gather', 'products')


def number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('invalid numeric trace field')
    return x


def integer(x):
    if type(x) is not int or x < 0:
        raise ValueError('invalid nonnegative integer trace field')
    return x


def union(intervals):
    result = []
    for a, b, *_ in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def duration(intervals):
    return sum(b-a for a, b in union(intervals))


def overlap(a, b):
    a, b = union(a), union(b)
    i = j = 0
    total = 0.
    while i < len(a) and j < len(b):
        total += max(0., min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def parse_traces(text):
    result = []
    for line in text.splitlines():
        if not line.startswith(PREFIX):
            continue
        raw = json.loads(line[len(PREFIX):])
        if raw.get('schema') != 1 or raw.get('instrumented') is not True or type(raw['stream_all']) is not bool:
            raise ValueError('invalid trace schema')
        # Reconstruct allowlisted fields: never copy an arbitrary raw-log object.
        row = {k: integer(raw[k]) for k in ('chunk', 'tokens', 'ring')}
        row.update(schema=1, instrumented=True, stream_all=raw['stream_all'], layers=[])
        if not row['tokens'] or not row['ring']:
            raise ValueError('empty trace configuration')
        seen = set()
        for layer in raw['layers']:
            l = integer(layer['layer'])
            if l not in (0, 24, 47) or l in seen:
                raise ValueError('unexpected or duplicate sampled layer')
            seen.add(l)
            r = {'layer': l, 'routed': None}
            routes = layer['routed']
            if routes is not None:
                r['routed'] = {k: integer(routes[k]) for k in ('groups', 'resident_groups', 'rows', 'resident_rows')}
                if routes['resident_groups'] > routes['groups'] or routes['resident_rows'] > routes['rows']:
                    raise ValueError('invalid residency accounting')
            for kind in KINDS:
                spans = []
                for p in layer[kind]:
                    if len(p) != 3:
                        raise ValueError('invalid event interval')
                    a, b, n = number(p[0]), number(p[1]), integer(p[2])
                    if b < a or (kind == 'h2d' and n == 0) or (kind != 'h2d' and n != 0):
                        raise ValueError('invalid interval bounds or bytes')
                    spans.append([a, b, n])
                r[kind] = spans
            r['h2d_bytes'] = sum(p[2] for p in r['h2d'])
            r['h2d_count'] = len(r['h2d'])
            r['slot_reuse_count'] = len(r['slot_wait'])
            for kind in KINDS:
                r[kind+'_interval_ms'] = duration(r[kind])
            ms = r['h2d_interval_ms']
            r['h2d_interval_GBps'] = r['h2d_bytes']/ms/1e6 if ms > 0 else None
            r['h2d_products_overlap_ms'] = overlap(r['h2d'], r['products'])
            row['layers'].append(r)
        result.append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'binary', 'prepared', 'state', 'report'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--repeatability', action='store_true',
                   help='diagnose 1024 off/off repeatability and inline-issuer off/on parity')
    a = p.parse_args()
    a.report.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix='qpn-supply-', dir=a.state))
    os.chmod(private, 0o700)
    tokenfile = a.state/'prompt-2048.txt'
    for f in (tokenfile, a.prepared/'pack/experts.bin', a.prepared/'dense.gguf', a.prepared/'ple-nvfp4.gguf'):
        if not f.is_file():
            raise ValueError('missing pinned private model or prompt')
    base = {k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
    base.update(STRATA_PREFILL_NVFP4='w4a8', STRATA_PREFILL_CPU_SHARE='0',
                STRATA_QPN_PREFILL_EXPERIMENTAL='0', STRATA_PREFILL_STREAM_MIN='1024')
    summary = dict(schema=1, scope='MMQ-only supply diagnosis, sampled layers 0/24/47; no QPN speed claim',
                   source_commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'], text=True).strip(),
                   model_sha256=sha(a.prepared/'pack/experts.bin'), prompt_sha256=sha(tokenfile),
                   binary_sha256=sha(a.binary), control_flags={k:v for k,v in base.items() if k.startswith('STRATA_')},
                   performance_only=True, numerical_quality_qualified=False,
                   limitations=['fresh processes; page-cache temperature uncontrolled; no cold-cache eviction',
                                'instrumented event intervals include submission/event overhead',
                                'products include gather/submission gaps; not hardware occupancy',
                                'sampled layers only; no new device routing readback',
                                'no full-model QPN or Q4TC execution; no warm-engine measurement'],
                   runs=[], comparisons=[], status='pending')
    report = a.report/'summary.json'
    def save():
        report.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    save()
    experiments = ([(1024, 'threaded-repeat', '1', ((False, 'off-a'), (False, 'off-b'))),
                    (1024, 'inline-trace', '0', ((False, 'off'), (True, 'trace')))]
                   if a.repeatability else
                   [(n, 'standard', '1', ((False, 'off'), (True, 'trace'))) for n in (768, 1024)])
    summary['repeatability_diagnostic'] = a.repeatability
    parity_failed = False
    for prefill, experiment, issuer, variants in experiments:
        pair = []
        for trace, variant in variants:
            label = f'{prefill}-{experiment}-{variant}'
            env = base.copy()
            env['STRATA_PREFILL_ISSUER'] = issuer
            out_logits = private/(label+'.f32')
            env.update(STRATA_DUMP_FIRST_LOGITS=str(out_logits), STRATA_QPN_SUPPLY_TRACE=str(int(trace)))
            cmd = [str(a.binary), '--pack', str(a.prepared/'pack'), '--native', str(a.prepared/'dense.gguf'),
                   '--native-dense-gguf', str(a.prepared/'dense.gguf'), '--ple-gguf', str(a.prepared/'ple-nvfp4.gguf'),
                   '--tokens-file', str(tokenfile), '--max-context', '4096', '--max-new', '16', '--greedy',
                   '--expert-cache', 'auto', '--expert-profile', str(a.source/'data/expert-profile.bin'),
                   '--resident-budget-gib', '24', '--pool-workers', '16', '--prefill', str(prefill), '--spec', '2',
                   '--suffix-draft', '0', '--adapt-every', '0', '--vram-reserve-mib', '1536', '--stats']
            log = private/(label+'.log')
            start = time.monotonic()
            with log.open('w') as f:
                try:
                    code = subprocess.run(['numactl','--cpunodebind=0','--membind=0',*cmd], env=env,
                                          stdout=f, stderr=subprocess.STDOUT, timeout=1100).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            row = dict(prefill=prefill, experiment=experiment, issuer=issuer, variant=variant,
                       trace=trace, returncode=code, wall_s=time.monotonic()-start)
            row.update(read_summary(log))
            summary['runs'].append(row)
            save()
            if code:
                row['startup_diagnostics'] = safe_failure_diagnostics(log)
                summary['status'] = 'failed_execution'; save(); return 2
            raw = log.read_text(errors='replace')
            if re.search(r'non[- ]finite|illegal memory access|CUDA error|device-side assert', raw, re.I):
                summary['status'] = 'failed_safety'; save(); return 2
            v = logits(out_logits)  # mandatory size and finite checks, private values
            pair.append(v)
            traces = parse_traces(raw)
            if bool(traces) != trace or row['prefill_ms'] is None or row['prefill_tokens'] < 1024 or row['qpn_groups'] != 0:
                summary['status'] = 'failed_trace_or_missing_work'; save(); return 2
            if trace:
                if not any(t['stream_all'] == (prefill == 1024) and
                           {l['layer'] for l in t['layers']} == {0,24,47} for t in traces):
                    summary['status'] = 'failed_unexercised_supply_path'; save(); return 2
                (a.report/(label+'-events.json')).write_text(json.dumps(traces, indent=2, allow_nan=False)+'\n')
                row['trace_chunks'] = len(traces)
            print(f'MMQ supply diagnostic prefill={prefill} trace={trace} completed', flush=True)
            save()
        metrics = {k:v for k,v in diff(*pair).items() if k not in ('argmax_base', 'argmax_candidate')}
        metrics.update(prefill=prefill, experiment=experiment, issuer=issuer)
        summary['comparisons'].append(metrics)
        if metrics['relative_l2'] > 1e-7 or not metrics['argmax_match']:
            parity_failed = True
            if not a.repeatability:
                summary['status'] = 'failed_trace_changed_output'; save(); return 3
        save()
    if parity_failed:
        summary['status'] = 'failed_repeatability_or_trace_parity'; save(); return 3
    summary['status'] = 'diagnostic_complete_quality_unqualified'
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
