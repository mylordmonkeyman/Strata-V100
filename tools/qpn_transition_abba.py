#!/usr/bin/env python3
"""MMQ-only A/B/B/A: transition fallback vs ungrouped, pinned 2047-token prefill.
Raw logs, logits, and prompt remain private. No independent oracle or QPN claim.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from qpn_full_model_ab import benchmark_env, diff, logits, read_summary, safe_failure_diagnostics, sha
from qpn_supply_profile import transition_bypass_count, parse_traces

MODEL_SHA = '1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5'
PROMPT_SHA = 'df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414'
ORDER = ('both-off', 'transition-fallback', 'transition-fallback', 'both-off')
SELECTIVE_ORDER = ('transition-fallback', 'selective-supply', 'selective-supply', 'transition-fallback')


def mode_flags(mode):
    if mode not in ORDER:
        raise ValueError('invalid ABBA mode')
    return dict(STRATA_QPN_DIAG_GG_DISABLE_STREAM='1' if mode == 'both-off' else '0',
                STRATA_QPN_DIAG_GG_DISABLE_STAGED='1' if mode == 'both-off' else '0',
                STRATA_QPN_DIAG_GG_TRANSITION_FALLBACK='1' if mode == 'transition-fallback' else '0')


def compare(a, b, ref, candidate):
    d = {k: v for k, v in diff(a, b).items() if k not in ('argmax_base', 'argmax_candidate')}
    d.update(reference=ref, candidate=candidate, bitwise_equal=(a.tobytes() == b.tobytes()))
    return d


def evaluate(summary, outputs, order=ORDER):
    if len(outputs) != 4 or len(summary['runs']) != 4:
        raise ValueError('incomplete ABBA')
    summary['comparisons'] = [compare(outputs[i], outputs[j], order[i], order[j])
                              for i, j in ((0, 3), (1, 2), (0, 1), (0, 2), (3, 1), (3, 2))]
    if not all(d['bitwise_equal'] and d['relative_l2'] == 0 and d['kl_base_vs_candidate'] == 0
               and d['argmax_match'] for d in summary['comparisons']):
        summary['status'] = 'failed_mmq_abba_bitwise_parity'
        return False
    for field in ('prefill_ms', 'ttft_ms'):
        medians = {mode: statistics.median(r[field] for r in summary['runs'] if r['mode'] == mode)
                   for mode in (order[0], order[1])}
        if any(not math.isfinite(x) or x <= 0 for x in medians.values()):
            summary['status'] = 'failed_timing'
            return False
        summary['median_' + field] = medians
        summary['median_' + field.removesuffix('_ms') + '_speedup'] = (
            medians[order[0]] / medians[order[1]])
    summary['status'] = 'mmq_abba_bitwise_parity_passed_quality_unqualified'
    return True


def supply_plan():
    return [('forced-staged', False), ('forced-staged', True),
            ('transition-fallback', False), ('transition-fallback', True)]


def supply_mode_flags(mode):
    """Forced-staged is grouped MMQ without the transition diagnostic counter."""
    if mode not in ('forced-staged', 'transition-fallback'):
        raise ValueError('invalid supply mode')
    flags = mode_flags('transition-fallback')
    if mode == 'forced-staged':
        flags['STRATA_QPN_DIAG_GG_TRANSITION_FALLBACK'] = '0'
    return flags


def validate_supply_traces(traces, mode):
    """Require every sampled layer in both pinned chunks; reject partial traces."""
    if mode not in ('forced-staged', 'transition-fallback'):
        raise ValueError('invalid supply mode')
    expected = [(0, 1024), (1024, 1023)]
    if [(t['chunk'], t['tokens']) for t in traces] != expected:
        raise ValueError('missing, duplicate or unexpected supply chunk')
    for i, t in enumerate(traces):
        if t['stream_all'] != (mode == 'transition-fallback' and i == 0):
            raise ValueError('unexpected supply regime')
        if {l['layer'] for l in t['layers']} != {0, 24, 47}:
            raise ValueError('incomplete sampled layers')
        for l in t['layers']:
            if not l['h2d_count'] or not l['products'] or l['routed'] is None:
                raise ValueError('missing transfer, compute or residency accounting')
            r = l['routed']
            l['resident_group_fraction'] = r['resident_groups'] / r['groups'] if r['groups'] else None
            l['resident_row_fraction'] = r['resident_rows'] / r['rows'] if r['rows'] else None
            l['routed_nonresident_groups'] = r['groups'] - r['resident_groups']
            # Counts only: routing lookahead is unavailable before DMA in stream-all.
            l['copies_beyond_routed_nonresident'] = max(0, l['h2d_count'] - l['routed_nonresident_groups'])
    return traces


def evaluate_supply(summary, outputs):
    if len(outputs) != 4 or len(summary['runs']) != 4:
        raise ValueError('incomplete supply control')
    summary['comparisons'] = [compare(outputs[i], outputs[i+1], mode + '-off', mode + '-trace')
                              for i, mode in ((0, 'forced-staged'), (2, 'transition-fallback'))]
    if not all(d['bitwise_equal'] for d in summary['comparisons']):
        summary['status'] = 'failed_supply_trace_bitwise_parity'
        return False
    summary['cross_supply_comparison'] = compare(outputs[0], outputs[2], 'forced-staged', 'transition-fallback')
    summary['status'] = 'supply_profile_parity_passed_quality_unqualified'
    return True


def selective_flags(mode):
    if mode not in SELECTIVE_ORDER:
        raise ValueError('invalid selective supply mode')
    flags = mode_flags('transition-fallback')
    flags['STRATA_QPN_MMQ_SELECTIVE_SUPPLY'] = str(int(mode == 'selective-supply'))
    return flags


def selective_accounting(raw, mode):
    pattern = (r'^strata qpn selective-supply: chunk=(\d+) tokens=(\d+) entries=(\d+) '
               r'copies=(\d+) skipped=(\d+) bytes=(\d+) skipped_bytes=(\d+) jobs=(\d+) skipped_jobs=(\d+)$')
    records = re.findall(pattern, raw, re.M)
    if len(records) != 1:
        raise ValueError('missing or duplicate selective supply accounting')
    r = dict(zip(('chunk', 'tokens', 'entries', 'copies', 'skipped', 'bytes', 'skipped_bytes', 'jobs', 'skipped_jobs'),
                 map(int, records[0])))
    if (r['chunk'] != 0 or r['tokens'] != 1024 or r['entries'] <= 0 or r['bytes'] <= 0
            or r['copies'] + r['skipped'] != r['entries'] or r['jobs'] > r['entries']
            or r['skipped_jobs'] > min(r['jobs'], r['skipped'])):
        raise ValueError('invalid selective accounting')
    if mode == 'selective-supply':
        if r['skipped'] <= 0 or r['skipped_bytes'] <= 0:
            raise ValueError('selective supply was not exercised')
    elif mode != 'transition-fallback' or r['skipped'] or r['skipped_bytes'] or r['skipped_jobs']:
        raise ValueError('baseline unexpectedly skipped transfers')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'binary', 'prepared', 'state', 'report'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--supply-profile', action='store_true',
                   help='profile repeatable forced-staged and transition-fallback off/on controls')
    p.add_argument('--selective-abba', action='store_true',
                   help='same-stream-all ABBA of default-off routing-aware DMA skipping')
    a = p.parse_args()
    if a.supply_profile and a.selective_abba:
        p.error('supply modes are mutually exclusive')
    order = SELECTIVE_ORDER if a.selective_abba else ORDER
    plan = supply_plan() if a.supply_profile else [(m, False) for m in order]
    a.report.mkdir(parents=True, exist_ok=True)
    a.state.mkdir(parents=True, exist_ok=True)
    prompt = a.state / 'prompt-2048.txt'
    pack = a.prepared / 'pack/experts.bin'
    for f in (a.binary, prompt, pack, a.prepared / 'dense.gguf', a.prepared / 'ple-nvfp4.gguf'):
        if f.is_symlink() or not f.is_file():
            raise ValueError('missing or symlinked private model/prompt/binary')
    if sha(pack) != MODEL_SHA or sha(prompt) != PROMPT_SHA:
        raise ValueError('private model/prompt SHA256 changed')
    env = benchmark_env()
    env.update(STRATA_PREFILL_NVFP4='w4a8', STRATA_PREFILL_CPU_SHARE='0',
               STRATA_QPN_PREFILL_EXPERIMENTAL='0', STRATA_PREFILL_STREAM_MIN='1024',
               STRATA_PREFILL_GROUP_GATHER='1', STRATA_PREFILL_ISSUER='1',
               STRATA_QPN_SUPPLY_TRACE='0')
    cmd = [str(a.binary), '--pack', str(a.prepared / 'pack'), '--native', str(a.prepared / 'dense.gguf'),
           '--native-dense-gguf', str(a.prepared / 'dense.gguf'), '--ple-gguf', str(a.prepared / 'ple-nvfp4.gguf'),
           '--tokens-file', str(prompt), '--max-context', '4096', '--max-new', '16', '--greedy',
           '--expert-cache', 'auto', '--expert-profile', str(a.source / 'data/expert-profile.bin'),
           '--resident-budget-gib', '24', '--pool-workers', '16', '--prefill', '1024', '--spec', '2',
           '--suffix-draft', '0', '--adapt-every', '0', '--vram-reserve-mib', '1536', '--stats']
    summary = dict(schema=1, scope='MMQ-only transition fallback vs both-off ABBA, 2047 prefill tokens',
                   source_commit=subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'],
                                                         text=True).strip(),
                   model_sha256=MODEL_SHA, prompt_sha256=PROMPT_SHA, binary_sha256=sha(a.binary),
                   order=list(ORDER), control_flags={k: v for k, v in env.items() if k.startswith('STRATA_')},
                   performance_only=True, numerical_quality_qualified=False,
                   limitations=['fresh processes, uncontrolled cache temperature, not warm-engine benchmarking',
                                'MMQ first-logit parity is not an independent GGUF FP64 oracle',
                                'no QPN execution, CUDA Graph or full-model sanitizer qualification'],
                   runs=[], comparisons=[], status='pending')
    if a.supply_profile:
        summary.update(scope='MMQ supply diagnostic: forced-staged and transition-fallback trace off/on',
                       order=[dict(mode=m, trace=t) for m, t in plan])
        summary['limitations'] += ['sampled layers 0/24/47 only; no full-model byte total',
            'event intervals include instrumentation overhead; not occupancy or PCIe link saturation',
            'no timing speedup inferred from instrumented controls; no realistic lookahead implemented']
    if a.selective_abba:
        summary.update(scope='MMQ stream-all ABBA: routing-aware selective supply vs matched transition fallback',
                       order=list(order))
        summary['limitations'] += ['experimental DMA scheduling; no full-model sanitizer or Graph qualification',
                                   'host staging jobs still drained; no skipped CPU reads or early routing prediction']
    report = a.report / 'summary.json'
    def save():
        report.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    save()
    outputs = []
    with tempfile.TemporaryDirectory(prefix='qpn-transition-abba-', dir=a.state) as private_dir:
        os.chmod(private_dir, 0o700)
        private = Path(private_dir)
        for i, (mode, trace) in enumerate(plan):
            trial = env.copy()
            trial.update(selective_flags(mode) if a.selective_abba else
                         supply_mode_flags(mode) if a.supply_profile else mode_flags(mode))
            trial['STRATA_PREFILL_STREAM_MIN'] = '4096' if mode == 'forced-staged' else '1024'
            trial['STRATA_QPN_SUPPLY_TRACE'] = str(int(trace))
            log = private / f'{i}.log'
            dump = private / f'{i}.f32'
            trial['STRATA_DUMP_FIRST_LOGITS'] = str(dump)
            start = time.monotonic()
            with log.open('w') as out:
                try:
                    code = subprocess.run(['numactl', '--cpunodebind=0', '--membind=0', *cmd],
                                          env=trial, stdout=out, stderr=subprocess.STDOUT,
                                          timeout=1100).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            row = dict(index=i, mode=mode, trace=trace, stream_min=trial['STRATA_PREFILL_STREAM_MIN'],
                       control_flags={k: v for k, v in trial.items() if k.startswith('STRATA_') and k != 'STRATA_DUMP_FIRST_LOGITS'},
                       returncode=code, wall_s=time.monotonic()-start)
            row.update(read_summary(log))
            summary['runs'].append(row)
            save()
            if code:
                row['startup_diagnostics'] = safe_failure_diagnostics(log)
                summary['status'] = 'failed_execution'
                save()
                return 2
            raw = log.read_text(errors='replace')
            if re.search(r'non[- ]finite|illegal memory access|CUDA error|device-side assert', raw, re.I):
                summary['status'] = 'failed_safety'
                save()
                return 2
            if mode in ('transition-fallback', 'selective-supply'):
                try:
                    count = transition_bypass_count(raw)
                except ValueError:
                    summary['status'] = 'failed_transition_counter'
                    save()
                    return 2
                row['transition_disabled_staged_layers'] = count
                if count != 48:
                    summary['status'] = 'failed_transition_not_fully_exercised'
                    save()
                    return 2
            elif 'strata qpn gg transition:' in raw:
                summary['status'] = 'failed_unexpected_transition_counter'
                save()
                return 2
            if (row['prefill_tokens'] != 2047 or row['prefill_ms'] is None or row['ttft_ms'] is None
                    or row['qpn_groups'] != 0 or row['qpn_reported_layers'] != 0
                    or row['qpn_plan_groups'] != 0 or not dump.is_file()):
                summary['status'] = 'failed_missing_work_or_qpn_execution'
                save()
                return 2
            v = logits(dump)
            if len(v) != 248320:
                summary['status'] = 'failed_output_shape'
                save()
                return 2
            try:
                traces = parse_traces(raw)
                if trace:
                    validate_supply_traces(traces, mode)
                    (a.report / f'{mode}-events.json').write_text(json.dumps(traces, indent=2, allow_nan=False) + '\n')
                    row['trace_chunks'] = len(traces)
                elif traces:
                    raise ValueError('unexpected tracing')
            except (ValueError, KeyError, TypeError):
                summary['status'] = 'failed_supply_trace_validation'
                save()
                return 2
            if a.selective_abba:
                try:
                    row['selective_supply'] = selective_accounting(raw, mode)
                except ValueError:
                    summary['status'] = 'failed_selective_supply_accounting'
                    save()
                    return 2
            outputs.append(v)
            save()
            print(f'MMQ ABBA {i+1}/4 {mode} completed', flush=True)
        if a.selective_abba:
            counters = [r['selective_supply'] for r in summary['runs']]
            total_bytes = {r['bytes'] + r['skipped_bytes'] for r in counters}
            if len({r['entries'] for r in counters}) != 1 or len(total_bytes) != 1:
                summary['status'] = 'failed_selective_unmatched_supply'
                save()
                return 2
        ok = evaluate_supply(summary, outputs) if a.supply_profile else evaluate(summary, outputs, order)
        save()
        return 0 if ok else 3


if __name__ == '__main__':
    raise SystemExit(main())
