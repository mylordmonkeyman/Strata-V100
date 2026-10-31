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
EARLY_STAGE_ORDER = ('selective-supply', 'early-stage', 'early-stage', 'selective-supply')
# Two labeled cache/page warmups, followed by three balanced four-run blocks.
# Every measured arm appears six times; this remains fresh-process, NOT warm-engine.
REPEAT_WARMUP_ORDER = ('selective-supply', 'early-stage')
REPEAT_MEASURED_ORDER = (EARLY_STAGE_ORDER +
                         ('early-stage', 'selective-supply', 'selective-supply', 'early-stage') +
                         EARLY_STAGE_ORDER)


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
    if mode not in (*SELECTIVE_ORDER, *EARLY_STAGE_ORDER):
        raise ValueError('invalid selective supply mode')
    flags = mode_flags('transition-fallback')
    flags['STRATA_QPN_MMQ_SELECTIVE_SUPPLY'] = str(int(mode != 'transition-fallback'))
    flags['STRATA_QPN_MMQ_EARLY_STAGE'] = str(int(mode == 'early-stage'))
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
    if mode in ('selective-supply', 'early-stage'):
        if r['skipped'] <= 0 or r['skipped_bytes'] <= 0:
            raise ValueError('selective supply was not exercised')
    elif mode != 'transition-fallback' or r['skipped'] or r['skipped_bytes'] or r['skipped_jobs']:
        raise ValueError('baseline unexpectedly skipped transfers')
    return r



def early_stage_accounting(raw, mode, supply):
    """Allowlisted CPU-omission counters only; private raw logs never exported."""
    if mode not in EARLY_STAGE_ORDER:
        raise ValueError('unexpected early-stage mode')
    pattern = (r'^strata qpn early-stage: chunk=(\d+) tokens=(\d+) '
               r'cpu_jobs_omitted=(\d+) cpu_bytes_omitted=(\d+)$')
    matches = re.findall(pattern, raw, re.M)
    if mode == 'selective-supply':
        if matches:
            raise ValueError('baseline unexpectedly emitted early-stage counters')
        return {'cpu_jobs_omitted': 0, 'cpu_bytes_omitted': 0}
    if len(matches) != 1:
        raise ValueError('missing or duplicate early-stage record')
    chunk, tokens, jobs, amount = map(int, matches[0])
    if (chunk != 0 or tokens != 1024 or jobs > supply['skipped_jobs']
            or amount > supply['skipped_bytes']
            or (jobs == 0) != (amount == 0)
            or (jobs and amount != jobs * 2764816)):
        raise ValueError('invalid or unmatched early-stage CPU omission')
    return {'cpu_jobs_omitted': jobs, 'cpu_bytes_omitted': amount}


def evaluate_early_stage(summary, outputs):
    """Exploratory ABBA, NOT a qualified full-request speedup."""
    if not evaluate(summary, outputs, EARLY_STAGE_ORDER):
        return False
    runs = summary['runs']
    for metric in ('entries', 'copies', 'skipped', 'bytes', 'skipped_bytes',
                   'jobs', 'skipped_jobs'):
        vals = [row['selective_supply'][metric] for row in runs]
        if metric in ('entries', 'jobs') and len(set(vals)) != 1:
            summary['status'] = 'failed_mismatched_host_work'
            return False
        if metric in ('copies', 'skipped') and max(vals) - min(vals) > 16:
            summary['status'] = 'failed_mismatched_H2D'
            return False
        if metric in ('bytes', 'skipped_bytes') and max(vals) - min(vals) > 16 * 2764816:
            summary['status'] = 'failed_mismatched_H2D'
            return False
    omitted = [row['early_stage']['cpu_jobs_omitted'] for row in runs
               if row['mode'] == 'early-stage']
    summary['cpu_jobs_omitted'] = omitted
    summary['cpu_bytes_omitted'] = [row['early_stage']['cpu_bytes_omitted']
                                    for row in runs if row['mode'] == 'early-stage']
    summary['prefill_latency_ratio'] = summary['median_prefill_speedup']
    summary['ttft_latency_ratio'] = summary['median_ttft_speedup']
    summary['exploratory_prefill_5pct_gate'] = bool(
        min(omitted) > 0 and summary['median_prefill_speedup'] >= 1.0 / 0.95)
    summary['full_request_performance_qualified'] = False
    summary['status'] = ('early_stage_initial_screen_positive_unqualified'
                         if summary['exploratory_prefill_5pct_gate'] else
                         'early_stage_initial_screen_no_material_benefit')
    return True


def evaluate_early_stage_repeat(summary, outputs):
    """Bounded matched-H2D repeat screen; no warm-engine/accepted-token qualification."""
    order = REPEAT_MEASURED_ORDER
    runs = summary['runs']
    if len(runs) != len(order) or len(outputs) != len(order) or len(summary['warmups']) != 2:
        summary['status'] = 'failed_incomplete_repeat'
        return False
    if any(r['mode'] != order[i] for i, r in enumerate(runs)):
        summary['status'] = 'failed_repeat_order'
        return False
    # Compare private logits in memory. Publish only an equality boolean and shape.
    first = outputs[0].tobytes()
    summary['first_logits_bitwise_all'] = all(v.tobytes() == first for v in outputs[1:])
    summary['n_first_logits'] = len(outputs[0])
    if not summary['first_logits_bitwise_all']:
        summary['status'] = 'failed_mmq_repeat_bitwise_parity'
        return False
    for metric in ('entries', 'copies', 'skipped', 'bytes', 'skipped_bytes',
                   'jobs', 'skipped_jobs'):
        if len({r['selective_supply'][metric] for r in runs}) != 1:
            summary['status'] = 'failed_repeat_mismatched_H2D_or_host_work'
            return False
    omitted = [r['early_stage']['cpu_jobs_omitted'] for r in runs
               if r['mode'] == 'early-stage']
    if not omitted or min(omitted) <= 0:
        summary['status'] = 'stopped_repeat_no_cpu_omission'
        return False
    summary['cpu_jobs_omitted_range'] = [min(omitted), max(omitted)]
    bytes_omitted = [r['early_stage']['cpu_bytes_omitted'] for r in runs
                     if r['mode'] == 'early-stage']
    summary['cpu_bytes_omitted_range'] = [min(bytes_omitted), max(bytes_omitted)]
    if any(r.get('decode_tokens', 0) < 64 for r in runs):
        summary['status'] = 'failed_repeat_insufficient_decode_tokens'
        return False
    for metric in ('prefill_ms', 'ttft_ms', 'wall_s', 'decode_toks_per_s'):
        med = {}
        p95 = {}
        for mode in ('selective-supply', 'early-stage'):
            vals = sorted(r[metric] for r in runs if r['mode'] == mode)
            if len(vals) != 6 or any(v is None or not math.isfinite(v) or v <= 0 for v in vals):
                summary['status'] = 'failed_repeat_timing'
                return False
            med[mode] = statistics.median(vals)
            p95[mode] = vals[math.ceil(.95 * len(vals)) - 1]  # conservative n=6 nearest rank
        summary['median_' + metric] = med
        summary['p95_' + metric] = p95
    blocks = []
    for start in (0, 4, 8):
        block = runs[start:start + 4]
        vals = {}
        for metric in ('prefill_ms', 'ttft_ms', 'wall_s'):
            a = statistics.median(r[metric] for r in block if r['mode'] == 'selective-supply')
            b = statistics.median(r[metric] for r in block if r['mode'] == 'early-stage')
            vals[metric + '_latency_reduction_pct'] = 100.0 * (1.0 - b / a)
        blocks.append(vals)
    summary['four_run_blocks'] = blocks
    a_wall = summary['median_wall_s']['selective-supply']
    b_wall = summary['median_wall_s']['early-stage']
    a_decode = summary['median_decode_toks_per_s']['selective-supply']
    b_decode = summary['median_decode_toks_per_s']['early-stage']
    summary['median_full_process_latency_reduction_pct'] = 100.0 * (1.0 - b_wall / a_wall)
    summary['median_decode_rate_change_pct'] = 100.0 * (b_decode / a_decode - 1.0)
    summary['all_blocks_full_process_5pct'] = all(
        b['wall_s_latency_reduction_pct'] >= 5.0 for b in blocks)
    summary['decode_regression_over_5pct'] = b_decode < 0.95 * a_decode
    summary['full_request_performance_qualified'] = False
    summary['status'] = ('repeat_screen_positive_unqualified'
                         if summary['all_blocks_full_process_5pct']
                         and not summary['decode_regression_over_5pct']
                         else 'repeat_screen_no_material_gain_or_decode_regression')
    return True

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'binary', 'prepared', 'state', 'report'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--supply-profile', action='store_true',
                   help='profile repeatable forced-staged and transition-fallback off/on controls')
    p.add_argument('--selective-abba', action='store_true',
                   help='same-stream-all ABBA of default-off routing-aware DMA skipping')
    p.add_argument('--early-stage-abba', action='store_true',
                   help='ABBA of MMQ selective H2D vs skipped CPU staging')
    p.add_argument('--early-stage-matched-abba', action='store_true',
                   help='same ABBA with published-route H2D (diagnostic only)')
    p.add_argument('--early-stage-matched-repeat', action='store_true',
                   help='two cache warmups plus 3 balanced ABBA blocks, matched H2D, 128 decode tokens')
    a = p.parse_args()
    repeat_mode = a.early_stage_matched_repeat
    early_abba = a.early_stage_abba or a.early_stage_matched_abba or repeat_mode
    if sum(map(bool, (a.supply_profile, a.selective_abba, a.early_stage_abba,
                      a.early_stage_matched_abba, repeat_mode))) > 1:
        p.error('supply modes are mutually exclusive')
    order = REPEAT_MEASURED_ORDER if repeat_mode else EARLY_STAGE_ORDER if early_abba else SELECTIVE_ORDER if a.selective_abba else ORDER
    plan = supply_plan() if a.supply_profile else [
        (m, False) for m in ((REPEAT_WARMUP_ORDER + order) if repeat_mode else order)]
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
           '--tokens-file', str(prompt), '--max-context', '4096', '--max-new', '128' if repeat_mode else '16', '--greedy',
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
    if early_abba:
        summary.update(schema=2,
                       scope='MMQ-only first screen: H2D-only vs route-aware CPU staging skip, canonical weights',
                       order=list(order), actual_qpn_executed=False,
                       full_request_performance_qualified=False)
        summary['limitations'] += [
            'four fresh processes with uncontrolled page cache/clocks: no warm session',
            'single 2047-token prompt, no longer-context or accepted-token quality gate',
            'CPU staging omission does not prove exposed critical-path time saved',
            'no CUDA Graph, Compute Sanitizer or broad model-quality qualification']
        if a.early_stage_matched_abba or repeat_mode:
            summary['scope'] = ('MMQ-only diagnostic: both arms wait for current-layer '
                                'route publication before H2D, canonical weights')
            summary['limitations'] += [
                'forced publication wait removes speculative H2D lookahead in both arms',
                'matched-H2D diagnostic cannot establish speedup of original asynchronous policy']
    if repeat_mode:
        summary.update(scope='MMQ-only 2 cache warmups + 3 balanced matched-H2D blocks; fresh processes',
                       order=list(order), warmup_order=list(REPEAT_WARMUP_ORDER), warmups=[],
                       full_request_performance_qualified=False)
        summary['limitations'] += [
            'six timed fresh processes per arm after two process warmups; not warm-engine sessions',
            'single 2047-token prompt; no independent held-out or longer-context dataset',
            '128 requested generated tokens, but no accepted-draft/target-pass counters',
            'three blocks, no independent confidence interval; cannot qualify production']
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
            trial.update(selective_flags(mode) if (a.selective_abba or early_abba) else
                         supply_mode_flags(mode) if a.supply_profile else mode_flags(mode))
            trial['STRATA_PREFILL_STREAM_MIN'] = '4096' if mode == 'forced-staged' else '1024'
            trial['STRATA_QPN_SUPPLY_TRACE'] = str(int(trace))
            if a.early_stage_matched_abba or repeat_mode:
                trial['STRATA_QPN_MMQ_MATCHED_H2D'] = '1'
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
            (summary['warmups'] if repeat_mode and i < len(REPEAT_WARMUP_ORDER)
             else summary['runs']).append(row)
            save()
            if code:
                row['startup_diagnostics'] = safe_failure_diagnostics(log)
                if early_abba:
                    # Fixed source-controlled phrases only: no raw log lines,
                    # layer/expert IDs, paths, logits or other private content.
                    private_text = log.read_text(errors='replace')
                    row['early_stage_failure_flags'] = {
                        'unsupported_phase': ('experimental early MMQ staging requires selective-supply=1'
                                              in private_text),
                        'stage_copy_counter_mismatch': ('experimental early MMQ stage/copy omission mismatch'
                                                        in private_text),
                        'source_copy_failed': ('the expert source could not copy experts' in private_text),
                    }
                summary['status'] = 'failed_execution'
                save()
                return 2
            raw = log.read_text(errors='replace')
            if repeat_mode:
                dm = re.findall(r'decode\s+(\d+)\s+tokens in', raw)
                row['decode_tokens'] = int(dm[-1]) if dm else 0
            if re.search(r'non[- ]finite|illegal memory access|CUDA error|device-side assert', raw, re.I):
                summary['status'] = 'failed_safety'
                save()
                return 2
            if mode in ('transition-fallback', 'selective-supply', 'early-stage'):
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
            if a.selective_abba or early_abba:
                try:
                    row['selective_supply'] = selective_accounting(raw, mode)
                    if early_abba:
                        row['early_stage'] = early_stage_accounting(
                            raw, mode, row['selective_supply'])
                except ValueError:
                    summary['status'] = 'failed_selective_supply_accounting'
                    save()
                    return 2
                if early_abba and mode == 'early-stage' and not row['early_stage']['cpu_jobs_omitted']:
                    # Opportunity absent at the first candidate; do not
                    # spend a further two model starts on a zero-copy result.
                    summary['status'] = 'stopped_zero_cpu_jobs_omitted'
                    save()
                    return 0
            if not (repeat_mode and i < len(REPEAT_WARMUP_ORDER)):
                outputs.append(v)
            save()
            print(f'MMQ matched repeat {i+1}/{len(plan)} {mode} completed', flush=True)
        if a.selective_abba or early_abba:
            counters = [r['selective_supply'] for r in summary['runs']]
            total_bytes = {r['bytes'] + r['skipped_bytes'] for r in counters}
            if len({r['entries'] for r in counters}) != 1 or len(total_bytes) != 1:
                summary['status'] = 'failed_selective_unmatched_supply'
                save()
                return 2
        ok = (evaluate_supply(summary, outputs) if a.supply_profile else
              evaluate_early_stage_repeat(summary, outputs) if repeat_mode else
              evaluate_early_stage(summary, outputs) if early_abba else
              evaluate(summary, outputs, order))
        save()
        return 0 if ok else 3


if __name__ == '__main__':
    raise SystemExit(main())
