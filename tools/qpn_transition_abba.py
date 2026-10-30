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
from qpn_supply_profile import transition_bypass_count

MODEL_SHA = '1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5'
PROMPT_SHA = 'df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414'
ORDER = ('both-off', 'transition-fallback', 'transition-fallback', 'both-off')


def mode_flags(mode):
    if mode not in ORDER:
        raise ValueError('invalid ABBA mode')
    return dict(STRATA_QPN_DIAG_GG_DISABLE_STREAM='1' if mode == 'both-off' else '0',
                STRATA_QPN_DIAG_GG_DISABLE_STAGED='1' if mode == 'both-off' else '0',
                STRATA_QPN_DIAG_GG_TRANSITION_FALLBACK='1' if mode == 'transition-fallback' else '0')


def compare(a, b, ref, candidate):
    d = {k: v for k, v in diff(a, b).items() if k not in ('argmax_base', 'argmax_candidate')}
    d.update(reference=ref, candidate=candidate, bitwise_equal=(a == b))
    return d


def evaluate(summary, outputs):
    if len(outputs) != 4 or len(summary['runs']) != 4:
        raise ValueError('incomplete ABBA')
    summary['comparisons'] = [compare(outputs[i], outputs[j], ORDER[i], ORDER[j])
                              for i, j in ((0, 3), (1, 2), (0, 1), (0, 2), (3, 1), (3, 2))]
    if not all(d['bitwise_equal'] and d['relative_l2'] == 0 and d['kl_base_vs_candidate'] == 0
               and d['argmax_match'] for d in summary['comparisons']):
        summary['status'] = 'failed_mmq_abba_bitwise_parity'
        return False
    for field in ('prefill_ms', 'ttft_ms'):
        medians = {mode: statistics.median(r[field] for r in summary['runs'] if r['mode'] == mode)
                   for mode in ('both-off', 'transition-fallback')}
        if any(not math.isfinite(x) or x <= 0 for x in medians.values()):
            summary['status'] = 'failed_timing'
            return False
        summary['median_' + field] = medians
        summary['median_' + field.removesuffix('_ms') + '_speedup'] = (
            medians['both-off'] / medians['transition-fallback'])
    summary['status'] = 'mmq_abba_bitwise_parity_passed_quality_unqualified'
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'binary', 'prepared', 'state', 'report'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
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
    report = a.report / 'summary.json'
    def save():
        report.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    save()
    outputs = []
    with tempfile.TemporaryDirectory(prefix='qpn-transition-abba-', dir=a.state) as private_dir:
        os.chmod(private_dir, 0o700)
        private = Path(private_dir)
        for i, mode in enumerate(ORDER):
            trial = env.copy()
            trial.update(mode_flags(mode))
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
            row = dict(index=i, mode=mode, returncode=code, wall_s=time.monotonic()-start)
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
            if mode == 'transition-fallback':
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
            outputs.append(v)
            save()
            print(f'MMQ ABBA {i+1}/4 {mode} completed', flush=True)
        ok = evaluate(summary, outputs)
        save()
        return 0 if ok else 3


if __name__ == '__main__':
    raise SystemExit(main())
