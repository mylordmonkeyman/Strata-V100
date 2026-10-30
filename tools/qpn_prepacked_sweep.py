#!/usr/bin/env python3
"""Prepacked real-weight/activation row sweep. Not a full-model speed result."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

BUCKETS = ((1,8),(9,16),(17,32),(33,64),(65,128),(129,256),(257,512),
           (1,32),(17,128),(33,512))
REGIMES = ('resident', 'serial-h2d', 'overlap-h2d')


def cases(counts):
    if len(counts) != 16 or any(type(n) is not int or not 1 <= n <= 1024 for n in counts):
        raise ValueError('invalid real capture counts')
    result = []
    for cap, bounds in [(0,b) for b in BUCKETS] + [(c,(1,c)) for c in (1,2,4,8,16)]:
        actual = [min(n,cap) if cap else n for n in counts]
        selected = sum(bounds[0] <= n <= bounds[1] for n in actual)
        if not selected:
            continue
        for regime in REGIMES:
            result.append(dict(name=f'{"decode-cap"+str(cap) if cap else "prefill"}-{bounds[0]}-{bounds[1]}-{regime}',
                               row_cap=cap, low=bounds[0], high=bounds[1], regime=regime,
                               expected_groups=selected, expected_rows=sum(actual),
                               scope='decode-shape proxy' if cap else 'original routed prefill fixture'))
    return result


def command(binary, fixture, cache, case, rounds=6, iterations=10):
    args = [str(binary), '--weights', str(fixture/'weights.bin'), '--activations', str(fixture/'activations.f32'),
            '--counts', str(fixture/'counts.txt'), '--qpn-packed-file', str(cache),
            '--qpn-fused-gu', '--qpn-chunkmajor', '--qpn-range', f'{case["low"]}:{case["high"]}',
            '--row-cap', str(case['row_cap']), '--rounds', str(rounds), '--iterations', str(iterations), '--graph-check']
    if case['regime'] != 'resident':
        args.append('--qpn-pinned-stream')
    if case['regime'] == 'overlap-h2d':
        args.append('--qpn-async-pipeline')
    return args


def validate(j, case, iterations=10, rounds=6):
    for flag in ('fused_gu','chunkmajor','disk_prepacked','explicit_range','numerics_pass',
                 'graph_checked','graph_replay_equal','graph_changed_input_equal','output_tail_guard_ok'):
        if j.get(flag) is not True:
            raise ValueError('missing numerical/layout/graph gate')
    if (j['qpn_groups'] != case['expected_groups'] or j['fallback_groups'] + j['qpn_groups'] != 16
            or j['routed_rows'] != case['expected_rows'] or j['row_cap'] != case['row_cap']
            or j['qpn_min'] != case['low'] or j['qpn_max'] != case['high']
            or j['iterations'] != iterations or j['rounds'] != rounds
            or j['repack_cpu_us'] != 0 or j['oracle_rows'] < 16 or j['original_routed_rows'] != 1100):
        raise ValueError('wrong selection, work, iterations or offline cache')
    for key in ('mixed_vs_mmq_output_l2','mmq_reference_l2','mixed_reference_l2'):
        if not math.isfinite(j[key]):
            raise ValueError('nonfinite numerical result')
    streamed = case['regime'] != 'resident'
    if j['pinned_stream'] != streamed or j['async_pipeline'] != (case['regime'] == 'overlap-h2d'):
        raise ValueError('wrong supply regime')
    if streamed:
        if j['control_h2d_bytes'] != j['hybrid_h2d_bytes'] or j['control_h2d_bytes'] != 44237056:
            raise ValueError('unequal transfer bytes')
    elif j['control_h2d_bytes'] or j['hybrid_h2d_bytes']:
        raise ValueError('resident case includes transfers')
    if case['regime'] == 'overlap-h2d' and j['pipeline_output_equal'] is not True:
        raise ValueError('pipeline output mismatch')
    timing_keys=['all_mmq_us','mixed_qpn_mmq_us','all_mmq_graph_us','mixed_graph_us']
    if case['regime']=='overlap-h2d': timing_keys += ['pipeline_all_mmq_us','pipeline_mixed_us']
    for key in timing_keys:
        if len(j[key]) != rounds or any(not math.isfinite(v) or v <= 0 for v in j[key]):
            raise ValueError('invalid timing samples')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','fixture','cache','private','report'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.report.mkdir(parents=True,exist_ok=True)
    counts=list(map(int,(a.fixture/'counts.txt').read_text().strip().split(',')))
    plan=cases(counts)
    report=dict(schema=1,scope='prepacked row-group sweep; decode cases are capped real-row proxies, not model decode',
                full_model_prefill_tested=False,full_model_decode_tested=False,
                binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),cases=[],sanitizers=[],status='pending')
    def save():
        (a.report/'sweep.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    for case in plan:
        run=subprocess.run(['numactl','--cpunodebind=0','--membind=0',*command(a.binary,a.fixture,a.cache,case)],
                           capture_output=True,timeout=600)
        row=dict(case=case,returncode=run.returncode)
        report['cases'].append(row)
        if run.returncode:
            row['diagnostic_sha256']=hashlib.sha256(run.stderr).hexdigest()
            report['status']='failed_execution';save();return 2
        try:
            j=json.loads(run.stdout);validate(j,case)
        except (ValueError,KeyError,TypeError):
            report['status']='failed_measurement_gate';save();return 2
        row['measurement']=j;save()
        print('Prepacked sweep completed '+case['name'],flush=True)
    # Exercise small and large selected groups plus two-slot reuse. Three iterations
    # are mandatory; raw sanitizer output stays in the private temporary directory.
    picks=[]
    for name in ('decode-cap1-1-1-overlap-h2d','prefill-17-32-overlap-h2d','prefill-129-256-overlap-h2d'):
        picks += [c for c in plan if c['name']==name]
    if len(picks)!=3:
        report['status']='failed_sanitizer_case_coverage';save();return 2
    for case in picks:
        for tool in ('memcheck','racecheck'):
            log=a.private/(case['name']+'-'+tool+'.log')
            cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','98','--log-file',str(log)]
            if tool=='memcheck': cmd+=['--leak-check','no']
            run=subprocess.run(['numactl','--cpunodebind=0','--membind=0',*cmd,
                                *command(a.binary,a.fixture,a.cache,case,rounds=1,iterations=3)],
                               capture_output=True,timeout=1200)
            text=log.read_text(errors='replace') if log.is_file() else ''
            expected='ERROR SUMMARY: 0 errors' if tool=='memcheck' else 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
            good=run.returncode==0 and expected in text
            if good:
                try: validate(json.loads(run.stdout),case,iterations=3,rounds=1)
                except (ValueError,KeyError,TypeError): good=False
            report['sanitizers'].append(dict(case=case['name'],tool=tool,passed=good,iterations=3))
            if not good:
                report['status']='failed_sanitizer';save();return 2
            save()
    report['status']='row_sweep_qualified_full_model_pending';save();return 0

if __name__=='__main__':
    raise SystemExit(main())
