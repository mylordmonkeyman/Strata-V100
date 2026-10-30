#!/usr/bin/env python3
"""Host-only conditional R1 critical-path sensitivity screen.

Input can contain a PRIVATE per-operation causal DAG: keep it local. Only a
fixed scalar allowlist is emitted; no prompts, logits, expert identities,
route records, tensors, or node IDs are written to the report. A complete
causal graph does not yet exist in the current Strata profiling artifacts.

This tool NEVER qualifies a QPN prototype; it exposes an idealized mathematical
hurdle from a supplied native-MMQ DAG, when independently supported.
"""
import argparse
import json
import math
from pathlib import Path

MODEL_SHA = '1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5'
CONTROLS = {'STRATA_MMQ_KQ': '1', 'STRATA_PREFILL_MMQ': '1',
            'STRATA_ARENA_LOCK': '0', 'STRATA_ARENA_PIN_GIB': '1'}
PHASES = frozenset(('prefill', 'verification', 'accepted_decode', 'other'))
KINDS = frozenset(('expert_compute', 'transfer', 'gather', 'cpu_stage',
                   'other_compute', 'other'))
EVIDENCE_FLAGS = ('complete_full_request_work', 'all_cpu_gpu_work_and_wait_edges',
                  'exact_static_group_eligibility', 'trace_off_parity_and_time_control',
                  'phase_classification_validated', 'observational_timing_calibrated',
                  'representative_heldout')
MAX_NODES = 200000
MAX_EDGES = 2000000


def finite_number(v, label, positive=False):
    if type(v) not in (int, float) or not math.isfinite(v) or v < 0 or (positive and v == 0):
        raise ValueError('invalid numeric ' + label)
    return float(v)


def longest_path(nodes, edges, factor):
    """Max completion time on fixed resource-order/causal DAG, O(V+E)."""
    n = len(nodes)
    successors = [[] for _ in range(n)]
    incoming = [0] * n
    for a, b in edges:
        successors[a].append(b)
        incoming[b] += 1
    durations = [x['duration_ms'] / factor if x['eligible'] else x['duration_ms']
                 for x in nodes] if factor is not None else [
                     0.0 if x['eligible'] else x['duration_ms'] for x in nodes]
    completion = [0.] * n
    from collections import deque
    ready = deque(i for i, count in enumerate(incoming) if count == 0)
    done = 0
    while ready:
        i = ready.popleft()
        completion[i] += durations[i]
        done += 1
        for j in successors[i]:
            completion[j] = max(completion[j], completion[i])
            incoming[j] -= 1
            if incoming[j] == 0:
                ready.append(j)
    if done != n:
        raise ValueError('causal DAG contains cycle')
    return max(completion, default=0.)


def screen(data, speed_factor=1.377, max_unattributed_fraction=.02):
    """Compute sensitivity, NEVER actual performance qualification.

    All evidence flags are assertions supplied by the local measurement tool,
    not independent proof. The no-eligibility result is scoped to that input.
    """
    finite_number(speed_factor, 'speed factor', positive=True)
    if speed_factor <= 1:
        raise ValueError('speed factor must exceed one')
    finite_number(max_unattributed_fraction, 'unattributed fraction')
    if max_unattributed_fraction > 1:
        raise ValueError('unattributed fraction must not exceed one')
    if type(data) is not dict or data.get('schema') != 1 or data.get('model_sha256') != MODEL_SHA:
        raise ValueError('unsupported private graph schema or model')
    if type(data.get('control_flags')) is not dict or any(
            data['control_flags'].get(k) != v for k, v in CONTROLS.items()):
        raise ValueError('unmatched MMQ controls')
    if data['control_flags'].get('STRATA_QPN_PREFILL_EXPERIMENTAL') != '0':
        raise ValueError('QPN execution must be off')
    wall = finite_number(data.get('full_request_latency_ms'), 'wall time', positive=True)
    flags = data.get('evidence_flags')
    if type(flags) is not dict or any(type(flags.get(k)) is not bool for k in EVIDENCE_FLAGS):
        raise ValueError('evidence completeness flags must be boolean')
    raw_nodes = data.get('nodes')
    raw_edges = data.get('edges')
    if type(raw_nodes) is not list or not 0 < len(raw_nodes) <= MAX_NODES:
        raise ValueError('missing or excessive private DAG nodes')
    if type(raw_edges) is not list or len(raw_edges) > MAX_EDGES:
        raise ValueError('invalid private DAG edges')
    nodes = []
    lane_order = {}
    lane_next = {}
    for i, n in enumerate(raw_nodes):
        if type(n) is not dict or set(n) != {'id', 'lane', 'order', 'phase', 'kind', 'duration_ms', 'eligible'}:
            raise ValueError('invalid private DAG node fields')
        if type(n['id']) is not int or n['id'] != i:
            raise ValueError('node IDs must be consecutive local indices')
        if type(n['lane']) is not int or not 0 <= n['lane'] < 64:
            raise ValueError('invalid resource lane')
        if type(n['order']) is not int or n['order'] < 0:
            raise ValueError('invalid lane order')
        if n['phase'] not in PHASES or n['kind'] not in KINDS or type(n['eligible']) is not bool:
            raise ValueError('invalid phase/kind/eligibility')
        if n['eligible'] and (n['kind'] != 'expert_compute' or n['phase'] != 'prefill'):
            raise ValueError('only exact prefill expert work may be marked eligible')
        duration = finite_number(n['duration_ms'], 'node duration')
        idx = (n['lane'], n['order'])
        if idx in lane_order:
            raise ValueError('duplicate lane/order')
        lane_order[idx] = i
        lane_next[n['lane']] = max(n['order'], lane_next.get(n['lane'], -1))
        nodes.append({'duration_ms':duration, 'eligible':n['eligible']})
    if any(any((lane, rank) not in lane_order for rank in range(highest+1))
           for lane, highest in lane_next.items()):
        raise ValueError('resource lane has missing sequence order')
    # Resource serialization is an explicit model requirement, not a guess from
    # event timestamps: for each lane its ordered operations cannot overlap.
    edges = set()
    for lane, highest in lane_next.items():
        edges.update((lane_order[(lane, i)], lane_order[(lane, i+1)])
                     for i in range(highest))
    for item in raw_edges:
        if type(item) is not list or len(item) != 2 or any(type(v) is not int for v in item):
            raise ValueError('invalid cross-resource edge')
        a, b = item
        if not 0 <= a < len(nodes) or not 0 <= b < len(nodes) or a == b:
            raise ValueError('invalid edge endpoints')
        edges.add((a,b))
    if len(edges) > MAX_EDGES:
        raise ValueError('excessive causal graph edges')
    base_cp = longest_path(nodes, edges, 1.)
    zero_cp = longest_path(nodes, edges, None)
    scenario_cp = longest_path(nodes, edges, speed_factor)
    eligible_n = sum(x['eligible'] for x in nodes)
    base_gap = abs(base_cp-wall)/wall
    unverified = [k for k in EVIDENCE_FLAGS if flags[k] is not True]
    calibrated = base_gap <= max_unattributed_fraction
    potential = max(0., base_cp-zero_cp)
    scenario = max(0., base_cp-scenario_cp)
    # No node ID, lane/order, inter-operation interval or route metadata output.
    output = {
        'schema': 1,
        'scope': 'host-only fixed-DAG sensitivity; no actual QPN runtime or latency qualification',
        'r1_feasibility': 'UNKNOWN',
        'production_qualified': False,
        'eligible_work_exposed_time_measured': False,
        'graph_sensitivity_validated': not unverified and calibrated,
        'missing_evidence_assertions': unverified,
        'observed_graph_vs_request_gap_fraction': base_gap,
        'graph_vs_request_tolerance_fraction': max_unattributed_fraction,
        'node_count': len(nodes),
        'resource_lane_count': len(lane_next),
        'explicit_causal_edge_count': len(raw_edges),
        'total_modeled_edge_count': len(edges),
        'eligible_prefill_operation_count': eligible_n,
        'full_request_latency_ms': wall,
        'modeled_baseline_critical_path_ms': base_cp,
        'modeled_zero_eligible_critical_path_ms': zero_cp,
        'conditional_max_latency_reduction_fraction': potential / wall,
        'assumed_fixture_only_speed_factor': speed_factor,
        'modeled_assumed_factor_critical_path_ms': scenario_cp,
        'conditional_assumed_factor_latency_reduction_fraction': scenario / wall,
        'qualifies_actual_qpn_kernel_speed': False,
        'notes': [
            'DAG intervals must represent actual nonoverlapping work; waits are causal edges, not double-counted durations.',
            'Complete resource ordering, host submission gaps, dependent waits, and real model grouping must be independently validated.',
            'Current three-layer supply trace and route-share aggregates do not satisfy this contract.',
            'Zero-time conditional sensitivity is an idealized fixed-DAG ceiling, not a measured QPN speedup.',
            'No route, expert or node identity is included in the published report.',
            'Do not interpret graph_screen as an R1 or R3 pass, or as evidence for a packed-cache prototype.'
        ]
    }
    if unverified or not calibrated:
        output['graph_screen'] = 'UNKNOWN_INCOMPLETE_CAUSAL_EVIDENCE'
    elif potential < 0.05*wall:
        output['graph_screen'] = 'CONDITIONAL_FIXED_DAG_BELOW_FIVE_PERCENT'
    else:
        output['graph_screen'] = 'CONDITIONAL_FIXED_DAG_POSSIBLE'
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dag', type=Path, required=True,
                        help='owner-only local timing DAG; NEVER upload')
    parser.add_argument('--aggregate-report', type=Path, required=True,
                        help='aggregate-only output file')
    parser.add_argument('--assumed-speed-factor', type=float, default=1.377)
    args = parser.parse_args()
    try:
        data = json.loads(args.private_dag.read_text())
        result = screen(data, speed_factor=args.assumed_speed_factor)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        parser.error('invalid local causal evidence (private contents suppressed)')
    args.aggregate_report.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_report.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print('Conditional graph screen:', result['graph_screen'], '(R1 UNKNOWN)')

if __name__ == '__main__':
    main()