#!/usr/bin/env python3
"""Sanitize opt-in R0 event-envelope diagnostics from a private original-MMQ log.

No inference of eligible exposed critical-path compute: event timing includes
instrumentation, queue gaps and possibly overlapped streams. Never upload input.
"""
import argparse
import json
import math
from pathlib import Path

PREFIX = 'strata qpn-r0: '
SCOPE = 'coalesced_layer_callback_envelopes_not_exposed_request'
COUNTS = ('event_pairs', 'observed_layers', 'product_layers', 'h2d_operations', 'h2d_bytes')
TIMES = ('trace_window_ms', 'h2d_envelope_ms', 'products_envelope_ms',
         'wait_envelope_ms', 'gather_envelope_ms', 'h2d_products_overlap_ms')
KEYS = frozenset(('schema', 'scope', 'instrumented',
                  'coalesced_layer_envelopes', 'callback_accounting_complete',
                  *COUNTS, *TIMES))


def nonnegative(value, name, integer=False):
    if integer:
        if type(value) is not int or value < 0:
            raise ValueError('invalid R0 ' + name)
    elif type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid R0 ' + name)
    return value


def inspect_record(record):
    if type(record) is not dict or set(record) != KEYS:
        raise ValueError('invalid R0 aggregate field allowlist')
    if (record['schema'] != 2 or record['scope'] != SCOPE or
            record['instrumented'] is not True or
            record['coalesced_layer_envelopes'] is not True or
            record['callback_accounting_complete'] is not True):
        raise ValueError('invalid R0 schema or instrumented scope')
    values = {k: nonnegative(record[k], k, integer=True) for k in COUNTS}
    values.update({k: nonnegative(record[k], k) for k in TIMES})
    # Each of five operation categories owns at most one pair per observed layer.
    # A coalesced pair can cover many callbacks; do not equate pairs with calls.
    if (values['product_layers'] > values['observed_layers'] or
            values['observed_layers'] > 48 or
            values['event_pairs'] > 5 * values['observed_layers'] or
            values['event_pairs'] < values['observed_layers'] or
            values['event_pairs'] < values['product_layers'] or
            (values['h2d_operations'] == 0) != (values['h2d_bytes'] == 0) or
            (values['h2d_operations'] > 0 and values['event_pairs'] == 0)):
        raise ValueError('R0 event/layer count inconsistency')
    h, p, ov, trace = (values['h2d_envelope_ms'], values['products_envelope_ms'],
                       values['h2d_products_overlap_ms'], values['trace_window_ms'])
    epsilon = 1e-3  # reported CUDA event timestamps are rounded
    if (ov > min(h, p) + epsilon or max(values[k] for k in TIMES) > trace + epsilon or
            h + p - ov > trace + epsilon):
        raise ValueError('R0 union/overlap inconsistency')
    return values


def scrub_log(text):
    records = []
    for line in text.splitlines():
        if line.startswith(PREFIX):
            records.append(inspect_record(json.loads(line[len(PREFIX):])))
    if not records:
        raise ValueError('missing opt-in R0 event-envelope records')
    # No raw routes, per-layer counts, input names, paths, timing traces, model
    # content, prompt lengths, tokens or logits can transit this allowlist.
    sums = {key: sum(rec[key] for rec in records) for key in (*COUNTS, *TIMES)}
    return {
        'schema': 2,
        'scope': 'R0 coalesced first/last layer event envelopes, not exposed critical-path time',
        'chunk_records': len(records),
        'event_pairs': sums['event_pairs'],
        'h2d_operations': sums['h2d_operations'],
        'h2d_bytes': sums['h2d_bytes'],
        'summed_per_chunk_envelopes_ms': {k: sums[k] for k in TIMES},
        'diagnostic_only': True,
        'coalesced_layer_envelopes': True,
        'full_request_latency_measured': False,
        'eligible_exposed_critical_path_ms': None,
        'qpn_candidate_executed': False,
        'r1_full_request_feasibility': 'UNKNOWN',
        'limitations': [
            'Coalesced first/last CUDA events include ALL intervening idle gaps',
            'CUDA event timing includes instrumentation and submission/queue gaps',
            'Product envelopes are not proved QPN-eligible or hardware occupancy',
            'Event overlap is NOT the amount of removable critical-path work',
            'Per-chunk envelopes cannot replace full-request wall latency',
            'Does not attribute host CPU staging, cache residency or accepted decode',
            'No QPN model execution or quality qualification',
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--private-log', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    summary = scrub_log(args.private_log.read_text(errors='replace'))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, sort_keys=True, indent=2,
                                      allow_nan=False) + '\n')
    print('R0 event envelope sanitized; full-request R1 remains UNKNOWN')


if __name__ == '__main__':
    main()
