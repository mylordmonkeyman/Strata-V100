#!/usr/bin/env python3
"""Host-only dual-format QPN residency capacity gate. Never exports route IDs.

This is a memory/coverage bound, NOT a speedup prediction. Canonical
resident weights must remain available for the existing native decode path.
Any private routing trace stays local; only aggregate scalar results are emitted.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

GIB = 1 << 30
MIB = 1 << 20
EXPERTS = 24576
LAYERS = 48
EXPERTS_PER_LAYER = 512
GROUP = 16
PACKED_BLOB = 2764816  # Verified 67,948,118,016-byte / 24,576-expert offline cache


def capacity_sweep(native_budget_bytes, extra_bytes, blob_bytes=PACKED_BLOB,
                   layers=LAYERS, group=GROUP, trace_counts=None, ranking=None):
    if min(native_budget_bytes, blob_bytes, layers, group) <= 0 or extra_bytes < 0:
        raise ValueError('invalid nonnegative capacity/geometry')
    if group > EXPERTS_PER_LAYER or layers > LAYERS:
        raise ValueError('unsupported group/layer geometry')
    base = min(native_budget_bytes // blob_bytes, layers * EXPERTS_PER_LAYER)
    desired = extra_bytes // blob_bytes
    if trace_counts is not None:
        if not ranking or len(ranking) != layers * EXPERTS_PER_LAYER or len(set(ranking)) != len(ranking):
            raise ValueError('complete valid profile ranking is required with routing trace')
        if any(not (0 <= l < layers and 0 <= e < EXPERTS_PER_LAYER) for l, e in ranking):
            raise ValueError('invalid profile expert indices')
        if any(v < 0 or not (0 <= l < layers and 0 <= e < EXPERTS_PER_LAYER)
               for (l, e), v in trace_counts.items()):
            raise ValueError('invalid trace counts')
    result = []
    for policy in ('additional_vram', 'fixed_total_budget'):
        # The extra format is a DUPLICATE. A 24-GiB budget is not proof of spare
        # live VRAM. A fixed-budget duplicate necessarily displaces native slots.
        count = min(desired, base if policy == 'additional_vram' else base // 2)
        native_slots = base if policy == 'additional_vram' else base - count
        packed_slots = min(count, native_slots)
        # Optimistic at-most-one-group-per-layer ceiling, not an actual
        # allocation, eligible-row count, or measured QPN dispatch.
        layer_ceiling = min(layers, packed_slots // group)
        row = dict(policy=policy, native_slots=native_slots,
                   packed_duplicate_slots=packed_slots,
                   extra_vram_bytes=(packed_slots * blob_bytes if policy == 'additional_vram' else 0),
                   canonical_slots_displaced=base-native_slots,
                   layers_with_one_group_ceiling=layer_ceiling,
                   eligible_layer_ceiling_fraction=layer_ceiling/layers,
                   qpn_groups_executed=None, model_speedup=None)
        if trace_counts is not None:
            selected = set(ranking[:native_slots])
            # Optimistic same-trace oracle, NOT a deployable cache policy.
            best = sorted(selected, key=lambda x: (-trace_counts.get(x, 0), x))[:packed_slots]
            per_layer = Counter(l for l, e in best)
            total_routes = sum(trace_counts.values())
            base_hits = sum(trace_counts.get(x, 0) for x in set(ranking[:base]))
            current_hits = sum(trace_counts.get(x, 0) for x in selected)
            packed_hits = sum(trace_counts.get(x, 0) for x in best)
            row.update(oracle_packed_routed_entries=packed_hits,
                       oracle_packed_routing_share=packed_hits/total_routes if total_routes else None,
                       canonical_route_entries_lost=max(0, base_hits-current_hits),
                       canonical_route_share_lost=(base_hits-current_hits)/total_routes if total_routes else None,
                       actual_selected_layers_with_16_slots=sum(v >= group for v in per_layer.values()),
                       oracle_uses_same_trace_for_selection=True)
        result.append(row)
    return result


def read_ranked_profile(path, layers=LAYERS):
    data = Path(path).read_bytes()
    if len(data) < 24 or data[:4] != b'STRP':
        raise ValueError('invalid STRP profile header')
    version, nl, ne, slots, ranked = struct.unpack_from('<5I', data, 4)
    if (version, nl, ne) != (1, layers, EXPERTS_PER_LAYER) or ranked > layers * ne or slots > ranked:
        raise ValueError('unexpected STRP geometry')
    if ranked != layers * ne or len(data) != 24 + 4 * ranked + 4 * layers * ne:
        raise ValueError('incomplete/truncated/extended STRP profile')
    values = [struct.unpack_from('<HH', data, 24+4*i) for i in range(ranked)]
    if len(set(values)) != len(values) or any(l >= layers or e >= ne for l, e in values):
        raise ValueError('duplicate/invalid ranked pair')
    return values, hashlib.sha256(data).hexdigest()


def read_private_routing_trace(path, layers=LAYERS):
    # Existing tools/make_profile.py format: layer i32, k i32, k expert
    # IDs (i32), then k weights (f32). Never output IDs or raw records.
    data = Path(path).read_bytes()
    counts = Counter()
    off = records = 0
    while off < len(data):
        if off+8 > len(data):
            raise ValueError('truncated private routing record')
        layer, k = struct.unpack_from('<ii', data, off)
        off += 8
        if layer < 0 or layer >= layers or k < 1 or k > EXPERTS_PER_LAYER or off+8*k > len(data):
            raise ValueError('invalid private routing record')
        ids = struct.unpack_from('<%di' % k, data, off)
        off += 8*k
        if len(set(ids)) != len(ids) or any(e < 0 or e >= EXPERTS_PER_LAYER for e in ids):
            raise ValueError('invalid expert routing IDs')
        counts.update((layer, e) for e in ids)
        records += 1
    if not records:
        raise ValueError('empty routing trace')
    return counts, records, hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path)
    p.add_argument('--private-trace', type=Path,
                   help='PRIVATE --dump-routing file; never upload the source')
    p.add_argument('--native-budget-gib', type=int, default=24)
    p.add_argument('--extra-mib', type=int, nargs='+', default=[256,512,1024,2048,3072])
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if args.native_budget_gib < 1 or any(m < 0 for m in args.extra_mib):
        p.error('invalid budget')
    if bool(args.private_trace) != bool(args.profile):
        p.error('routing trace requires the matching profile and vice versa')
    ranking = counts = None
    provenance = dict(private_trace_used=False)
    if args.private_trace:
        ranking, profile_sha = read_ranked_profile(args.profile)
        counts, record_count, trace_sha = read_private_routing_trace(args.private_trace)
        provenance = dict(private_trace_used=True, trace_records=record_count,
                          private_trace_sha256=trace_sha, profile_sha256=profile_sha)
    output = dict(schema=1,
        scope='host-only dual-format residency memory and optimistic routing ceiling',
        performance_qualified=False, numerical_quality_qualified=False,
        canonical_decode_weights_preserved=True, packed_expert_blob_bytes=PACKED_BLOB,
        layers=LAYERS, slots_per_qpn_group=GROUP, original_expert_slots=EXPERTS,
        native_budget_bytes=args.native_budget_gib*GIB,
        minimum_packed_vram_for_one_group_per_layer_bytes=LAYERS*GROUP*PACKED_BLOB,
        **provenance,
        limitations=[
            'input extra capacity is hypothetical; live post-allocation V100 headroom was not measured',
            'group ceiling ignores routed row counts, eligibility, dynamic scheduling and kernel overhead',
            'profile ranking is not a held-out routing hit-rate distribution',
            'private-trace oracle selects on the same trace; optimistic not predictive',
            'route-entry shares are not expert-DMA counts, time saved, or speedup',
            'no cache/model changes and no GPU-resident packed QPN measured'],
        scenarios=[])
    for mib in args.extra_mib:
        output['scenarios'].append({
            'requested_packed_budget_mib':mib,
            'policies':capacity_sweep(args.native_budget_gib*GIB, mib*MIB,
                                     trace_counts=counts, ranking=ranking)})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print('theoretical minimum for one 16-slot group/layer: '
          f"{output['minimum_packed_vram_for_one_group_per_layer_bytes']/GIB:.3f} GiB")
    for row in output['scenarios']:
        print(f"{row['requested_packed_budget_mib']} MiB: "+'; '.join(
            f"{x['policy']} {x['packed_duplicate_slots']} duplicates, at most "
            f"{x['layers_with_one_group_ceiling']}/{LAYERS} layers"
            for x in row['policies']))


if __name__ == '__main__':
    main()
