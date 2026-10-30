#!/usr/bin/env python3
"""Rev4.2 R1: private train/held-out route feasibility; emit aggregates ONLY.

Private JSON schema: {"schema":1,"prefill":[{"layers":[{"layer":0,
"counts":[[expert_id, routed_rows],...]}]}],"decode":[...]}.
Each prefill entry is one prompt chunk and each decode entry is one verification
window. IDs, individual layer statistics and prompt text are never exported.
This is an optimistic host bound, not an actual cache allocator or QPN dispatch.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from qpn_resident_cache_feasibility import PACKED_BLOB, LAYERS, EXPERTS_PER_LAYER, GROUP, MIB, read_ranked_profile


def private_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_private_sample(path):
    obj = json.loads(Path(path).read_text())
    if obj.get('schema') != 1 or not isinstance(obj.get('prefill'), list) or not obj['prefill'] or not isinstance(obj.get('decode'), list):
        raise ValueError('invalid private sample schema')
    output = {'prefill': [], 'decode': []}
    for phase in output:
        for chunk in obj[phase]:
            if not isinstance(chunk, dict) or not isinstance(chunk.get('layers'), list) or not chunk['layers']:
                raise ValueError('invalid chunk')
            layer_map = {}
            for item in chunk['layers']:
                l = item.get('layer')
                counts = item.get('counts')
                if type(l) is not int or not 0 <= l < LAYERS or l in layer_map or not isinstance(counts, list):
                    raise ValueError('invalid layer or duplicate')
                seen = {}
                for pair in counts:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ValueError('invalid expert row tuple')
                    e, rows = pair
                    if type(e) is not int or type(rows) is not int or not 0 <= e < EXPERTS_PER_LAYER or not 1 <= rows <= 32768 or e in seen:
                        raise ValueError('invalid expert rows or duplicate')
                    seen[e] = rows
                layer_map[l] = seen
            output[phase].append(layer_map)
    return output


def build_groups(train, resident, low=33, high=512, group=GROUP):
    # Predictions are trained on DIFFERENT prompts from those evaluated.
    train_score = Counter()
    for chunk in train['prefill']:
        for l, counts in chunk.items():
            for e, rows in counts.items():
                if low <= rows <= high and (l,e) in resident:
                    train_score[(l,e)] += rows
    groups = []
    for l in range(LAYERS):
        pairs = sorted(((train_score[(ll,e)],(ll,e)) for ll,e in resident if ll == l and train_score[(ll,e)] > 0),
                       key=lambda z:(-z[0],z[1]))
        for at in range(0,len(pairs)-group+1,group):
            chosen = [x[1] for x in pairs[at:at+group]]
            groups.append((sum(x[0] for x in pairs[at:at+group]),l,tuple(chosen)))
    return sorted(groups,key=lambda x:(-x[0],x[1],x[2]))


def score_holdout(test, selected, resident, low=33, high=512, group=GROUP):
    selected = set(selected)
    totals = Counter()
    for chunk in test['prefill']:
        for l, counts in chunk.items():
            totals['prefill_routed_entries'] += sum(counts.values())
            valid_packed = [(rows,e) for e,rows in counts.items() if (l,e) in selected and low <= rows <= high]
            valid_packed.sort(reverse=True)
            # Optimistic group-repacking upper bound, NOT an actual grouping plan.
            n = len(valid_packed)//group
            totals['upper_bound_complete_qpn_groups'] += n
            totals['upper_bound_qpn_entries'] += sum(x[0] for x in valid_packed[:n*group])
            # Per-chunk canonical fallback: if a packed expert is routed but
            # isn't part of a complete QPN group, it needs canonical bytes.
            grouped = {e for _,e in valid_packed[:n*group]}
            totals['prefill_packed_canonical_fallback_expert_uses'] += sum(
                (l,e) in selected and e not in grouped for e in counts)
            totals['canonical_nonresident_routed_entries'] += sum(
                rows for e,rows in counts.items() if (l,e) not in resident)
    per_request_decode = set()
    for chunk in test['decode']:
        for l,counts in chunk.items():
            totals['decode_routed_entries'] += sum(counts.values())
            per_request_decode.update((l,e) for e in counts if (l,e) in selected)
    totals['decode_distinct_packed_experts_to_restore'] = len(per_request_decode)
    totals['optimistic_prefill_restore_bytes'] = totals['prefill_packed_canonical_fallback_expert_uses']*PACKED_BLOB
    totals['optimistic_decode_restore_bytes'] = len(per_request_decode)*PACKED_BLOB
    total = totals['prefill_routed_entries']
    return dict(totals, optimistic_qpn_entry_share=(totals['upper_bound_qpn_entries']/total if total else None))


def evaluate(train,test,ranking,capacities_mib, native_budget_gib=24,low=33,high=512):
    if not 1 <= low <= high <= 512 or not 1 <= native_budget_gib <= 32 or any(type(x) is not int or x<0 or x>3072 for x in capacities_mib):
        raise ValueError('unsupported budgets/rows')
    native_slots = min(len(ranking), (native_budget_gib*(1<<30))//PACKED_BLOB)
    resident = set(ranking[:native_slots])
    candidates = build_groups(train,resident,low,high)
    results = []
    for mib in capacities_mib:
        n_groups = min(len(candidates), (mib*MIB//PACKED_BLOB)//GROUP)
        selected = {p for _,_,group in candidates[:n_groups] for p in group}
        scores = score_holdout(test,selected,resident,low,high)
        results.append(dict(packed_budget_mib=mib, selected_packed_experts=len(selected),
                            predicted_packed_groups=n_groups, resident_canonical_ids_retained=native_slots,
                            canonical_ids_evicted=0, **scores))
    return dict(native_slots=native_slots, observed_training_candidate_groups=len(candidates),
                training_prefill_chunks=len(train['prefill']),
                heldout_prefill_chunks=len(test['prefill']),heldout_decode_windows=len(test['decode']),
                scenarios=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('private-training','private-heldout','profile','report'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--budget-mib',nargs='+',type=int,default=[128,256,512,737,1024,2048])
    p.add_argument('--native-budget-gib',type=int,default=24)
    p.add_argument('--min-rows',type=int,default=33)
    p.add_argument('--max-rows',type=int,default=512)
    a=p.parse_args()
    if a.private_training.resolve()==a.private_heldout.resolve():
        p.error('training and heldout traces must be different files')
    if private_sha(a.private_training) == private_sha(a.private_heldout):
        p.error('identical training and heldout route inputs are not an independent evaluation')
    try:
        ranking, profile_sha = read_ranked_profile(a.profile)
        train=read_private_sample(a.private_training)
        heldout=read_private_sample(a.private_heldout)
        result=evaluate(train,heldout,ranking,a.budget_mib,a.native_budget_gib,a.min_rows,a.max_rows)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
        p.error('invalid private sample, profile, or R1 geometry (contents not printed)')
    result.update(schema=1,scope='heldout optimistic single-representation residency R1; not GPU measurement',
        model_quality_qualified=False,performance_qualified=False,
        selected_rows=[a.min_rows,a.max_rows],
        profile_sha256=profile_sha,training_sha256=private_sha(a.private_training),
        heldout_sha256=private_sha(a.private_heldout),
        assumptions=['no extra VRAM, original canonical resident expert identities kept, packed subset is a replacement',
                     'selection trained on separate prompts; all 16 packed experts must be jointly eligible',
                     'group counts optimistically assume eligible packed experts can be regrouped',
                     'decode restores distinct packed experts once per request, prefill restorations counted per chunk',
                     'source load, transition latency, graph generation, cache-hit dynamics and exposed compute not measured',
                     'routed-entry share is not a prefill-time share or performance gain'])
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('Aggregate-only heldout R1 produced; no raw IDs or samples exported')

if __name__=='__main__':
    main()

