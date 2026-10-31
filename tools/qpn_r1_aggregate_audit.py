#!/usr/bin/env python3
"""Host-only audit of two *published aggregate* R1 evidence files.

Never ingests route IDs, model outputs or private traces. Reconstructs an explicit
allowlist; unrecognized input keys are NEVER copied into the output. The different
MMQ supply and held-out runs must not be combined into a QPN performance result.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

MODEL_SHA = '1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5'
PROMPT_SHA = 'df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414'
REQUIRED_ENV = {'STRATA_MMQ_KQ': '1', 'STRATA_PREFILL_MMQ': '1',
                'STRATA_ARENA_LOCK': '0', 'STRATA_ARENA_PIN_GIB': '1'}
BUDGETS = (128, 256, 512, 737, 1024, 2048)
ORDER = ('transition-fallback', 'selective-supply', 'selective-supply', 'transition-fallback')
FIXTURE_SPEED = 1.377  # isolated resident fixture, deliberately NOT model speed
HELDOUT_SOURCE = '2e79bd5ccef53066f706cb321366a2d04084aac7'
SUPPLY_SOURCE = 'd777b85efadb5a359ebbb7ebb68e88e475f20cc9'
PACKED_BLOB = 2764816  # pinned native expert packed-body bytes
MIB = 1 << 20


def check(ok, reason):
    if not ok:
        raise ValueError('invalid safe aggregate: ' + reason)


def integer(value, reason):
    check(type(value) is int and value >= 0, reason)
    return value


def finite(value, reason):
    check(type(value) in (float, int) and math.isfinite(value) and value >= 0, reason)
    return value


def flags(control):
    check(type(control) is dict and all(control.get(k) == v for k, v in REQUIRED_ENV.items()),
          'four mandatory unchanged MMQ controls')
    check(control.get('STRATA_QPN_PREFILL_EXPERIMENTAL') == '0', 'QPN dispatch must remain off')


def digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def audit(heldout, supply):
    check(type(heldout) is dict and heldout.get('schema') == 1, 'routing schema')
    check(type(supply) is dict and supply.get('schema') == 1, 'MMQ supply schema')
    check(heldout.get('source_commit') == HELDOUT_SOURCE, 'pinned heldout source commit')
    check(supply.get('source_commit') == SUPPLY_SOURCE, 'pinned supply source commit')
    for name, obj in (('heldout', heldout), ('supply', supply)):
        check(obj.get('model_sha256') == MODEL_SHA, name + ' model SHA')
        flags(obj.get('control_flags'))
    check(supply.get('prompt_sha256') == PROMPT_SHA, 'MMQ supply prompt SHA')
    check(heldout.get('performance_qualified') is False and
          heldout.get('model_quality_qualified') is False and
          heldout.get('actual_qpn_decode_executed') is False and
          heldout.get('actual_accepted_token_decode_qualified') is False,
          'unqualified heldout experiment must remain unqualified')
    requests = heldout.get('requests')
    check(type(requests) is list and len(requests) == 2, 'two independent synthetic request records')
    prompts = []
    for i, (record, expected) in enumerate(zip(requests, (1024, 2048))):
        check(type(record) is dict and record.get('input_tokens') == expected,
              'captured token geometry')
        check(record.get('first_logit_equal') is True and record.get('verification_windows') == 16,
              'paired original MMQ capture parity/completeness')
        check(digest(record.get('prompt_sha256')), 'prompt provenance digest')
        prompts.append(record['prompt_sha256'])
    check(prompts[0] != prompts[1], 'distinct synthetic prompt hashes')
    check(heldout.get('common_observed_resident_experts') == 9293, 'stable measured residency')
    scenarios = heldout.get('scenarios')
    check(type(scenarios) is list and len(scenarios) == len(BUDGETS), 'complete budget sweep')
    coverage = []
    for expected_budget, item in zip(BUDGETS, scenarios):
        check(type(item) is dict and item.get('replacement_payload_budget_mib') == expected_budget,
              'replacement budget ordering')
        selected = integer(item.get('selected_packed_experts'), 'selected count')
        groups = integer(item.get('upper_bound_complete_qpn_groups'), 'optimistic group count')
        entries = integer(item.get('upper_bound_qpn_entries'), 'optimistic covered entries')
        total = integer(item.get('prefill_routed_entries'), 'heldout routed entries')
        share = finite(item.get('optimistic_qpn_entry_share'), 'optimistic entry share')
        check(total == 982560 and 0 <= entries <= total, 'heldout complete routed work')
        check(groups <= selected // 16 and selected % 16 == 0,
              'full 16-expert pack group bound')
        check(selected <= heldout['common_observed_resident_experts'] and
              selected * PACKED_BLOB <= expected_budget * MIB,
              'packed selection must fit its REPLACEMENT payload budget')
        check(groups != 0 or entries == 0,
              'zero eligible complete groups must cover zero routed entries')
        check(math.isclose(entries / total, share, rel_tol=0, abs_tol=1e-12),
              'share/entry arithmetic')
        prefill_native_uses = integer(
            item.get('prefill_packed_canonical_fallback_expert_uses'),
            'canonical fallback demand')
        restores = integer(item.get('verification_distinct_packed_experts_to_restore'),
                           'native verification consumers')
        check(restores <= selected, 'native consumers cannot exceed selected experts')
        coverage.append({'replacement_budget_mib': expected_budget,
                         'optimistic_entry_share_fraction': share,
                         'optimistic_complete_groups': groups,
                         'canonical_fallback_uses': prefill_native_uses,
                         'native_verification_consumers': restores,
                         'no_transition_packed_only_reference_native_screen': (
                             'NO_PACKED_WORK' if selected == 0 or groups == 0
                             else 'CONFLICTS_WITH_REFERENCE_NATIVE_CONSUMERS'
                             if prefill_native_uses > 0 or restores > 0
                             else 'NOT_REFUTED_BY_AGGREGATE_BUT_UNQUALIFIED')})
    check(supply.get('order') == list(ORDER), 'matched supply ABBA order')
    check(supply.get('performance_only') is True and
          supply.get('numerical_quality_qualified') is False and
          supply.get('status') == 'mmq_abba_bitwise_parity_passed_quality_unqualified',
          'supply experiment remains unqualified')
    runs = supply.get('runs')
    check(type(runs) is list and len(runs) == 4, 'four supply processes')
    results = {'transition-fallback': [], 'selective-supply': []}
    first_chunk = {'transition-fallback': [], 'selective-supply': []}
    for i, (record, name) in enumerate(zip(runs, ORDER)):
        check(type(record) is dict and record.get('index') == i and record.get('mode') == name,
              'run identity/order')
        check(record.get('returncode') == 0 and record.get('prefill_tokens') == 2047 and
              record.get('qpn_groups') == 0 and record.get('trace') is False,
              'full matched MMQ-only work')
        flags(record.get('control_flags'))
        results[name].append(finite(record.get('prefill_ms'), 'prefill milliseconds'))
        transfer = record.get('selective_supply')
        check(type(transfer) is dict and transfer.get('chunk') == 0 and
              transfer.get('tokens') == 1024, 'first-chunk transfer provenance')
        entries = integer(transfer.get('entries'), 'scheduled entry count')
        moved = integer(transfer.get('bytes'), 'transferred bytes')
        skipped = integer(transfer.get('skipped_bytes'), 'skipped bytes')
        check(entries == 15347 and moved + skipped == 42431631152,
              'constant first-chunk supply work')
        copies = integer(transfer.get('copies'), 'completed H2D copy count')
        skips = integer(transfer.get('skipped'), 'skipped copy count')
        jobs = integer(transfer.get('jobs'), 'staging job count')
        jobs_skipped = integer(transfer.get('skipped_jobs'), 'skipped H2D staging jobs')
        check(copies + skips == entries and moved == copies * PACKED_BLOB and
              skipped == skips * PACKED_BLOB and jobs_skipped <= jobs,
              'expert copy/byte/job consistency')
        if name == 'transition-fallback':
            check(skipped == 0, 'unchanged control H2D')
        else:
            check(skipped > 0, 'selective H2D skip exercised')
        first_chunk[name].append((moved, skipped))
    comp = supply.get('comparisons')
    check(type(comp) is list and len(comp) == 6 and all(
        type(c) is dict and c.get('bitwise_equal') is True and
        c.get('n_logits') == 248320 and c.get('relative_l2') == 0
        for c in comp), 'all six first-logit MMQ parity comparisons')
    baseline = statistics.median(results['transition-fallback'])
    selective = statistics.median(results['selective-supply'])
    medians = supply.get('median_prefill_ms')
    check(type(medians) is dict and
          all(math.isclose(medians.get(name), value, rel_tol=0, abs_tol=1e-8)
              for name, value in (('transition-fallback', baseline),
                                  ('selective-supply', selective))),
          'published median prefill times must match measured arms')
    fractional = (baseline - selective) / baseline
    skipped_fraction = statistics.mean(x[1] / 42431631152 for x in first_chunk['selective-supply'])
    eligible_fraction = (1 - 1 / FIXTURE_SPEED)
    required = .05 / eligible_fraction
    for row in coverage:
        row['conditional_time_share_to_entry_share_ratio'] = (
            required / row['optimistic_entry_share_fraction']
            if row['optimistic_entry_share_fraction'] > 0 else None)
    return {
        'schema': 1,
        'scope': 'safe aggregate-only cross-evidence compatibility audit; NOT a model performance comparison',
        'model_sha256': MODEL_SHA,
        'evidence_flags': {
            'paired_synthetic_route_capture_validated': True,
            'representative_heldout_qualified': False,
            'native_mmq_supply_abba_first_logit_parity_validated': True,
            'qpn_candidate_ever_executed_in_these_runs': False,
            'eligible_exposed_full_request_time_measured': False,
            'actual_accepted_token_decode_qualified': False,
            'qpn_model_numerical_quality_qualified': False,
        },
        'routing_scenarios': coverage,
        'supply_observation': {
            'first_chunk_selective_h2d_reduction_fraction': skipped_fraction,
            'median_prefill_control_ms': baseline,
            'median_prefill_selective_ms': selective,
            'median_prefill_reduction_fraction': fractional,
            'timing_qualified_as_reproducible_speedup': False,
        },
        'conditional_sensitivity': {
            'fixture_only_assumed_speed_factor': FIXTURE_SPEED,
            'minimum_full_request_eligible_critical_path_share_at_zero_overhead': required,
            'is_measured_model_speed_factor': False,
        },
        'static_single_representation_admission': 'STOPPED',
        'r1_full_request_feasibility': 'UNKNOWN',
        'notes': [
            'Route share is NOT time share and cannot be converted to removable work.',
            'The heldout pair is independently authored but synthetic and not representative qualification.',
            'Selective H2D counts are first-chunk only; timings span complete prefill in other processes.',
            'Native MMQ supply evidence does not establish QPN model acceleration or exposed compute.',
            'Demand proxy restore bytes do not establish measured or necessary DMA.',
            'Static packed-only with unchanged native verifier conflicts with observed canonical consumers; a new compatible one-layout reader or sequential transition would be a different, unqualified design.',
            'No same-expert dual-format VRAM residency or duplicate-copy experiment is permitted.',
            'No further QPN variant, Q4 conversion, packed-cache prototype, or GPU rerun is authorized.',
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--heldout-aggregate', required=True, type=Path)
    ap.add_argument('--supply-aggregate', required=True, type=Path)
    ap.add_argument('--report', required=True, type=Path)
    a = ap.parse_args()
    heldout_bytes = a.heldout_aggregate.read_bytes()
    supply_bytes = a.supply_aggregate.read_bytes()
    result = audit(json.loads(heldout_bytes), json.loads(supply_bytes))
    result['input_digests'] = {
        'joint_heldout_aggregate_sha256': hashlib.sha256(heldout_bytes).hexdigest(),
        'selective_supply_aggregate_sha256': hashlib.sha256(supply_bytes).hexdigest(),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print('R1 aggregate audit: static STOPPED, full-request timing UNKNOWN')


if __name__ == '__main__':
    main()