#!/usr/bin/env python3
"""Host-only Q4TC geometry/VRAM screen from pinned public source facts.

No GGUF, model weights, routes, private traces or GPU runtime are needed.
This is not Q4TC implementation or numerical/performance qualification.
"""
import argparse
import json
from pathlib import Path

MIB = 1024 * 1024
Q4_K_BLOCK = 256
Q4_K_BYTES = 144
IQ4_NL_BLOCK = 32
IQ4_NL_BYTES = 18
QPN_INPUT_TILE = 256
QAD_SCALE_TAIL_BYTES = 16
MODEL_SHA = '1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5'
STRATA_REFERENCE = '2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926'
SUPERFAST_REFERENCE = '403f06f580e3a2451061b444ad7e14fa897636f0'

# Already-published safe aggregate counts, NOT private expert identities.
# Published historical NVFP4 heldout aggregate, not Q4TC measured eligibility.
HISTORICAL_NVFP4_SELECTED = 768
HISTORICAL_NVFP4_PREFILL_CANONICAL_DEMANDS = 1198
HISTORICAL_NVFP4_VERIFICATION_NATIVE_CONSUMERS = 203
REPLACEMENT_BUDGET_MIB = 2048


def row_major_quant_bytes(k, n, block_len, block_bytes):
    if any(type(v) is not int or v <= 0 for v in (k, n, block_len, block_bytes)):
        raise ValueError('invalid positive quantized geometry')
    if k % block_len:
        raise ValueError('input dimension is not canonical block aligned')
    return (k // block_len) * n * block_bytes


def geometry_screen():
    gu = row_major_quant_bytes(2560, 640, Q4_K_BLOCK, Q4_K_BYTES)
    down = row_major_quant_bytes(640, 2560, IQ4_NL_BLOCK, IQ4_NL_BYTES)
    body = 2 * gu + down
    assert (gu, down, body) == (921600, 921600, 2764800)
    # Superfast's qpn_takes/repacker currently require K % QK_K == 0.
    eligible_gu = (2560 % QPN_INPUT_TILE == 0 and 640 % 32 == 0)
    eligible_down = (640 % QPN_INPUT_TILE == 0 and 2560 % 32 == 0)
    full_tiles, tail = divmod(640, QPN_INPUT_TILE)
    padded_k = ((640 + QPN_INPUT_TILE - 1) // QPN_INPUT_TILE) * QPN_INPUT_TILE
    padded_down = row_major_quant_bytes(padded_k, 2560, IQ4_NL_BLOCK, IQ4_NL_BYTES)
    padding = padded_down - down

    return {
        'schema': 1,
        'scope': 'host-only single-representation Q4TC geometry; no Q4TC code or model execution',
        'decision': 'Q4TC_SOURCE_GEOMETRY_PARTIAL_NO_PROTOTYPE',
        'r1_feasibility': 'UNKNOWN',
        'production_qualified': False,
        'source': {
            'strata_commit': STRATA_REFERENCE,
            'superfast_commit': SUPERFAST_REFERENCE,
            'original_model_sha256': MODEL_SHA,
            'superfast_source_requirement': 'K divisible by 256; output rows divisible by 32',
        },
        'canonical_q4': {
            'gate_up_type': 'Q4_K',
            'down_type': 'IQ4_NL',
            'gate_k': 2560, 'gate_n': 640,
            'down_k': 640, 'down_n': 2560,
            'q4k_block_len': Q4_K_BLOCK, 'q4k_block_bytes': Q4_K_BYTES,
            'iq4_nl_block_len': IQ4_NL_BLOCK, 'iq4_nl_block_bytes': IQ4_NL_BYTES,
            'gate_body_bytes': gu,
            'up_body_bytes': gu,
            'down_body_bytes': down,
            'three_projection_body_bytes': body,
            'original_qad_native_blob_body_plus_scale_tail_bytes': body + QAD_SCALE_TAIL_BYTES,
            'qad_scale_tail_bytes_missing_from_all_q4_native_layout': QAD_SCALE_TAIL_BYTES,
        },
        'source_qpn_geometry_only': {
            'unmodified_superfast_gate_eligible': eligible_gu,
            'unmodified_superfast_up_eligible': eligible_gu,
            'unmodified_superfast_down_eligible': eligible_down,
            'down_full_256_column_tiles': full_tiles,
            'down_remainder_columns': tail,
            'down_complete_tile_fraction': (full_tiles * QPN_INPUT_TILE) / 640,
            'mixed_packed_gate_up_canonical_down_nominal_mac_fraction': (2 * gu) / body,
            'custom_partial_128_down_nominal_mac_fraction_if_implemented': (
                2 * gu + down * (full_tiles * QPN_INPUT_TILE / 640)) / body,
            'notes': 'Eligibility is shape only. Current unmodified Superfast repacker rejects the WHOLE K=640 down tensor; partial support requires new code.',
        },
        'padded_down_hypothesis_not_equal_byte': {
            'padded_k': padded_k,
            'padding_columns': padded_k - 640,
            'padded_down_body_bytes': padded_down,
            'extra_per_expert_bytes': padding,
            'extra_total_body_fraction': padding / body,
            'extra_down_body_fraction': padding / down,
            'extra_selected_experts_mib': padding * HISTORICAL_NVFP4_SELECTED / MIB,
            'notes': 'Padding changes the artifact and nominal arithmetic. Valid zero-valued padded blocks and scale-tail math require proof; no conversion performed.',
        },
        'single_representation_replacement_only': {
            'maximum_resident_copies_of_any_expert_weight': 1,
            'replacement_budget_mib_not_extra_vram': REPLACEMENT_BUDGET_MIB,
            'ideal_q4_body_expert_capacity_in_2gib': (REPLACEMENT_BUDGET_MIB * MIB) // body,
            'historical_nvfp4_selected_packed_experts_not_q4_qualified': HISTORICAL_NVFP4_SELECTED,
            'historical_nvfp4_prefill_canonical_format_demands_not_conversions':
                HISTORICAL_NVFP4_PREFILL_CANONICAL_DEMANDS,
            'historical_nvfp4_distinct_native_verification_consumers':
                HISTORICAL_NVFP4_VERIFICATION_NATIVE_CONSUMERS,
            'any_format_transition_or_fallback_timing_measured': False,
            'cross_phase_correctness_qualified': False,
            'notes': 'One resident weight representation per expert at all times, '
                     'including during changes of format. Capacity is only geometry, '
                     'not a validated Q4 model allocation. Earlier NVFP4 demands are '
                     'separate historical evidence, not a Q4 routing prediction.',
        },
        'qualification_gaps': [
            'No tensor-core speed or Q4 model quality has been measured.',
            'Source QPN eligibility also requires mask, route, contiguous view, and target architecture.',
            'A canonical down fallback does not resolve canonical gate/up demand in native verification.',
            'The QAD FP32 scale tail is absent from ordinary all-Q4 native expert layout; a distinct ABI or proven scale-folding is required.',
            'Single-representation replacement must serve native verification from the same resident format, or evict it before a change; full restoration/state and exposed-time costs are unknown.',
            'Cross-quantization quality and within-Q4 same-quant MMQ-vs-TC performance are separately gated.',
            'Licensing review remains root-license only; no upstream kernel source is copied.',
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    report = geometry_screen()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(report['decision'] + '; R1 UNKNOWN; no GPU action')


if __name__ == '__main__':
    main()
