"""Synthetic, host-only Q4_K/IQ4_NL dequantization and projection oracle.

This is NOT a quantizer, a model fixture, or a GPU/backend parity result.
Only public GGUF block layouts and caller-supplied synthetic bytes are used.
"""
from __future__ import annotations

import math
import struct

Q4_K_BLOCK, Q4_K_BYTES = 256, 144
IQ4_NL_BLOCK, IQ4_NL_BYTES = 32, 18
IQ4_NL_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10,
                  1, 13, 25, 38, 53, 69, 89, 113)


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite synthetic block/activation/scale")


def decode_q4_k(block: bytes) -> tuple[float, ...]:
    if len(block) != Q4_K_BYTES:
        raise ValueError("Q4_K block must be exactly 144 bytes")
    d, dmin = struct.unpack_from("<ee", block)
    _finite(d, dmin)
    s = block[4:16]
    sc = [s[i] & 63 for i in range(4)] + [
        (s[8 + i] & 15) | ((s[i] >> 2) & 48) for i in range(4)]
    mn = [s[4 + i] & 63 for i in range(4)] + [
        (s[8 + i] >> 4) | ((s[4 + i] >> 2) & 48) for i in range(4)]
    qs = block[16:]
    result: list[float] = []
    for group in range(4):
        q = qs[group * 32:(group + 1) * 32]
        for half in range(2):
            j = group * 2 + half
            shift = half * 4
            result.extend(d * sc[j] * ((byte >> shift) & 15) - dmin * mn[j] for byte in q)
    assert len(result) == Q4_K_BLOCK
    return tuple(result)


def decode_iq4_nl(block: bytes) -> tuple[float, ...]:
    if len(block) != IQ4_NL_BYTES:
        raise ValueError("IQ4_NL block must be exactly 18 bytes")
    (d,) = struct.unpack_from("<e", block)
    _finite(d)
    q = block[2:]
    return tuple(d * IQ4_NL_VALUES[byte & 15] for byte in q) + tuple(
        d * IQ4_NL_VALUES[byte >> 4] for byte in q)


def decode_row(data: bytes, kind: str, k: int) -> tuple[float, ...]:
    if kind == "Q4_K":
        width, size, decoder = Q4_K_BLOCK, Q4_K_BYTES, decode_q4_k
    elif kind == "IQ4_NL":
        width, size, decoder = IQ4_NL_BLOCK, IQ4_NL_BYTES, decode_iq4_nl
    else:
        raise ValueError("unsupported synthetic quant type")
    if type(k) is not int or k <= 0 or k % width or len(data) != (k // width) * size:
        raise ValueError("unaligned or mismatched quantized row")
    return tuple(value for off in range(0, len(data), size)
                 for value in decoder(data[off:off + size]))


def scaled_dot(data: bytes, kind: str, k: int, activation: list[float], scale: float) -> float:
    _finite(scale, *activation)
    if len(activation) != k:
        raise ValueError("activation length mismatch")
    return math.fsum(w * x for w, x in zip(decode_row(data, kind, k), activation)) * scale


def swiglu_from_unscaled(gate: float, up: float, s_gate: float, s_up: float) -> float:
    _finite(gate, up, s_gate, s_up)
    g = gate * s_gate
    u = up * s_up
    return (g / (1.0 + math.exp(-g))) * u
