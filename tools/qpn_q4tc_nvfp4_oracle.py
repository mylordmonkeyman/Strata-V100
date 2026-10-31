"""Independent CPU-only decoder of pinned ggml's 36-byte NVFP4 block.

This operates only on caller-supplied bytes. No checkpoint, CUDA, or GPU state.
The 4 UE4M3 bytes scale 16 E2M1 codes each; each subblock's low
nibbles precede its high nibbles in logical element order.
"""
from __future__ import annotations

import math

NVFP4_BLOCK = 64
NVFP4_BYTES = 36
# Pinned ggml's kvalues_fp4 = 2 * E2M1, including signed-zero codes.
E2M1_DOUBLED = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def ue4m3_half(code: int) -> float:
    """Pinned ggml UE4M3 value divided by two for doubled E2M1 table."""
    if type(code) is not int or not 0 <= code <= 255:
        raise ValueError("UE4M3 code must be an unsigned byte")
    if code == 0 or code == 0x7F:
        return 0.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    raw = math.ldexp(float(mantissa), -9) if exponent == 0 else math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return raw * 0.5


def decode_nvfp4_row(data: bytes, k: int) -> tuple[float, ...]:
    """Decode row-major 64-value blocks without reading any external weights."""
    if type(k) is not int or k <= 0 or k % NVFP4_BLOCK or len(data) != k // NVFP4_BLOCK * NVFP4_BYTES:
        raise ValueError("invalid NVFP4 row shape/length")
    out: list[float] = []
    for b in range(0, len(data), NVFP4_BYTES):
        scales = data[b:b + 4]
        codes = data[b + 4:b + NVFP4_BYTES]
        for sub in range(4):
            scale = ue4m3_half(scales[sub])
            packed = codes[sub * 8:(sub + 1) * 8]
            out.extend(E2M1_DOUBLED[v & 15] * scale for v in packed)
            out.extend(E2M1_DOUBLED[v >> 4] * scale for v in packed)
    return tuple(out)
