#!/usr/bin/env python3
"""Host-only reference for upstream MMQ Q8_1 D4 activation layout.

Diagnostic only. Consumes caller-supplied synthetic bytes; never reads a model,
checkpoint, CUDA buffer, prompt, route, or user artifact. This is NOT a
numerical-quality or end-to-end QPN qualification.

Pinned source: ggml-org/llama.cpp@3cf03257f219afbe7334045ff7c6a06ac68c627d
  ggml/src/ggml-cuda/mmq.cuh (block_q8_1_mmq)
  ggml/src/ggml-cuda/quantize.cu (quantize_mmq_q8_1 scatter/non-scatter)
"""
import math
import struct

D4_BLOCK_BYTES = 144
D4_BLOCK_VALUES = 128


def decode_d4_to_f16(payload: bytes, rows: int, columns: int,
                     *, ne1: int | None = None) -> bytes:
    """Reconstruct row-major binary16 [rows,columns] from column-block-major D4.

    D4 storage: block index = (column//128)*ne1 + row; each block has
    four FP32 scales followed by 128 signed int8 codes. ne1 is the
    producer's physical row pitch, NOT necessarily the number of rows
    selected for this stage. Caller must independently verify correct
    producer layout (D4, not DS4/D2S6).

    Invalid/non-FP16-representable values are rejected rather than silently
    creating inf. A future device model consumer requires its own guards.
    """
    if (type(rows) is not int or type(columns) is not int or rows <= 0 or
            columns <= 0 or columns % D4_BLOCK_VALUES):
        raise ValueError('D4 requires positive rows and 128-aligned columns')
    if ne1 is None:
        ne1 = rows
    if type(ne1) is not int or ne1 < rows:
        raise ValueError('D4 physical row pitch must contain output rows')
    if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) != (
            columns // D4_BLOCK_VALUES) * ne1 * D4_BLOCK_BYTES:
        raise ValueError('D4 payload length does not match producer pitch')
    out = bytearray(rows * columns * 2)
    for colblock in range(columns // D4_BLOCK_VALUES):
        for row in range(rows):
            base = (colblock * ne1 + row) * D4_BLOCK_BYTES
            scales = struct.unpack_from('<4f', payload, base)
            for i in range(D4_BLOCK_VALUES):
                scale = scales[i // 32]
                value = scale * struct.unpack_from('<b', payload, base + 16 + i)[0]
                if not math.isfinite(value):
                    raise ValueError('non-finite D4 reconstructed value')
                try:
                    struct.pack_into('<e', out,
                                     (row * columns + colblock * D4_BLOCK_VALUES + i) * 2,
                                     value)
                except (OverflowError, struct.error) as exc:
                    raise ValueError('D4 value outside finite FP16 range') from exc
    return bytes(out)
