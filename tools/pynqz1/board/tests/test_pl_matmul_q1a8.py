"""PL MATMUL_Q1A8 driver tests that do not require a PYNQ board."""

from __future__ import annotations

import struct
from types import SimpleNamespace

from board.kernels import pl
from board.kernels.pl import matmul_q1a8
from board.kernels.registry import KernelRegistry
from proto.ops import GOP_COPY, GOP_MATMUL_Q1A8, Q1_BLOCK, Q1_BLOCK_BYTES, Q8_BLOCK
from tests.golden import kernels as golden


def test_quantize_q8_0_matches_golden():
    values = tuple((i - 17) / 13.0 for i in range(Q8_BLOCK * 2))
    quants, scale_bits = matmul_q1a8._quantize_q8_0(values)
    exp_quants, exp_scales = golden.quantize_q8_0(values)

    assert quants == exp_quants
    assert scale_bits == [
        struct.unpack("<H", struct.pack("<e", scale))[0]
        for scale in exp_scales
    ]


def test_pack_rowblock_matches_axis_format():
    row_count = 3
    weight_row_bytes = Q1_BLOCK_BYTES
    weights = bytearray()
    row_bits = []
    for row in range(row_count):
        weight_scale = 0x1234 + row
        bits = bytes((row * 17 + i) & 0xFF for i in range(Q1_BLOCK // 8))
        row_bits.append(bits)
        weights += struct.pack("<H", weight_scale) + bits

    act_quants = [i - 64 for i in range(Q1_BLOCK)]
    act_scale_bits = [0x2000, 0x2001, 0x2002, 0x2003]
    packed = bytearray(matmul_q1a8._rowblock_nbytes(Q1_BLOCK))

    matmul_q1a8._pack_rowblock_into(
        packed,
        bytes(weights),
        0,
        row_count,
        weight_row_bytes,
        act_quants,
        act_scale_bits,
        Q1_BLOCK,
    )

    scale_beats = (matmul_q1a8.ROWS_PER_BLOCK + 3) // 4
    wbits_beats = (matmul_q1a8.ROWS_PER_BLOCK + 1) // 2
    assert packed[:8] == struct.pack("<Q", 0x1234 | (0x1235 << 16) | (0x1236 << 32))
    assert packed[8 : scale_beats * 8] == bytes((scale_beats - 1) * 8)

    cursor = scale_beats * 8
    for q8_index in range(Q1_BLOCK // Q8_BLOCK):
        bit_base = q8_index * (Q8_BLOCK // 8)
        assert packed[cursor : cursor + Q8_BLOCK] == bytes(
            value & 0xFF for value in act_quants[q8_index * Q8_BLOCK : (q8_index + 1) * Q8_BLOCK]
        )
        cursor += Q8_BLOCK

        assert packed[cursor : cursor + 8] == struct.pack("<Q", act_scale_bits[q8_index])
        cursor += 8

        for beat in range(wbits_beats):
            word = 0
            for local in range(2):
                lane = beat * 2 + local
                bits = 0
                if lane < row_count:
                    bits = int.from_bytes(row_bits[lane][bit_base : bit_base + 4], "little")
                word |= bits << (local * 32)
            assert packed[cursor : cursor + 8] == struct.pack("<Q", word)
            cursor += 8

    assert cursor == len(packed)


def test_register_all_registers_matmul_only_for_matmul_overlay():
    registry = KernelRegistry()
    overlay = SimpleNamespace(
        ip_dict={"axi_dma_0": {}, "q1a8_kernel_top_0": {}},
        axi_dma_0=SimpleNamespace(),
        q1a8_kernel_top_0=SimpleNamespace(),
    )

    pl.register_all(registry, overlay)

    assert isinstance(registry.get(GOP_MATMUL_Q1A8), matmul_q1a8.PLMatmulQ1A8)
    assert GOP_COPY not in registry
