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


def test_pack_cell_matches_axis_format():
    weight_scale = 0x1234
    weight_bits = bytes(range(Q1_BLOCK // 8))
    weight_row = struct.pack("<H", weight_scale) + weight_bits
    assert len(weight_row) == Q1_BLOCK_BYTES

    act_quants = [i - 64 for i in range(Q1_BLOCK)]
    act_scale_bits = [0x2000, 0x2001, 0x2002, 0x2003]
    packed = bytearray((Q1_BLOCK // Q8_BLOCK) * matmul_q1a8.SUBBLOCK_BYTES)

    matmul_q1a8._pack_cell_into(
        packed,
        weight_row,
        act_quants,
        act_scale_bits,
        Q1_BLOCK,
    )

    for q8_index in range(Q1_BLOCK // Q8_BLOCK):
        base = q8_index * matmul_q1a8.SUBBLOCK_BYTES
        bit_base = q8_index * 4
        expected_bits = int.from_bytes(weight_bits[bit_base : bit_base + 4], "little")
        assert packed[base : base + 8] == struct.pack("<II", expected_bits, 0)
        assert packed[base + 8 : base + 40] == bytes(
            value & 0xFF for value in act_quants[q8_index * Q8_BLOCK : (q8_index + 1) * Q8_BLOCK]
        )
        assert packed[base + 40 : base + 48] == struct.pack(
            "<HHI",
            weight_scale,
            act_scale_bits[q8_index],
            0,
        )


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
