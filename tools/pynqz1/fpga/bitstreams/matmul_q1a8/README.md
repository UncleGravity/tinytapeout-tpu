# matmul_q1a8

The W1A8 matmul bitstream. One Q1A8 output cell per kernel kick, driven by:
- **AXI4-Lite control** plane (host writes registers via the PS GP0)
- **AXI4-Stream data** plane (host streams 48-byte packed sub-blocks via AXI DMA -> HP0)

## Topology

```
ps7/M_AXI_GP0 -> axi_lite_interconnect -> { axi_dma.S_AXI_LITE,
                                            q1a8_kernel_top.S_AXI }
ps7/S_AXI_HP0 <- axi_mem_interconnect  <- axi_dma.M_AXI_MM2S
axi_dma.M_AXIS_MM2S -> q1a8_kernel_top.S_AXIS
```

## Register map

See `../../rtl/q1a8/q1a8_kernel_top.v` for the authoritative definitions.

| Offset | Name           | Access | Meaning                              |
|--------|----------------|--------|--------------------------------------|
| 0x00   | ID             | RO     | `0xB05A_1000`                        |
| 0x04   | VERSION        | RO     | `0x0000_0001`                        |
| 0x08   | CTRL           | RW     | `bit[0]` = start-kernel strobe       |
| 0x0C   | STATUS         | RO     | `bit[0]` busy, `bit[1]` done_latched |
| 0x10   | NUM_SUBBLOCKS  | RW     | K/32 for the next kernel             |
| 0x14   | RESULT         | RO     | fp32 accumulator                     |
| 0x18   | CYCLES         | RO     | cycles taken by the last kernel      |

## Sub-block byte layout (host packs this into DDR, DMA streams it)

48 bytes per sub-block. See `pack_subblock_bytes` in `bench.py`.

| Offset (bytes) | Field                                             |
|----------------|---------------------------------------------------|
| 0..3           | `weight_bits[31:0]`                               |
| 4..7           | (reserved, write 0)                               |
| 8..39          | `acts[0..31]` (32 int8s, little-endian per byte)  |
| 40..41         | `weight_scale` (fp16)                             |
| 42..43         | `act_scale` (fp16)                                |
| 44..47         | (reserved, write 0)                               |

## Build

```sh
fpga/bitstreams/matmul_q1a8/build.sh            # build, fetch .bit/.hwh into out/
fpga/bitstreams/matmul_q1a8/build.sh --install  # ...and push to the board
```

build.sh pushes both this folder AND the shared `fpga/rtl/` tree to the
Vivado VM so the tcl can `add_files [file join $proj_root rtl q1a8 *.v]`.

## Run on the board

```sh
# Sanity: ID/VERSION + one K=2048 kernel against the truncating-fp32 golden.
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  ~/pynqz1/fpga/bitstreams/matmul_q1a8/bench.py verify
```

```sh
# Throughput: 1000 back-to-back kernels, report PS-side us/kernel + PL cycles.
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  ~/pynqz1/fpga/bitstreams/matmul_q1a8/bench.py bench --iters 1000
```

The `bench` output separates **PL compute** (from the CYCLES register, free
perf counter) from **PS overhead** (wall time minus PL compute). For the
first iteration we expect PS overhead to dominate - that's the signal
that the next optimization target is reducing per-kernel host work
(batched commands, descriptor-driven kernels, etc.).
