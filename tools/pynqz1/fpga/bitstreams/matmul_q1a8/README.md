# matmul_q1a8

The W1A8 matmul bitstream (v4, dual-stream). One kernel kick computes a full
matmul of `NUM_ROWBLOCKS * 8` output rows for one activation column:
- **AXI4-Lite control** plane (host writes registers via the PS GP0)
- **AXI4-Stream weights** in: `axi_dma_0` MM2S -> `S_AXIS`, fed from HP1 (a
  dedicated DDR port for the bandwidth-dominant weight stream)
- **AXI4-Stream acts** in: `axi_dma_1` MM2S -> `S_AXIS_ACTS`, fed from HP0,
  sent once per column (the kernel BRAMs them and reuses across rowblocks)
- **AXI4-Stream results** out: `M_AXIS` (8 fp32/rowblock) -> `axi_dma_0` S2MM
  -> HP0

The host packs the per-column weight + acts streams once, fires one start
strobe, and collects results from a CMA buffer the S2MM DMA fills. No PS
round-trip per rowblock.

Clock: `FCLK_CLK0` is requested at 150 MHz; the PS PLL delivers ~142.857 MHz
(1000/7). The enable-gated `acc_flat` accumulator is declared multicycle in
`tcl/timing.xdc` (see notes there); the reducer fp32 datapath is the remaining
worst-case-timing item.

## Topology

```
ps7/M_AXI_GP0 -> axi_lite_interconnect -> { axi_dma_0.S_AXI_LITE,
                                            axi_dma_1.S_AXI_LITE,
                                            q1a8_kernel_top.S_AXI }
ps7/S_AXI_HP1 <- axi_mem_interconnect_hp1 <- axi_dma_0.M_AXI_MM2S   (weights)
ps7/S_AXI_HP0 <- axi_mem_interconnect     <- { axi_dma_0.M_AXI_S2MM (results),
                                               axi_dma_1.M_AXI_MM2S (acts) }
axi_dma_0.M_AXIS_MM2S    -> q1a8_kernel_top.S_AXIS        (weights)
axi_dma_1.M_AXIS_MM2S    -> q1a8_kernel_top.S_AXIS_ACTS   (acts)
q1a8_kernel_top.M_AXIS   -> axi_dma_0.S_AXIS_S2MM         (results)
```

## Register map

See `../../rtl/q1a8/q1a8_kernel_top.v` for the authoritative definitions.

| Offset | Name           | Access | Meaning                              |
|--------|----------------|--------|--------------------------------------|
| 0x00   | ID             | RO     | `0xB05A_2000`                        |
| 0x04   | VERSION        | RO     | `0x0000_0004` (v4 = dual-stream)     |
| 0x08   | CTRL           | WO     | `bit[0]` = start-kernel strobe       |
| 0x0C   | STATUS         | RO     | `bit[0]` busy, `bit[1]` done_latched |
| 0x10   | NUM_Q1_BLOCKS  | RW     | K/128                                |
| 0x14   | NUM_ROWBLOCKS  | RW     | ceil(M / 8)                          |
| 0x18   | CYCLES         | RO     | busy-cycle count of last run         |
| 0x1C   | ROWS           | RO     | lanes per rowblock (8)               |

## Stream layout

Authoritative byte counts live in `proto/q1a8_layout.py`. v4 splits weights
and acts into two streams.

**Weights** (`S_AXIS`) carry `NUM_ROWBLOCKS * NUM_Q1_BLOCKS` Q1 blocks in
order. Per Q1 block (`PACKED_PER_Q1_BLOCK` = 144 bytes for ROWS=8):
- `ceil(ROWS/4)` beats of fp16 weight scales, four rows per beat
- four Q8 sub-blocks, each `ceil(ROWS/2)` beats of uint32 weight bits, two
  rows per beat

**Acts** (`S_AXIS_ACTS`) are sent once per column (the kernel loads them into
BRAM and reuses across rowblocks). Per Q1 block (`ACTS_PER_Q1_BLOCK` = 160 B):
four Q8 sub-blocks of four int8-act beats + one fp16 act-scale beat.

All ROWS lanes are always active. When `M % ROWS != 0` the host packer
zero-pads inactive lanes in the final rowblock; their results come out as
fp32 zero and the host driver discards them.

## Output layout

The master AXIS stream emits `NUM_ROWBLOCKS * 4` 64-bit beats, lane-major:

| Beat | High 32           | Low 32            |
|------|-------------------|-------------------|
| 0    | lane 1 fp32       | lane 0 fp32       |
| 1    | lane 3 fp32       | lane 2 fp32       |
| 2    | lane 5 fp32       | lane 4 fp32       |
| 3    | lane 7 fp32       | lane 6 fp32       |
| ...  | next rowblock...                       |

`TLAST` is asserted on the final beat of the final rowblock.

## Build

```sh
fpga/bitstreams/matmul_q1a8/build.sh            # build, fetch .bit/.hwh into out/
fpga/bitstreams/matmul_q1a8/build.sh --install  # ...and push to the board
```

build.sh pushes both this folder AND the shared `fpga/rtl/` tree to the
Vivado VM so the tcl can `add_files [file join $proj_root rtl q1a8 *.v]`.

## Run on the board

```sh
# Sanity: ID/VERSION + one matmul against the truncating-fp32 golden.
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  ~/pynqz1/fpga/bitstreams/matmul_q1a8/bench.py verify --rowblocks 8
```

```sh
# Throughput: full matmul perf, report us/matmul + us/rowblock + DMA MB/s.
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  ~/pynqz1/fpga/bitstreams/matmul_q1a8/bench.py bench --rowblocks 32 --iters 200
```
