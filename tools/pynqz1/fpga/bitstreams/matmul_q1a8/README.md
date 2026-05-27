# matmul_q1a8

The W1A8 matmul bitstream. One kernel kick computes a full matmul of
`NUM_ROWBLOCKS * 8` output rows for one activation column:
- **AXI4-Lite control** plane (host writes registers via the PS GP0)
- **AXI4-Stream data** plane in (host streams packed weights+acts via AXI DMA MM2S -> HP0)
- **AXI4-Stream data** plane out (kernel emits 8 fp32 results/rowblock -> AXI DMA S2MM -> HP0)

The host packs a per-column wire stream once, fires one start strobe, and
collects results from a CMA buffer the S2MM DMA fills. There is no PS round-
trip per rowblock.

## Topology

```
ps7/M_AXI_GP0 -> axi_lite_interconnect -> { axi_dma.S_AXI_LITE,
                                            q1a8_kernel_top.S_AXI }
ps7/S_AXI_HP0 <- axi_mem_interconnect  <- { axi_dma.M_AXI_MM2S,
                                            axi_dma.M_AXI_S2MM }
axi_dma.M_AXIS_MM2S      -> q1a8_kernel_top.S_AXIS
q1a8_kernel_top.M_AXIS   -> axi_dma.S_AXIS_S2MM
```

## Register map

See `../../rtl/q1a8/q1a8_kernel_top.v` for the authoritative definitions.

| Offset | Name           | Access | Meaning                              |
|--------|----------------|--------|--------------------------------------|
| 0x00   | ID             | RO     | `0xB05A_2000`                        |
| 0x04   | VERSION        | RO     | `0x0000_0003`                        |
| 0x08   | CTRL           | WO     | `bit[0]` = start-kernel strobe       |
| 0x0C   | STATUS         | RO     | `bit[0]` busy, `bit[1]` done_latched |
| 0x10   | NUM_Q1_BLOCKS  | RW     | K/128                                |
| 0x14   | NUM_ROWBLOCKS  | RW     | ceil(M / 8)                          |
| 0x18   | CYCLES         | RO     | busy-cycle count of last run         |
| 0x1C   | ROWS           | RO     | lanes per rowblock (8)               |

## Stream layout

The slave AXIS stream carries `NUM_ROWBLOCKS * NUM_Q1_BLOCKS` Q1 blocks in
order. For each Q1 block (304 bytes for ROWS=8):

- `ceil(ROWS/4)` beats of fp16 weight scales, four rows per beat
- Four Q8 sub-block groups, each with:
  - four activation beats (`32 x int8`)
  - one activation-scale beat (fp16 in low 16 bits)
  - `ceil(ROWS/2)` beats of uint32 weight bits, two rows per beat

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
