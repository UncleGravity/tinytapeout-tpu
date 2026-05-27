# matmul_q1a8 rowblock

The W1A8 matmul bitstream. One kernel kick computes up to 8 output rows for
one activation column, driven by:
- **AXI4-Lite control** plane (host writes registers via the PS GP0)
- **AXI4-Stream data** plane (host streams rowblock-packed data via AXI DMA -> HP0)

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
| 0x00   | ID             | RO     | `0xB05A_2000`                        |
| 0x04   | VERSION        | RO     | `0x0000_0002`                        |
| 0x08   | CTRL           | RW     | `bit[0]` = start-kernel strobe       |
| 0x0C   | STATUS         | RO     | `bit[0]` busy, `bit[1]` done_latched |
| 0x10   | NUM_Q1_BLOCKS  | RW     | K/128 for the next rowblock          |
| 0x14   | ROW_COUNT      | RW     | active lanes in this rowblock        |
| 0x18   | RESULT_INDEX   | RW     | lane index read through RESULT       |
| 0x1C   | RESULT         | RO     | fp32 result for RESULT_INDEX         |
| 0x20   | CYCLES         | RO     | cycles taken by the last kernel      |
| 0x24   | ROWS           | RO     | lanes per rowblock                   |

## Stream layout

For each Q1 block:

- `ceil(ROWS/4)` beats of fp16 weight scales, four rows per beat.
- Four Q8 sub-block groups, each with:
  - four activation beats (`32 x int8`)
  - one activation-scale beat
  - `ceil(ROWS/2)` beats of uint32 weight bits, two rows per beat

For `ROWS=8`, one Q1 block is 304 bytes instead of 8 independent single-cell
streams at 1536 bytes.

## Build

```sh
fpga/bitstreams/matmul_q1a8/build.sh            # build, fetch .bit/.hwh into out/
fpga/bitstreams/matmul_q1a8/build.sh --install  # ...and push to the board
```

build.sh pushes both this folder AND the shared `fpga/rtl/` tree to the
Vivado VM so the tcl can `add_files [file join $proj_root rtl q1a8 *.v]`.

## Run on the board

```sh
# Sanity: ID/VERSION + one K=2048 rowblock against the truncating-fp32 golden.
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  ~/pynqz1/fpga/bitstreams/matmul_q1a8/bench.py verify
```

```sh
# Throughput: 1000 back-to-back rowblocks, report PS-side us/rowblock + PL cycles.
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  ~/pynqz1/fpga/bitstreams/matmul_q1a8/bench.py bench --iters 1000
```
