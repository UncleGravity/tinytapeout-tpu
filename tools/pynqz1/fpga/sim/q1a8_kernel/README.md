# Q1A8 kernel

The full compute path for one output cell, from a 64-bit AXIS input down
to a registered fp32 result. Composes the packer (`axis_to_subblock`),
the streamer (`q1a8_streamer`), the cell (`q1a8_cell`), and the reducer
(`q1a8_reducer`) into the module the eventual bitstream will instantiate.

## Pack format

One Q8 sub-block = 48 bytes = 6 × 64-bit AXIS beats:

| Beat | Bits 63:32         | Bits 31:0                            |
|------|--------------------|--------------------------------------|
| 0    | (reserved, 0)      | `weight_bits[31:0]`                  |
| 1    | `acts[ 63: 0]` (acts_packed)                              |
| 2    | `acts[127:64]`                                            |
| 3    | `acts[191:128]`                                           |
| 4    | `acts[255:192]`                                           |
| 5    | (reserved) + `act_scale[15:0]` | `weight_scale[15:0]` (low 16)|

The host writes a flat byte buffer of `48 * num_subblocks` bytes and a
single AXI DMA programs streams it into the kernel.

## Run

```sh
cd tools/pynqz1/fpga/sim/q1a8_kernel
make
```

Five tests. If any fail, the failure message tells you which layer to
look at (see test.py header).
