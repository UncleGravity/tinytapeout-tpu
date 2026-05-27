# Q1A8 multi-rowblock kernel

End-to-end matmul kernel: one start strobe processes `NUM_ROWBLOCKS` rowblocks
of `NUM_Q1_BLOCKS * 4 * 32` MACs each. A 64-bit AXIS slave carries the packed
input stream; a 64-bit AXIS master emits 8 fp32 results per rowblock (2 fp32
per beat, 4 beats per rowblock, lane-major), with TLAST asserted on the final
beat of the final rowblock.

## Input pack format (per Q1 block)

- `ceil(ROWS/4)` beats of fp16 weight scales, four rows per beat
- Four Q8 sub-block groups, each with:
  - four activation beats (`32 x int8`)
  - one activation-scale beat (fp16 in low 16 bits)
  - `ceil(ROWS/2)` beats of uint32 weight bits, two rows per beat

All ROWS lanes are always active; the host packer zero-pads inactive lanes
in the final rowblock when `M % ROWS != 0`.

## Output format (per rowblock)

| Beat | High 32     | Low 32      |
|------|-------------|-------------|
| 0    | lane 1 fp32 | lane 0 fp32 |
| 1    | lane 3 fp32 | lane 2 fp32 |
| 2    | lane 5 fp32 | lane 4 fp32 |
| 3    | lane 7 fp32 | lane 6 fp32 |

## Run

```sh
cd tools/pynqz1/fpga/sim/q1a8_kernel
make
```
