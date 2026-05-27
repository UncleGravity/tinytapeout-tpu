# Q1A8 rowblock kernel

The full compute path for one activation column against a block of output
rows. The 64-bit AXIS input carries one shared Q8 activation sub-block plus
one Q1 weight sub-block per row lane. This replaces the old one-output-cell
stream, which repeated activations for every row.

## Pack format

For each Q1 block:

- `ceil(ROWS/4)` beats of fp16 weight scales, four rows per beat.
- Four Q8 sub-block groups, each with:
  - four activation beats (`32 x int8`)
  - one activation-scale beat
  - `ceil(ROWS/2)` beats of uint32 weight bits, two rows per beat

## Run

```sh
cd tools/pynqz1/fpga/sim/q1a8_kernel
make
```
