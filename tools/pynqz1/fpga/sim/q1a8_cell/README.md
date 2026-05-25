# Q1A8 cell

One full Q1A8 matmul output cell. Composes the validated `q1a8_reducer`
with a new `fp32_add` accumulator into the unit that becomes one (row, col)
position of the eventual systolic array.

## Protocol

1. Pulse `start_cell` for one cycle. `acc` clears to +0, `busy` goes high.
2. For each of K/32 sub-blocks, raise `valid_in` along with operands. The
   reducer + accumulator pipeline absorbs them one cycle apart.
3. On the final sub-block's `valid_in` cycle, also raise `last_in`.
4. Two cycles later, `cell_done` pulses for one cycle. On that cycle `acc`
   holds the final fp32 value and `busy` drops.

For Bonsai-1.7B with K=2048, that's 64 sub-blocks per cell.

## Files

Synthesizable Verilog lives in `../../rtl/q1a8/`; this folder is just
the test setup:

| Local              | What                                                       |
|--------------------|------------------------------------------------------------|
| `test.py`          | cocotb tests with bit-exact Python golden                  |
| `Makefile`         | cocotb runner; points at `rtl/q1a8/`                       |

| `../../rtl/q1a8/...`  | Module                                                  |
|-----------------------|---------------------------------------------------------|
| `q1a8_cell.v`         | reducer + accumulator + start/done sequencing           |
| `fp32_add.v`          | truncating fp32 adder, same simplifications as fp32_mul |
| (plus reducer chain)  | fp16_to_fp32, int_to_fp32, fp32_mul, q1a8_reducer       |

## Run

```sh
nix develop /Users/angel/Documents/asic/tt-tpu/tools/pynqz1
cd tools/pynqz1/fpga/sim/q1a8_cell
make
```

The Makefile runs four tests:

| Test                  | What it proves                                                  |
|-----------------------|-----------------------------------------------------------------|
| `test_one_subblock`   | acc receives one contribution, cell_done fires, busy clears     |
| `test_full_k2048`     | 64-sub-block stream matches the bit-exact golden                |
| `test_with_gaps`      | idle cycles between sub-blocks don't perturb the accumulation   |
| `test_back_to_back`   | 8 cells in sequence each clear & accumulate independently       |
