# Q1A8 reducer

One-cycle reducer that consumes a Q8 sub-block (32 weight bits, 32 int8
acts, two fp16 scales) and produces one fp32 contribution to a matmul
output cell:

    contribution = (fp32) weight_scale * act_scale * sum_i (b_i ? +a_i : -a_i)

This is the inner-most compute unit of the W1A8 matmul. Each cell of the
eventual systolic array is one of these plus an fp32 accumulator wrapped
around it.

## Files

This folder holds the test setup only - the synthesizable Verilog lives
in `../../rtl/q1a8/`:

| Local         | What                                                            |
|---------------|-----------------------------------------------------------------|
| `test.py`     | cocotb testbench with a bit-exact Python golden                 |
| `Makefile`    | cocotb driver (defaults to verilator); points at `rtl/q1a8/`    |

| `../../rtl/q1a8/...`  | Module under test                                       |
|-----------------------|---------------------------------------------------------|
| `q1a8_reducer.v`      | the reducer module (integer reduce + fp scaling)        |
| `fp16_to_fp32.v`      | combinational fp16 -> fp32 conversion                   |
| `int_to_fp32.v`       | combinational signed int -> fp32 (parameterized)        |
| `fp32_mul.v`          | combinational truncating fp32 multiplier                |

The fp32 multiplier is truncating (round-toward-zero) and doesn't model
Inf / NaN. For real silicon swap it for the Xilinx Floating Point Operator
IP - same I/O, fully IEEE 754. Until then the truncating version lets the
testbench compare hardware against a bit-exact Python golden without
depending on Vivado at simulation time.

## Run

From inside `nix develop` (which now includes `cocotb` and `verilator`):

```sh
cd tools/pynqz1/fpga/sim/q1a8_reducer
make            # SIM=verilator (default)
make SIM=icarus # if icarus is available
```

The Makefile runs four tests:

| Test                  | What it proves                                              |
|-----------------------|-------------------------------------------------------------|
| `test_zero`           | reset path, zero-input path                                 |
| `test_identity_scale` | conditional-add tree + int->fp32, with scales = 1.0         |
| `test_corners`        | extreme sub_sum values on both signs (+4096, -4096, etc.)   |
| `test_random`         | 1000 random sub-blocks vs the Python golden                 |

Failures print raw fp32 bits, decoded floats, and the integer `sub_sum`
so you can localize which stage diverged (integer reduce, fp16 convert,
fp32 multiply, etc.).
