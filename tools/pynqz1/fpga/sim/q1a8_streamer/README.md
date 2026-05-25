# Q1A8 streamer

Wraps `q1a8_cell` with the stream-driven FSM that the real AXI-DMA-fed
kernel will use. Two interfaces:

| Interface       | Signals                                                   |
|-----------------|-----------------------------------------------------------|
| Kernel control  | `start_kernel`, `num_subblocks`, `kernel_done`, `busy`, `result` |
| Sub-block input | AXIS-style `s_valid` / `s_ready` + 320-bit parallel data  |

The stream input is the size of one Q8 sub-block presented in parallel
(32 weight bits + 32 int8 acts + 2 fp16 scales = 320 bits). The real AXI
DMA wrapper will serialize this onto 64-bit AXIS beats; that packing logic
is its own module and is *not* what this sim validates.

What this sim validates:
- The kernel-control sequencing (start_kernel → busy → kernel_done pulse)
- The handshake (s_ready high when busy and remaining > 0; backpressure
  when s_valid drops in the middle of a kernel)
- The remaining-counter math (last_in raised on the right sub-block;
  kernel_done fires after the cell drains)
- Multiple kernels in sequence (state cleanly resets between runs)

## Files

| File                 | What                                                     |
|----------------------|----------------------------------------------------------|
| `q1a8_streamer.v`    | The FSM + handshake (~80 lines)                          |
| `test.py`            | cocotb tests with bit-exact Python golden                |
| `Makefile`           | cocotb runner; pulls cell + reducer .v files from siblings |

## Run

```sh
nix develop /Users/angel/Documents/asic/tt-tpu/tools/pynqz1
cd tools/pynqz1/fpga/sim/q1a8_streamer
make
```

Five tests:

| Test                    | What it proves                                            |
|-------------------------|-----------------------------------------------------------|
| `test_one_subblock`     | minimum length (num_subblocks=1) kernel completes        |
| `test_k2048_no_stalls`  | 64 sub-blocks streamed back-to-back match the golden     |
| `test_k2048_with_stalls`| 64 sub-blocks with random 0-3 cycle stalls between each  |
| `test_multiple_kernels` | 5 kernels in sequence each clear & accumulate cleanly    |
| `test_varied_lengths`   | num_subblocks=1, 2, 16, 33, 64, 100 - counter edge cases |
