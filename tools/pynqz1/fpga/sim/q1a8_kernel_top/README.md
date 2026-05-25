# Q1A8 kernel top

The synthesizable top that goes into the Vivado block design. Wraps
`q1a8_kernel` with a hand-rolled AXI4-Lite slave that exposes the kernel
as a register file, plus the AXIS data port for AXI DMA.

This sim is the last RTL-only milestone before Vivado. It exercises the
register-file glue (start strobe, busy/done latching, result register,
cycle counter) under the same handshake the bitstream will see from the
PS, but in seconds-fast cocotb iteration.

## Register map

| Offset | Name           | Access | Meaning                                            |
|--------|----------------|--------|----------------------------------------------------|
| 0x00   | ID             | RO     | `0xB05A_1000`                                      |
| 0x04   | VERSION        | RO     | `0x0000_0001`                                      |
| 0x08   | CTRL           | RW     | `bit[0]` = start-kernel strobe                     |
| 0x0C   | STATUS         | RO     | `bit[0]` = busy, `bit[1]` = done_latched           |
| 0x10   | NUM_SUBBLOCKS  | RW     | sub-block count for the next kernel                |
| 0x14   | RESULT         | RO     | fp32 accumulator                                   |
| 0x18   | CYCLES         | RO     | cycles taken by the last kernel                    |

## Files

Synthesizable Verilog at `../../rtl/q1a8/q1a8_kernel_top.v` (~180 lines).

| Local       | What                                                   |
|-------------|--------------------------------------------------------|
| `test.py`   | cocotb tests + bit-exact golden + AXI-Lite helpers     |
| `Makefile`  | cocotb runner; points at `rtl/q1a8/`                   |

## Run

```sh
cd tools/pynqz1/fpga/sim/q1a8_kernel_top
make
```

Five tests:

| Test                      | What it proves                                        |
|---------------------------|-------------------------------------------------------|
| `test_register_constants` | AXI-Lite read decode + ID/VERSION magic constants     |
| `test_full_kernel`        | full path: configure -> start -> stream -> poll -> read |
| `test_cycles_counter`     | the free perf counter increments only while busy      |
| `test_back_to_back`       | 3 kernels in sequence each reset cleanly              |
| `test_varied_lengths`     | num_subblocks=1, 16, 64 via the register interface    |
