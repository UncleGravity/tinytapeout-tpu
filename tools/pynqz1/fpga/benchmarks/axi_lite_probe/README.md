# AXI-Lite control-plane probe

A minimal bitstream that exposes one hand-written AXI4-Lite slave (`rtl/
axi_lite_regs.v`) with a small register file: two RO magic constants, a
scratch RW, a control bit, and a free-running counter. No DMA, no HP ports,
no compute — the *only* PL logic is the register file.

The point is not the registers. The point is to prove that the **path**
the W1A8 kernel will use — custom Verilog → AXI-Lite slave → Vivado block
design → PYNQ MMIO read/write — works end-to-end on its own. Debugging a
real compute kernel later means debugging the math, not also debugging the
control plane.

## Register map

| Offset | Name    | Access | Meaning                                          |
|--------|---------|--------|--------------------------------------------------|
| 0x00   | ID      | RO     | `0xCAFE_0001` magic constant                     |
| 0x04   | VERSION | RO     | `0x0000_0001`                                    |
| 0x08   | SCRATCH | RW     | any 32-bit value (byte-strobe honored)           |
| 0x0C   | CTRL    | RW     | `bit[0]`=run; `bit[1]`=clear-counter strobe      |
| 0x10   | COUNTER | RO     | free-running counter while `CTRL.run=1`          |

## Build

```sh
fpga/benchmarks/axi_lite_probe/build.sh           # build + fetch artifacts
fpga/benchmarks/axi_lite_probe/build.sh --install # ...also push to the board
```

## Run

```sh
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  ~/pynqz1/fpga/benchmarks/axi_lite_probe/bench.py
```

The probe runs seven check groups in order. Each prints `ok` or `FAIL`;
the script exits non-zero if any check failed. The most informative one
is `[5] counter rate` — it computes how many counter ticks elapsed over
a 50 ms host-side sleep and compares against the expected 100 MHz, which
catches both "PL is dead-clocked" and "AXI-Lite reads return stale data".
