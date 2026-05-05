## How it works

This is a tiny weight-stationary systolic array for signed int8 × 1-bit
matrix multiplies. The fabric is a 2×2 grid of processing elements
(`w1a8_pe`); each PE stores one 1-bit weight and on every cycle computes
`psum_out = psum_in + (weight ? +act_in : -act_in)` while forwarding the
activation and a valid bit by one cycle. Activations are skewed in from
the left, partial sums accumulate down each column, and a per-row
accumulator on the bottom captures the final sum for that row.

The host talks to the tile through a single-byte command port:

| `ui_in[2:0]` | Command | Effect                                                                 |
|--------------|---------|------------------------------------------------------------------------|
| 0            | STATUS  | `uo_out` = status byte                                                 |
| 1            | CLEAR   | reset FSM, clear acts/results (weights kept)                           |
| 2            | LDW     | write `COLS` packed weight bits from `uio_in` into row `arg`           |
| 3            | LDA     | write one int8 activation from `uio_in` into `act_mem[arg]`            |
| 4            | SEED    | write one byte from `uio_in` into the row-`arg_row`/byte-`arg_byte` accumulator slot |
| 5            | START   | shift stored weights into the PEs and run one compute pass             |
| 6            | RDP     | `uo_out` = accumulator byte at `arg_row`/`arg_byte`                    |
| 7            | NOP     | `uo_out` = status byte                                                 |

`ui_in[7:3]` carries the 5-bit argument (row index, column index, or
`{row, byte}` depending on command). The status byte exposes
`busy / done / weight_done / all_rows_done / idle / error` so the host
can poll instead of relying on cycle counts. Because the host can `RDP`
one run's accumulator and `SEED` it back into the next, dot products
wider than `COLS` can be tiled — that's how the Bonsai Q1_0 × Q8_0
fixtures stitch 32-wide blocks together (see
`test/bonsai_fixture.py::replay_q8_block`).

Top-level module is `tt_um_unclegravity_tpu` in `src/rtl/project.v`,
which wires the command decoder, FSM, the three scratchpads (weight,
activation, accumulator) and the array together.

## How to test

Using Nix is HIGHLY recommended.

The full cocotb suite lives in `test/` and runs against both RTL and
gate-level netlists:

### Option 1 (boo)
```sh
# Enter devshell with all dependencies installed
nix develop

# RTL simulation (icarus + cocotb)
cd test && make -B 

# Gate-level (needs a prior harden)
nix run .#harden
cd test && make -B GATES=yes
```
The Nix devshell wires up `PDK_ROOT` and the toolchain.

### Option 2 (ez mode):
```sh
nix run .#test      # RTL Tests
nix run .#test-gl   # Gate level tests (requires prior harden)
```

To exercise the tile by hand from a cocotb test, the protocol is:

1. Pulse `rst_n` low, then issue `CLEAR` (cmd 1).
2. For each row, drive `uio_in` with the packed weight bits and issue
   `LDW` with `arg = row`.
3. For each column, drive `uio_in` with the int8 activation and issue
   `LDA` with `arg = col`.
4. (Optional) Use `SEED` to preload the accumulator with a partial sum
   from a previous tile.
5. Issue `START` and poll `STATUS` until `all_rows_done` (bit 3) is set.
6. Issue `RDP` with `arg = {row, byte}` and read the accumulator byte
   off `uo_out`.

Helpers for all of this are in `test/tt_protocol.py` and `test/common.py`.
Waveforms land in `sim_build/rtl/tb.fst` (or `sim_build/gl/tb.fst` for
gate-level) and can be opened with `gtkwave sim_build/rtl/tb.fst tb.gtkw`
or `surfer sim_build/rtl/tb.fst`.

## External hardware

Requires a RP2350. Either reprogramming the one in the dev board, or an external one.
