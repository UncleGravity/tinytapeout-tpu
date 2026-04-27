# Tests

All tests use [cocotb](https://docs.cocotb.org/en/stable/) to drive the DUT and check the outputs.
[More Info](https://tinytapeout.com/hdl/testing/).

The Makefile requires `PDK_ROOT` to be set; the Nix devshell (`nix develop`) sets this automatically.

## RTL simulation
```sh
make -B # default RTL simulation
```
This runs the cocotb suite plus some Bonsai Q1_0 fixtures in `test/fixtures/`.

## Gate-level simulation 
```sh
make -B GATES=yes
```

The Makefile picks up the netlist from `../runs/wokwi/final/pnl/<top_module>.pnl.v` automatically. If you want to test a different netlist, drop it in as `gate_level_netlist.v` and it'll be used instead.

If you wish to save the waveform in VCD format instead of FST, edit `tb.v` to use `$dumpfile("tb.vcd");` and then run:

```sh
make -B FST=
```

## Regenerate Fixtures

```sh
make -C tools/bonsai_fixture generate
make -C tools/bonsai_fixture generate-row-tile
```

Use `tools/bonsai_fixture/bonsai_fixture --help`-style usage by running the
tool without arguments.

## Full Tensor One-Off

Generates full `blk.0.attn_q.weight` tensor. Compares GGUF and systolic array outputs.

```sh
set -euo pipefail; out=/tmp/tt-tpu-bonsai-full-tensor-fixtures; rm -rf "$out"; mkdir -p "$out"; for row in $(seq 0 2 2046); do tools/bonsai_fixture/bonsai_fixture generate-row-tile models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf "$out/blk0_attn_q_rows${row}_$((row+1))_all_groups.json" blk.0.attn_q.weight "$row" 2 4; done
```
```sh
cd test && BONSAI_FULL_TENSOR_TESTS=1 BONSAI_FULL_TENSOR_FIXTURE_DIR=/tmp/tt-tpu-bonsai-full-tensor-fixtures make -B
```

## How to view the waveform file

Waveforms are written under `sim_build/rtl/tb.fst` (RTL) or `sim_build/gl/tb.fst` (gate-level).

Using GTKWave

```sh
gtkwave sim_build/rtl/tb.fst tb.gtkw
```

Using Surfer

```sh
surfer sim_build/rtl/tb.fst
```
