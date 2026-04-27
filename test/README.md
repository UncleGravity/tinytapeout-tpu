# Tests

All tests use [cocotb](https://docs.cocotb.org/en/stable/) to drive the DUT and check the outputs.
[More Info](https://tinytapeout.com/hdl/testing/).

The Makefile requires `PDK_ROOT` to be set; the Nix devshell (`nix develop`) sets this automatically.

## RTL simulation
```sh
make -B # default RTL simulation
```
This runs the cocotb suite plus Bonsai Q1_0 x Q8_0 fixtures in `test/fixtures/`.
Fixture tests reset the RTL seed at each 32-wide Q8_0 block, then apply the
recorded scales in Python and compare the reconstructed float against GGML's
`vec_dot` reference.

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
make -C tools/bonsai_fixture generate-group
make -C tools/bonsai_fixture generate-row-tile
make -C tools/bonsai_fixture generate-tensor
```

Use `tools/bonsai_fixture/bonsai_fixture --help`-style usage by running the
tool without arguments.

## Full Tensor One-Off

Generates the full `blk.0.attn_q.weight` tensor fixture as one large uncommitted file. Compares scaled GGML references against outputs reconstructed from systolic-array Q8-block sums.

```sh
make -C tools/bonsai_fixture generate-tensor
```
```sh
cd test && BONSAI_FULL_TENSOR_TESTS=1 BONSAI_FULL_TENSOR_FIXTURE=/tmp/tt-tpu-bonsai-full-tensor.json make -B
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
