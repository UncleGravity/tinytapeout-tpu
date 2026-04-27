![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# Tiny Tapeout 1-Bit TPU

Simple 1-Bit Systolic Array for learning purposes.

> Based on the [official TT Verilog template](https://github.com/TinyTapeout/tt10-verilog-template).

TODO: expand

## 2D Preview
![png](https://unclegravity.github.io/tt-tpu/gds_render.png)

## TLDR:
```sh
# Install Nix
curl -fsSL https://install.determinate.systems/nix | sh -s -- install

# Use librelane binary cache (will take forever to build otherwise)
nix run nixpkgs#cachix -- use librelane

# Enable GitHub Pages with Actions deployment (docs workflow will fail otherwise)
nix run nixpkgs#gh -- api -X POST repos/{owner}/{repo}/pages -f build_type=workflow

# Start developing
nix develop                             # Enter dev shell with all dependencies
./tt/tt_tool.py --harden --no-docker    # Harden silicon
./tt/tt_fpga.py harden                  # Harden FPGA bitstream
./tt/tt_fpga.py configure --port /dev/<your-tty> --upload --clockrate 12000000
cd test && make -B                      # RTL test
cd test && make -B GATES=yes            # gate-level test (needs prior harden)
```

## What you get

- `nix develop` — full EDA toolchain (librelane, yosys, iverilog, klayout, openroad, magic, netgen, verilator, nextpnr, icestorm) plus `tt-support-tools` and the sky130A PDK, installed automatically on first enter.
- `nix run .#test` / `.#harden` / `.#fpga` — the common workflows as one-liners.
- Works on `aarch64-darwin`, `x86_64-linux`, `aarch64-linux`.

| Command | What it does |
|---|---|
| `nix run .#test`   | RTL simulation (cocotb + icarus), equivalent to `cd test && make -B` |
| `nix run .#harden` | ASIC harden flow for Tiny Tapeout (`tt/tt_tool.py --harden --no-docker`) |
| `nix run .#fpga`   | FPGA bitstream build for the TT FPGA breakout (`tt/tt_fpga.py harden`) |

## CI

The upstream GitHub Actions workflows (`.github/workflows/{gds,docs,test,fpga}.yaml`) are unchanged from Tiny Tapeout.

## Resources

- [Tiny Tapeout](https://tinytapeout.com) · [FAQ](https://tinytapeout.com/faq/) · [Discord](https://tinytapeout.com/discord)
- [Local hardening guide](https://www.tinytapeout.com/guides/local-hardening/) · [FPGA breakout guide](https://tinytapeout.com/guides/fpga-breakout/)
- [LibreLane docs](https://librelane.readthedocs.io/en/stable/)
- [Submit to the next shuttle](https://app.tinytapeout.com/)
