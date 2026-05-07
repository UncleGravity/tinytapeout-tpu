![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# Tiny Tapeout 1-Bit TPU

Simple 1-Bit Systolic Array.

Do you want your matrix multiplications to go faster? Look elsewhere.

This project implements a 2x2 weight stationary systolic array that computes matrices with binary {-1,+1} weights and INT8 activations. Designed to fit in a single 1x1 [Tiny Tapeout](https://tinytapeout.com/) tile.

Comes with custom llama-cpp backend to run the worlds slowest forward pass. The first AI decelerator.

Enjoy?

Note:
To prototype, requires [Tiny Tapeout FPGA Dev Kit](https://store.tinytapeout.com/products/FPGA-Development-Kit-p813805747)

## 2D Preview
![png](https://unclegravity.github.io/tinytapeout-tpu/gds_render.png)

## 3D Viewer
[Open 3D viewer](https://gds-viewer.tinytapeout.com/?model=https://unclegravity.github.io/tinytapeout-tpu/tinytapeout.oas&pdk=sky130A)

## TLDR:
```sh
# Install Nix
curl -fsSL https://install.determinate.systems/nix | sh -s -- install

# Use librelane binary cache (makes librelane build much faster)
nix run nixpkgs#cachix -- use librelane

# Start developing
nix develop                 # Enter dev shell with all dependencies installed

# RTL
nix run .#harden            # Harden silicon
nix run .#test              # RTL tests
nix run .#test-gl           # Gate level tests

# FPGA
nix run .#tt-firmware       # Flash custom RP2350 firmware
nix run .#fpga-harden       # Harden FPGA bitstream
nix run .#fpga-flash        # Upload bitstream to FPGA (needs custom RP2350 fw mentioned above)

# Optional
nix run .#view              # View RTL digram (like vivado but worse)

# Only if you want to fork it:
# Enable GitHub Pages with Actions deployment
# (docs workflow will fail otherwise)
nix run nixpkgs#gh -- api -X POST repos/{owner}/{repo}/pages -f build_type=workflow
```

## Inference
You can run any 1-bit GGUF model using llama-cpp + a small custom backend made for this project.
It will be SUPER slow because:
- We're only using a single Tiny Tapeout tile, so the systolic array is 2x2.
- Most importantly, we're using the RP2350 as the transport, which is limited to 1MB/s

```sh
# After running all the commands in the TLDR
uvx hf download prism-ml/Bonsai-1.7B-gguf --local-dir ./models/Bonsai-1.7B
nix run .#llama-cli-bonsai -- \
  --model models/Bonsai-1.7B/Bonsai-1.7B.gguf \
  --device Bonsai \
  --temp 0 --no-warmup
  --prompt "One eternity later..."
```

## CI
The upstream GitHub Actions workflows (`.github/workflows/{gds,docs,test,fpga}.yaml`) are unchanged from Tiny Tapeout.

## Resources
- [Tiny Tapeout](https://tinytapeout.com) · [FAQ](https://tinytapeout.com/faq/) · [Discord](https://tinytapeout.com/discord)
- [Local hardening guide](https://www.tinytapeout.com/guides/local-hardening/) · [FPGA breakout guide](https://tinytapeout.com/guides/fpga-breakout/)
- [LibreLane docs](https://librelane.readthedocs.io/en/stable/)
- [Submit to the next shuttle](https://app.tinytapeout.com/)
