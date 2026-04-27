# Bonsai Fixture Tool

This tool extracts small, deterministic GGUF/Q1_0 fixtures from the local
Bonsai model for RTL tests.

Use C++ here, not Python. The fixture generator links against the `ggml`
package from `flake.nix`, so GGUF parsing and type definitions come from the
same library family as llama.cpp.

The pin is the repository `flake.lock`: `flake.nix` uses the pinned `nixpkgs`
input, and the dev shell currently exposes `ggml` through that pinned package
set. Check the active version with:

```sh
nix develop -c pkg-config --modversion ggml
```

Build and inspect the first attention Q projection tensor:

```sh
nix develop -c make -C tools/bonsai_fixture inspect
```

Generate a fixture for a specific tensor, output row pair, and Q1_0 group:

```sh
nix develop -c tools/bonsai_fixture/bonsai_fixture \
  generate-group \
  models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  test/fixtures/bonsai_blk0_attn_q_r42_r43_g7.json \
  blk.0.attn_q.weight \
  42 \
  7 \
  2 \
  4
```

The final two arguments are optional and default to the current RTL tile:

```text
tile_rows = 2
tile_cols = 4
```

`tile_cols` must divide the 128-wide Q1_0 group size.
It must also divide the 32-wide Q8_0 activation block size, because GGML
applies activation scales at that boundary.

Generate all Q1_0 groups for one complete output row range:

```sh
nix develop -c tools/bonsai_fixture/bonsai_fixture \
  generate-row-tile \
  models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  test/fixtures/bonsai_blk0_attn_q_rows0_1_all_groups.json \
  blk.0.attn_q.weight \
  0 \
  2 \
  4
```

For the current `blk.0.attn_q.weight` shape, that emits 16 Q1_0
groups and 512 `2x4` tile transactions.

Current scope:

- prove the dev shell can compile/link against `ggml`
- load Bonsai GGUF metadata
- inspect a selected Q1_0 tensor shape, byte size, and file offset
- emit JSON transactions for any selected Q1_0 tensor row pair/group
- emit all groups for one complete output row range
- emit deterministic Q8_0 activation scales and GGML `Q1_0 x Q8_0`
  scaled float references
- self-check scaled references against GGML's CPU `vec_dot` implementation

Next scope:

- use Q8_0 activations captured from a real token forward pass
