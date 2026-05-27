# RTL library

Single source of truth for the synthesizable Verilog used by both the
cocotb sims under `../sim/` and the buildable bitstreams under
`../bitstreams/`. Nothing under here is bitstream- or test-specific - it
all gets reused across multiple downstream consumers.

## Layout

| Folder       | What                                                          |
|--------------|---------------------------------------------------------------|
| `q1a8/`      | the W1A8 rowblock compute pipeline plus the shared fp16/int/fp32 helpers |
| `common/`    | generic plumbing reusable across designs (AXI-Lite slave skeletons, etc.) |

## Consumers

- `../sim/q1a8_*/Makefile` — each sim points its `VERILOG_SOURCES` at the
  modules it exercises.
- `../bitstreams/<design>/tcl/build.tcl` — Vivado projects include the
  needed `.v` files from here plus their own design-specific top wrapper.

When a new module is needed by exactly one design, write it locally in
that design's folder (`bitstreams/<design>/rtl/`). Promote it here only
when a second consumer appears.
