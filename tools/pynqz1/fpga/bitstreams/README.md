# Bitstreams

Every folder under here builds a `.bit` + `.hwh` overlay loadable by
PYNQ. Each design is self-contained: its own Tcl, build script, board
runner, and any design-specific RTL not shared via `../rtl/`.

## Layout

| Folder            | What                                                       |
|-------------------|------------------------------------------------------------|
| `matmul_q1a8/`    | TODO: the W1A8 matmul kernel - ps7 + axi_dma + axi_lite + q1a8_kernel |
| `benchmarks/`     | standalone plumbing tests; no shipping function           |
| `benchmarks/dma_loopback/`      | single-HP AXI DMA loopback (bandwidth + plumbing baseline) |
| `benchmarks/dma_dual_loopback/` | two-HP concurrent loopback (measures DDR ceiling)          |
| `benchmarks/axi_lite_probe/`    | hand-written AXI-Lite slave + register-protocol probe      |

## Conventions per design

```
<design>/
  rtl/                 design-specific top wrappers (most RTL lives in ../../rtl/)
  tcl/build.tcl        Vivado batch script
  build.sh             pushes sources to the Vivado VM, fetches .bit, optionally pushes to board
  bench.py             board-side runner / verifier
  README.md
```
