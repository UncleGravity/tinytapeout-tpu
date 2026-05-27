# PL Kernels

PL kernel drivers for Vivado overlays. The current W1A8 matmul driver is
`matmul_q1a8.py`, which packs rowblocks for the row-parallel Q1A8 bitstream.

```
pl/
├── loopback.py      # COPY driver for the DMA loopback benchmark overlay
├── matmul_q1a8.py   # MATMUL_Q1A8 rowblock driver
└── __init__.py     # register_all(registry)
```

A `pl.register_all(registry, overlay)` call from `board.daemon.__main__`
installs only the PL kernels exposed by the loaded overlay.
