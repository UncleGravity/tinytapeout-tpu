# PL Kernels

Empty for now. Once a Vivado overlay exposes the W1A8 core, the AXI driver
and kernel implementations land here:

```
pl/
├── driver.py       # bitfile load, AXI register map, command ring
├── matmul_q1a8.py  # implements Kernel using driver.py
└── __init__.py     # register_all(registry)
```

A `pl.register_all(registry)` call from `board.daemon.__main__` will install
PL kernels with higher priority than the equivalent PS kernels.
