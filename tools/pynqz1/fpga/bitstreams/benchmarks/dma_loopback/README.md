# DMA loopback

A zero-compute AXI DMA loopback bitstream. Validates the integration path
between PS (Linux + daemon + kernel registry) and PL (overlay + AXI HP DMA)
without any RTL math in the way. If COPY works through this bitstream, the
plumbing is good — every later compute kernel only has to worry about
correctness.

## Build (drives Vivado on a remote VM)

```sh
fpga/benchmarks/dma_loopback/build.sh           # build + fetch artifacts
fpga/benchmarks/dma_loopback/build.sh --install # ...also push to the board
```

`build.sh --help` lists env overrides (VM host, Vivado path, board host, etc.).

If you want to invoke Vivado directly instead:

```sh
cd fpga/benchmarks/dma_loopback
vivado -mode batch -source tcl/build.tcl
# Produces out/dma_loopback.bit + out/dma_loopback.hwh
```

## Three levels of test

PYNQ's overlay loader needs root (for the PL device node) and
`XILINX_XRT=/usr` so XRT can locate its install, and it runs from the
pynq-venv interpreter — not the system python3. Wrap every invocation
that touches the overlay in `sudo env XILINX_XRT=/usr
/usr/local/share/pynq-venv/bin/python …`.

**Level 1 — sanity.** Loads the overlay, DMAs one buffer through the
loopback, asserts byte-equality:

```sh
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  ~/pynqz1/fpga/benchmarks/dma_loopback/bench.py verify
```

**Level 2 — bandwidth sweep.** Same wrapper, different subcommand:

```sh
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  ~/pynqz1/fpga/benchmarks/dma_loopback/bench.py bench
```

**Level 3 — end-to-end through the daemon.** `pynq-deploy daemon`
already wraps the daemon in the same `sudo env XILINX_XRT=/usr …`
incantation, so you just pass `--bitfile` and it registers
``PLLoopback`` as ``GOP_COPY`` (overriding the PS COPY kernel). The
existing C++ smoke test's COPY case then drives the PL DMA:

```sh
nix run .#deploy -- daemon --bitfile /home/xilinx/overlays/dma_loopback.bit &
PYNQ_HOST=pynq nix run .#smoke
```
