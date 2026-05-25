# Dual-HP DMA loopback

Two `axi_dma` instances, each looping through its own `axis_loopback` on its
own AXI HP port (HP0 and HP1). The single-port `dma_loopback` proves the
plumbing; this one answers the question the W1A8 plan depends on: **do two
concurrent PL streams hit DDR aggregate bandwidth, or does one HP port's
throughput cap them both?**

If `both` ≈ `hp0 + hp1`, weights on HP0 and activations on HP1 is safe.
If `both` ≈ `max(hp0, hp1)`, DDR is the wall and there's no point planning
around two ports.

## Build

```sh
fpga/benchmarks/dma_dual_loopback/build.sh           # build + fetch
fpga/benchmarks/dma_dual_loopback/build.sh --install # ...and push to board
```

## Run

Same wrapping as `dma_loopback` (root, XRT path, pynq-venv):

```sh
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  '~/pynqz1/fpga/benchmarks/dma_dual_loopback/bench.py' verify

```

```sh
ssh -t xilinx@pynq sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python \
  '~/pynqz1/fpga/benchmarks/dma_dual_loopback/bench.py' bench
```

The `bench` output reports `hp0`, `hp1`, `both` MiB/s and a `scale` column
(= `both / (hp0 + hp1)`). 1.0 means perfect concurrent scaling; 0.5 means
fully serialized at the DDR controller.
