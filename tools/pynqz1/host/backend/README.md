# PYNQ ggml Backend

`libggml-pynq.so` is the host-side ggml boundary for `bonsaid`.

The first backend version owns remote buffers and the first graph ops:

- it registers one PYNQ accelerator device;
- `ggml_backend_dev_memory()` reports the daemon allocator memory;
- ggml buffer allocation creates one daemon tensor handle per ggml buffer arena;
- tensor init binds each ggml tensor to an offset inside its arena handle;
- tensor set/get/memset/clear use the runtime tensor RPCs;
- tensor views share the remote allocation and carry an offset;
- contiguous same-type ggml `CPY` nodes lower to runtime graph version 1
  `COPY` ops and keep output resident until tensor get.
- contiguous 2D `Q1_0 x F32 -> F32` ggml `MUL_MAT` nodes lower to the
  runtime `MATMUL_Q1A8` op. The runtime executes that boundary on the PS
  until the PL W1A8 kernel replaces it.
- contiguous F32 `ADD`, `MUL`, `SCALE`, `SiLU`, `SwiGLU`, and `RMS_NORM`
  nodes lower to PS glue ops so FFN/residual regions can stay resident
  between matmuls.

The daemon endpoint defaults to `127.0.0.1:50055`. Override it for a board
runtime with:

```sh
export PYNQ_HOST=pynq
export PYNQ_PORT=50055
```

From `tools/pynqz1`, the packaged smoke test requires a running daemon and
verifies `HELLO`, memory reporting, remote tensor allocation, upload/download,
a ggml view write, a ggml `CPY` graph, direct `MUL_MAT` lowering, and
`MUL_MAT` plus F32 glue through the ggml scheduler:

```sh
nix run .#pynq-backend-smoke
```

`bonsaid` can use a native PS shared library for `MATMUL_Q1A8` and the F32
glue ops when `PYNQ_PS_LIB` points at `libbonsai_ps.so`. Build it on the board
before starting the daemon:

```sh
nix run .#pynq-board -- build-native
nix run .#pynq-board -- daemon
```

For llama.cpp dynamic loading, use the packaged wrapper:

```sh
nix run .#llama-cli-pynq -- --help
```

Set `PYNQ_TRACE=1` on the host process when diagnosing model load or graph
placement. The backend writes flushed trace lines to stderr for device memory
queries, remote buffer reservations, tensor arena bindings,
uploads/downloads with cumulative byte counts, and each lowered graph op or
`RUN_GRAPH` call:

```sh
PYNQ_TRACE=1 nix run .#llama-cli-pynq -- --list-devices
```

Set `PYNQ_PROFILE=1` on the board daemon to emit one compact JSON line per
`RUN_GRAPH` with per-op `read_us`, `compute_us`, `write_us`, byte counts, and
shape fields. Native PS calls also include `native_marshal_us` and
`native_kernel_us`, so matmul/kernel time can be separated from ctypes buffer
setup. This is intended for timing hot graph regions without the full host-side
trace stream.
