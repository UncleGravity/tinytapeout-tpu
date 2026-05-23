# PYNQ ggml Backend

`libggml-pynq.so` is the host-side ggml boundary for `bonsaid`.

The first backend version owns remote buffers and the first graph ops:

- it registers one PYNQ accelerator device;
- `ggml_backend_dev_memory()` reports the daemon allocator memory;
- ggml buffer allocation creates daemon tensor handles;
- tensor set/get/memset/clear use the runtime tensor RPCs;
- tensor views share the remote allocation and carry an offset;
- contiguous same-type ggml `CPY` nodes lower to runtime graph version 1
  `COPY` ops and keep output resident until tensor get.
- contiguous 2D `Q1_0 x F32 -> F32` ggml `MUL_MAT` nodes lower to the
  runtime `MATMUL_Q1A8` op. The runtime executes that boundary on the PS
  until the PL W1A8 kernel replaces it.
- contiguous F32 `ADD`, `MUL`, `SCALE`, `SiLU`, and `RMS_NORM` nodes lower
  to PS glue ops so FFN/residual regions can stay resident between matmuls.

The daemon endpoint defaults to `127.0.0.1:50055`. Override it for a board
runtime with:

```sh
export PYNQ_BONSAID_HOST=pynq
export PYNQ_BONSAID_PORT=50055
```

From `tools/pynqz1`, the packaged smoke test requires a running daemon and
verifies `HELLO`, memory reporting, remote tensor allocation, upload/download,
a ggml view write, a ggml `CPY` graph, direct `MUL_MAT` lowering, and
`MUL_MAT` plus F32 glue through the ggml scheduler:

```sh
nix run .#pynq-backend-smoke
```

For llama.cpp dynamic loading, use the packaged wrapper:

```sh
nix run .#llama-cli-pynq -- --help
```

Set `PYNQ_TRACE=1` on the host process when diagnosing model load or graph
placement. The backend writes flushed trace lines to stderr for device memory
queries, remote buffer reservations, tensor handle allocations,
uploads/downloads with cumulative byte counts, and each lowered graph op or
`RUN_GRAPH` call:

```sh
PYNQ_TRACE=1 nix run .#llama-cli-pynq -- --list-devices
```
