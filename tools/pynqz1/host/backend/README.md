# PYNQ ggml Backend

`libggml-pynq.so` is the host-side ggml boundary for `bonsaid`.

The first backend version owns remote buffers and the first graph ops:

- it registers one PYNQ accelerator device;
- `ggml_backend_dev_memory()` reports the daemon allocator memory;
- ggml buffer allocation creates one daemon tensor handle per ggml buffer arena;
- tensor init binds each ggml tensor to an offset inside its arena handle;
- tensor set/get/memset/clear use the runtime tensor RPCs;
- tensor views share the remote allocation and carry an offset;
- contiguous ggml `CPY` nodes lower to runtime `COPY` ops (same-type byte
  copy, plus F32->F16 converting copies) and keep output resident.
- contiguous 2D `Q1_0 x F32 -> F32` ggml `MUL_MAT` nodes lower to the runtime
  `MATMUL_Q1A8` op, executed on the PL W1A8 kernel (PS fallback if no overlay).
- F32 glue ops lower to PS kernels so FFN/residual/attention regions stay
  resident between matmuls: `ADD`, `MUL`, `SCALE`, `SiLU`, `SwiGLU`,
  `RMS_NORM`, `ROPE`, `FLASH_ATTN_EXT`, `GET_ROWS`, `SET_ROWS`. An `RMS_NORM`
  immediately followed by its weight `MUL` is fused into one `RMS_NORM_MUL`.

The daemon endpoint defaults to `127.0.0.1:50055`. Override it for a board
runtime with:

```sh
export PYNQ_HOST=pynq
export PYNQ_PORT=50055
```

From `tools/pynqz1`, the `backend-smoke` flake check spawns its own daemon and
verifies `HELLO`, memory reporting, remote tensor allocation, upload/download,
a ggml view write, a ggml `CPY` graph, direct `MUL_MAT` lowering, and
`MUL_MAT` plus F32 glue through the ggml scheduler:

```sh
nix flake check        # or: nix build .#checks.<system>.backend-smoke
```

`bonsaid` can use a native PS shared library for `MATMUL_Q1A8` and the F32
glue ops when `PYNQ_PS_LIB` points at `libbonsai_ps.so`. Build it on the board
before starting the daemon:

```sh
nix run .#deploy -- build-native
nix run .#deploy -- daemon
```

For llama.cpp dynamic loading, use the packaged wrapper:

```sh
nix run .#llama -- --help
```

Set `PYNQ_TRACE=1` on the host process when diagnosing model load or graph
placement. The backend writes flushed trace lines to stderr for device memory
queries, remote buffer reservations, tensor arena bindings,
uploads/downloads with cumulative byte counts, and each lowered graph op or
`RUN_GRAPH` call:

```sh
PYNQ_TRACE=1 nix run .#llama -- --list-devices
```

Set `PYNQ_PROFILE` (`1` for stderr, or a path) on the board daemon to emit
NDJSON per-op spans with per-section `*_us` timings (read/compute/write/
quantize/run_chunk/…), byte counts, and shape fields, analysed by
`pynq-profile`. When `PYNQ_PROFILE` is unset the daemon skips per-op span/event
recording entirely (no profiling overhead) and only the RUN_GRAPH counters are
returned — so profile a hot run, then re-measure unprofiled for true tok/s.
