# PYNQ ggml Backend

`libggml-pynq.so` is the host-side ggml boundary for `bonsaid`.

The first backend version is buffer-only:

- it registers one PYNQ accelerator device;
- `ggml_backend_dev_memory()` reports the daemon allocator memory;
- ggml buffer allocation creates daemon tensor handles;
- tensor set/get/memset/clear use the runtime tensor RPCs;
- tensor views share the remote allocation and carry an offset;
- contiguous same-type ggml `CPY` nodes lower to runtime graph version 1
  `COPY` ops and keep output resident until tensor get.

The daemon endpoint defaults to `127.0.0.1:50055`. Override it for a board
runtime with:

```sh
export PYNQ_BONSAID_HOST=pynq
export PYNQ_BONSAID_PORT=50055
```

From `tools/pynqz1`, the packaged smoke test requires a running daemon and
verifies `HELLO`, memory reporting, remote tensor allocation, upload/download,
a ggml view write, and a ggml `CPY` graph lowered through `RUN_GRAPH`:

```sh
nix run .#pynq-backend-smoke
```

For llama.cpp dynamic loading, use the packaged wrapper:

```sh
nix run .#llama-cli-pynq -- --help
```
