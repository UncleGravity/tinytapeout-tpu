# PYNQ-Z1 Runtime Protocol

This is the host-visible protocol between `libggml-pynq.so` and `bonsaid`.
It is intentionally independent of ggml structs, PYNQ Python objects, and PL
physical addresses.

## Transport

The first implementation uses one TCP connection per client. Requests and
responses share the same frame format:

```text
u8  magic[4]       "BPNQ"
u16 version        1
u16 flags          0 for now
u32 json_len       big endian
u32 payload_len    big endian
u8  json[json_len] UTF-8 JSON object
u8  payload[payload_len]
```

JSON carries control metadata. Payload carries bulk tensor bytes.

Successful responses:

```json
{
  "id": 1,
  "ok": true,
  "result": {}
}
```

Error responses:

```json
{
  "id": 1,
  "ok": false,
  "error": {
    "code": "out_of_bounds",
    "message": "range [8, 24) exceeds tensor size 16"
  }
}
```

## Commands

### `HELLO`

Probe ABI, memory, and current capabilities.

Request:

```json
{ "id": 1, "op": "HELLO" }
```

Response result:

```json
{
  "abi_version": 1,
  "server": "bonsaid",
  "overlay_id": "fake-local",
  "memory": {
    "total_bytes": 67108864,
    "free_bytes": 67108864,
    "used_bytes": 0,
    "slab_count": 2,
    "tensor_count": 0
  },
  "capabilities": [
    "ALLOC_TENSOR",
    "UPLOAD_TENSOR",
    "DOWNLOAD_TENSOR",
    "FREE_TENSOR",
    "RUN_GRAPH"
  ],
  "graph_ops": [
    "COPY",
    "MATMUL_Q1A8",
    "ADD_F32",
    "MUL_F32",
    "SCALE_F32",
    "SILU_F32",
    "RMS_NORM_F32"
  ]
}
```

### `MEMORY`

Return current allocator accounting.

Request:

```json
{ "id": 2, "op": "MEMORY" }
```

### `ALLOC_TENSOR`

Allocate board-resident tensor storage and return an opaque handle.

Request:

```json
{
  "id": 3,
  "op": "ALLOC_TENSOR",
  "nbytes": 3145728,
  "shape": [2048, 1536],
  "dtype": "q1_0",
  "usage": "weight",
  "layout": "raw",
  "alignment": 64
}
```

Response result:

```json
{
  "tensor": {
    "handle": 1,
    "nbytes": 3145728,
    "shape": [2048, 1536],
    "dtype": "q1_0",
    "usage": "weight",
    "layout": "raw",
    "extent_count": 1
  }
}
```

Handles are the only addresses exposed to the host. `bonsaid` may represent a
single tensor with one or more board-private DDR extents.

### `UPLOAD_TENSOR`

Write payload bytes into an allocated tensor.

Request metadata:

```json
{ "id": 4, "op": "UPLOAD_TENSOR", "handle": 1, "offset": 0 }
```

The frame payload is copied into the tensor at `offset`.

### `DOWNLOAD_TENSOR`

Read tensor bytes back to the host.

Request:

```json
{ "id": 5, "op": "DOWNLOAD_TENSOR", "handle": 1, "offset": 0, "size": 4096 }
```

The response payload contains the requested bytes.

### `FREE_TENSOR`

Free a tensor handle and return updated memory accounting.

Request:

```json
{ "id": 6, "op": "FREE_TENSOR", "handle": 1 }
```

### `RUN_GRAPH`

Run a lowered device op list against board-resident tensors. Graphs reference
tensor handles instead of ggml structs or physical addresses. The first graph
version is PS-executed and supports `COPY`, the reference form of the
resident-weight matmul that the PL path will replace, and a small set of F32
glue ops used to keep FFN/residual graph regions resident.

Request:

```json
{
  "id": 7,
  "op": "RUN_GRAPH",
  "graph_version": 1,
  "ops": [
    {
      "op": "COPY",
      "src": 1,
      "dst": 2,
      "nbytes": 4096,
      "src_offset": 0,
      "dst_offset": 0
    }
  ],
  "outputs": [2]
}
```

`COPY` reads `nbytes` from `src` at optional `src_offset` and writes those
bytes to `dst` at optional `dst_offset`. `bonsaid` validates handles and tensor
ranges before copying.

`MATMUL_Q1A8` reads contiguous 2D ggml `Q1_0` weights and contiguous F32
activations from resident tensor handles, quantizes each activation row with
ggml Q8_0 block rules on the PS, and writes a contiguous F32 ggml `MUL_MAT`
output. It is the first stable graph boundary for the PL W1A8 kernel:

```json
{
  "op": "MATMUL_Q1A8",
  "weights": 3,
  "acts": 4,
  "dst": 5,
  "rows": 2048,
  "cols": 1,
  "k": 2048,
  "weights_offset": 0,
  "acts_offset": 0,
  "dst_offset": 0
}
```

`k` must be a positive multiple of the `Q1_0` block size, 128. `rows` is
`src0.ne[1]`, `cols` is `src1.ne[1]`, and output cells are written in ggml
order with rows contiguous inside each column.

`ADD_F32` and `MUL_F32` operate on contiguous F32 tensors. `src0` is always the
full output shape. `src1` is either the same shape or a row vector broadcast
across `cols` when `src1_broadcast` is true:

```json
{
  "op": "MUL_F32",
  "src0": 6,
  "src1": 7,
  "dst": 8,
  "rows": 2048,
  "cols": 16,
  "src1_broadcast": true,
  "src0_offset": 0,
  "src1_offset": 0,
  "dst_offset": 0
}
```

`SCALE_F32` writes `dst[i] = src[i] * scale + bias` for a contiguous F32 range.
`SILU_F32` writes `dst[i] = src[i] / (1 + exp(-src[i]))`.

```json
{
  "op": "SCALE_F32",
  "src": 8,
  "dst": 9,
  "elements": 32768,
  "scale": 0.5,
  "bias": 0.0,
  "src_offset": 0,
  "dst_offset": 0
}
```

`RMS_NORM_F32` treats the contiguous tensor as `cols` vectors of length `rows`
and normalizes each vector independently:

```json
{
  "op": "RMS_NORM_F32",
  "src": 10,
  "dst": 11,
  "rows": 2048,
  "cols": 16,
  "eps": 0.000001,
  "src_offset": 0,
  "dst_offset": 0
}
```

Response result:

```json
{
  "graph_version": 1,
  "op_count": 1,
  "outputs": [2],
  "counters": {
    "ps_ops": 1,
    "pl_ops": 0,
    "bytes_read": 4096,
    "bytes_written": 4096,
    "elapsed_us": 173
  }
}
```

`RUN_GRAPH` does not implicitly download outputs. The backend or host tool
chooses which output tensor bytes to retrieve with `DOWNLOAD_TENSOR`.
