Flow

llama-server
- (me) run with device = pynq and custom .so
- build ggml cgraph
- scheduler assigns each node to cpu / pynq / other backend
    - pynq-backend decides which OPs to handle (supports_op) and which ones to offload (offload_op)
- scheduler groups adjacent nodes with same backend into graph splits
    - for each split:
       copy required input tensors into that backend if needed
       call backend->graph_compute(split)
       copy outputs later only if another backend needs them
- Translate GGML splits graph into generic pynq tpu shape
- Send lowered graph to pynqz1

-------

I would treat the current TinyTapeout backend as a v0 proof, not as the shape to stretch. It proves ggml dynamic backend loading, op claiming, and Q1A8 correctness, but it is still host-memory based: `set/get/cpy` are null, the device buffer type is CPU, and `matmul.cpp` dereferences host tensor data directly. See [ggml-bonsai.cpp](/Users/angel/Documents/asic/tt-tpu/tools/bonsai_backend/src/ggml-bonsai.cpp:207), [ggml-bonsai.cpp](/Users/angel/Documents/asic/tt-tpu/tools/bonsai_backend/src/ggml-bonsai.cpp:312), and [matmul.cpp](/Users/angel/Documents/asic/tt-tpu/tools/bonsai_backend/src/matmul.cpp:154).

For PYNQ v1, I’d make the first milestone an architecture-correct device backend, even if it is not fast yet.

**V1 Shape**

`libggml-pynq.so` should expose a real non-host ggml buffer type. When llama.cpp loads the model, weights selected for PYNQ are allocated as PYNQ tensors, uploaded once, and tracked by handles. The board daemon owns the physical addresses and slab allocator; the host only sees tensor handles.

The host-visible RPC should stay small:

```text
HELLO -> abi, overlay id, memory, capabilities
ALLOC_TENSOR(shape, type, usage, layout_hint) -> tensor_handle
UPLOAD_TENSOR(handle, offset, bytes, flags)
RUN_GRAPH(op_list) -> status, counters
DOWNLOAD_TENSOR(handle, offset, size) -> bytes
FREE_TENSOR(handle)
```

`RUN_GRAPH` should be a high-level lowered op list, not raw ggml structs and not TT-style per-tile commands. For v1:

```text
VIEW / RESHAPE / PERMUTE / TRANSPOSE  metadata only
COPY / CONT                           board-side copy
MATMUL_Q1A8                           resident Q1_0 weights, board Q8 activations, F32 output
RMS_NORM, ADD, MUL, SCALE, SILU        PS kernels first
ROPE, GET_ROWS, KV update              next PS kernels
```

The `MATMUL_Q1A8` op should reference tensor handles and carry logical shape/stride info. Internally, `bonsaid` can quantize activations to Q8_0, submit PL work, then either let PL write final F32 or initially let PS fold the `d_w * d_a * sub_sum` result. The current host lowerer already encodes the important Q1/Q8 grouping rules: Q1 groups of 128, Q8 blocks of 32, and scale folding order in [matmul.cpp](/Users/angel/Documents/asic/tt-tpu/tools/bonsai_backend/src/matmul.cpp:39).

**Call Flow**

1. User runs llama with `GGML_BACKEND_PATH=.../libggml-pynq.so --device PYNQ`.
2. Backend connects to `bonsaid`, does `HELLO`, reports real memory.
3. ggml allocates PYNQ buffers. Backend returns fake host pointers for ggml bookkeeping, but never dereferences them.
4. Model load calls tensor upload hooks. Q1_0 weights are repacked into board-native layout during upload.
5. llama builds a cgraph for prompt/decode.
6. ggml scheduler assigns nodes. Large splits only happen if PYNQ supports the glue ops between matmuls; matmul-only support will still produce small splits.
7. `graph_compute` lowers the PYNQ split into `RUN_GRAPH`.
8. `bonsaid` executes PS metadata/glue ops and PL matmuls against DDR-resident tensors.
9. Host downloads only boundary tensors the CPU needs, ideally just logits.

**Implementation Order**

1. Build `bonsaid` plus RPC first, with slab allocation from several 8-32 MiB CMA buffers. Your memory notes make this important: `cma=320M` gives about 296 MiB usable, while full Q1_0 weights are about 236 MiB, leaving roughly 60 MiB for everything else ([memory.txt](/Users/angel/Documents/asic/tt-tpu/tools/pynqz1/docs/memory.txt:17), [memory.txt](/Users/angel/Documents/asic/tt-tpu/tools/pynqz1/docs/memory.txt:28)).
2. Implement the ggml PYNQ buffer type and upload/download path before PL acceleration.
3. Add a fake/reference board-side `MATMUL_Q1A8` path to validate handles, scheduling, and graph lowering.
4. Replace that matmul with one PL command over DDR-resident tensors.
5. Add PS glue ops until scheduler splits become layer-ish instead of matmul-ish.
6. Only then scale the array, add double buffering, more HP ports, command rings, and compiled graph plans.

The key design constraint: do not let physical addresses or tile microcode leak into `libggml-pynq.so`. The backend compiles ggml into device ops; `bonsaid` lowers device ops into board-private descriptors; PL consumes only board-private physical descriptors. That keeps the black box clean enough to evolve from “one correct GEMV” to “whole layer per token” without rewriting the llama.cpp-facing contract.
