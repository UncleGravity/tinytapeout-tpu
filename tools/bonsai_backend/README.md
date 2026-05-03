# Bonsai llama.cpp Backend

This directory contains an out-of-tree `ggml-bonsai` backend. The backend
registers as a fake accelerator, accepts host buffers, and offloads
`GGML_OP_MUL_MAT` nodes whose inputs can be converted to F32 and whose output
is F32.

Non-matmul ops stay on the regular CPU backend through ggml's scheduler. Matmul splits assigned to Bonsai are computed by the Bonsai module.

## Out-of-tree dynamic backend

For manual CMake development, the loop is:

1. Build llama.cpp with dynamic backend loading enabled, but without compiling Bonsai into llama.cpp.
2. Build Bonsai as a separate `MODULE` from this directory.
3. Point llama.cpp at the module with `GGML_BACKEND_PATH`.

```sh
nix develop -c cmake -S tools/bonsai_backend/llama.cpp \
  -B tools/bonsai_backend/llama.cpp/build-dl \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_CURL=OFF \
  -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_BONSAI=OFF

nix develop -c cmake --build tools/bonsai_backend/llama.cpp/build-dl --target llama-cli

nix develop -c cmake -S tools/bonsai_backend \
  -B tools/bonsai_backend/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_BUILD_DIR=$PWD/tools/bonsai_backend/llama.cpp/build-dl \
  -DBONSAI_BACKEND_SOURCE=$PWD/tools/bonsai_backend/src/ggml-bonsai.cpp

nix develop -c cmake --build tools/bonsai_backend/build
```

Verify the dynamic loader can see Bonsai:

```sh
nix develop -c env \
  GGML_BACKEND_PATH=$PWD/tools/bonsai_backend/build/bin/libggml-bonsai.so \
  tools/bonsai_backend/llama.cpp/build-dl/bin/llama-cli --list-devices
```

Expected signal:

```text
load_backend: loaded Bonsai backend from ...
Available devices:
  Bonsai: Bonsai fake accelerator (0 MiB, 0 MiB free)
```

Run a model with:

```sh
nix develop -c env \
  GGML_BACKEND_PATH=$PWD/tools/bonsai_backend/build/bin/libggml-bonsai.so \
  BONSAI_TRACE_MATMUL=1 BONSAI_TRACE_LIMIT=8 \
  tools/bonsai_backend/llama.cpp/build-dl/bin/llama-cli \
  -m /Users/angel/Documents/asic/tt-tpu/models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  -p Hello -n 2 -c 128 -b 16 -ub 16 -t 4 \
  --device Bonsai --temp 0 --no-warmup --single-turn --no-display-prompt --verbosity 2
```

The standalone module only links against `libggml-base`. It deliberately does
not link against `libggml-cpu`, because with `GGML_BACKEND_DL=ON` the CPU
backend is a loadable module itself on macOS.

## Nix package

The root flake exposes:

```sh
nix build .#bonsai-llama-cpp-dl
nix build .#bonsai-backend
nix build .#llama-cli-bonsai
```

`.#llama-cli-bonsai` is a wrapper that sets `GGML_BACKEND_PATH` and the dynamic
library search path before execing the matched `llama-cli` build:

```sh
nix run .#llama-cli-bonsai -- \
  -m $PWD/models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  -p Hello -n 2 -c 128 -b 16 -ub 16 -t 4 \
  --device Bonsai \
  --temp 0 --no-warmup --single-turn --no-display-prompt --verbosity 2
```

The package requires a pinned llama.cpp source from the top-level flake:

```nix
inputs.llama-cpp-src = {
  url = "github:ggml-org/llama.cpp/<commit>";
  flake = false;
};

bonsai = pkgs.callPackage ./tools/bonsai_backend/package.nix {
  llamaCppSrc = llama-cpp-src;
};
```

Because the package has no `./llama.cpp` fallback, `nix build
.#llama-cli-bonsai` does not depend on a local checkout or submodule. Newly
added Bonsai package files still need to be tracked or staged so flakes can see
them.

## In-tree build

```sh
nix develop -c cmake -S tools/bonsai_backend/llama.cpp \
  -B tools/bonsai_backend/llama.cpp/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_CURL=OFF \
  -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_BONSAI=ON

nix develop -c cmake --build tools/bonsai_backend/llama.cpp/build --target llama-cli
```
