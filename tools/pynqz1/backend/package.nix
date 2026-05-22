{
  lib,
  stdenv,
  cmake,
  ninja,
  nlohmann_json,
  writeShellScriptBin,
  symlinkJoin,
  llamaCppSrc,
}:

let
  commonLlamaFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DGGML_BACKEND_DL=ON"
    "-DLLAMA_BUILD_COMMON=ON"
    "-DLLAMA_BUILD_TOOLS=ON"
    "-DLLAMA_BUILD_TESTS=OFF"
    "-DLLAMA_BUILD_EXAMPLES=OFF"
    "-DLLAMA_BUILD_SERVER=ON"
    "-DLLAMA_CURL=OFF"
    "-DLLAMA_TOOLS_INSTALL=OFF"
    "-DGGML_METAL=OFF"
    "-DGGML_ACCELERATE=OFF"
    "-DGGML_BLAS=OFF"
  ];

  dynamicLibraryPathVar =
    if stdenv.hostPlatform.isDarwin then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH";

  llama-cpp-dl = stdenv.mkDerivation {
    pname = "pynq-llama-cpp-dl";
    version = "pinned";
    src = llamaCppSrc;

    nativeBuildInputs = [
      cmake
      ninja
    ];

    cmakeFlags = commonLlamaFlags;

    buildPhase = ''
      runHook preBuild
      cmake --build . --target llama-cli
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall

      mkdir -p "$out/bin" "$out/lib" "$out/include/ggml"
      cp bin/llama-cli "$out/bin/"

      if [ -e bin/libggml-cpu.so ]; then
        cp -P bin/libggml-cpu.so "$out/bin/"
      fi

      find bin -maxdepth 1 \( -name 'lib*.so*' -o -name 'lib*.dylib' \) \
        ! -name 'libggml-cpu.so' \
        -exec cp -P {} "$out/lib/" \;

      cp -R "$src/ggml/include/." "$out/include/ggml/"

      runHook postInstall
    '';

    meta = {
      description = "Pinned llama.cpp host build with dynamic ggml backends for PYNQ";
      platforms = lib.platforms.darwin ++ lib.platforms.linux;
    };
  };

  pynq-backend = stdenv.mkDerivation {
    pname = "ggml-pynq-backend";
    version = "local";
    src = ./.;

    nativeBuildInputs = [
      cmake
      ninja
    ];

    buildInputs = [
      nlohmann_json
    ];

    cmakeFlags = [
      "-DLLAMA_CPP_DIR=${llamaCppSrc}"
      "-DLLAMA_CPP_BUILD_DIR=${llama-cpp-dl}"
      "-DPYNQ_BUILD_SMOKE_TESTS=ON"
    ];

    installPhase = ''
      runHook preInstall

      mkdir -p "$out/bin" "$out/lib"
      cp backend/libggml-pynq.so "$out/lib/"
      cp test/pynq-backend-smoke "$out/bin/"

      runHook postInstall
    '';

    passthru = {
      inherit llama-cpp-dl;
    };

    meta = {
      description = "Out-of-tree ggml backend that stores tensors in bonsaid";
      platforms = lib.platforms.darwin ++ lib.platforms.linux;
    };
  };

  llama-cli-pynq-unwrapped = writeShellScriptBin "llama-cli-pynq" ''
    set -e
    export GGML_BACKEND_PATH="${pynq-backend}/lib/libggml-pynq.so"
    export ${dynamicLibraryPathVar}="${llama-cpp-dl}/lib:${llama-cpp-dl}/bin:${pynq-backend}/lib''${${dynamicLibraryPathVar}:+:''${${dynamicLibraryPathVar}}}"
    exec "${llama-cpp-dl}/bin/llama-cli" "$@"
  '';

  llama-cli-pynq = symlinkJoin {
    name = "llama-cli-pynq";
    paths = [ llama-cli-pynq-unwrapped ];
    meta.mainProgram = "llama-cli-pynq";
    passthru = {
      inherit llama-cpp-dl pynq-backend;
    };
  };
in
{
  inherit llama-cpp-dl pynq-backend llama-cli-pynq;
  default = llama-cli-pynq;
}
