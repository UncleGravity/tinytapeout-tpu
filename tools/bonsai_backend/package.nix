{
  lib,
  stdenv,
  cmake,
  ninja,
  verilator,
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
    "-DGGML_BONSAI=OFF"
  ];

  dynamicLibraryPathVar =
    if stdenv.hostPlatform.isDarwin then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH";

  llama-cpp-dl = stdenv.mkDerivation {
    pname = "bonsai-llama-cpp-dl";
    version = "pinned";
    src = llamaCppSrc;

    nativeBuildInputs = [
      cmake
      ninja
    ];

    cmakeFlags = commonLlamaFlags;

    buildPhase = ''
      runHook preBuild
      cmake --build .
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
      description = "llama.cpp host built with ggml dynamic backend loading for Bonsai";
      platforms = lib.platforms.darwin ++ lib.platforms.linux;
    };
  };

  bonsai-backend = stdenv.mkDerivation {
    pname = "ggml-bonsai-backend";
    version = "local";
    src = ./.;

    nativeBuildInputs = [
      cmake
      ninja
      verilator
    ];

    cmakeFlags = [
      "-DLLAMA_CPP_DIR=${llamaCppSrc}"
      "-DLLAMA_CPP_BUILD_DIR=${llama-cpp-dl}"
    ];

    installPhase = ''
      runHook preInstall

      mkdir -p "$out/lib"
      cp bin/libggml-bonsai.so "$out/lib/"

      runHook postInstall
    '';

    meta = {
      description = "Out-of-tree ggml Bonsai backend module";
      platforms = lib.platforms.darwin ++ lib.platforms.linux;
    };
  };

  bonsai-llama-cli-unwrapped = writeShellScriptBin "bonsai-llama-cli" ''
    set -e
    export GGML_BACKEND_PATH="${bonsai-backend}/lib/libggml-bonsai.so"
    export ${dynamicLibraryPathVar}="${llama-cpp-dl}/lib:${llama-cpp-dl}/bin:${bonsai-backend}/lib''${${dynamicLibraryPathVar}:+:''${${dynamicLibraryPathVar}}}"
    exec "${llama-cpp-dl}/bin/llama-cli" "$@"
  '';

  bonsai-llama-cli = symlinkJoin {
    name = "bonsai-llama-cli";
    paths = [ bonsai-llama-cli-unwrapped ];
    passthru = {
      inherit llama-cpp-dl bonsai-backend;
    };
  };
in
{
  inherit llama-cpp-dl bonsai-backend bonsai-llama-cli;
  default = bonsai-llama-cli;
}
