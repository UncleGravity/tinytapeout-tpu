{
  description = "PYNQ-Z1 Bonsai accelerator development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/a95a11e5b834057e684712963f90bbb730f4745c";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, llama-cpp-src }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f (import nixpkgs { inherit system; }));
    in
    {
      devShells = forAllSystems (pkgs:
        let
          python = pkgs.python3.withPackages (ps: [
            ps.numpy
            ps.pytest
          ]);

          hostTools = [
            pkgs.git
            pkgs.gnumake
            pkgs.iperf3
            pkgs.tcl
          ];

          buildTools = [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.llvmPackages.clang
            pkgs.clang-tools
            pkgs.nlohmann_json
            pkgs.ruff
          ];

          docsTools = [
            pkgs.graphviz
            pkgs.plantuml
          ];

          hdlTools = [
            pkgs.verilator
          ];

          linuxOnlyTools = pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.ethtool
            pkgs.openssh
            pkgs.rsync
          ];
        in
        {
          default = pkgs.mkShell {
            packages =
              [ python ]
              ++ hostTools
              ++ buildTools
              ++ docsTools
              ++ hdlTools
              ++ linuxOnlyTools;

            shellHook = ''
              echo "PYNQ-Z1 dev shell"
              echo "PYNQ itself is board-side; run PYNQ scripts on the board Python environment."
            '';
          };
        });

      packages = forAllSystems (pkgs:
        let
          system = pkgs.stdenv.hostPlatform.system;
          src = ./.;
          python = pkgs.python3.withPackages (ps: [
            ps.numpy
            ps.pytest
          ]);

          boardTransportTools = pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.openssh
            pkgs.rsync
          ];
          pynqBackend = pkgs.callPackage ./host/backend/package.nix {
            llamaCppSrc = llama-cpp-src;
          };
          bonsaiPsNative = pkgs.stdenv.mkDerivation {
            pname = "bonsai-ps-native";
            version = "0.1.0";
            src = ./runtime/native;
            dontConfigure = true;
            buildPhase =
              let
                sharedFlag = if pkgs.stdenv.isDarwin then "-dynamiclib" else "-shared";
              in
              ''
                runHook preBuild
                $CC -O3 -std=c99 -fPIC ${sharedFlag} \
                  -o libbonsai_ps.so bonsai_ps.c -lm
                runHook postBuild
              '';
            installPhase = ''
              runHook preInstall
              mkdir -p "$out/lib"
              cp libbonsai_ps.so "$out/lib/"
              runHook postInstall
            '';
          };

          mkTool = name: script: toolInputs:
            pkgs.writeShellApplication {
              inherit name;
              runtimeInputs = [ python ] ++ toolInputs;
              text = ''
                exec "${python}/bin/python" "${src}/${script}" "$@"
              '';
            };
        in
        {
          pynqctl = mkTool "pynqctl" "tools/pynqctl.py" [ ];
          pynq-board = mkTool "pynq-board" "tools/pynq_board.py" boardTransportTools;
          bonsai-ps-native = bonsaiPsNative;
          inherit (pynqBackend) llama-cpp-dl llama-cli-pynq pynq-backend;
          default = self.packages.${system}.pynqctl;
        });

      apps = forAllSystems (pkgs:
        let
          system = pkgs.stdenv.hostPlatform.system;
        in
        {
          pynqctl = {
            type = "app";
            program = "${self.packages.${system}.pynqctl}/bin/pynqctl";
            meta.description = "Control a PYNQ-Z1 bonsaid runtime";
          };
          pynq-board = {
            type = "app";
            program = "${self.packages.${system}.pynq-board}/bin/pynq-board";
            meta.description = "Deploy and exercise bonsaid on a PYNQ board";
          };
          llama-cli-pynq = {
            type = "app";
            program = "${self.packages.${system}.llama-cli-pynq}/bin/llama-cli-pynq";
            meta.description = "Run llama-cli with libggml-pynq available";
          };
          pynq-backend-smoke = {
            type = "app";
            program = "${self.packages.${system}.pynq-backend}/bin/pynq-backend-smoke";
            meta.description = "Round-trip ggml tensors through bonsaid";
          };
          default = self.apps.${system}.pynqctl;
        });

      checks = forAllSystems (pkgs:
        let
          system = pkgs.stdenv.hostPlatform.system;
          python = pkgs.python3.withPackages (ps: [
            ps.numpy
            ps.pytest
          ]);

          pythonFiles = [
            "tools/__init__.py"
            "tools/alloc_probe.py"
            "tools/mem_bandwidth.py"
            "host/__init__.py"
            "host/cli/__init__.py"
            "host/cli/deploy.py"
            "host/cli/pynqctl.py"
            "host/transport/__init__.py"
            "host/transport/client.py"
            "proto/__init__.py"
            "proto/framing.py"
            "proto/ops.py"
            "proto/tests/__init__.py"
            "proto/tests/test_parity.py"
            "runtime/__init__.py"
            "runtime/allocator.py"
            "runtime/bonsaid.py"
            "runtime/graph.py"
            "runtime/ps_native.py"
            "tests/conftest.py"
            "tests/test_deploy.py"
            "tests/test_pynqctl.py"
            "tests/test_rpc.py"
          ];

          compileTargets =
            pkgs.lib.concatMapStringsSep " \\\n                "
              (path: ''"$src/${path}"'')
              pythonFiles;
          pynqBackend = pkgs.callPackage ./host/backend/package.nix {
            llamaCppSrc = llama-cpp-src;
          };
          bonsaiPsNative = self.packages.${system}.bonsai-ps-native;
        in
        {
          python-syntax = pkgs.runCommand "pynqz1-python-syntax"
            {
              src = ./.;
              nativeBuildInputs = [ python ];
            }
            ''
              export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
              python -m py_compile \
                ${compileTargets}
              touch "$out"
            '';

          runtime-tests = pkgs.runCommand "pynqz1-runtime-tests"
            {
              src = ./.;
              nativeBuildInputs = [ python bonsaiPsNative ];
            }
            ''
              cp -R "$src" source
              chmod -R u+w source
              cd source
              export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
              export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
              export PYNQ_PS_LIB="${bonsaiPsNative}/lib/libbonsai_ps.so"
              python -m pytest -q tests
              touch "$out"
            '';

          backend-smoke = pkgs.runCommand "pynqz1-backend-smoke"
            {
              src = ./.;
              nativeBuildInputs = [
                python
                pynqBackend.pynq-backend
              ];
            }
            ''
              export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
              export PYNQ_BONSAID_HOST=127.0.0.1
              export PYNQ_BONSAID_PORT=50055
              export PYNQ_PS_LIB="${bonsaiPsNative}/lib/libbonsai_ps.so"
              python "$src/runtime/bonsaid.py" \
                --host "$PYNQ_BONSAID_HOST" \
                --port "$PYNQ_BONSAID_PORT" \
                --allocator fake \
                --overlay none \
                --overlay-id backend-smoke \
                --heap-mib 8 \
                --slab-mib 1 > "$TMPDIR/bonsaid.log" 2>&1 &
              daemon_pid=$!
              trap 'kill "$daemon_pid" 2>/dev/null || true' EXIT

              attempts=0
              until pynq-backend-smoke; do
                attempts=$((attempts + 1))
                if [ "$attempts" -ge 30 ]; then
                  cat "$TMPDIR/bonsaid.log" >&2
                  exit 1
                fi
                sleep 0.1
              done
              touch "$out"
            '';
        });

      formatter = forAllSystems (pkgs: pkgs.nixpkgs-fmt);
    };
}
