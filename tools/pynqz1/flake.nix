{
  description = "PYNQ-Z1 Bonsai accelerator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/a95a11e5b834057e684712963f90bbb730f4745c";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, llama-cpp-src }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs   = nixpkgs.legacyPackages.${system};
        python = pkgs.python3.withPackages (ps: [
          ps.numpy ps.pytest
          (ps.cocotb.overridePythonAttrs (o: {
            meta = o.meta // { broken = false; };
            # ghdl has no aarch64-darwin build; we only run Verilog sims anyway.
            doCheck = false;
            nativeCheckInputs = [];
          }))
        ]);

        backend = pkgs.callPackage ./host/backend/package.nix {
          llamaCppSrc = llama-cpp-src;
        };

        # `deploy` always needs samba rsync. macOS's openrsync
        # silently ignores --chmod and leaves the board tree read-only.
        transportTools = [ pkgs.rsync ];

        # One recipe for every Python CLI. ``meta.mainProgram`` lets
        # `nix run .#name` find the binary without an apps entry.
        mkPyTool = name: script: extra: pkgs.writeShellApplication {
          inherit name;
          runtimeInputs = [ python ] ++ extra;
          text = ''exec "${python}/bin/python" "${./.}/${script}" "$@"'';
          meta.mainProgram = name;
        };

        # Tag a C++-built derivation with mainProgram so `nix run .#X` picks it up.
        withMainProgram = drv: name: drv // {
          meta = (drv.meta or {}) // { mainProgram = name; };
        };
      in {
        packages = rec {
          # User-facing entry points. Short, memorable names.
          pynqctl = mkPyTool "pynqctl"      "host/cli/pynqctl.py"      [];
          deploy  = mkPyTool "pynq-deploy"  "host/cli/deploy.py"       transportTools;
          profile = mkPyTool "pynq-profile" "host/cli/pynq-profile.py" [];

          # `nix run .#daemon` — deploy daemon with canonical sizing.
          #
          # heap-mib 288 leaves room for:
          #   - 231 MiB Bonsai-1.7B Q1_0 weights
          #   - ~14 MiB KV cache (ctx=128, 28 layers, F16)
          #   - ~9 MiB activation/compute scratch
          #   - alignment/fragmentation headroom (each slab loses ≤ slab-mib
          #     worth at allocation boundaries)
          # Stays under the ~296 MiB CMA budget on cma=320M boards.
          # Extra args are forwarded and (because argparse takes the last
          # value) override the baked-in flags:
          #   `nix run .#daemon -- --heap-mib 200`
          daemon = pkgs.writeShellApplication {
            name = "pynq-daemon";
            text = ''exec ${pkgs.lib.getExe deploy} daemon --heap-mib 270 --slab-mib 32 "$@"'';
            meta.mainProgram = "pynq-daemon";
          };

          # `nix run .#bench` — daemon + llama-cli + profile pull + summary.
          bench = pkgs.writeShellApplication {
            name = "pynq-bench";
            runtimeInputs = [
              daemon pynqctl profile llama
              pkgs.openssh pkgs.coreutils
            ];
            text = builtins.readFile ./host/cli/bench.sh;
            meta.mainProgram = "pynq-bench";
          };

          # C++ binaries surfaced for `nix run`.
          llama = withMainProgram backend.llama-cli-pynq "llama-cli-pynq";
          smoke = withMainProgram backend.pynq-backend   "pynq-backend-smoke";

          # Lower-level building blocks (rarely run directly).
          inherit (backend) llama-cpp-dl pynq-backend;

          default = pynqctl;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.cmake pkgs.ninja pkgs.pkg-config pkgs.llvmPackages.clang
            pkgs.clang-tools pkgs.nlohmann_json
            pkgs.ruff
            pkgs.graphviz pkgs.plantuml
            pkgs.verilator
            pkgs.gnumake pkgs.tcl pkgs.iperf3
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux (transportTools ++ [ pkgs.ethtool ]);

          shellHook = ''
            export PATH="${self.packages.${system}.pynqctl}/bin:${self.packages.${system}.deploy}/bin:${self.packages.${system}.profile}/bin:$PATH"
            echo "pynq devshell ready."
            echo "  pynqctl, pynq-deploy, pynq-profile on PATH"
            echo "  PYNQ_HOST + PYNQ_PORT set the daemon target"
            echo "  PYNQ_TRACE=path / PYNQ_PROFILE=path enable telemetry"
          '';
        };

        checks = {
          # Compile every tracked .py file. Auto-discovered.
          python-syntax = pkgs.runCommand "pynqz1-python-syntax" {
            src = ./.; nativeBuildInputs = [ python ];
          } ''
            export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
            find "$src" -name '*.py' -not -path '*/__pycache__/*' \
              -print0 | xargs -0 python -m py_compile
            touch "$out"
          '';

          # ruff against the whole tree.
          lint = pkgs.runCommand "pynqz1-lint" {
            src = ./.; nativeBuildInputs = [ pkgs.ruff ];
          } ''
            cp -R "$src" source && chmod -R u+w source && cd source
            ruff check .
            touch "$out"
          '';

          # Full pytest pass — unit + e2e. Builds libbonsai_ps.so via its Makefile.
          tests = pkgs.runCommand "pynqz1-tests" {
            src = ./.; nativeBuildInputs = [ python pkgs.gnumake pkgs.stdenv.cc ];
          } ''
            cp -R "$src" source && chmod -R u+w source && cd source
            export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
            export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
            python -m pytest -q
            touch "$out"
          '';

          # C++ backend correctness against a spawned daemon.
          # All spawn/poll logic lives in tests/e2e/run_backend_smoke.sh
          # so it's runnable locally without nix.
          backend-smoke = pkgs.runCommand "pynqz1-backend-smoke" {
            src = ./.;
            nativeBuildInputs = [ python pkgs.gnumake pkgs.stdenv.cc backend.pynq-backend ];
          } ''
            cp -R "$src" source && chmod -R u+w source && cd source
            make -C board/kernels/ps OUT_DIR=. CC="$CC"
            export PYNQ_HOST=127.0.0.1
            export PYNQ_PORT=50055
            bash tests/e2e/run_backend_smoke.sh
            touch "$out"
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      });
}
