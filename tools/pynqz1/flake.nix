{
  description = "PYNQ-Z1 Bonsai accelerator development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
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
            pkgs.openssh
            pkgs.rsync
            pkgs.iperf3
            pkgs.tcl
          ];

          buildTools = [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.llvmPackages.clang
            pkgs.clang-tools
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

          mkTool = name: script:
            pkgs.writeShellApplication {
              inherit name;
              runtimeInputs = [ python ];
              text = ''
                exec "${python}/bin/python" "${src}/${script}" "$@"
              '';
            };
        in
        {
          pynqctl = mkTool "pynqctl" "tools/pynqctl.py";
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
          default = self.apps.${system}.pynqctl;
        });

      checks = forAllSystems (pkgs:
        let
          python = pkgs.python3.withPackages (ps: [
            ps.numpy
            ps.pytest
          ]);

          pythonFiles = [
            "tools/alloc_probe.py"
            "tools/__init__.py"
            "tools/mem_bandwidth.py"
            "tools/pynqctl.py"
            "dma_loopback/dma_bandwidth.py"
            "runtime/__init__.py"
            "runtime/allocator.py"
            "runtime/bonsai_rpc.py"
            "runtime/bonsaid.py"
            "tests/conftest.py"
            "tests/test_pynqctl.py"
            "tests/test_rpc.py"
          ];

          compileTargets =
            pkgs.lib.concatMapStringsSep " \\\n                "
              (path: ''"$src/${path}"'')
              pythonFiles;
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
              nativeBuildInputs = [ python ];
            }
            ''
              cp -R "$src" source
              chmod -R u+w source
              cd source
              export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
              export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
              python -m pytest -q tests
              touch "$out"
            '';
        });

      formatter = forAllSystems (pkgs: pkgs.nixpkgs-fmt);
    };
}
