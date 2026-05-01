{
  description = "Tiny Tapeout verilog development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    librelane.url = "github:librelane/librelane";
    verilog-viewer.url = "github:UncleGravity/verilog-viewer";
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/a95a11e5b834057e684712963f90bbb730f4745c";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, librelane, verilog-viewer, llama-cpp-src }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-darwin"
        "x86_64-linux"
        "aarch64-linux"
      ];

      # Wrap a shell command as a flake app that re-enters the devshell so the
      # user can run `nix run .#<name>` without first running `nix develop`.
      mkDevApp = pkgs: name: cmd: {
        type = "app";
        program = toString (pkgs.writeShellScript "tt-${name}" ''
          set -e
          exec nix develop --command bash -c ${pkgs.lib.escapeShellArg cmd}
        '');
      };
    in {
      apps = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          bonsai = pkgs.callPackage ./tools/bonsai_backend/package.nix {
            llamaCppSrc = llama-cpp-src;
          };
        in {
          test   = mkDevApp pkgs "test"   "cd test && make -B";
          harden = mkDevApp pkgs "harden" "./tt/tt_tool.py --create-user-config && ./tt/tt_tool.py --harden --no-docker";
          fpga   = mkDevApp pkgs "fpga"   "./tt/tt_fpga.py harden";
          check  = mkDevApp pkgs "check"  "./tt/tt_tool.py --check-docs";
          bonsai-llama-cli = {
            type = "app";
            program = "${bonsai.bonsai-llama-cli}/bin/bonsai-llama-cli";
          };
          view = {
            type = "app";
            program = toString (pkgs.writeShellScript "tt-view" ''
              set -e
              cd "$(git rev-parse --show-toplevel)"
              TOP=""
              if [ -f src/config_merged.json ]; then
                TOP=$(${pkgs.jq}/bin/jq -r '.DESIGN_NAME // empty' src/config_merged.json)
              fi
              exec ${verilog-viewer.packages.${system}.default}/bin/verilog-viewer \
                --rtl 'src/rtl/**/*.{v,sv}' \
                ''${TOP:+--top "$TOP"} \
                "$@"
            '');
          };
        }
      );

      packages = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          bonsai = pkgs.callPackage ./tools/bonsai_backend/package.nix {
            llamaCppSrc = llama-cpp-src;
          };
        in {
          bonsai-llama-cpp-dl = bonsai.llama-cpp-dl;
          bonsai-backend = bonsai.bonsai-backend;
          bonsai-llama-cli = bonsai.bonsai-llama-cli;
        }
      );

      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system}; # nixpkgs
          llPkgs = librelane.legacyPackages.${system}; # get librelane pkgs
          python = llPkgs.python3.withPackages (ps: [ ps.cocotb ]); # use librelane python
        in {
          default = pkgs.mkShell {

            # ------------------------------------------------------------------
            # DEPENDENCIES
            inputsFrom = [ librelane.devShells.${system}.default ];
            buildInputs = [
              pkgs.nextpnr
              pkgs.icestorm
              pkgs.pdk-ciel

              # Tiny Tapeout tools dependencies
              python
              pkgs.cairosvg
              pkgs.ggml
              pkgs.pkg-config
              pkgs.cmake
              pkgs.ninja

              # Testing
              pkgs.surfer
            ];

            # ------------------------------------------------------------------
            # SHELL HOOK - runs after calling `nix develop`
            shellHook = ''
              FLAKE_ROOT=$(git rev-parse --show-toplevel)

              # Clone tt-support-tools if not present
              if [ ! -d "$FLAKE_ROOT/tt/.git" ]; then
                echo "Cloning tt-support-tools..."
                git clone https://github.com/TinyTapeout/tt-support-tools.git "$FLAKE_ROOT/tt"
              fi

              # Install tt deps
              if [ ! -d "$FLAKE_ROOT/tt/.venv" ]; then
                echo "Installing tt-support-tools python dependencies..."
                uv venv --project="$FLAKE_ROOT/tt" --python=${python} # Create venv
                uv --project="$FLAKE_ROOT/tt" sync --python=${python} # Install python dependencies
              fi
              source "$FLAKE_ROOT/tt/.venv/bin/activate"

              # Install sky130A PDK if not present
              export PDK_ROOT="$FLAKE_ROOT/.pdk"
              if [ ! -d "$PDK_ROOT/sky130A" ]; then
                echo "Installing sky130A PDK..."
                ciel enable --pdk-family sky130 8afc8346a57fe1ab7934ba5a6056ea8b43078e71
              fi

              # Ensure Nix-managed Python packages are available in subshells
              export PYTHONPATH="''${NIX_PYTHONPATH:-}''${PYTHONPATH:+:$PYTHONPATH}"

            '';
            # ------------------------------------------------------------------
          };
        }
      );
    };
}
