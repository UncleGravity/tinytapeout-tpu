{
  description = "ggml Bonsai out-of-tree backend for llama.cpp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/a95a11e5b834057e684712963f90bbb730f4745c";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, llama-cpp-src }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-darwin"
        "x86_64-linux"
        "aarch64-linux"
      ];
    in {
      packages = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          bonsai = pkgs.callPackage ./package.nix {
            llamaCppSrc = llama-cpp-src;
          };
        in {
          inherit (bonsai) llama-cpp-dl bonsai-backend llama-cli-bonsai;
          default = bonsai.default;
        }
      );
    };
}
