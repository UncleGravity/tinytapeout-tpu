{ pkgs ? import <nixpkgs> {} }:
let
  pico-sdk-full = pkgs.pico-sdk.override { withSubmodules = true; };
  # pyusb talks to the vendor-class rp_streamer firmware over libusb;
  # pyserial stays around for the TT-MicroPython REPL scaffold scripts.
  pyenv = pkgs.python3.withPackages (ps: [ ps.pyserial ps.pyusb ]);
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    # firmware build
    gcc-arm-embedded
    pico-sdk-full
    cmake
    ninja
    git

    # host + flashing
    pyenv
    mpremote
  ];
  shellHook = ''
    export PICO_SDK_PATH=${pico-sdk-full}/lib/pico-sdk
  '';
}
