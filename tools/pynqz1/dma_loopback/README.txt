PYNQ-Z1 DMA loopback overlay

Goal:
- Measure PL-to-DDR bandwidth through AXI DMA and the PS HP0 port.
- Data path: DDR -> AXI DMA MM2S -> axis_loopback -> AXI DMA S2MM -> DDR.

Build on Vivado VM from repo root:
nix develop -c ssh 10.211.55.3 'powershell -NoProfile -Command "& \"C:\\AMDDesignTools\\2025.2.1\\Vivado\\bin\\vivado.bat\" -mode batch -source \"C:\\Mac\\Home\\Documents\\asic\\tt-tpu\\tools\\pynqz1\\dma_loopback\\build.tcl\""'

Copy/run on PYNQ after build:
nix develop -c scp tools/pynqz1/dma_loopback/out/dma_loopback.bit tools/pynqz1/dma_loopback/out/dma_loopback.hwh tools/pynqz1/dma_loopback/dma_bandwidth.py xilinx@192.168.1.252:/home/xilinx/
nix develop -c ssh xilinx@192.168.1.252 'sudo env XILINX_XRT=/usr PYNQ_PYTHON=python3.10 /usr/local/share/pynq-venv/bin/python /home/xilinx/dma_bandwidth.py'

Reported payload MiB/s is one-way stream payload.
Reported DDR traffic MiB/s is 2x payload because DMA reads input and writes output.

Verified result, 2026-05-18:
size_mib    payload_MiB/s  ddr_traffic_MiB/s
1                    480.2              960.5
4                    660.2             1320.4
16                   728.8             1457.7
32                   742.6             1485.2

The 64-bit AXI-stream path at 100 MHz has a theoretical one-way payload limit
of about 763 MiB/s. The 32 MiB result is close enough for this first overlay.
