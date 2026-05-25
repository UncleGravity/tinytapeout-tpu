#!/usr/bin/env bash
# Build the dma_dual_loopback bitstream on the Vivado VM and pull it home.
# With --install, also scp .bit/.hwh + bench.py to the PYNQ board.

# set -euo pipefail
set -euxo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_NAME="dma_dual_loopback"

# Vivado VM (Windows, OpenSSH server).
VIVADO_HOST="${VIVADO_HOST:-10.211.55.3}"
VIVADO_BAT="${VIVADO_BAT:-C:\\AMDDesignTools\\2025.2.1\\Vivado\\bin\\vivado.bat}"
VIVADO_WORK_PARENT="${VIVADO_WORK_PARENT:-C:/work}"

# PYNQ board.
PYNQ_HOST="${PYNQ_HOST:-pynq}"
PYNQ_USER="${PYNQ_USER:-xilinx}"
PYNQ_OVERLAY_DIR="${PYNQ_OVERLAY_DIR:-/home/${PYNQ_USER}/overlays}"
PYNQ_BENCH_DIR="${PYNQ_BENCH_DIR:-/home/${PYNQ_USER}/pynqz1/fpga/benchmarks/${PROJ_NAME}}"

INSTALL=0
case "${1:-}" in
--install) INSTALL=1 ;;
-h | --help)
    cat <<EOF
usage: $(basename "$0") [--install]

Build dma_dual_loopback.{bit,hwh} on the Vivado VM and copy them into ./out/.
With --install, also scp them to ${PYNQ_USER}@${PYNQ_HOST}:${PYNQ_OVERLAY_DIR}/
and ship bench.py to ${PYNQ_BENCH_DIR}/.

Env overrides:
  VIVADO_HOST          (${VIVADO_HOST})
  VIVADO_BAT           (${VIVADO_BAT})
  VIVADO_WORK_PARENT   (${VIVADO_WORK_PARENT})
  PYNQ_HOST            (${PYNQ_HOST})
  PYNQ_USER            (${PYNQ_USER})
  PYNQ_OVERLAY_DIR     (${PYNQ_OVERLAY_DIR})
  PYNQ_BENCH_DIR       (${PYNQ_BENCH_DIR})
EOF
    exit 0
    ;;
"") ;;
*)
    echo "unknown argument: $1" >&2
    exit 2
    ;;
esac

VM_PROJECT="${VIVADO_WORK_PARENT}/${PROJ_NAME}"

echo "==> [1/3] push source to ${VIVADO_HOST}:${VM_PROJECT}"
PS_PREP="New-Item -ItemType Directory -Force '${VIVADO_WORK_PARENT}' | Out-Null; Remove-Item -Recurse -Force '${VM_PROJECT}' -ErrorAction SilentlyContinue; exit 0"
ssh "${VIVADO_HOST}" "powershell -NoProfile -Command \"${PS_PREP}\""
scp -rq "${PROJ_DIR}" "${VIVADO_HOST}:${VIVADO_WORK_PARENT}/"

echo "==> [2/3] vivado batch synth+impl (this is the slow part)"
PS_BUILD="cd '${VM_PROJECT}'; & '${VIVADO_BAT}' -mode batch -source tcl/build.tcl"
ssh "${VIVADO_HOST}" "powershell -NoProfile -Command \"${PS_BUILD}\""

echo "==> [3/3] fetch .bit and .hwh"
mkdir -p "${PROJ_DIR}/out"
scp -q "${VIVADO_HOST}:${VM_PROJECT}/out/${PROJ_NAME}.bit" \
    "${VIVADO_HOST}:${VM_PROJECT}/out/${PROJ_NAME}.hwh" \
    "${PROJ_DIR}/out/"
echo "    -> ${PROJ_DIR}/out/${PROJ_NAME}.{bit,hwh}"

if [[ ${INSTALL} -eq 1 ]]; then
    echo "==> install to ${PYNQ_USER}@${PYNQ_HOST}"
    ssh "${PYNQ_USER}@${PYNQ_HOST}" "mkdir -p '${PYNQ_OVERLAY_DIR}' '${PYNQ_BENCH_DIR}'"
    scp -q "${PROJ_DIR}/out/${PROJ_NAME}.bit" \
        "${PROJ_DIR}/out/${PROJ_NAME}.hwh" \
        "${PYNQ_USER}@${PYNQ_HOST}:${PYNQ_OVERLAY_DIR}/"
    scp -q "${PROJ_DIR}/bench.py" \
        "${PYNQ_USER}@${PYNQ_HOST}:${PYNQ_BENCH_DIR}/"
    echo "    overlay : ${PYNQ_OVERLAY_DIR}/${PROJ_NAME}.{bit,hwh}"
    echo "    bench   : ${PYNQ_BENCH_DIR}/bench.py"
fi

echo "==> done"
