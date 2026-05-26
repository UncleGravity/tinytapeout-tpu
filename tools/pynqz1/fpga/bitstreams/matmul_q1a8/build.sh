#!/usr/bin/env bash
# Build the matmul_q1a8 bitstream on the Vivado VM and pull it home.
#
# Pushes the bitstream folder AND the shared fpga/rtl/ tree to the VM, so
# the tcl can include the synthesizable modules from `../rtl/q1a8/` the
# same way they live locally.

set -euxo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_NAME="matmul_q1a8"
SHARED_RTL_DIR="$(cd "${PROJ_DIR}/../../rtl" && pwd)"

# Vivado VM.
VIVADO_HOST="${VIVADO_HOST:-10.211.55.3}"
VIVADO_BAT="${VIVADO_BAT:-C:\\AMDDesignTools\\2025.2.1\\Vivado\\bin\\vivado.bat}"
VIVADO_WORK_PARENT="${VIVADO_WORK_PARENT:-C:/work}"

# PYNQ board.
PYNQ_HOST="${PYNQ_HOST:-pynq}"
PYNQ_USER="${PYNQ_USER:-xilinx}"
PYNQ_OVERLAY_DIR="${PYNQ_OVERLAY_DIR:-/home/${PYNQ_USER}/overlays}"
PYNQ_BENCH_DIR="${PYNQ_BENCH_DIR:-/home/${PYNQ_USER}/pynqz1/fpga/bitstreams/${PROJ_NAME}}"

INSTALL=0
case "${1:-}" in
--install) INSTALL=1 ;;
-h | --help)
    cat <<EOF
usage: $(basename "$0") [--install]

Build ${PROJ_NAME}.{bit,hwh} on the Vivado VM, copy into ./out/.
With --install, also scp them to the PYNQ board.

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

echo "==> [1/4] push project to ${VIVADO_HOST}:${VM_PROJECT}"
PS_PREP="New-Item -ItemType Directory -Force '${VIVADO_WORK_PARENT}' | Out-Null; Remove-Item -Recurse -Force '${VM_PROJECT}' -ErrorAction SilentlyContinue; exit 0"
ssh "${VIVADO_HOST}" "powershell -NoProfile -Command \"${PS_PREP}\""
scp -rq "${PROJ_DIR}" "${VIVADO_HOST}:${VIVADO_WORK_PARENT}/"

echo "==> [2/4] push shared rtl/ alongside the project"
scp -rq "${SHARED_RTL_DIR}" "${VIVADO_HOST}:${VM_PROJECT}/"

echo "==> [3/4] vivado batch synth+impl (slow)"
PS_BUILD="cd '${VM_PROJECT}'; & '${VIVADO_BAT}' -mode batch -source tcl/build.tcl"
ssh "${VIVADO_HOST}" "powershell -NoProfile -Command \"${PS_BUILD}\""

echo "==> [4/4] fetch .bit and .hwh"
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
