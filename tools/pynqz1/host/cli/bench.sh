# pynq-bench: start the board daemon, run a canned llama-cli pass, pull
# the board's profile back, and summarize host + board traces.
#
# Env overrides:
#   PYNQ_HOST          board hostname        (default: pynq)
#   PYNQ_PORT          daemon port           (default: 50055)
#   PYNQ_MODEL         path to .gguf model   (default: ~/models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf)
#   PYNQ_BOARD_PROFILE board-side trace path (default: /tmp/board.ndjson)
#   PYNQ_BITFILE       PL bitstream on the board (default: /home/xilinx/overlays/matmul_q1a8.bit)

# Export so every child (pynq-daemon, pynqctl, llama-cli-pynq) sees them
# without having to prefix each invocation.
export PYNQ_HOST="${PYNQ_HOST:-pynq}"
export PYNQ_PORT="${PYNQ_PORT:-50055}"
PYNQ_MODEL="${PYNQ_MODEL:-$HOME/models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf}"
PYNQ_BOARD_PROFILE="${PYNQ_BOARD_PROFILE:-/tmp/board.ndjson}"
PYNQ_BITFILE="${PYNQ_BITFILE:-/home/xilinx/overlays/matmul_q1a8.bit}"

HOST_PROFILE="$(mktemp -t pynq-host.XXXXXX.ndjson)"
LOCAL_BOARD_PROFILE="$(mktemp -t pynq-board.XXXXXX.ndjson)"
DAEMON_LOG="$(mktemp -t pynq-daemon.XXXXXX.log)"
DAEMON_PID=""

cleanup() {
    rc=$?
    echo
    echo "--- cleanup (exit=$rc) ---"

    # Close the local SSH tunnel; that sends SIGHUP to the remote process tree.
    if [[ -n "${DAEMON_PID}" ]]; then
        kill "${DAEMON_PID}" 2>/dev/null || true
        wait "${DAEMON_PID}" 2>/dev/null || true
    fi
    # Belt-and-suspenders: kill any lingering daemon on the board.
    ssh -o BatchMode=yes "xilinx@${PYNQ_HOST}" "sudo pkill -f board.daemon" \
        2>/dev/null || true

    if scp -q "xilinx@${PYNQ_HOST}:${PYNQ_BOARD_PROFILE}" \
        "${LOCAL_BOARD_PROFILE}" 2>/dev/null; then
        echo "board profile -> ${LOCAL_BOARD_PROFILE}"
    else
        echo "warning: could not fetch ${PYNQ_BOARD_PROFILE} from board" >&2
        LOCAL_BOARD_PROFILE=""
    fi

    files=()
    [[ -s "${HOST_PROFILE}" ]] && files+=("${HOST_PROFILE}")
    [[ -n "${LOCAL_BOARD_PROFILE}" && -s "${LOCAL_BOARD_PROFILE}" ]] &&
        files+=("${LOCAL_BOARD_PROFILE}")
    if ((${#files[@]} > 0)); then
        echo
        pynq-profile summary "${files[@]}" || true
    fi
    exit "$rc"
}
trap cleanup EXIT

echo "--- spawning daemon (log: ${DAEMON_LOG}) ---"
# The board's PYNQ_PROFILE sink opens in append mode, so wipe any prior
# run's data before the daemon (re)opens it.
# Otherwise per-op compute accumulates across runs and the
# summary % goes over 100.
ssh -o BatchMode=yes "xilinx@${PYNQ_HOST}" \
    "sudo rm -f ${PYNQ_BOARD_PROFILE}" 2>/dev/null || true
PYNQ_PROFILE="${PYNQ_BOARD_PROFILE}" pynq-daemon --bitfile "${PYNQ_BITFILE}" >"${DAEMON_LOG}" 2>&1 &
DAEMON_PID=$!

echo "--- waiting for daemon on ${PYNQ_HOST}:${PYNQ_PORT} ---"
READY=""
for _ in $(seq 1 150); do
    if pynqctl hello >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 0.2
done

if [[ -z "${READY}" ]]; then
    echo "ERROR: daemon did not come up in 30s. Last log lines:" >&2
    tail -50 "${DAEMON_LOG}" >&2
    exit 1
fi

echo "--- running llama-cli (model: ${PYNQ_MODEL}) ---"
# -ngl 99 offloads ALL transformer layers (including KV cache) to PYNQ.
# Without it, model.dev_layer(il) = CPU for every layer and ggml's FA
# auto-detect disables FA on a layer/KV device mismatch — see
# llama-context.cpp:449 ("layer N is assigned to device CPU but the
# Flash Attention tensor is assigned to device PYNQ"). With FA off,
# attention is decomposed into MUL_MAT:f16xf32+SOFT_MAX+CONT splits.
PYNQ_PROFILE="${HOST_PROFILE}" \
    llama-cli-pynq \
    -m "${PYNQ_MODEL}" \
    -p Hello \
    -n 2 \
    -c 32 \
    -b 16 -ub 16 \
    -t 4 \
    -ngl 99 \
    --device PYNQ \
    --temp 0 \
    --no-warmup \
    --single-turn \
    --no-display-prompt
