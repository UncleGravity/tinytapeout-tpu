#!/usr/bin/env bash
# Spawn a localhost daemon, wait for it to accept connections, then run
# the C++ backend smoke binary against it. Used by the `backend-smoke`
# nix check and runnable standalone for local debugging.

set -euo pipefail

: "${PYNQ_HOST:=127.0.0.1}"
: "${PYNQ_PORT:=50055}"
: "${PYNQ_SMOKE_HEAP_MIB:=8}"
: "${PYNQ_SMOKE_SLAB_MIB:=1}"

LOG="${TMPDIR:-/tmp}/bonsaid-smoke.$$.log"
trap '[[ -n "${daemon_pid:-}" ]] && kill "$daemon_pid" 2>/dev/null || true' EXIT

python -m board.daemon \
    --host "$PYNQ_HOST" \
    --port "$PYNQ_PORT" \
    --allocator fake \
    --overlay none \
    --overlay-id backend-smoke \
    --heap-mib "$PYNQ_SMOKE_HEAP_MIB" \
    --slab-mib "$PYNQ_SMOKE_SLAB_MIB" >"$LOG" 2>&1 &
daemon_pid=$!

export PYNQ_HOST PYNQ_PORT

attempts=0
until pynq-backend-smoke; do
    attempts=$((attempts + 1))
    if (( attempts >= 30 )); then
        echo "backend smoke gave up after $attempts attempts; daemon log:" >&2
        cat "$LOG" >&2
        exit 1
    fi
    sleep 0.1
done
