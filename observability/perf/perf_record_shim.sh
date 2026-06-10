#!/usr/bin/env bash
# perf_record_shim.sh - capture a sampling profile of the GreenBoost CUDA
# shim during model load + first 10 s of inference, then convert to a
# flamegraph.  Audit references: F-L6-09, F-L6-11.
#
# Requires linux-tools-common (perf) and the flamegraph.pl + stackcollapse-perf.pl
# scripts from https://github.com/brendangregg/FlameGraph in $PATH.
#
# Usage:
#   PID=$(pgrep -f ollama)   # or whatever loads libgreenboost_cuda.so
#   sudo ./perf_record_shim.sh "$PID" 10
#
# Output:
#   /tmp/greenboost-shim.svg  - interactive SVG flamegraph

set -euo pipefail

PID="${1:-}"
DURATION="${2:-10}"

if [[ -z "$PID" ]]; then
    echo "Usage: $0 <pid> [duration-seconds]" >&2
    exit 1
fi

if ! command -v perf >/dev/null 2>&1; then
    echo "perf not found - install linux-tools-common (Debian/Ubuntu) or linux-perf" >&2
    exit 1
fi

if [[ "$EUID" -ne 0 ]] && ! capsh --print 2>/dev/null | grep -q "cap_perfmon"; then
    echo "ERROR: perf record requires root or CAP_PERFMON. Re-run with sudo." >&2
    exit 1
fi

TMPDIR_WORK=$(mktemp -d -t greenboost-shim.XXXXXX)
trap 'rm -rf "$TMPDIR_WORK"' EXIT
OUT_DATA="$TMPDIR_WORK/shim.perf"
OUT_SCRIPT="$TMPDIR_WORK/shim.txt"
OUT_FOLDED="$TMPDIR_WORK/shim.folded"
OUT_SVG="$TMPDIR_WORK/shim.svg"

echo "[*] Recording $DURATION s of pid=$PID (libgreenboost_cuda.so loaded?)..."
perf record -F 99 --call-graph dwarf -p "$PID" -o "$OUT_DATA" -- sleep "$DURATION"

echo "[*] Resolving symbols..."
perf script -i "$OUT_DATA" > "$OUT_SCRIPT"

if command -v stackcollapse-perf.pl >/dev/null 2>&1 \
   && command -v flamegraph.pl >/dev/null 2>&1; then
    stackcollapse-perf.pl "$OUT_SCRIPT" > "$OUT_FOLDED"
    flamegraph.pl --title "GreenBoost shim flamegraph (pid=$PID)" "$OUT_FOLDED" > "$OUT_SVG"
    echo "[*] Flamegraph: $OUT_SVG"
else
    echo "[*] FlameGraph scripts not in PATH - perf script output at $OUT_SCRIPT"
    echo "    Install: git clone https://github.com/brendangregg/FlameGraph"
fi
