#!/bin/bash
# GreenBoost CUDA shim deploy — rebuilds and reinstalls libgreenboost_cuda.so
# Run with sudo: sudo ./deploy_fix.sh [--debug] [--no-restart]
#
# Options:
#   --debug      Enable GREENBOOST_DEBUG=1 in Ollama service (verbose shim logging)
#   --no-restart Don't restart Ollama after deploy

set -e

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM_SRC="$MODULE_DIR/libgreenboost_cuda.so"
SHIM_DST="/usr/local/lib/libgreenboost_cuda.so"
SVC="/etc/systemd/system/ollama.service"

[[ $EUID -eq 0 ]] || { echo "Run with sudo: sudo $0 $*"; exit 1; }

DEBUG=0
RESTART=1
for arg in "$@"; do
    [[ "$arg" == "--debug" ]]      && DEBUG=1
    [[ "$arg" == "--no-restart" ]] && RESTART=0
done

echo "[deploy] Rebuilding CUDA shim..."
make -C "$MODULE_DIR" shim

echo "[deploy] Installing to $SHIM_DST..."
cp "$SHIM_SRC" "$SHIM_DST"
ldconfig

if [[ -f "$SVC" ]]; then
    if [[ $DEBUG -eq 1 ]]; then
        sed -i 's/GREENBOOST_DEBUG=0/GREENBOOST_DEBUG=1/' "$SVC"
        echo "[deploy] Debug mode ENABLED in Ollama service"
    else
        sed -i 's/GREENBOOST_DEBUG=1/GREENBOOST_DEBUG=0/' "$SVC"
        echo "[deploy] Debug mode off"
    fi
    systemctl daemon-reload
fi

if [[ $RESTART -eq 1 ]]; then
    echo "[deploy] Restarting Ollama..."
    systemctl restart ollama
    echo "[deploy] Done. Watch logs:"
    echo "  journalctl -u ollama -f"
else
    echo "[deploy] Skipping Ollama restart (--no-restart). Restart manually:"
    echo "  sudo systemctl restart ollama"
fi

echo ""
echo "[deploy] Verify shim loaded (look for libcudart loaded + cudaMallocAsync=hooked):"
echo "  journalctl -u ollama --since 'now' -f | grep GreenBoost"
