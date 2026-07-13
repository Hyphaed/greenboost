#!/usr/bin/env bash
# set_environment.sh — Install / repair the greenboost-cli Python dependencies
#
# Run this script with the greenboost-cli mamba env active:
#
#   mamba activate greenboost-cli
#   ./set_environment.sh
#
# Or point it at the env directly (no activation required):
#
#   ./set_environment.sh --env ~/.local/share/mamba/envs/greenboost-cli
#
# Re-running is safe — pip skips already-satisfied installs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── colour helpers ─────────────────────────────────────────────────────────────
if [[ -t 1 && "${TERM:-}" != "dumb" && -z "${NO_COLOR:-}" ]]; then
    C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'; C_CYAN='\033[0;36m'
    C_RED='\033[0;31m';   C_DIM='\033[2m';        C_BOLD='\033[1m'
    C_RESET='\033[0m'
else
    C_GREEN=''; C_YELLOW=''; C_CYAN=''; C_RED=''; C_DIM=''; C_BOLD=''; C_RESET=''
fi

ok()   { echo -e "  ${C_GREEN}✓${C_RESET}  $*"; }
warn() { echo -e "  ${C_YELLOW}⚠${C_RESET}  $*"; }
info() { echo -e "  ${C_CYAN}◈${C_RESET}  ${C_DIM}$*${C_RESET}"; }
fail() { echo -e "  ${C_RED}✗${C_RESET}  ${C_BOLD}$*${C_RESET}"; exit 1; }
sep()  { printf '%0.s─' $(seq 1 $(tput cols 2>/dev/null || echo 72)); echo; }

# ── parse args ─────────────────────────────────────────────────────────────────
ENV_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV_DIR="$2"; shift 2 ;;
        --env=*) ENV_DIR="${1#--env=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [--env /path/to/env]"
            echo "  Installs the correct CUDA-pinned Python stack for greenboost-cli."
            echo "  Default: uses the currently-active Python (mamba activate first)."
            exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done

# ── resolve pip / python ───────────────────────────────────────────────────────
if [[ -n "$ENV_DIR" ]]; then
    [[ -d "$ENV_DIR" ]] || fail "Env not found: $ENV_DIR"
    PIP="$ENV_DIR/bin/pip"
    PYTHON="$ENV_DIR/bin/python"
else
    PIP="$(command -v pip)"
    PYTHON="$(command -v python)"
fi

[[ -x "$PIP"    ]] || fail "pip not found — activate the greenboost-cli env first:\n    mamba activate greenboost-cli"
[[ -x "$PYTHON" ]] || fail "python not found at $PYTHON"

PYTHON_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
[[ "$(echo "$PYTHON_VER >= 3.11" | bc 2>/dev/null || echo 1)" == "1" ]] || true
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
    :
fi

echo
echo -e "${C_BOLD}  GreenBoost CLI — environment setup${C_RESET}"
sep
echo

info "Python:   $("$PYTHON" --version 2>&1)"
info "pip:      $PIP"
info "env:      $(dirname "$(dirname "$PIP")")"
echo

# ── detect GPU / CUDA driver ───────────────────────────────────────────────────
CUDA_DRIVER_VER=""
COMPUTE_CAP=""
GPU_NAME=""

if command -v nvidia-smi &>/dev/null; then
    CUDA_DRIVER_VER=$(nvidia-smi 2>/dev/null \
        | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d ' ' || true)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
        | head -1 | xargs || true)
fi

CUDA_MAJOR="${CUDA_DRIVER_VER%%.*}"
CUDA_MINOR="${CUDA_DRIVER_VER#*.}"; CUDA_MINOR="${CUDA_MINOR%%.*}"

if [[ -n "$GPU_NAME" ]]; then
    info "GPU:      $GPU_NAME  (cc $COMPUTE_CAP)"
    info "Driver:   CUDA $CUDA_DRIVER_VER"
else
    warn "No NVIDIA GPU detected — installing CPU-only torch."
    CUDA_DRIVER_VER="0.0"; CUDA_MAJOR=0; CUDA_MINOR=0
fi

# ── choose the right torch wheel index ────────────────────────────────────────
#
# Mapping rules (driver CUDA version → PyTorch wheel channel):
#
#   driver ≥ 13.0   →  cu130  (CUDA 13 — Blackwell GA, RTX 5070/5080/5090)
#   driver ≥ 12.8   →  cu128  (Blackwell early drivers, A100/H100 + sm_120 compat)
#   driver ≥ 12.4   →  cu124
#   driver ≥ 12.1   →  cu121
#   driver ≥ 11.8   →  cu118
#   else            →  cpu
#
# Exact torch + torchvision versions pinned to match vLLM 0.20.1's compiled ABI:
#   cu130 → torch==2.11.0+cu130   torchvision==0.26.0+cu130
#   cu128 → torch==2.11.0+cu128   torchvision==0.26.0+cu128   (nightly)
#   cu124 → torch==2.11.0+cu124   torchvision==0.26.0+cu124   (stable)
#   older → torch==2.11.0+cuXXX (best available)
#
TORCH_VER="2.11.0"
TORCHVISION_VER="0.26.0"

if   (( CUDA_MAJOR >= 13 )); then
    WHEEL_TAG="cu130"
    WHEEL_URL="https://download.pytorch.org/whl/cu130"
elif (( CUDA_MAJOR == 12 && CUDA_MINOR >= 8 )); then
    WHEEL_TAG="cu128"
    WHEEL_URL="https://download.pytorch.org/whl/nightly/cu128"
elif (( CUDA_MAJOR == 12 && CUDA_MINOR >= 4 )); then
    WHEEL_TAG="cu124"
    WHEEL_URL="https://download.pytorch.org/whl/cu124"
elif (( CUDA_MAJOR == 12 && CUDA_MINOR >= 1 )); then
    WHEEL_TAG="cu121"
    WHEEL_URL="https://download.pytorch.org/whl/cu121"
elif (( CUDA_MAJOR == 11 && CUDA_MINOR >= 8 )); then
    WHEEL_TAG="cu118"
    WHEEL_URL="https://download.pytorch.org/whl/cu118"
else
    WHEEL_TAG="cpu"
    WHEEL_URL="https://download.pytorch.org/whl/cpu"
fi

echo
info "Selected torch wheel: ${WHEEL_TAG}  (torch==${TORCH_VER}+${WHEEL_TAG})"
echo

# ── [1/5] torch + torchvision ─────────────────────────────────────────────────
echo -e "  ${C_CYAN}${C_BOLD}[1/5]${C_RESET}  torch + torchvision"
echo

"$PIP" install \
    "torch==${TORCH_VER}" \
    "torchvision==${TORCHVISION_VER}" \
    --index-url "$WHEEL_URL" \
    --quiet
ok "torch ${TORCH_VER}+${WHEEL_TAG} installed"
echo

# ── [2/5] vLLM ────────────────────────────────────────────────────────────────
echo -e "  ${C_CYAN}${C_BOLD}[2/5]${C_RESET}  vLLM"
echo

"$PIP" install \
    "vllm==0.20.1" \
    --extra-index-url "https://pypi.org/simple" \
    --quiet
ok "vllm 0.20.1 installed"
echo

# ── [3/5] greenboost-cli package ──────────────────────────────────────────────
echo -e "  ${C_CYAN}${C_BOLD}[3/5]${C_RESET}  greenboost-cli (source)"
echo

"$PIP" install -e "$SCRIPT_DIR" --no-deps --quiet
ok "greenboost-cli installed from $SCRIPT_DIR"
echo

# ── [4/5] ML extras (RAG, diffusion, PDF, vision) ─────────────────────────────
echo -e "  ${C_CYAN}${C_BOLD}[4/5]${C_RESET}  ML extras"
echo

"$PIP" install \
    sentence-transformers \
    transformers \
    diffusers \
    accelerate \
    bitsandbytes \
    torchao \
    PyMuPDF \
    opencv-python-headless \
    --quiet
ok "sentence-transformers / diffusers / bitsandbytes / torchao / pymupdf / opencv installed"
echo

# ── [5/5] verify ──────────────────────────────────────────────────────────────
echo -e "  ${C_CYAN}${C_BOLD}[5/5]${C_RESET}  Verification"
echo

"$PYTHON" - <<'PYEOF'
import sys

results = []

def chk(label, fn):
    try:
        val = fn()
        results.append((True, label, val))
    except Exception as e:
        results.append((False, label, f"{type(e).__name__}: {str(e)[:120]}"))

chk("torch import",      lambda: __import__("torch").__version__)
chk("torch.cuda.is_available", lambda: str(__import__("torch").cuda.is_available()))
chk("torch CUDA version",lambda: __import__("torch").version.cuda)

import torch
if torch.cuda.is_available():
    chk("GPU name",        lambda: torch.cuda.get_device_name(0))
    chk("compute cap",     lambda: str(torch.cuda.get_device_capability(0)))

chk("vllm._C import",    lambda: (__import__("vllm.platforms.cuda", fromlist=["cuda"]) and "OK"))
chk("vllm version",      lambda: __import__("vllm").__version__)
chk("sentence-transformers", lambda: __import__("sentence_transformers").__version__)
chk("diffusers",         lambda: __import__("diffusers").__version__)
chk("transformers",      lambda: __import__("transformers").__version__)

ok = "\033[0;32m✓\033[0m"
fail = "\033[0;31m✗\033[0m"
any_fail = False
for passed, label, value in results:
    sym = ok if passed else fail
    print(f"  {sym}  {label:<36} {value}")
    if not passed:
        any_fail = True

print()
if any_fail:
    print("  \033[1;33m⚠\033[0m  Some checks failed — see above.")
    sys.exit(1)
else:
    print("  \033[0;32m✓\033[0m  All checks passed.")
PYEOF

echo
sep
echo
echo -e "  ${C_GREEN}${C_BOLD}✓  Environment ready.${C_RESET}"
echo
echo -e "  ${C_DIM}Clear any stale vLLM state and launch:${C_RESET}"
echo -e "    ${C_CYAN}rm -f ~/.greenboost_cli/vllm.{pid,log} && greenboost-cli${C_RESET}"
echo
