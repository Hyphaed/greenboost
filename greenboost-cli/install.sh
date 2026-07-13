#!/usr/bin/env bash
# install.sh — GreenBoost CLI installer
#
# Usage:
#   bash install.sh                     # interactive wizard (TTY required)
#   bash install.sh install             # install greenboost-cli
#   bash install.sh uninstall           # remove greenboost-cli
#   bash install.sh status              # show installation status
#   bash install.sh update              # refresh code + sync new deps
#   bash install.sh repair [variant]    # diagnose + fix broken torch / ML stack
#                                       #   variant: cu128 | cu124 | cu121 | cu118 | cpu
#                                       #   (default: auto-detect — Blackwell → cu128)
#   bash install.sh help                # command reference
#
# Environment manager priority:  mamba → conda → uv
# mamba/conda is preferred because it resolves CUDA/GPU deps reliably
# (torch, diffusers, sentence-transformers all work with conda channels).
#
# Installs into:
#   env     → ~/.local/share/greenboost-cli/env
#   symlink → ~/.local/bin/greenboost-cli
#             ~/.local/bin/gb
#
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GB_DEFAULT_VENV_DIR="$HOME/.local/share/greenboost-cli/env"
VENV_DIR="$GB_DEFAULT_VENV_DIR"
BIN_DIR="$HOME/.local/bin"
SYMLINK="$BIN_DIR/greenboost-cli"
SYMLINK_GB="$BIN_DIR/gb"
GB_VERSION="1.0.0"
CONFIG_FILE="$HOME/.greenboost_cli/settings.json"
PY_VER="3.12"
# Named mamba/conda env to reuse if it already exists
GB_NAMED_ENV="greenboost-cli"
# Set to 1 when an existing named env is adopted (prevents rm -rf on uninstall)
GB_USING_NAMED_ENV=0

# ── Colours ───────────────────────────────────────────────────────────────────
_gb_truecolor() { [[ "${COLORTERM:-}" =~ ^(truecolor|24bit)$ ]]; }

if _gb_truecolor; then
    C_VIOLET='\033[38;2;108;113;196m'
    C_LIME='\033[38;2;166;227;161m'
    C_GRAY='\033[38;2;208;207;204m'
    C_CYAN='\033[38;2;137;220;235m'
    C_AMBER='\033[38;2;249;226;175m'
    C_RED='\033[38;2;243;139;168m'
else
    C_VIOLET='\033[0;34m'; C_LIME='\033[0;32m'; C_GRAY='\033[0;37m'
    C_CYAN='\033[0;36m';   C_AMBER='\033[1;33m'; C_RED='\033[0;31m'
fi
C_BOLD='\033[1m'; C_DIM='\033[2m'; C_RESET='\033[0m'

has_color() { [[ -n "${TERM:-}" && "${TERM}" != "dumb" ]]; }
if ! has_color || [[ -n "${NO_COLOR:-}" ]]; then
    C_VIOLET=""; C_LIME=""; C_GRAY=""; C_CYAN=""; C_AMBER=""
    C_RED=""; C_BOLD=""; C_DIM=""; C_RESET=""
fi

# ── UI helpers ────────────────────────────────────────────────────────────────
gb_ok()   { echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}$*${C_RESET}"; }
gb_fail() { echo -e "  ${C_RED}✗${C_RESET}  ${C_BOLD}$*${C_RESET}"; }
gb_warn() { echo -e "  ${C_AMBER}⚠${C_RESET}  $*"; }
gb_info() { echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}$*${C_RESET}"; }

die() {
    local msg="$1" cause="${2:-}" fix="${3:-}"
    echo -e ""
    echo -e "  ${C_RED}✗${C_RESET}  ${C_BOLD}${msg}${C_RESET}"
    [[ -n "$cause" ]] && echo -e "  ${C_DIM}  Likely cause: ${cause}${C_RESET}"
    [[ -n "$fix"   ]] && echo -e "  ${C_AMBER}  Fix: ${fix}${C_RESET}"
    echo -e ""
    exit 1
}

# This installer resolves everything (env dir, ~/.local/bin symlinks, config)
# from $HOME. Under sudo, $HOME is /root — the install/update silently targets
# a root-owned environment that the user's own `greenboost-cli` command never
# sees, while looking like it succeeded. Refuse early instead of failing quietly.
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    die "Do not run install.sh as root/sudo." \
        "\$HOME becomes /root under sudo, so install/update silently targets a root-owned environment instead of your own — the real greenboost-cli command never sees the change." \
        "Run as your normal user:  bash install.sh ${1:-install}"
fi

gb_separator() {
    local cols; cols=$(tput cols 2>/dev/null || echo 64)
    echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 $((cols - 2))))${C_RESET}"
}

gb_step() { echo -e ""; echo -e "  ${C_CYAN}${C_BOLD}[$1/$2]${C_RESET}  ${C_GRAY}${C_BOLD}$3${C_RESET}"; }

gb_header() {
    local cols; cols=$(tput cols 2>/dev/null || echo 64)
    local title="  GreenBoost CLI v${GB_VERSION} — Unified AI Coding Assistant"
    echo -e ""
    echo -e "${C_VIOLET}${C_BOLD}  ╔$(printf '═%.0s' $(seq 1 $((cols-4))))╗${C_RESET}"
    echo -e "${C_VIOLET}${C_BOLD}  ║${C_RESET}${C_GRAY}${C_BOLD}${title}$(printf ' %.0s' $(seq 1 $((cols-4-${#title}-1))))${C_VIOLET}${C_BOLD}║${C_RESET}"
    echo -e "${C_VIOLET}${C_BOLD}  ╚$(printf '═%.0s' $(seq 1 $((cols-4))))╝${C_RESET}"
    echo -e ""
}

GB_SPIN_FRAMES=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
gb_spin() {
    local pid=$1 msg="$2" i=0
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r  ${C_LIME}%s${C_RESET}  ${C_DIM}%s${C_RESET}" \
            "${GB_SPIN_FRAMES[$((i % ${#GB_SPIN_FRAMES[@]}))]}" "$msg"
        sleep 0.08
        (( i++ )) || true
    done
    printf "\r  ${C_LIME}✓${C_RESET}  ${C_GRAY}%s${C_RESET}\n" "$msg"
}

confirm() {
    local prompt="$1" default="${2:-n}" yn
    if [[ "$default" == "y" ]]; then
        read -rp "$(echo -e "  ${C_AMBER}?${C_RESET}  ${prompt} [Y/n] ")" yn
        yn="${yn:-y}"
    else
        read -rp "$(echo -e "  ${C_AMBER}?${C_RESET}  ${prompt} [y/N] ")" yn
        yn="${yn:-n}"
    fi
    [[ "$yn" =~ ^[Yy]$ ]]
}

is_interactive() { [[ -t 0 && -t 1 ]]; }

# ── Package manager detection ─────────────────────────────────────────────────
# Returns global PKG_MANAGER ("mamba" | "conda" | "uv") and UV / CONDA_CMD.
detect_pkg_manager() {
    PKG_MANAGER=""
    UV=""
    CONDA_CMD=""

    if command -v mamba &>/dev/null; then
        PKG_MANAGER="mamba"
        CONDA_CMD="$(command -v mamba)"
        return 0
    fi

    if command -v conda &>/dev/null; then
        PKG_MANAGER="conda"
        CONDA_CMD="$(command -v conda)"
        return 0
    fi

    if command -v uv &>/dev/null; then
        PKG_MANAGER="uv"
        UV="$(command -v uv)"
        return 0
    fi

    PKG_MANAGER="uv"   # will try to install uv
    return 1
}

# ── Named env resolution ──────────────────────────────────────────────────────
# Locate the named mamba/conda env entirely through mamba/conda's own APIs —
# no hardcoded paths, so it works with miniforge3, mambaforge, micromamba,
# custom --prefix installs, etc.
resolve_named_env() {
    [[ "$PKG_MANAGER" =~ ^(mamba|conda)$ ]] || return 1
    local found=""

    # Method 1: env list --json — authoritative; returns every known env path.
    # grep for a JSON string value that ends with /<name>.
    found=$(
        "$CONDA_CMD" env list --json 2>/dev/null \
        | grep -o "\"[^\"]*/${GB_NAMED_ENV}\"" \
        | tr -d '"' \
        | head -1
    )

    # Method 2: ask every available conda/mamba binary for its base dir, then
    # probe base/envs/<name>.  This finds envs created under miniforge3, mambaforge,
    # micromamba, or any custom prefix without any hardcoded path guessing.
    # mamba info --base prints "  base environment : /path" (multi-word);
    # conda info --base prints just "/path".  Both are handled by extracting $NF.
    if [[ -z "$found" ]]; then
        local _cmd _base
        for _cmd in mamba conda micromamba; do
            command -v "$_cmd" &>/dev/null || continue
            _base=$("$_cmd" info --base 2>/dev/null | awk '{print $NF}' | tail -1)
            [[ -n "$_base" && -d "$_base/envs/$GB_NAMED_ENV/bin" ]] \
                && { found="$_base/envs/$GB_NAMED_ENV"; break; }
        done
    fi

    if [[ -n "$found" && -d "$found/bin" ]]; then
        VENV_DIR="$found"
        GB_USING_NAMED_ENV=1
        return 0
    fi
    return 1
}

# Detect CUDA version from nvidia-smi (returns e.g. "12.1")
detect_cuda_ver() {
    local ver=""
    if command -v nvidia-smi &>/dev/null; then
        ver=$(nvidia-smi 2>/dev/null \
              | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' \
              | head -1 || true)
    fi
    echo "${ver:-12.1}"
}

# Map the nvidia driver's CUDA version to the nearest pytorch-cuda conda package.
# nvidia-smi reports the *maximum* CUDA the driver supports, which is often higher
# than what PyTorch packages exist for.  Supported: 11.8, 12.1, 12.4, 12.8.
# 12.8 is the first build that supports Blackwell (RTX 5070, sm_120).
map_pytorch_cuda_ver() {
    local ver="$1"
    local major minor
    major="${ver%%.*}"
    minor="${ver#*.}"; minor="${minor%%.*}"
    if   (( major >= 13 )); then
        echo "13.0"
    elif (( major == 12 )) && (( minor >= 8 )); then
        echo "12.8"
    elif (( major == 12 )) && (( minor >= 4 )); then
        echo "12.4"
    elif (( major == 12 )) && (( minor >= 1 )); then
        echo "12.1"
    elif (( major == 11 )) && (( minor >= 8 )); then
        echo "11.8"
    else
        echo "12.1"   # safe default
    fi
}

# Return the PyTorch wheel index URL for `pip install --index-url …`.
# Accepts both compact ("128", "124", "121", "118") and dotted ("12.8", "12.4")
# forms, plus "cpu". Used by cmd_repair when reinstalling torch outside the
# conda channel.
pytorch_wheel_index() {
    local v="${1#cu}"      # strip leading "cu" if present (e.g. "cu128" → "128")
    v="${v//./}"           # strip dots (e.g. "12.8" → "128")
    case "$v" in
        130)  echo "https://download.pytorch.org/whl/cu130" ;;
        128)  echo "https://download.pytorch.org/whl/nightly/cu128" ;;
        124)  echo "https://download.pytorch.org/whl/cu124" ;;
        121)  echo "https://download.pytorch.org/whl/cu121" ;;
        118)  echo "https://download.pytorch.org/whl/cu118" ;;
        cpu)  echo "https://download.pytorch.org/whl/cpu"   ;;
        *)    echo "https://download.pytorch.org/whl/cu121" ;;
    esac
}

# Detect Blackwell (sm_120) — needs torch built with CUDA 12.8 or newer.
gpu_is_blackwell() {
    command -v nvidia-smi &>/dev/null || return 1
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | grep -qE '^12\.'
}

# Detect if $VENV_DIR is a conda env (has conda-meta/) or a plain venv
is_conda_env() { [[ -d "$VENV_DIR/conda-meta" ]]; }

# ── install ───────────────────────────────────────────────────────────────────
cmd_install() {
    gb_header

    # ── [1/5] Package manager ─────────────────────────────────────────────────
    gb_step 1 5 "Package manager  (mamba → conda → uv)"
    echo -e ""

    if ! detect_pkg_manager; then
        # Nothing found — offer to install uv as last resort
        gb_warn "mamba, conda, and uv not found."
        echo -e ""
        echo -e "  ${C_DIM}mamba/conda gives the best GPU/CUDA support (recommended for${C_RESET}"
        echo -e "  ${C_DIM}RAG + diffusion).  uv is a lightweight pip-based fallback.${C_RESET}"
        echo -e ""
        if is_interactive; then
            echo -e "  ${C_GRAY}Options:${C_RESET}"
            printf "  ${C_LIME}  1${C_RESET}  ${C_GRAY}Install mamba (miniforge)${C_RESET}  ${C_DIM}https://github.com/conda-forge/miniforge${C_RESET}\n"
            printf "  ${C_LIME}  2${C_RESET}  ${C_GRAY}Install uv (lightweight)${C_RESET}   ${C_DIM}https://docs.astral.sh/uv${C_RESET}\n"
            echo -e ""
            local choice
            read -rp "$(echo -e "  ${C_AMBER}?${C_RESET}  Choose [1/2/cancel] ")" choice
            echo -e ""
            case "${choice:-}" in
                1)
                    gb_info "Installing miniforge (mamba)…"
                    local mf_url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
                    local mf_sh="/tmp/miniforge_install.sh"
                    curl -L "$mf_url" -o "$mf_sh" \
                        || wget -O "$mf_sh" "$mf_url" \
                        || die "Download failed" "no curl/wget" "Install manually: $mf_url"
                    bash "$mf_sh" -b -p "$HOME/miniforge3"
                    export PATH="$HOME/miniforge3/bin:$PATH"
                    command -v mamba &>/dev/null \
                        || die "mamba install failed" "" "Try: bash $mf_sh"
                    PKG_MANAGER="mamba"
                    CONDA_CMD="$(command -v mamba)"
                    rm -f "$mf_sh"
                    gb_ok "mamba installed at $HOME/miniforge3"
                    ;;
                2)
                    gb_info "Installing uv via official installer…"
                    if command -v brew &>/dev/null; then
                        brew install uv
                    elif command -v curl &>/dev/null; then
                        curl -LsSf https://astral.sh/uv/install.sh | sh
                    elif command -v wget &>/dev/null; then
                        wget -qO- https://astral.sh/uv/install.sh | sh
                    else
                        die "Cannot install uv" "No curl/wget/brew found" \
                            "Install manually: https://docs.astral.sh/uv"
                    fi
                    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
                    command -v uv &>/dev/null \
                        || die "uv install failed" "" "Install manually: https://docs.astral.sh/uv"
                    PKG_MANAGER="uv"
                    UV="$(command -v uv)"
                    gb_ok "uv installed"
                    ;;
                *)
                    gb_info "Cancelled."
                    echo -e ""
                    exit 0
                    ;;
            esac
        else
            die "No package manager available" \
                "mamba/conda/uv all missing" \
                "Install mamba: https://github.com/conda-forge/miniforge"
        fi
    fi

    case "$PKG_MANAGER" in
        mamba)
            gb_ok "Using mamba $(mamba --version 2>&1 | head -1) — best CUDA support"
            ;;
        conda)
            gb_ok "Using conda $(conda --version 2>&1)"
            gb_info "Tip: install mamba for faster solves:  conda install -c conda-forge mamba"
            ;;
        uv)
            gb_ok "Using uv $(uv --version 2>&1 | awk '{print $2}') — pip-based"
            gb_info "For GPU workloads, mamba/conda gives better CUDA dependency resolution."
            ;;
    esac

    # ── [2/5] Python ≥ 3.11 ──────────────────────────────────────────────────
    gb_step 2 5 "Python ${PY_VER}"
    echo -e ""

    case "$PKG_MANAGER" in
        mamba|conda)
            gb_info "Python ${PY_VER} will be installed into the conda environment."
            ;;
        uv)
            if "$UV" python find "$PY_VER" &>/dev/null; then
                gb_ok "Python ${PY_VER} available: $("$UV" python find "$PY_VER")"
            else
                gb_info "Python ${PY_VER} not found locally — uv will download it."
            fi
            ;;
    esac

    # ── [3/5] Create environment ──────────────────────────────────────────────
    # Prefer the existing named mamba env 'greenboost-cli' if it exists.
    if resolve_named_env; then
        gb_step 3 5 "Using existing '$GB_NAMED_ENV' environment"
        echo -e ""
        gb_ok "Found existing env at $VENV_DIR"
        gb_info "Skipping environment creation — reusing '$GB_NAMED_ENV'."
    else
        gb_step 3 5 "Create environment at $VENV_DIR"
        echo -e ""

        if [[ -d "$VENV_DIR" ]]; then
            gb_info "Removing existing environment…"
            rm -rf "$VENV_DIR"
            gb_ok "Removed $VENV_DIR"
        fi
        mkdir -p "$(dirname "$VENV_DIR")"

        case "$PKG_MANAGER" in
            mamba)
                gb_info "Creating mamba environment (python=${PY_VER})…"
                "$CONDA_CMD" create -p "$VENV_DIR" python="$PY_VER" -y -q \
                    || die "Failed to create mamba environment" \
                           "mamba not in PATH or network error" \
                           "mamba create -p $VENV_DIR python=${PY_VER} -y"
                gb_ok "mamba environment created"
                ;;
            conda)
                gb_info "Creating conda environment (python=${PY_VER})…"
                "$CONDA_CMD" create -p "$VENV_DIR" python="$PY_VER" -y -q \
                    || die "Failed to create conda environment" \
                           "conda error" \
                           "conda create -p $VENV_DIR python=${PY_VER} -y"
                gb_ok "conda environment created"
                ;;
            uv)
                gb_info "Creating venv (python=${PY_VER})…"
                "$UV" venv "$VENV_DIR" --python "$PY_VER" --quiet \
                    || die "Failed to create virtualenv" \
                           "Python ${PY_VER} unavailable" \
                           "uv python install ${PY_VER}"
                gb_ok "venv created"
                ;;
        esac
    fi

    PYTHON="$VENV_DIR/bin/python"
    PIP="$VENV_DIR/bin/pip"

    # ── [4/5] Install package + extras ───────────────────────────────────────
    gb_step 4 5 "Install greenboost-cli"
    echo -e ""

    if "$PIP" show greenboost-cli 2>/dev/null \
            | grep -qF "Editable project location: $SCRIPT_DIR"; then
        gb_ok "greenboost-cli already installed (editable from $SCRIPT_DIR) — skipping"
    else
        gb_info "Installing core package from $SCRIPT_DIR…"
        "$PIP" install -e "$SCRIPT_DIR" --quiet &
        gb_spin $! "Installing dependencies (anthropic, openai, rich…)"
    fi

    # ── PyTorch via conda channels (GPU-aware) ────────────────────────────────
    if [[ "$PKG_MANAGER" == "mamba" || "$PKG_MANAGER" == "conda" ]]; then
        local _torch_ver
        _torch_ver=$("$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || true)
        if [[ -n "$_torch_ver" ]] \
                && "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            echo -e ""
            gb_ok "PyTorch already installed (${_torch_ver}, CUDA available) — skipping"
        else
            CUDA_VER="$(detect_cuda_ver)"
            PT_CUDA_VER="$(map_pytorch_cuda_ver "$CUDA_VER")"
            echo -e ""
            if [[ "$CUDA_VER" != "$PT_CUDA_VER" ]]; then
                gb_info "CUDA version detected: ${CUDA_VER}  →  using pytorch-cuda=${PT_CUDA_VER} (latest supported)"
            else
                gb_info "CUDA version detected: ${CUDA_VER}"
            fi
            echo -e ""
            if is_interactive; then
                if confirm "Install PyTorch via conda channels? (best CUDA support, ~2 GB)" "y"; then
                    gb_info "Installing pytorch + pytorch-cuda=${PT_CUDA_VER} via pytorch / nvidia / conda-forge channels…"
                    local _pt_log; _pt_log="$(mktemp /tmp/gb_pytorch_XXXXXX.log)"
                    "$CONDA_CMD" install pytorch "pytorch-cuda=${PT_CUDA_VER}" \
                        -c pytorch -c nvidia -c conda-forge -p "$VENV_DIR" -y -q \
                        >"$_pt_log" 2>&1 &
                    local _pt_pid=$!
                    gb_spin $_pt_pid "Installing PyTorch with CUDA ${PT_CUDA_VER}…"
                    if ! wait $_pt_pid; then
                        gb_warn "conda PyTorch install failed — trying pip fallback…"
                        gb_info "conda log: $_pt_log"
                        local _cu_tag="cu${PT_CUDA_VER/./}"
                        "$PIP" install torch --index-url \
                            "https://download.pytorch.org/whl/${_cu_tag}" --quiet \
                            && gb_ok "PyTorch installed via pip (CUDA ${PT_CUDA_VER})" \
                            || gb_warn "PyTorch install failed — install later: pip install torch"
                    fi
                    rm -f "$_pt_log"
                fi
            fi
        fi
    fi

    # ── Optional pip extras ───────────────────────────────────────────────────
    echo -e ""
    gb_info "Optional extras (install later with:  pip install 'greenboost-cli[extra]')"
    echo -e ""

    EXTRAS=""
    if is_interactive; then
        if _pkg_importable sentence_transformers; then
            gb_ok "RAG extras already installed — skipping"
        elif confirm "Install RAG extras? (sentence-transformers — ~500 MB)" "y"; then
            EXTRAS="${EXTRAS},rag"
        fi
        if _pkg_importable diffusers && _pkg_importable torchao; then
            gb_ok "Diffusion extras already installed — skipping"
        elif confirm "Install Diffusion extras? (diffusers, torchao — ~1 GB)" "y"; then
            EXTRAS="${EXTRAS},diffusion"
        fi
        if _pkg_importable fitz; then
            gb_ok "PDF extras already installed — skipping"
        elif confirm "Install PDF extras? (pymupdf)" "y"; then
            EXTRAS="${EXTRAS},pdf"
        fi
        if _pkg_importable cv2; then
            gb_ok "OpenCV extras already installed — skipping"
        elif confirm "Install OpenCV extras? (postprocessing for diffusion)" "y"; then
            EXTRAS="${EXTRAS},opencv"
        fi
    fi

    if [[ -n "$EXTRAS" ]]; then
        EXTRAS="${EXTRAS#,}"   # strip leading comma
        gb_info "Installing extras: ${EXTRAS}…"
        "$PIP" install -e "${SCRIPT_DIR}[${EXTRAS}]" --quiet &
        gb_spin $! "Installing extras (${EXTRAS})…"
    fi

    # Verify entry point
    [[ -x "$VENV_DIR/bin/greenboost-cli" ]] \
        || die "greenboost-cli entry point missing" \
               "pyproject.toml may be missing the script entry" \
               "Check [project.scripts] in pyproject.toml"

    # ── [5/5] Symlinks ────────────────────────────────────────────────────────
    gb_step 5 5 "Create symlinks in $BIN_DIR"
    echo -e ""

    mkdir -p "$BIN_DIR"
    ln -sf "$VENV_DIR/bin/greenboost-cli" "$SYMLINK"
    gb_ok "Symlink: $SYMLINK → $VENV_DIR/bin/greenboost-cli"

    if [[ -x "$VENV_DIR/bin/gb" ]]; then
        ln -sf "$VENV_DIR/bin/gb" "$SYMLINK_GB"
        gb_ok "Symlink: $SYMLINK_GB → $VENV_DIR/bin/gb"
    else
        ln -sf "$VENV_DIR/bin/greenboost-cli" "$SYMLINK_GB"
        gb_ok "Symlink: $SYMLINK_GB → $VENV_DIR/bin/greenboost-cli (alias)"
    fi

    if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
        echo -e ""
        gb_warn "$BIN_DIR is not in your PATH."
        echo -e "  ${C_DIM}Add to ~/.bashrc or ~/.zshrc:${C_RESET}"
        echo -e "    ${C_AMBER}export PATH=\"\$HOME/.local/bin:\$PATH\"${C_RESET}"
    fi

    # Smoke test
    echo -e ""
    gb_info "Smoke test…"
    if "$VENV_DIR/bin/greenboost-cli" --version &>/dev/null; then
        gb_ok "greenboost-cli --version OK"
    else
        gb_warn "Smoke test failed — check output above"
    fi

    # Run setup wizard
    echo -e ""
    # ── Verify torch / sentence-transformers / transformers all import ────
    # If the install left them in an inconsistent state (partial wheels,
    # CUDA-driver mismatch with the conda-resolved torch), auto-repair from
    # the right PyTorch wheel index. Quiet on the happy path.
    echo -e ""
    gb_info "Verifying ML stack import…"
    PIP="$VENV_DIR/bin/pip"
    PYTHON="$VENV_DIR/bin/python"
    if ! _torch_ensure_healthy "install"; then
        gb_warn "ML stack still degraded — RAG / skill / sharding will not work."
        gb_info "Manual variant override:  bash install.sh repair cu121"
    fi

    if is_interactive; then
        if confirm "Run setup wizard now? (choose backend + model)" "y"; then
            echo -e ""
            "$VENV_DIR/bin/greenboost-cli" --setup
        fi
    fi

    echo -e ""
    gb_separator
    echo -e ""
    echo -e "  ${C_LIME}${C_BOLD}✓  Installation complete!${C_RESET}"
    echo -e ""
    echo -e "  ${C_GRAY}Quick start:${C_RESET}"
    echo -e "    ${C_CYAN}greenboost-cli web${C_RESET}          ${C_DIM}# open web dashboard (http://localhost:7821)${C_RESET}"
    echo -e "    ${C_CYAN}greenboost-cli${C_RESET}  or  ${C_CYAN}gb${C_RESET}     ${C_DIM}# start interactive REPL${C_RESET}"
    echo -e "    ${C_CYAN}greenboost-cli --setup${C_RESET}      ${C_DIM}# configure backend & model${C_RESET}"
    echo -e "    ${C_CYAN}greenboost-cli help${C_RESET}         ${C_DIM}# full command reference${C_RESET}"
    echo -e ""
    echo -e "  ${C_GRAY}Env manager:${C_RESET}  ${C_DIM}${PKG_MANAGER} → ${VENV_DIR}${C_RESET}"
    echo -e "  ${C_GRAY}Uninstall:${C_RESET}    ${C_CYAN}bash ${SCRIPT_DIR}/install.sh uninstall${C_RESET}"
    echo -e ""
}

# ── update ────────────────────────────────────────────────────────────────────
cmd_update() {
    gb_header

    detect_pkg_manager || true
    resolve_named_env || true   # sets VENV_DIR to named env if it exists

    if [[ ! -d "$VENV_DIR" ]]; then
        gb_fail "GreenBoost CLI is not installed. Run: bash install.sh install"
        exit 1
    fi

    PIP="$VENV_DIR/bin/pip"

    local _old_ver
    _old_ver="$("$VENV_DIR/bin/greenboost-cli" --version 2>/dev/null || echo 'unknown')"

    # ── [1/2] Refresh the package code ───────────────────────────────────────
    gb_step 1 2 "Refresh greenboost-cli code"
    echo -e ""
    gb_info "Installed version: ${_old_ver}"
    gb_info "Source: ${SCRIPT_DIR}"
    echo -e ""
    # --no-deps: update only the greenboost-cli package itself.
    # Large already-installed packages (torch, sentence-transformers, etc.) are untouched.
    "$PIP" install -e "$SCRIPT_DIR" --no-deps --quiet &
    gb_spin $! "Refreshing package (skipping installed deps)…"

    # ── [2/2] Install missing/new dependencies ────────────────────────────────
    gb_step 2 2 "Sync new dependencies"
    echo -e ""
    gb_info "Installing any new dependencies added since last install…"
    gb_info "(existing packages are kept as-is; nothing is re-downloaded)"
    echo -e ""
    # --upgrade-strategy only-if-needed: installs packages that are missing or
    # no longer satisfy constraints, but never upgrades ones that already do.
    local _dep_log; _dep_log="$(mktemp /tmp/gb_update_XXXXXX.log)"
    "$PIP" install -e "$SCRIPT_DIR" --upgrade-strategy only-if-needed --quiet \
        >"$_dep_log" 2>&1 &
    local _dep_pid=$!
    gb_spin $_dep_pid "Syncing dependencies…"
    if ! wait $_dep_pid; then
        gb_warn "Dependency sync had issues — see: $_dep_log"
    else
        rm -f "$_dep_log"
    fi

    local _new_ver
    _new_ver="$("$VENV_DIR/bin/greenboost-cli" --version 2>/dev/null || echo 'unknown')"

    # ── Verify ML stack survived the update ─────────────────────────────────
    # A dep sync can break torch/sentence-transformers if a transitive
    # constraint nudges torch to a new version with mismatched C++ syms.
    # Run the same probe Install runs and auto-repair if needed.
    echo -e ""
    gb_info "Verifying ML stack import…"
    PYTHON="$VENV_DIR/bin/python"
    if ! _torch_ensure_healthy "update"; then
        gb_warn "ML stack degraded after update — RAG / skill / sharding will not work."
        gb_info "Manual variant override:  bash install.sh repair cu121"
    fi

    echo -e ""
    gb_separator
    echo -e ""
    gb_ok "Update complete."
    if [[ "$_old_ver" != "$_new_ver" ]]; then
        gb_info "Version: ${_old_ver}  →  ${_new_ver}"
    else
        gb_info "Version: ${_new_ver}  (code refreshed, no version bump)"
    fi
    echo -e ""
}

# ── Torch health helpers (shared by install / update / repair) ───────────────
#
# `_torch_health_probe` runs the import probe and returns 0 (healthy) or 1
# (broken). It writes a short human summary to stdout — callers decide
# whether to show it.
#
# `_torch_pick_variant` chooses the right PyTorch wheel index for this host:
#   - explicit override (env REPAIR_VARIANT or arg) wins
#   - Blackwell GPU (compute capability 12.x) → cu128
#   - else, derive from `detect_cuda_ver` / `map_pytorch_cuda_ver`
#
# `_torch_force_reinstall <variant>` force-reinstalls torch+torchvision from
# the variant's wheel index (--no-deps so it doesn't disturb anything else).
# Returns 0 on success, 1 on failure.
#
# Callers must have $VENV_DIR / $PIP / $PYTHON set.

_pkg_importable() {
    "$PYTHON" -c "import $1" >/dev/null 2>&1
}

_torch_health_probe() {
    "$PYTHON" -c "
import sys
errs = []
try:
    import torch  # noqa: F401
except Exception as e:
    errs.append(('torch', type(e).__name__, str(e)[:160]))
else:
    try:
        import sentence_transformers  # noqa: F401
    except Exception as e:
        errs.append(('sentence_transformers', type(e).__name__, str(e)[:160]))
    try:
        import transformers  # noqa: F401
    except Exception as e:
        errs.append(('transformers', type(e).__name__, str(e)[:160]))
if errs:
    for name, etype, msg in errs:
        sys.stderr.write(f'    {name}: {etype}: {msg}\n')
    sys.exit(1)
sys.exit(0)
" 2>&1
    return $?
}

_torch_pick_variant() {
    local override="${1:-${REPAIR_VARIANT:-}}"
    if [[ -n "$override" ]]; then
        echo "${override#cu}"
        return 0
    fi
    if gpu_is_blackwell; then
        echo "128"
        return 0
    fi
    local cuda_ver; cuda_ver=$(detect_cuda_ver)
    local mapped; mapped=$(map_pytorch_cuda_ver "$cuda_ver")
    echo "${mapped//./}"
}

_torch_force_reinstall() {
    local variant="$1"
    local wheel_url; wheel_url=$(pytorch_wheel_index "$variant")
    local _log; _log="$(mktemp /tmp/gb_torch_XXXXXX.log)"

    # IMPORTANT: do NOT pass --no-deps. The PyTorch wheels declare
    # nvidia-cuda-cupti-* / nvidia-cuda-runtime-* as dependencies and the
    # binary will SIGSEGV / ImportError at runtime if the matching CUDA
    # runtime libs aren't installed alongside. We do pass --force-reinstall
    # to torch + torchvision so the .so files actually get refreshed even
    # if pip thinks the version constraint is already met.
    "$PIP" install --force-reinstall --quiet \
        torch torchvision --index-url "$wheel_url" \
        --extra-index-url "https://pypi.org/simple" \
        >"$_log" 2>&1 &
    local _pid=$!
    gb_spin $_pid "Reinstalling torch + CUDA runtime from ${wheel_url} (~2-3 GB)…"
    if ! wait $_pid; then
        gb_warn "torch reinstall failed — last lines of log:"
        tail -8 "$_log" | sed 's/^/    /' >&2
        gb_info "Full log: $_log"
        return 1
    fi
    rm -f "$_log"
    return 0
}

# Run a probe; if broken, attempt one automatic repair. Returns 0 if
# torch is healthy at the end (either it was, or repair succeeded).
# Quiet on success — only narrates if it had to repair.
_torch_ensure_healthy() {
    local context="${1:-install}"   # purely cosmetic — appears in messages

    if _torch_health_probe >/dev/null 2>&1; then
        gb_ok "torch + sentence-transformers + transformers all import cleanly"
        return 0
    fi

    gb_warn "ML stack import failed after ${context} — auto-repairing"
    local probe_err; probe_err=$(_torch_health_probe 2>&1 || true)
    echo -e "$probe_err" | head -3 | sed 's/^/    /'
    echo -e ""

    local variant; variant=$(_torch_pick_variant)
    gb_info "Selected PyTorch wheel variant: cu${variant}"
    if _torch_force_reinstall "$variant"; then
        if _torch_health_probe >/dev/null 2>&1; then
            gb_ok "Repair successful — ML stack now imports cleanly"
            return 0
        fi
        gb_warn "Reinstall completed but probe still fails. Try a different variant:"
        gb_info "  bash install.sh repair cu121"
        gb_info "  bash install.sh repair cpu"
        return 1
    fi
    return 1
}

# ── repair ────────────────────────────────────────────────────────────────────
# Diagnose-and-fix flow for partial-install or broken-symbol cases.
# Common symptoms it handles:
#   - AttributeError: module 'torch._C._linalg' has no attribute 'linalg__powsum'
#   - ImportError: libtorch_cpu.so: undefined symbol cuptiActivityEnableDriverApi
#   - sentence-transformers / transformers refusing to import
#   - torch present but built for an older CUDA than the GPU (e.g. Blackwell
#     sm_120 with a torch built for cu124 → "no kernel image" at runtime)
#
# Strategy: probe; force-reinstall torch + torchvision from the correct
# PyTorch wheel index (cu128 for Blackwell, otherwise the conda-resolved
# CUDA version, with `cpu` as an explicit override); then re-verify.
cmd_repair() {
    gb_header

    detect_pkg_manager || true
    resolve_named_env || true

    if [[ ! -d "$VENV_DIR" ]]; then
        gb_fail "GreenBoost CLI is not installed. Run: bash install.sh install"
        exit 1
    fi

    PIP="$VENV_DIR/bin/pip"
    PYTHON="$VENV_DIR/bin/python"

    # Honour an explicit variant override:  bash install.sh repair cu128
    local variant_override="${REPAIR_VARIANT:-${2:-}}"

    # ── [1/3] Probe ──────────────────────────────────────────────────────────
    gb_step 1 3 "Probe environment"
    echo -e ""
    gb_info "Env:    $VENV_DIR"
    gb_info "Python: $($PYTHON -V 2>&1)"
    echo -e ""

    local probe_rc=0
    _torch_health_probe || probe_rc=$?
    if [[ $probe_rc -eq 0 ]]; then
        echo -e ""
        gb_ok "All ML imports succeed — nothing to repair."
        echo -e ""
        return 0
    fi
    echo -e ""

    # ── [2/3] Reinstall torch + torchvision ──────────────────────────────────
    gb_step 2 3 "Reinstall torch + torchvision"
    echo -e ""
    local variant; variant=$(_torch_pick_variant "$variant_override")
    gb_info "Selected PyTorch wheel variant: cu${variant}"
    gb_info "This is a ~2-3 GB download. Other packages stay as-is."
    echo -e ""

    if ! _torch_force_reinstall "$variant"; then
        die "torch reinstall failed" \
            "Wheel index may be wrong for this GPU/driver combo" \
            "Try:  bash install.sh repair cpu   (or cu121 / cu124 / cu128)"
    fi

    # ── [3/3] Verify ─────────────────────────────────────────────────────────
    gb_step 3 3 "Verify"
    echo -e ""
    local verify_rc=0
    "$PYTHON" -c "
import torch
print(f'  torch  : {torch.__version__}  (cuda built: {torch.version.cuda})')
print(f'  cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device : {torch.cuda.get_device_name(0)}  '
          f'(cc {torch.cuda.get_device_capability(0)})')
import sentence_transformers
print(f'  sentence-transformers: {sentence_transformers.__version__}')
import transformers
print(f'  transformers         : {transformers.__version__}')
" || verify_rc=$?

    echo -e ""
    if [[ $verify_rc -ne 0 ]]; then
        gb_warn "Verify failed — ML stack still broken. Try a different variant:"
        echo -e "  ${C_CYAN}bash install.sh repair cu121${C_RESET}"
        echo -e "  ${C_CYAN}bash install.sh repair cpu${C_RESET}"
        exit 1
    fi

    gb_separator
    echo -e ""
    gb_ok "Repair complete — gb rag-search / skill-route should work now."
    if command -v gb &>/dev/null; then
        gb_info "Sanity check:  gb rag-search 'hello' --top-k 1 --json"
    fi
    echo -e ""
}

# ── uninstall ─────────────────────────────────────────────────────────────────
cmd_uninstall() {
    gb_header
    echo -e "  ${C_GRAY}This will remove:${C_RESET}"
    echo -e "  ${C_DIM}  Symlink : $SYMLINK${C_RESET}"
    echo -e "  ${C_DIM}  Symlink : $SYMLINK_GB${C_RESET}"
    echo -e "  ${C_DIM}  Env     : $VENV_DIR${C_RESET}"
    echo -e ""

    if is_interactive; then
        confirm "Remove greenboost-cli and its environment?" "n" || {
            gb_info "Cancelled."
            echo -e ""
            exit 0
        }
    fi
    echo -e ""

    [[ -L "$SYMLINK"    || -f "$SYMLINK"    ]] && { rm -f "$SYMLINK";    gb_ok "Removed symlink: $SYMLINK"; }
    [[ -L "$SYMLINK_GB" || -f "$SYMLINK_GB" ]] && { rm -f "$SYMLINK_GB"; gb_ok "Removed symlink: $SYMLINK_GB"; }
    detect_pkg_manager || true
    if resolve_named_env; then
        # Named env is managed by the user via mamba — only pip-uninstall the package
        if [[ -x "$VENV_DIR/bin/pip" ]]; then
            "$VENV_DIR/bin/pip" uninstall greenboost-cli -y 2>/dev/null || true
            gb_ok "Removed greenboost-cli package from '$GB_NAMED_ENV' env"
        fi
        gb_info "Named env '$GB_NAMED_ENV' at $VENV_DIR preserved (managed by mamba)"
    elif [[ -d "$VENV_DIR" ]]; then
        rm -rf "$VENV_DIR"
        gb_ok "Removed env: $VENV_DIR"
    fi

    echo -e ""
    gb_ok "greenboost-cli uninstalled."
    gb_info "Config preserved at: $CONFIG_FILE  (remove manually if desired)"
    echo -e ""
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
    gb_header

    if [[ -L "$SYMLINK" && -x "$SYMLINK" ]]; then
        local _target; _target=$(readlink "$SYMLINK")
        echo -e "  ${C_LIME}●${C_RESET}  ${C_GRAY}greenboost-cli installed${C_RESET}  ${C_DIM}${_target}${C_RESET}"
    elif [[ -L "$SYMLINK" ]]; then
        echo -e "  ${C_RED}✗${C_RESET}  ${C_GRAY}symlink broken${C_RESET}  ${C_DIM}$SYMLINK${C_RESET}"
    else
        echo -e "  ${C_DIM}○${C_RESET}  ${C_GRAY}greenboost-cli not installed${C_RESET}"
    fi

    if [[ -L "$SYMLINK_GB" && -x "$SYMLINK_GB" ]]; then
        local _target_gb; _target_gb=$(readlink "$SYMLINK_GB")
        echo -e "  ${C_LIME}●${C_RESET}  ${C_GRAY}gb symlink present${C_RESET}  ${C_DIM}${_target_gb}${C_RESET}"
    else
        echo -e "  ${C_DIM}○${C_RESET}  ${C_GRAY}gb symlink absent${C_RESET}"
    fi

    if [[ -d "$VENV_DIR" ]]; then
        local _py_ver _env_type
        _py_ver=$("$VENV_DIR/bin/python" --version 2>&1 || echo "unknown")
        if is_conda_env; then
            _env_type="conda/mamba"
        else
            _env_type="venv (uv/pip)"
        fi
        echo -e "  ${C_LIME}●${C_RESET}  ${C_GRAY}env present${C_RESET}  ${C_DIM}$VENV_DIR  ($_py_ver, $_env_type)${C_RESET}"
    else
        echo -e "  ${C_DIM}○${C_RESET}  ${C_GRAY}env absent${C_RESET}"
    fi

    if [[ -f "$CONFIG_FILE" ]]; then
        echo -e "  ${C_LIME}●${C_RESET}  ${C_GRAY}Config found${C_RESET}  ${C_DIM}$CONFIG_FILE${C_RESET}"
    else
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}No config yet${C_RESET}  ${C_DIM}run: greenboost-cli --setup${C_RESET}"
    fi

    # Package manager in use
    detect_pkg_manager || true
    echo -e ""
    case "$PKG_MANAGER" in
        mamba) echo -e "  ${C_LIME}●${C_RESET}  ${C_GRAY}mamba available${C_RESET}  ${C_DIM}$(mamba --version 2>&1 | head -1)${C_RESET}" ;;
        conda) echo -e "  ${C_LIME}●${C_RESET}  ${C_GRAY}conda available${C_RESET}  ${C_DIM}$(conda --version 2>&1)${C_RESET}" ;;
        uv)
            if command -v uv &>/dev/null; then
                echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}uv only — no mamba/conda${C_RESET}  ${C_DIM}(GPU extras need conda channels)${C_RESET}"
            else
                echo -e "  ${C_RED}✗${C_RESET}  ${C_GRAY}no package manager found${C_RESET}"
            fi
            ;;
    esac

    echo -e ""
    if [[ -L "$SYMLINK" && -x "$SYMLINK" ]]; then
        gb_separator
        echo -e ""
        echo -e "  ${C_GRAY}Launch:${C_RESET}  ${C_CYAN}greenboost-cli${C_RESET}  or  ${C_CYAN}gb${C_RESET}"
        echo -e "  ${C_GRAY}Setup:${C_RESET}   ${C_CYAN}greenboost-cli --setup${C_RESET}"
        echo -e ""
    fi
}

# ── help ──────────────────────────────────────────────────────────────────────
cmd_help() {
    gb_header

    echo -e "  ${C_CYAN}${C_BOLD}Commands${C_RESET}"
    echo -e ""
    printf "  ${C_LIME}${C_BOLD}%-14s${C_RESET}  ${C_GRAY}%s${C_RESET}\n" \
        "install"   "Full install (first-time setup)"
    printf "  ${C_LIME}${C_BOLD}%-14s${C_RESET}  ${C_GRAY}%s${C_RESET}\n" \
        "uninstall" "Remove greenboost-cli and its environment"
    printf "  ${C_LIME}${C_BOLD}%-14s${C_RESET}  ${C_GRAY}%s${C_RESET}\n" \
        "status"    "Show installation and package-manager status"
    printf "  ${C_LIME}${C_BOLD}%-14s${C_RESET}  ${C_GRAY}%s${C_RESET}\n" \
        "update"    "Update to latest version"
    printf "  ${C_LIME}${C_BOLD}%-14s${C_RESET}  ${C_GRAY}%s${C_RESET}\n" \
        "repair"    "Diagnose + reinstall a broken torch / ML stack"
    printf "  ${C_LIME}${C_BOLD}%-14s${C_RESET}  ${C_GRAY}%s${C_RESET}\n" \
        "help"      "Show this help"
    echo -e ""
    gb_separator
    echo -e ""
    echo -e "  ${C_CYAN}${C_BOLD}Package manager priority${C_RESET}"
    echo -e ""
    echo -e "  ${C_LIME}  1${C_RESET}  ${C_GRAY}mamba${C_RESET}   ${C_DIM}Best GPU/CUDA support — conda channels resolve torch/diffusers${C_RESET}"
    echo -e "  ${C_LIME}  2${C_RESET}  ${C_GRAY}conda${C_RESET}   ${C_DIM}Conda without mamba (slower solves)${C_RESET}"
    echo -e "  ${C_LIME}  3${C_RESET}  ${C_GRAY}uv${C_RESET}      ${C_DIM}Pip-only — works fine for cloud/CPU use${C_RESET}"
    echo -e ""
    echo -e "  ${C_DIM}mamba recommended: conda install -c conda-forge mamba${C_RESET}"
    echo -e ""
    gb_separator
    echo -e ""
    echo -e "  ${C_CYAN}${C_BOLD}After installation${C_RESET}"
    echo -e ""
    echo -e "  ${C_GRAY}Launch REPL:${C_RESET}        ${C_CYAN}greenboost-cli${C_RESET}  or  ${C_CYAN}gb${C_RESET}"
    echo -e "  ${C_GRAY}Setup wizard:${C_RESET}       ${C_CYAN}greenboost-cli --setup${C_RESET}"
    echo -e "  ${C_GRAY}Download models:${C_RESET}    ${C_CYAN}gb${C_RESET}  then  ${C_CYAN}/download-models${C_RESET}"
    echo -e "  ${C_GRAY}Non-interactive:${C_RESET}    ${C_CYAN}greenboost-cli -p \"your prompt\"${C_RESET}"
    echo -e "  ${C_GRAY}GreenBoost status:${C_RESET}  ${C_CYAN}gb${C_RESET}  then  ${C_CYAN}/gb-status${C_RESET}"
    echo -e "  ${C_GRAY}Web dashboard:${C_RESET}      ${C_CYAN}gb${C_RESET}  then  ${C_CYAN}/dashboard${C_RESET}"
    echo -e ""
    gb_separator
    echo -e ""
    echo -e "  ${C_CYAN}${C_BOLD}Paths${C_RESET}"
    echo -e ""
    echo -e "  ${C_DIM}Env     ${C_RESET}${C_GRAY}$VENV_DIR${C_RESET}"
    echo -e "  ${C_DIM}Symlink ${C_RESET}${C_GRAY}$SYMLINK${C_RESET}"
    echo -e "  ${C_DIM}Alias   ${C_RESET}${C_GRAY}$SYMLINK_GB${C_RESET}"
    echo -e "  ${C_DIM}Config  ${C_RESET}${C_GRAY}$CONFIG_FILE${C_RESET}"
    echo -e "  ${C_DIM}Source  ${C_RESET}${C_GRAY}$SCRIPT_DIR${C_RESET}"
    echo -e ""
}

# ── wizard ────────────────────────────────────────────────────────────────────
cmd_wizard() {
    gb_header

    echo -e "  ${C_CYAN}${C_BOLD}What would you like to do?${C_RESET}"
    echo -e ""
    printf "  ${C_LIME}${C_BOLD}%s${C_RESET}  ${C_GRAY}%-14s${C_RESET}  ${C_DIM}%s${C_RESET}\n" \
        "1" "Install"   "First-time setup"
    printf "  ${C_LIME}${C_BOLD}%s${C_RESET}  ${C_GRAY}%-14s${C_RESET}  ${C_DIM}%s${C_RESET}\n" \
        "2" "Uninstall" "Remove greenboost-cli"
    printf "  ${C_LIME}${C_BOLD}%s${C_RESET}  ${C_GRAY}%-14s${C_RESET}  ${C_DIM}%s${C_RESET}\n" \
        "3" "Status"    "Show installation info"
    printf "  ${C_LIME}${C_BOLD}%s${C_RESET}  ${C_GRAY}%-14s${C_RESET}  ${C_DIM}%s${C_RESET}\n" \
        "4" "Update"    "Update to latest version"
    printf "  ${C_LIME}${C_BOLD}%s${C_RESET}  ${C_GRAY}%-14s${C_RESET}  ${C_DIM}%s${C_RESET}\n" \
        "5" "Repair"    "Fix broken torch / ML stack"
    printf "  ${C_LIME}${C_BOLD}%s${C_RESET}  ${C_GRAY}%-14s${C_RESET}  ${C_DIM}%s${C_RESET}\n" \
        "6" "Help"      "Show all commands"
    echo -e ""

    local choice
    read -r -p "$(echo -e "  ${C_AMBER}❯${C_RESET} ")" choice
    echo -e ""

    case "$choice" in
        1) cmd_install   ;;
        2) cmd_uninstall ;;
        3) cmd_status    ;;
        4) cmd_update    ;;
        5) cmd_repair    ;;
        6) cmd_help      ;;
        *) gb_info "Cancelled."; echo -e ""; exit 0 ;;
    esac
}

# ── Entry point ───────────────────────────────────────────────────────────────
COMMAND="${1:-}"

case "$COMMAND" in
    install)        cmd_install   ;;
    uninstall)      cmd_uninstall ;;
    status)         cmd_status    ;;
    update)         cmd_update    ;;
    repair|fix|doctor) cmd_repair  "$@" ;;
    help|--help|-h) cmd_help      ;;
    "")
        if is_interactive; then
            cmd_wizard
        else
            cmd_help
        fi
        ;;
    *)
        gb_fail "Unknown command: $COMMAND"
        echo -e "  ${C_DIM}Run: bash install.sh help${C_RESET}"
        echo -e ""
        exit 1
        ;;
esac
