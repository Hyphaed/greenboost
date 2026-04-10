#!/usr/bin/env bash
# push_to_gitlab.sh — push main (+ optional tags) to GitLab
# Usage: ./push_to_gitlab.sh [tag1 tag2 ...]
#
# All regular pushes go to origin (backup drive).
# This script is the only sanctioned path to GitLab.
set -euo pipefail

# ---- Brand palette ---------------------------------------------------------
_gb_truecolor() { [[ "${COLORTERM:-}" =~ ^(truecolor|24bit)$ ]]; }
if _gb_truecolor; then
    C_VIOLET='\033[38;2;108;113;196m'
    C_LIME='\033[38;2;230;255;60m'
    C_GRAY='\033[38;2;208;207;204m'
    C_CYAN='\033[38;2;48;200;255m'
    C_AMBER='\033[38;2;255;191;0m'
    C_RED='\033[38;2;255;92;50m'
else
    C_VIOLET='\033[0;34m'
    C_LIME='\033[0;32m'
    C_GRAY='\033[0;37m'
    C_CYAN='\033[0;36m'
    C_AMBER='\033[1;33m'
    C_RED='\033[0;31m'
fi
C_BOLD='\033[1m'
C_DIM='\033[2m'
C_RESET='\033[0m'

GB_SPIN_FRAMES=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")

gb_ok()      { echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}$*${C_RESET}"; }
gb_fail()    { echo -e "  ${C_RED}✗${C_RESET}  $*" >&2; }
gb_warn_ui() { echo -e "  ${C_AMBER}⚠${C_RESET}  $*"; }
gb_info()    { echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}$*${C_RESET}"; }
gb_step()    { echo -e "\n${C_CYAN}${C_BOLD}  [$1/$2]${C_RESET} ${C_GRAY}${C_BOLD}$3${C_RESET}"; }

gb_spin() {
    local pid=$1 msg="$2" i=0 start_time
    start_time=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
        local elapsed=$(( $(date +%s) - start_time ))
        local time_str
        if (( elapsed >= 60 )); then
            time_str=$(printf "%d:%02d" $((elapsed / 60)) $((elapsed % 60)))
        else
            time_str="${elapsed}s"
        fi
        printf "\r  ${C_LIME}%s${C_RESET}  ${C_DIM}%s${C_RESET} ${C_GRAY}[%s]${C_RESET}   " \
            "${GB_SPIN_FRAMES[$((i % ${#GB_SPIN_FRAMES[@]}))]}" "$msg" "$time_str"
        sleep 0.08
        (( i++ )) || true
    done
    printf "\r  ${C_LIME}✓${C_RESET}  ${C_GRAY}%s${C_RESET}                             \n" "$msg"
}

# ---- Sanity checks ---------------------------------------------------------
REMOTE="gitlab"

if ! git remote get-url "$REMOTE" &>/dev/null; then
    gb_fail "Remote '${REMOTE}' not found. Run: git remote add gitlab https://gitlab.com/IsolatedOctopi/greenboost"
    exit 1
fi

BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
if [[ "$BRANCH" != "main" ]]; then
    gb_warn_ui "Current branch is '${BRANCH}', not 'main'."
    read -rp "  ${C_AMBER}${C_BOLD}❯${C_RESET}  Push '${BRANCH}' to GitLab? [y/N]: " confirm
    [[ "${confirm,,}" == "y" ]] || { gb_info "Aborted."; exit 0; }
fi

TAGS=("$@")
TOTAL=$(( 1 + ${#TAGS[@]} ))

# ---- Push main -------------------------------------------------------------
gb_step 1 "$TOTAL" "Pushing ${BRANCH} → gitlab"

git push "$REMOTE" "$BRANCH" &
gb_spin $! "git push gitlab ${BRANCH}"

gb_ok "Branch '${BRANCH}' pushed to GitLab"

# ---- Push tags (if any) ----------------------------------------------------
IDX=2
for tag in "${TAGS[@]}"; do
    gb_step "$IDX" "$TOTAL" "Pushing tag ${tag} → gitlab"

    if ! git tag -l "$tag" | grep -q "^${tag}$"; then
        gb_fail "Tag '${tag}' does not exist locally — skipping"
        (( IDX++ )) || true
        continue
    fi

    git push "$REMOTE" "$tag" &
    gb_spin $! "git push gitlab ${tag}"

    gb_ok "Tag '${tag}' pushed to GitLab"
    (( IDX++ )) || true
done

echo ""
gb_info "Done. GitLab remote: $(git remote get-url gitlab)"
