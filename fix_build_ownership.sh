#!/usr/bin/env bash
# Give the GreenBoost build trees back to the invoking user.
#
# Why this is needed
# ------------------
# Full Install runs as root but builds INSIDE this source checkout, so every
# artifact lands root-owned. A later non-root `cmake --build` then cannot
# overwrite them and fails with a linker/permission error that names no cause.
# Found 2026-08-18: 511 root-owned files under
# third_party/llama.cpp/build-synapse made the engine unbuildable except as
# root, which is also why a llama-kv-cache patch could not be deployed.
#
# gb_synapse.build_engine() now chowns the tree back automatically after a root
# build, so this script is a ONE-OFF repair for trees that are already poisoned.
#
# It changes ownership only. It deletes nothing.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_USER="${SUDO_USER:-$(id -un)}"

if [[ $EUID -ne 0 ]]; then
    echo "This needs root (it is changing ownership of root-owned files)."
    echo "Run:  sudo $0"
    exit 1
fi

if ! id -u "$TARGET_USER" >/dev/null 2>&1; then
    echo "Cannot resolve user '$TARGET_USER' , run with sudo from your own shell."
    exit 1
fi

echo "Repo:        $REPO"
echo "Giving back to: $TARGET_USER"
echo

total=0
for d in \
    "$REPO/third_party/llama.cpp/build-synapse" \
    "$REPO/third_party/llama.cpp/build" \
    "$REPO/third_party/gb_cutlass" \
    "$REPO/build"
do
    [[ -d "$d" ]] || continue
    n="$(find "$d" ! -user "$TARGET_USER" 2>/dev/null | wc -l)"
    if [[ "$n" -gt 0 ]]; then
        printf '  %-58s %6s file(s)\n' "${d#$REPO/}" "$n"
        chown -R "$TARGET_USER":"$TARGET_USER" "$d"
        total=$((total + n))
    fi
done

# Stray root-owned files anywhere else in the checkout (build_info, *.o, logs
# a root run dropped). Deliberately skips .git , if root touched that, chowning
# it silently is not the right call and you want to know.
stray="$(find "$REPO" -path "$REPO/.git" -prune -o ! -user "$TARGET_USER" -print 2>/dev/null | wc -l)"
if [[ "$stray" -gt 0 ]]; then
    printf '  %-58s %6s file(s)\n' "(elsewhere in the checkout)" "$stray"
    find "$REPO" -path "$REPO/.git" -prune -o ! -user "$TARGET_USER" -print0 2>/dev/null \
        | xargs -0 -r chown "$TARGET_USER":"$TARGET_USER"
    total=$((total + stray))
fi

git_root="$(find "$REPO/.git" ! -user "$TARGET_USER" 2>/dev/null | wc -l)"
if [[ "${git_root:-0}" -gt 0 ]]; then
    echo
    echo "  NOTE: $git_root file(s) under .git are root-owned and were NOT touched."
    echo "        Fix deliberately if you want it:  sudo chown -R $TARGET_USER:$TARGET_USER $REPO/.git"
fi

echo
if [[ "$total" -eq 0 ]]; then
    echo "Nothing to fix , the tree is already yours."
else
    echo "Done , $total file(s) returned to $TARGET_USER."
    echo "You can now rebuild without sudo:"
    echo "  python3 -c \"import gb_synapse as gs; gs.build_engine(src_dir='third_party/llama.cpp')\""
fi
