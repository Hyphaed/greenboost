#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
#
# Run with: bash tests/bench/pull_placement_sweep_model.sh
#
# WHY THIS EXISTS: Phase 2's placement sweep (--n-cpu-moe x -ot x
# GGML_OP_OFFLOAD_MIN_BATCH, Finding 9) needs a MoE model that is NOT on
# gb_synapse.ARCH_CPU_SPLIT_BROKEN ({"qwen35moe"} as of this session). Every
# MoE-shaped model already pulled locally either IS qwen35moe (same crash)
# or turned out dense on closer inspection (see the plan's Phase 2 status).
# Pulling a new model is real disk (~26 GB) and bandwidth cost, an explicit
# decision, not something done silently on your behalf.
#
# WHY MIXTRAL-8x7B: the most mature, longest-supported MoE architecture in
# llama.cpp (years of testing, unlike qwen35moe's newer, less-tested arch
# that's currently crashing on this exact box). Genuinely too large to fit
# this box's ~10 GB VRAM budget (46.7B total / 12.9B active params, Q4_K_M
# ~26 GB), so real --n-cpu-moe placement decisions actually get made,
# unlike a small MoE that would just fit fully in VRAM and never touch the
# sweep's interesting cases at all.
#
# This script only PULLS the model (gb_synapse.py's own download path,
# HuggingFace -> local GGUF, resumable, standard). It does NOT serve it,
# does NOT run the sweep, does NOT touch GPU state. Safe to run any time,
# including while other GPU work is active (this is a CPU/network/disk
# operation, no CUDA calls).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

REPO_SPEC="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF:Q4_K_M"

echo "=================================================================="
echo "Pulling $REPO_SPEC for Phase 2's placement sweep"
echo "=================================================================="
echo
echo "Free disk space:"
df -h /usr/local 2>/dev/null || df -h "$HOME" 2>/dev/null
echo
echo "Expected size: ~26 GB. This will take a while depending on your"
echo "connection. Safe to Ctrl-C and re-run later, gb_synapse.py's own"
echo "download path is resumable."
echo

python3 -c "
import gb_synapse as gs
entry = gs.pull('$REPO_SPEC')
print()
print('==================================================================')
print('PULLED OK')
print('==================================================================')
print(f'name:      {entry.name}')
print(f'arch:      {entry.arch}')
print(f'is_moe:    {entry.is_moe}')
print(f'n_bytes:   {entry.n_bytes / 2**30:.1f} GiB')
print(f'broken:    {entry.arch in gs.ARCH_CPU_SPLIT_BROKEN}  (must be False for the sweep to be useful)')
print()
print('Next step once this is done, run the placement sweep against it')
print('(see the plan file, Phase 2.1) using:')
print(f'  python3 tests/bench/run_gguf_decode.py --model \"{entry.name}\" ...')
"
