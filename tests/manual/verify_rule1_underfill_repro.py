#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Manual repro harness for workflow/known-issues.md's OPEN `gb_needs_overflow()`
Rule #1 underfill item (2026-08-10 update).

The original repro used Qwen3-VL-8B-{Thinking,Instruct}-FP8 via
SynapseTorchBackend; that checkpoint is no longer in this box's manifest, and
every torch-backend model still present (Qwen3-0.6B, mamba2-130m/2.7b-hf) is
far smaller than this box's ~9.3 GB per-model VRAM budget, so none of them
could force the overflow-with-real-headroom condition the bug needs.

`gb_needs_overflow()` / `gb_smart_overflow_alloc()` (greenboost_cuda_shim.c)
are shim-level CUDA allocator hooks , they run identically no matter which
Python-side backend (llama.cpp, torch, ...) called cudaMalloc, so the
reference workload (Qwen3.6-27B-Fable-Fusion, llama.cpp/GGUF, ~15 GB weights
, comfortably large enough to raise real headroom questions during load) is a
legitimate substitute repro target, even though it isn't the ORIGINAL
observation's backend. Treat a PASS here as real evidence for THIS repro
path only; a torch-backend re-test remains the more faithful option if a
mid-sized (~8-16 GB) FP8/safetensors checkpoint is ever pulled.

Usage (needs the GPU + kmod loaded, no root):
    python3 tests/manual/verify_rule1_underfill_repro.py [--ctx N] [--no-restore]

Serves MODEL fresh (stopping any existing serve of it first), polls dataflux
for `shim_transition`/`rule1_underfill` events for POLL_WINDOW_S seconds after
the server reports ready, classifies each transition as TRANSIENT (clears
within TRANSIENT_S of firing , matches the load-phase blip already observed
during this repo's 2026-08-10 M4 re-test, fb_phys_used_pct=85 at the
from=false to=true edge, self-resolved) or SUSTAINED (a real, ongoing
violation), cross-checks the final state against gb_semantics's governed
`rule1_underfilled` segment (the authoritative verdict , a raw shim_transition
event alone is not proof, see the same M4 re-test), and prints a PASS/
TRANSIENT/SUSTAINED verdict. Restores the reference serve config (ctx=65536)
afterward unless --no-restore is passed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gb_dataflux as gdf
import gb_semantics
import gb_synapse as gs

MODEL = "Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF"
# The reference workload's own documented serving config (CLAUDE.md's
# Reference Workload Rule; this repo's live launch command as of 2026-08-10)
# , restored at the end unless --no-restore is passed.
RESTORE_CTX = 65536

POLL_WINDOW_S = 90      # how long to watch dataflux after the server is ready
POLL_INTERVAL_S = 3
TRANSIENT_S = 15        # a rule1_underfill that clears within this long of
                         # firing is treated as a load-phase blip, not a
                         # sustained violation


def _serve(ctx: int) -> float:
    print(f"[repro] stopping any existing serve of {MODEL}...")
    try:
        gs.stop(MODEL)
    except Exception as e:
        print(f"[repro]   (nothing to stop, or stop failed harmlessly: {e})")
    start_ts = time.time()
    print(f"[repro] serving {MODEL} at ctx={ctx}...")
    st = gs.serve(MODEL, ctx=ctx, use_cluster=True)
    print(f"[repro] ready: llama_pid={st.llama_pid} port={st.port} "
          f"kv_type={st.kv_type} n_gpu_layers={st.n_gpu_layers}")
    return start_ts


def _poll_transitions(start_ts: float) -> list[dict]:
    print(f"[repro] polling dataflux for shim_transition/rule1_underfill "
          f"for {POLL_WINDOW_S}s...")
    seen: list[dict] = []
    seen_ts: set = set()
    deadline = time.time() + POLL_WINDOW_S
    while time.time() < deadline:
        for e in gdf.read_events(since_hours=0.05):
            ts = e.get("ts")
            if (e.get("kind") == "shim_transition"
                    and e.get("stage") == "rule1_underfill"
                    and ts is not None and ts >= start_ts
                    and ts not in seen_ts):
                seen_ts.add(ts)
                seen.append(e)
                print(f"[repro]   {e.get('from')}→{e.get('to')} at "
                      f"fb_phys_used_pct={e.get('fb_phys_used_pct')} "
                      f"t2_pressure={e.get('t2_pressure')} ts={ts}")
        time.sleep(POLL_INTERVAL_S)
    return seen


def _classify(transitions: list[dict]) -> str:
    """PASS (never fired), TRANSIENT (fired but self-cleared quickly, the
    known MODEL_LOAD-phase blip pattern), or SUSTAINED (a real, ongoing
    Rule #1 violation , the bug this script exists to catch)."""
    to_true = [e for e in transitions if e.get("to") is True]
    if not to_true:
        return "PASS"
    for e in to_true:
        cleared = any(
            e2.get("to") is False and 0 < (e2["ts"] - e["ts"]) <= TRANSIENT_S
            for e2 in transitions
        )
        if not cleared:
            return "SUSTAINED"
    return "TRANSIENT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=RESTORE_CTX,
                     help="ctx to serve at for this repro attempt "
                          "(default: the reference workload's own ctx)")
    ap.add_argument("--no-restore", action="store_true",
                     help="leave this repro's serve running instead of "
                          "restoring the reference config afterward")
    args = ap.parse_args()

    start_ts = _serve(args.ctx)
    transitions = _poll_transitions(start_ts)
    verdict = _classify(transitions)

    print(f"\n[repro] raw shim_transition verdict: {verdict} "
          f"({len(transitions)} transition(s) observed)")

    try:
        governed = gb_semantics.answer(
            "is Rule #1 satisfied right now — VRAM fill and rule1_underfilled state")
        seg = governed.get("segments", {}).get("rule1_underfilled", {})
        print(f"[repro] governed rule1_underfilled segment: "
              f"matched={seg.get('matched')} (the authoritative check , "
              f"cross-reference against the raw verdict above, don't trust "
              f"either alone)")
    except Exception as e:
        print(f"[repro] gb_semantics check failed (non-fatal): {e}")

    if not args.no_restore:
        print(f"\n[repro] restoring reference config (ctx={RESTORE_CTX})...")
        _serve(RESTORE_CTX)
    else:
        print(f"\n[repro] --no-restore passed , leaving this repro's serve "
              f"(ctx={args.ctx}) running for inspection.")

    print(f"\n{'=' * 70}\nFINAL VERDICT: {verdict}\n{'=' * 70}")
    if verdict == "SUSTAINED":
        print("Rule #1 underfill REPRODUCED , see workflow/known-issues.md's "
              "OPEN gb_needs_overflow() entry before touching "
              "greenboost_cuda_shim.c.")
        return 1
    print("Not reproduced this run (see this script's own docstring for why "
          "a different backend than the original observation limits how "
          "conclusively a PASS/TRANSIENT result here closes that entry).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
