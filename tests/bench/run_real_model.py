#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
tests/bench/run_real_model.py , Phase 0 measurement harness (roadmap Track A/C/D
prerequisite).

WHY: gb_prefetch.py / gb_moe.py are unit-tested against CPU
synthetic models only. Their keep_resident / lookahead / prefetch_topn /
hot_threshold defaults are explicitly unvalidated against a real checkpoint
under the shim. This script loads a real HF causal LM (dense or MoE) through
gb_llm, optionally attaches gb_prefetch / gb_moe, runs a short decode, and
emits one JSON line of measurements to results.jsonl so different configs
can be diffed.

Usage:
    env GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \\
        python3 tests/bench/run_real_model.py --model <hf-id> \\
            [--moe] [--prefetch] [--keep-resident 2] [--lookahead 1] \\
            [--max-new-tokens 64] [--prompt "..."] [--tag baseline]

Without --prefetch/--moe this is a pure shim-overflow baseline run (Phase 0
"shim ON, prefetch OFF"). Run again with --prefetch (dense) or --moe (MoE
routing) to compare. Run the whole binary with LD_PRELOAD unset for a
"shim OFF" baseline (will OOM on models that don't fit local VRAM , expected,
record the OOM in results.jsonl rather than crashing the sweep).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Repo root must win over the installed /usr/local/lib/greenboost copy (set via
# global PYTHONPATH) so this harness exercises the dev modules under test, not
# a stale deploy that may be missing files entirely (e.g. gb_prefetch.py /
# gb_moe.py are untracked-but-not-yet-deployed as of this writing).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RESULTS_PATH = Path(__file__).parent / "results.jsonl"


def _read_ebpf_stats() -> dict:
    """Best-effort: parse `greenboost faults --llm` machine-readable output.
    Returns {} if the eBPF tracer or CLI isn't available , never raises."""
    try:
        out = subprocess.run(["greenboost", "faults", "--llm"], capture_output=True,
                              text=True, timeout=5).stdout
    except Exception:
        return {}
    stats = {}
    for line in out.splitlines():
        m = re.match(r"\s*([A-Za-z0-9_ /→>-]+?)\s+([0-9.]+)\s*/s\s*$", line)
        if m:
            key = re.sub(r"[^a-z0-9]+", "_", m.group(1).strip().lower()).strip("_")
            stats[f"ebpf_{key}"] = float(m.group(2))
    return stats


def _read_residency_heat() -> dict:
    """Best-effort: parse `greenboost residency --llm` hot/warm/cold MB
    breakdown. Returns {} if debugfs/residency export is unavailable."""
    try:
        out = subprocess.run(["greenboost", "residency", "--llm"], capture_output=True,
                              text=True, timeout=5).stdout
    except Exception:
        return {}
    heat = {}
    for key in ("Hot", "Warm", "Cold", "T2 (DDR)", "T3 (NVMe)"):
        m = re.search(rf"{re.escape(key)}\s+(\d+)\s*MB", out)
        if m:
            slug = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            heat[f"residency_{slug}_mb"] = int(m.group(1))
    return heat


def _gpu_metrics_snapshot() -> dict:
    """Best-effort GpuMetrics snapshot via gb_init (None fields if telemetry
    isn't running, e.g. shimless baseline)."""
    try:
        import gb_init
    except Exception:
        return {}
    snap = gb_init.snapshot()
    if snap is None:
        return {}
    out = {
        "fb_used_mb": snap.fb_used_mb,
        "fb_free_mb": snap.fb_free_mb,
        "fb_total_mb": snap.fb_total_mb,
        "shim_phase": snap.shim_phase,
    }
    if snap.gb is not None:
        out["gb_t2_used_mb"] = getattr(snap.gb, "t2_used_mb", None)
        out["gb_t3_used_mb"] = getattr(snap.gb, "t3_used_mb", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--budget-gb", type=float, default=None,
                     help="force gb_quant's fit-to-VRAM budget low enough that the model "
                          "overflows into T2 (default: gb_llm's auto budget, ~92%% free VRAM, "
                          "which may let small models fit entirely in T1 and never overflow)")
    ap.add_argument("--moe", action="store_true", help="attach gb_moe.GbMoEManager")
    ap.add_argument("--prefetch", action="store_true", help="attach gb_prefetch.LayerPrefetcher (dense)")
    ap.add_argument("--keep-resident", type=int, default=2)
    ap.add_argument("--lookahead", type=int, default=1)
    ap.add_argument("--hot-threshold", type=float, default=0.05)
    ap.add_argument("--prefetch-topn", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Explain what a GPU memory tier is in two sentences.")
    ap.add_argument("--tag", default="run", help="label stamped into results.jsonl for A/B grouping")
    args = ap.parse_args()

    import gb_llm

    record = {
        "tag": args.tag,
        "model": args.model,
        "budget_gb": args.budget_gb,
        "moe": args.moe,
        "prefetch": args.prefetch,
        "keep_resident": args.keep_resident,
        "lookahead": args.lookahead,
        "hot_threshold": args.hot_threshold,
        "prefetch_topn": args.prefetch_topn,
        "timestamp": time.time(),
    }

    t_load0 = time.monotonic()
    try:
        model, tok = gb_llm.load_causal_lm(args.model, budget_gb=args.budget_gb)
    except Exception as exc:
        record["error"] = f"load_failed: {exc}"
        _append(record)
        print(json.dumps(record, indent=2), file=sys.stderr)
        return 1
    record["load_s"] = round(time.monotonic() - t_load0, 2)

    prefetcher = None
    moe_mgr = None
    if args.prefetch:
        import gb_init
        prefetcher = gb_init.make_layer_prefetcher(
            model, keep_resident=args.keep_resident, lookahead=args.lookahead)
        if prefetcher is not None:
            prefetcher.attach()
    if args.moe:
        from gb_moe import GbMoEManager as _Mgr
        moe_mgr = _Mgr(model, hot_threshold=args.hot_threshold,
                        prefetch_topn=args.prefetch_topn)
        record["moe_blocks_found"] = moe_mgr.attach()

    record["gpu_before"] = _gpu_metrics_snapshot()

    t_gen0 = time.monotonic()
    try:
        text = gb_llm.generate(model, tok, args.prompt, max_new_tokens=args.max_new_tokens)
    except Exception as exc:
        record["error"] = f"generate_failed: {exc}"
        _append(record)
        print(json.dumps(record, indent=2), file=sys.stderr)
        return 1
    gen_s = time.monotonic() - t_gen0
    record["gen_s"] = round(gen_s, 3)
    record["tok_per_s"] = round(args.max_new_tokens / gen_s, 2) if gen_s > 0 else None
    record["output_preview"] = text[:120]

    record["gpu_after"] = _gpu_metrics_snapshot()
    record["ebpf_stats"] = _read_ebpf_stats()
    record["residency"] = _read_residency_heat()
    if prefetcher is not None:
        record["prefetcher_status"] = prefetcher.status()
    if moe_mgr is not None:
        record["moe_status"] = moe_mgr.status()

    _append(record)
    print(json.dumps(record, indent=2))
    return 0


def _append(record: dict) -> None:
    with RESULTS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
