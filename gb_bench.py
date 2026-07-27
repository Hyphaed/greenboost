#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb_bench.py, Speed Program Phase 0 telemetry: the single emit site for the
"bench_result" dataflux kind.

WHY THIS LIVES AT ROOT, NOT IN tests/bench/: checks/check_dataflux_coverage.py
deliberately excludes tests/ from its kind-literal parity scan (tests/
fixtures use fake kind values, not real emit sites, see that file's own
comment). A real telemetry emit call belongs in a root-level module, the same
way every other emitting subsystem here does (gb_placement.py, gb_moe.py,
gb_quant.py, ...), not buried in the harness script that happens to call it.
tests/bench/run_real_model.py and the gb_pathbench C harness's Python
reporter both import this instead of calling gb_dataflux.emit() directly, so
there is exactly one place that defines what a "bench_result" event contains.

Records path bandwidth (gb_pathbench) or end-to-end tok/s (run_real_model.py)
before/after measurements, keyed by config_hash so a later phase's change can
be diffed against its own baseline row via dataflux_group('bench'), see the
KindSpec doc in gb_dataflux_kinds.py.
"""
from __future__ import annotations

import hashlib
import json
import time


def config_hash(config: dict) -> str:
    """Stable short hash of the knobs that define a run. Excludes measured
    outputs and timestamps deliberately, this identifies the CONFIG being
    compared, not one instance of its result."""
    basis = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(basis.encode()).hexdigest()[:12]


def emit_bench_result(
    *,
    path: str,
    config: dict,
    model: str | None = None,
    tag: str | None = None,
    bandwidth_gb_s: float | None = None,
    prefill_tok_s: float | None = None,
    decode_tok_s: float | None = None,
    load_s: float | None = None,
    error: str | None = None,
) -> None:
    """Best-effort dataflux emit for one benchmark measurement. Never raises,
    a missing gb_dataflux import (e.g. a minimal/CI env) just means no event
    was recorded, not a failed benchmark run.

    path: which harness produced this ("gb_pathbench", "run_real_model", ...).
    config: the full knob dict for this run (hashed via config_hash() so
        Phase N's change can be diffed against its own baseline row).
    """
    try:
        import gb_dataflux
    except Exception:
        return
    event = {
        "kind": "bench_result",
        "status": "error" if error else "ok",
        "path": path,
        "config_hash": config_hash(config),
        "model": model,
        "tag": tag,
        "bandwidth_gb_s": bandwidth_gb_s,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "load_s": load_s,
        "error": error,
        "ts": time.time(),
    }
    gb_dataflux.emit(event)
