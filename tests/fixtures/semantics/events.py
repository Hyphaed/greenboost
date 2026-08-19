# SPDX-License-Identifier: GPL-2.0-only
"""tests/fixtures/semantics/events.py — the frozen GB-Semantics eval fixture.

Models one concrete scenario, chosen to be the exact trap the whole layer
exists to catch: DavidAU/Qwen3.6-27B-Fable-Fusion (MTP-Q4_K_M) served on a
single 12 GB RTX 5070, mid-generation. The shim's VIRTUAL VRAM view
(`fb_used_pct`) reports 97% "full" (T2/T3/feeder bytes folded in per the
Cluster Compute Architecture's single-virtual-device design) while the
REAL PHYSICAL occupancy (`fb_phys_used_pct`) is only 42.3% — i.e. Rule #1
(target 85-92% physical) is being VIOLATED right now, and an agent reading
the wrong field would confidently report the opposite of the truth.

Event bodies here carry every field EXCEPT `ts` — the freshness gates built
into gb_semantics.py's resolvers (see `_latest_event`'s `max_age_s`) are
themselves part of what's under test, so a literally-frozen historical
timestamp would fail them by construction. `load(node="host")` stamps
`ts=time.time()` at call time, keeping every other value pinned.
"""
from __future__ import annotations

import time

NODE = "host"

# The core scenario: Rule #1 violation (physical VRAM well under target,
# while overflow to T2/T3 is genuinely active), fp8-floor quant, hybrid-arch
# KV, and a cold prompt cache.
_TEMPLATE: list[dict] = [
    {
        "node": NODE, "label": "system_snapshot", "kind": "snapshot", "status": "ok",
        "n_items": 0, "items": [], "duration_s": 0.0,
        # THE trap pair: virtual says "97% full", physical says "42.3%".
        "fb_used_pct": 97.0, "fb_used_mb": 11800, "fb_free_mb": 427, "fb_total_mb": 12227,
        "fb_phys_used_pct": 42.3, "fb_phys_used_mb": 5175, "fb_phys_total_mb": 12227,
        "gpu_util_pct": 61.0,
        "t2_pressure": 0.72,       # continuous fraction (t2_pressure_fraction's scale)
        "t3_pressure": 0.05,
        "t3_used_mb": 0,           # weights NEVER on T3 — placement floor holds here
        "kv_used_mb": 900, "kv_t2_mb": 2100,   # most KV spilled to T2 (kv_resident_share should be low)
    },
    {
        "node": NODE, "label": "gb_quant", "kind": "kernel_backend", "status": "ok",
        "n_items": 1, "items": ["fable-fusion-27b"], "duration_s": 0.0,
        "bits": 8, "backend": "triton_int8", "skipped_prequantized": False, "bf16_kept": 0,
    },
    {
        "node": NODE, "label": "gb_synapse", "kind": "tok_s_measured", "status": "ok",
        "n_items": 1, "items": ["fable-fusion-27b"], "duration_s": 0.0,
        "model": "fable-fusion-27b", "tok_s": 5.2,
    },
    {
        "node": NODE, "label": "gb_attn", "kind": "turboquant_activate", "status": "ok",
        "n_items": 1, "items": ["fable-fusion-27b"], "duration_s": 0.0,
        "k_bits": 4, "v_bits": 3, "mode": "asymmetric",
    },
    {
        "node": NODE, "label": "gb_synapse", "kind": "prompt_cache", "status": "ok",
        "n_items": 1, "items": ["fable-fusion-27b"], "duration_s": 0.0,
        "model": "fable-fusion-27b",
        "ttft_ms": 4200.0, "hit_pct": 3.0, "reused_tokens": 40,  # cold cache
        # engine_prompt_ms deliberately well below ttft_ms: the gap between the
        # two is the diagnostic (queueing + contention, not prefill), and a
        # fixture where they matched would let a resolver confusing the two
        # pass. prompt_tokens is the depth those figures were measured at.
        "engine_prompt_ms": 1350.0, "prompt_tokens": 1333,
    },
    # GB-1. Two cache_index events, not one, because slot_pin_rate_pct is a
    # RATE: a single event can only ever produce 0% or 100%, which would let a
    # resolver that reads the newest event instead of the window pass. The mix
    # here (one pinned, one reassigned) is the interesting case , a conversation
    # kept its slot while another was displaced.
    {
        "node": NODE, "label": "synapse", "kind": "cache_index", "status": "ok",
        "n_items": 1, "items": ["fable-fusion-27b"], "duration_s": 0.0,
        "model": "fable-fusion-27b", "conv": "aaaaaaaaaaaaaaaa", "slot": 0,
        "decision": "reassigned", "chunks": 6, "chunks_before": 4,
        "changed_chunk": None, "n_slots": 4, "tracked": 2, "identity": "inferred",
        "applied": False,
    },
    {
        "node": NODE, "label": "synapse", "kind": "cache_index", "status": "ok",
        "n_items": 1, "items": ["fable-fusion-27b"], "duration_s": 0.0,
        "model": "fable-fusion-27b", "conv": "bbbbbbbbbbbbbbbb", "slot": 1,
        # changed_chunk=2 is the whole point of the field: an edit landed on
        # the third chunk, so everything after it has to be re-prefilled.
        "decision": "pinned-edited", "chunks": 6, "chunks_before": 6,
        "changed_chunk": 2, "n_slots": 4, "tracked": 2, "identity": "explicit",
        "applied": False,
    },
    {
        "node": "feeder1 (192.0.2.10)", "label": "cluster_dispatch", "kind": "chunk_remote",
        "status": "ok", "n_items": 0, "items": [], "duration_s": 0.0,
    },
]


def load(node: str = NODE, t2_pressure: "float | None" = None,
         t3_used_mb: "int | None" = None, prompt_cache_hit_pct: "float | None" = None) -> list[dict]:
    """Fresh copies of the fixture events, timestamped now, with a few
    optional overrides for evals that need a DIFFERENT scenario (e.g. a
    kmod-missing case, or a healthy-cache case) without a second fixture file."""
    now = time.time()
    out = []
    # Stamp in list order, a millisecond apart, so "the latest event of this
    # kind" is DETERMINISTIC. Stamping every event with the same instant made
    # that a tie, and `max(key=ts)` breaks a tie by list position — which meant
    # a resolver reading the newest event of a kind with two fixture entries
    # silently got the FIRST one. Ordering is part of what these resolvers
    # depend on, so the fixture has to model it.
    n = len(_TEMPLATE)
    for i, ev in enumerate(_TEMPLATE):
        e = dict(ev)
        e["ts"] = now - (n - 1 - i) * 0.001
        if e["kind"] == "snapshot":
            if t2_pressure is not None:
                e["t2_pressure"] = t2_pressure
            if t3_used_mb is not None:
                e["t3_used_mb"] = t3_used_mb
        if e["kind"] == "prompt_cache" and prompt_cache_hit_pct is not None:
            e["hit_pct"] = prompt_cache_hit_pct
        out.append(e)
    return out


def write(log_path, **overrides) -> None:
    """Append this fixture's events (as JSONL) to `log_path` — the path
    tests/conftest.py's autouse `_isolate_dataflux_log_globally` fixture
    already points GREENBOOST_DATAFLUX_LOG at, so gb_dataflux.read_events()
    (and therefore gb_semantics' resolvers) see exactly these events and
    nothing from the real system log."""
    import json
    from pathlib import Path
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for ev in load(**overrides):
            f.write(json.dumps(ev) + "\n")
