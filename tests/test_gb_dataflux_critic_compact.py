# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""dataflux_critic(compact=True): a readable summary instead of a 57k-character report that overflowed the MCP client."""
import json

import gb_dataflux as gdf


def _report(n=32):
    inc = [{"event": {"ts": 1793175000.0 + i, "kind": "semantic_transition" if i % 3 else "synapse_serve", "node": "host",
                      "label": "gb_semantics", "status": "error"},
            "snapshot_before": {"x": "v" * 300}, "snapshot_after": {"x": "v" * 300}, "cluster_context": {"c": 1},
            "diagnosis_hints": ["cluster rule: 0 feeder-side work items in the window while the host GPU ran at 100% " + "x" * 200
                                if i % 2 else "no rule matched , compare snapshot_before/after deltas"]} for i in range(n)]
    return {"window_days": 0.5, "events_scanned": 2241, "snapshots": [{"s": "y" * 400}] * 400, "incident_count": n, "total_incidents": n,
            "truncated": False, "incidents": inc, "recommendations": ["r" * 900] * 12,
            "segments_evaluated": 41, "segments_matched": ["decode_tail_heavy", "serve_healthy"]}


def test_the_compact_report_is_small_and_keeps_the_signal():
    rep = _report()
    full = len(json.dumps(rep))
    c = gdf.compact_critic(rep)
    assert full > 100_000 and len(json.dumps(c)) < 8_000
    assert c["incident_count"] == 32 and sum(c["by_kind"].values()) == 32
    assert c["by_hint"][0]["count"] == 16 and len(c["recent_incidents"]) == 8 and len(c["recommendations"]) == 8
    assert "snapshots" not in c and all("snapshot_before" not in i for i in c["recent_incidents"])
    assert c["recent_incidents"][-1]["ts"] == 1793175031.0                      # the most recent ones, in order
    assert all(len(r) <= 400 for r in c["recommendations"]) and "compact=False" in c["compact"]


def test_max_incidents_is_respected_and_an_empty_report_is_fine():
    assert len(gdf.compact_critic(_report(), max_incidents=3)["recent_incidents"]) == 3
    c = gdf.compact_critic({"window_days": 1, "incidents": [], "recommendations": []})
    assert c["incident_count"] is None and c["recent_incidents"] == [] and c["by_kind"] == {}
