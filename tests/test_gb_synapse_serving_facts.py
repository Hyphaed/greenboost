#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse.py's build_serving_facts()/resolve_serving_preset()
(NemoClaw audit, Phase 7 integration): real facts wired from
gb_readiness/gb_topology/gb_cluster, and the cluster.feeder_count fact
that closes the "host alone satisfies anyNode" gap found live.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_cluster
import gb_synapse as gs


def test_build_serving_facts_has_controller_and_host_node():
    facts = gs.build_serving_facts()
    assert "controller" in facts
    assert "host" in facts["nodes"]


def test_build_serving_facts_feeder_count_zero_when_no_feeders(monkeypatch):
    monkeypatch.setattr(gb_cluster, "feeders", lambda probe=True: [])
    facts = gs.build_serving_facts()
    assert facts["controller"]["cluster.feeder_count"] == 0


def test_build_serving_facts_counts_only_online_feeders(monkeypatch):
    online = gb_cluster.Feeder(ip="10.0.0.2", port=9740)
    online.online = True
    offline = gb_cluster.Feeder(ip="10.0.0.3", port=9740)
    offline.online = False
    monkeypatch.setattr(gb_cluster, "feeders", lambda probe=True: [online, offline])
    facts = gs.build_serving_facts()
    assert facts["controller"]["cluster.feeder_count"] == 1
    assert "10.0.0.2" in facts["nodes"]
    assert "10.0.0.3" not in facts["nodes"]


def test_build_serving_facts_never_raises_when_readiness_fails(monkeypatch):
    import gb_readiness
    monkeypatch.setattr(gb_readiness, "build_report", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    facts = gs.build_serving_facts()  # must not raise
    assert isinstance(facts, dict)


def test_resolve_serving_preset_returns_selected_and_facts(monkeypatch):
    monkeypatch.setattr(gb_cluster, "feeders", lambda probe=True: [])
    result = gs.resolve_serving_preset()
    assert result["selected"] == "host-only"
    assert "facts" in result
    assert result["facts"]["controller"]["cluster.feeder_count"] == 0


def test_resolve_serving_preset_never_raises_on_bad_kmod_state(monkeypatch):
    import gb_readiness
    monkeypatch.setattr(gb_readiness, "build_report", lambda: {
        "observations": [{"id": "kmod.greenboost.loaded", "value": False, "determined": True}],
        "capabilities": [],
    })
    monkeypatch.setattr(gb_cluster, "feeders", lambda probe=True: [])
    result = gs.resolve_serving_preset()
    assert result["selected"] is None
    assert "no eligible preset" in result["error"]
