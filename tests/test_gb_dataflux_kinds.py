#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for the gb_dataflux_kinds registry and its wiring into
gb_dataflux.summarize()/_is_incident()/critic_report():

  * _is_incident() is now registry-driven, not hardcoded to shim_transition
    alone , a kind's registered incident_when statuses make it incident-
    eligible with zero code change here.
  * summarize()'s new "by_kind" rollup covers every kind automatically.
  * critic_report()'s incident list is no longer silently capped at 50 ,
    total_incidents/truncated make a cap visible.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_dataflux as gdf
import gb_dataflux_kinds as kinds


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    p = tmp_path / "dataflux.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(p))
    gdf._READ_EVENTS_MEMO.clear()
    yield p
    gdf._READ_EVENTS_MEMO.clear()


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_registry_covers_every_known_group():
    assert set(kinds.GROUPS) == {
        "placement", "quant", "synapse", "cluster", "shim",
        "pipeline", "health", "agent", "eval", "bench", "gaming",
    }


def test_schema_returns_all_kinds():
    s = kinds.schema()
    assert "shim_decision" in s
    assert s["shim_decision"]["group"] == "shim"
    assert "warn" in s["shim_decision"]["incident_when"]


def test_schema_single_kind():
    s = kinds.schema("quantize")
    assert s["kind"] == "quantize"
    assert s["group"] == "quant"


def test_schema_unknown_kind_returns_empty():
    assert kinds.schema("totally_made_up_kind") == {}


def test_kinds_in_group():
    shim_kinds = kinds.kinds_in_group("shim")
    assert set(shim_kinds) == {
        "snapshot", "shim_decision", "shim_transition", "tier_move", "mem_pool_trim",
        "reclaim", "ssm_state"}


# ---------------------------------------------------------------------------
# _is_incident() driven by the registry
# ---------------------------------------------------------------------------

def test_is_incident_true_for_any_error_status():
    assert gdf._is_incident({"kind": "quantize", "status": "error"}) is True


def test_is_incident_true_for_registered_warn_kind():
    # quant_budget_fallback registers incident_when=("warn",)
    assert gdf._is_incident({"kind": "quant_budget_fallback", "status": "warn"}) is True


def test_is_incident_false_for_unregistered_warn_kind():
    # "quantize" has no incident_when entries , a warn status there is not
    # (yet) incident-worthy.
    assert gdf._is_incident({"kind": "quantize", "status": "warn"}) is False


def test_is_incident_false_for_unknown_kind():
    assert gdf._is_incident({"kind": "not_a_real_kind", "status": "warn"}) is False


def test_is_incident_previously_hardcoded_shim_transition_still_works():
    assert gdf._is_incident({"kind": "shim_transition", "status": "warn"}) is True
    assert gdf._is_incident({"kind": "shim_transition", "status": "ok"}) is False


# ---------------------------------------------------------------------------
# summarize()'s "by_kind" rollup
# ---------------------------------------------------------------------------

def test_summarize_by_kind_covers_every_emitted_kind(log_path):
    gdf.emit({"kind": "quantize", "status": "ok"})
    gdf.emit({"kind": "quantize", "status": "ok"})
    gdf.emit({"kind": "synapse_stall", "status": "warn"})
    gdf.emit({"kind": "synapse_serve", "status": "error"})

    events = gdf.read_events()
    s = gdf.summarize(events)

    assert s["by_kind"]["quantize"]["count"] == 2
    assert s["by_kind"]["quantize"]["errors"] == 0
    assert s["by_kind"]["synapse_serve"]["errors"] == 1
    assert s["by_kind"]["synapse_stall"]["count"] == 1


# ---------------------------------------------------------------------------
# critic_report()'s limit/truncated
# ---------------------------------------------------------------------------

def test_critic_report_truncated_flag_and_total(log_path):
    now = time.time()
    for i in range(5):
        gdf.emit({"kind": "quantize", "status": "error", "ts": now - i})

    rep = gdf.critic_report(days=1.0, limit=2)

    assert rep["total_incidents"] == 5
    assert rep["incident_count"] == 2
    assert rep["truncated"] is True
    assert len(rep["incidents"]) == 2


def test_critic_report_not_truncated_when_under_limit(log_path):
    now = time.time()
    gdf.emit({"kind": "quantize", "status": "error", "ts": now})

    rep = gdf.critic_report(days=1.0, limit=50)

    assert rep["total_incidents"] == 1
    assert rep["truncated"] is False
