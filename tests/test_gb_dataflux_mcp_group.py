#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_dataflux_mcp.py's new dataflux_schema/dataflux_group tools and
dataflux_events' from_ts/to_ts/cursor extension , the P5 fix for 22 event
kinds that were emitted but had no dedicated query path (only reachable via
dataflux_events(kind=...) if a caller already knew the exact name).
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_dataflux as gdf
import gb_dataflux_mcp as mcp


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    p = tmp_path / "dataflux.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(p))
    gdf._READ_EVENTS_MEMO.clear()
    yield p
    gdf._READ_EVENTS_MEMO.clear()


def test_dataflux_schema_full_registry():
    s = mcp.dataflux_schema()
    assert "shim_transition" in s          # a previously-orphaned kind
    assert "turboquant_activate" in s      # another one
    assert s["placement"]["group"] == "placement"


def test_dataflux_schema_single_kind():
    s = mcp.dataflux_schema("tensor_split")
    assert s["kind"] == "tensor_split"
    assert s["group"] == "placement"


def test_dataflux_group_shim_returns_previously_orphaned_kinds(log_path):
    gdf.emit({"kind": "shim_transition", "status": "warn", "node": "host"})
    gdf.emit({"kind": "shim_decision", "status": "ok", "node": "host"})
    gdf.emit({"kind": "quantize", "status": "ok"})  # different group , must not leak in

    result = mcp.dataflux_group("shim", days=1)

    kinds_seen = {e["kind"] for e in result["events"]}
    assert kinds_seen == {"shim_transition", "shim_decision"}
    assert result["by_kind"]["shim_transition"]["count"] == 1
    assert "quantize" not in result["by_kind"]


def test_dataflux_group_placement_returns_all_four_kinds(log_path):
    for k in ("placement", "tensor_split", "capacity_fit", "synapse_engine_placement"):
        gdf.emit({"kind": k, "status": "ok"})

    result = mcp.dataflux_group("placement", days=1)

    assert result["event_count"] == 4
    assert set(result["kinds"]) == {
        "placement", "tensor_split", "capacity_fit", "synapse_engine_placement"}


def test_dataflux_group_unknown_group_returns_empty():
    result = mcp.dataflux_group("not_a_real_group", days=1)
    assert result["kinds"] == []
    assert result["event_count"] == 0


def test_dataflux_events_from_to_ts_bounds(log_path):
    base = time.time() - 1000
    gdf.emit({"kind": "quantize", "ts": base})
    gdf.emit({"kind": "quantize", "ts": base + 100})
    gdf.emit({"kind": "quantize", "ts": base + 200})

    result = mcp.dataflux_events(days=1, from_ts=base + 50, to_ts=base + 150)

    assert len(result) == 1
    assert result[0]["ts"] == base + 100


def test_dataflux_events_cursor_pages_backward(log_path):
    base = time.time() - 1000
    for i in range(5):
        gdf.emit({"kind": "quantize", "seq": i, "ts": base + i})

    first_page = mcp.dataflux_events(days=1, limit=2)
    assert [e["seq"] for e in first_page] == [4, 3]

    second_page = mcp.dataflux_events(days=1, limit=2, cursor=first_page[-1]["ts"])
    assert [e["seq"] for e in second_page] == [2, 1]
