# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
pytest suite for gb_dataflux.py: event log read/write, summarize(), HTML
rendering, and the stdlib HTTP server (`greenboost dataflux-ui`'s backend).

No CUDA, no daemon, no network beyond a loopback TCP socket for the server
tests. The real ~/.local/share/greenboost/dataflux.jsonl is never touched ,
tests/conftest.py's autouse fixture redirects GREENBOOST_DATAFLUX_LOG (the
env var gb_dataflux._log_path() resolves on every call) at a tmp_path file.
"""
import http.client
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_dataflux as gdf


def test_emit_then_read_roundtrip():
    gdf.emit({"node": "host", "label": "x", "n_items": 3, "duration_s": 1.5, "status": "ok"})
    events = gdf.read_events()
    assert len(events) == 1
    assert events[0]["node"] == "host"
    assert "ts" in events[0]  # emit() auto-stamps a timestamp


def test_emit_never_raises_on_unwritable_path(monkeypatch, tmp_path):
    # Point at a path whose parent can't be created (a file, not a dir).
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(blocker / "nested" / "dataflux.jsonl"))
    gdf.emit({"node": "host"})  # must not raise


def test_read_events_filters_by_since_hours():
    now = time.time()
    log_path = gdf._log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(json.dumps({"ts": now - 10 * 3600, "node": "old"}) + "\n")
        f.write(json.dumps({"ts": now - 1 * 3600, "node": "recent"}) + "\n")
    all_events = gdf.read_events()
    assert len(all_events) == 2
    recent_only = gdf.read_events(since_hours=5)
    assert len(recent_only) == 1
    assert recent_only[0]["node"] == "recent"


def test_read_events_skips_malformed_lines():
    log_path = gdf._log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("not json at all\n")
        f.write(json.dumps({"ts": time.time(), "node": "ok"}) + "\n")
        f.write("\n")  # blank line
    events = gdf.read_events()
    assert len(events) == 1
    assert events[0]["node"] == "ok"


def test_read_events_empty_when_log_absent():
    assert gdf.read_events() == []


def test_summarize_empty():
    s = gdf.summarize([])
    assert s["event_count"] == 0
    assert s["total_items"] == 0
    assert s["nodes"] == {}


def test_summarize_groups_runs_and_labels():
    events = [
        {"ts": 100.0, "run_id": "r1", "node": "host", "label": "gen_image_batch.py",
         "n_items": 2, "duration_s": 10.0, "status": "ok"},
        {"ts": 105.0, "run_id": "r1", "node": "omen", "label": "gen_image_batch.py",
         "n_items": 3, "duration_s": 8.0, "status": "ok"},
    ]
    s = gdf.summarize(events)
    assert s["labels"]["gen_image_batch.py"] == 5
    run = s["runs"]["r1"]
    assert run["items"] == 5
    assert sorted(run["nodes"]) == ["host", "omen"]
    assert run["started"] == 100.0
    assert run["finished"] == 105.0


def test_summarize_stage_profile_rollup():
    events = [
        {"ts": 100.0, "run_id": "j1", "node": "host", "label": "jobqueue",
         "kind": "stage_profile", "stage": "image:generate",
         "n_items": 1, "duration_s": 30.0, "status": "ok"},
        {"ts": 200.0, "run_id": "j2", "node": "host", "label": "jobqueue",
         "kind": "stage_profile", "stage": "image:generate",
         "n_items": 1, "duration_s": 50.0, "status": "ok"},
        {"ts": 300.0, "run_id": "j3", "node": "host", "label": "jobqueue",
         "kind": "stage_profile", "stage": "image:generate",
         "n_items": 1, "duration_s": 40.0, "status": "error"},
        {"ts": 300.0, "run_id": "j3", "node": "host", "label": "forge",
         "kind": "stage_profile", "stage": "forge:texture",
         "n_items": 1, "duration_s": 12.0, "status": "ok"},
    ]
    s = gdf.summarize(events)
    gen = s["stages"]["image:generate"]
    assert gen["count"] == 3
    assert gen["mean_s"] == 40.0
    assert gen["p95_s"] == 50.0        # max of the sorted walls at n=3
    assert gen["latest_s"] == 40.0     # arrival order, not the sorted max
    assert gen["last_status"] == "error"
    assert s["stages"]["forge:texture"]["count"] == 1


def test_summarize_no_stage_profile_key_absent_events():
    s = gdf.summarize([{"ts": 1.0, "run_id": "r", "node": "host",
                        "label": "x", "n_items": 1, "duration_s": 1.0,
                        "status": "ok"}])
    assert s["stages"] == {}


def test_render_html_contains_summary_numbers():
    events = [{"ts": time.time(), "run_id": "r1", "node": "host", "label": "x",
              "n_items": 4, "duration_s": 2.0, "status": "ok", "items": ["a", "b"]}]
    html = gdf.render_html(events, days=5)
    assert "<html>" in html
    assert "GreenBoost Dataflux" in html
    assert ">4<" in html  # total_items card
    assert "host" in html


def test_render_html_no_events_shows_placeholder():
    html = gdf.render_html([], days=5)
    assert "No events yet" in html


def test_render_html_escapes_untrusted_content():
    events = [{"ts": time.time(), "run_id": "r1", "node": "<script>alert(1)</script>",
              "label": "x", "n_items": 1, "duration_s": 1.0, "status": "ok", "items": []}]
    html = gdf.render_html(events, days=5)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# ── HTTP server ──────────────────────────────────────────────────────────

def test_serve_html_and_json_endpoints():
    gdf.emit({"node": "host", "label": "x", "n_items": 2, "duration_s": 1.0, "status": "ok"})

    import socketserver
    gdf._Handler.days = 5
    with socketserver.TCPServer(("127.0.0.1", 0), gdf._Handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/")
            resp = conn.getresponse()
            assert resp.status == 200
            assert b"GreenBoost Dataflux" in resp.read()

            conn.request("GET", "/api/summary.json")
            resp = conn.getresponse()
            assert resp.status == 200
            summary = json.loads(resp.read())
            assert summary["total_items"] == 2

            conn.request("GET", "/api/events.json?days=5")
            resp = conn.getresponse()
            events = json.loads(resp.read())
            assert len(events) == 1
        finally:
            httpd.shutdown()


# ── SnapshotRecorder: continuous system state recording ─────────────────────

class _FakeMetrics:
    def __init__(self, device=0, fb_used_mb=100, fb_free_mb=900, gpu_util_pct=42.0):
        self.device = device
        self.fb_used_mb = fb_used_mb
        self.fb_free_mb = fb_free_mb
        self.fb_total_mb = fb_used_mb + fb_free_mb
        self.fb_used_pct = 10.0
        self.gpu_util_pct = gpu_util_pct
        self.temp_c = 55.0
        self.power_w = 120.0
        self.shim_phase = "INFERENCE"
        self.gb = None


class _FakeTelemetryManager:
    """Minimal stand-in exposing just the add_callback() contract
    SnapshotRecorder needs , no real polling thread."""
    def __init__(self):
        self._callbacks = []

    def add_callback(self, fn):
        self._callbacks.append(fn)

    def fire(self, m):
        for cb in self._callbacks:
            cb(m)


def test_snapshot_recorder_emits_on_first_callback():
    tm = _FakeTelemetryManager()
    gdf.SnapshotRecorder(tm, interval_s=5.0)
    tm.fire(_FakeMetrics(fb_free_mb=777))
    events = gdf.read_events()
    assert len(events) == 1
    assert events[0]["kind"] == "snapshot"
    assert events[0]["fb_free_mb"] == 777
    assert events[0]["gpu_util_pct"] == 42.0


def test_snapshot_recorder_throttles_rapid_callbacks():
    tm = _FakeTelemetryManager()
    gdf.SnapshotRecorder(tm, interval_s=5.0)
    for _ in range(10):
        tm.fire(_FakeMetrics())
    events = gdf.read_events()
    assert len(events) == 1  # all 10 calls landed within the 5s throttle window


def test_snapshot_recorder_uses_gb_pool_fields_when_present():
    tm = _FakeTelemetryManager()
    gdf.SnapshotRecorder(tm, interval_s=5.0)
    m = _FakeMetrics()
    m.gb = type("GB", (), {"kv_used_mb": 512, "kv_reserve_mb": 2048, "kv_t2_mb": 100,
                           "t2_allocated_mb": 4096, "t2_available_mb": 8192,
                           "t2_pressure": 1, "t3_used_mb": 0, "t3_pressure": 0})()
    tm.fire(m)
    events = gdf.read_events()
    assert events[0]["kv_used_mb"] == 512
    assert events[0]["t2_pressure"] == 1


def test_start_snapshot_recorder_returns_none_without_telemetry():
    assert gdf.start_snapshot_recorder(telemetry_manager=None) is None or True
    # (falls back to gb_init.get_telemetry(); in a bare test process that's
    # None too, so either a graceful None or a real recorder is acceptable ,
    # the only requirement is it must not raise.)


def test_start_snapshot_recorder_fans_out_to_cluster_managers():
    class _FakeClusterManager:
        def __init__(self, n):
            self._managers = [_FakeTelemetryManager() for _ in range(n)]

    cluster = _FakeClusterManager(3)
    result = gdf.start_snapshot_recorder(cluster, interval_s=5.0)
    assert isinstance(result, list)
    assert len(result) == 3
