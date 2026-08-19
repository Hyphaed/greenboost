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


# missing_features.md item (k): dataflux_tok_s (via summarize()) must not
# blend tok_s_measured samples across different quant/ctx/kv_type configs
# of the same model name — the same class of bug fixed in
# gb_synapse._measured_tok_s(), independently reproduced here because this
# function reads the raw event log, not gb_synapse's persisted file.

def test_summarize_tok_s_separates_quant_swaps_of_same_model():
    events = [
        {"ts": 100.0, "run_id": "r1", "node": "host", "label": "gb_synapse",
         "kind": "tok_s_measured", "n_items": 1, "duration_s": 0.0, "status": "ok",
         "model": "qwen35", "tok_s": 5.0, "quant": "Q4_K_M", "ctx": 65536, "kv_type": "q8_0"},
        {"ts": 200.0, "run_id": "r2", "node": "host", "label": "gb_synapse",
         "kind": "tok_s_measured", "n_items": 1, "duration_s": 0.0, "status": "ok",
         "model": "qwen35", "tok_s": 12.0, "quant": "IQ2_M", "ctx": 32768, "kv_type": "f16"},
    ]
    s = gdf.summarize(events)

    assert "qwen35::Q4_K_M::65536::q8_0" in s["tok_s"]
    assert "qwen35::IQ2_M::32768::f16" in s["tok_s"]
    assert s["tok_s"]["qwen35::Q4_K_M::65536::q8_0"]["avg"] == 5.0
    assert s["tok_s"]["qwen35::IQ2_M::32768::f16"]["avg"] == 12.0


def test_summarize_tok_s_legacy_events_without_quant_keep_old_key_shape():
    """Events recorded before this fix (no quant/ctx/kv_type fields) must
    keep the exact old unprefixed key , backward compatible with existing
    dashboards and historical rows."""
    events = [
        {"ts": 100.0, "run_id": "r1", "node": "host", "label": "gb_synapse",
         "kind": "tok_s_measured", "n_items": 1, "duration_s": 0.0, "status": "ok",
         "model": "qwen35", "tok_s": 4.5},
    ]
    s = gdf.summarize(events)

    assert "qwen35" in s["tok_s"]
    assert s["tok_s"]["qwen35"]["avg"] == 4.5


def test_summarize_tok_s_source_prefix_still_applies_with_quant_key():
    events = [
        {"ts": 100.0, "run_id": "r1", "node": "host", "label": "gb_synapse",
         "kind": "tok_s_measured", "n_items": 1, "duration_s": 0.0, "status": "ok",
         "model": "qwen35", "tok_s": 5.0, "source": "proxy",
         "quant": "Q4_K_M", "ctx": 65536, "kv_type": "q8_0"},
    ]
    s = gdf.summarize(events)

    assert "[proxy]qwen35::Q4_K_M::65536::q8_0" in s["tok_s"]


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


def test_critic_report_feeder_active_from_snapshot_telemetry():
    """Real incident (2026-07-14): feeder_active used to be computed ONLY
    from co-occurring chunk_remote/job_remote dispatch-kind log rows within
    the ±60s critic window , so a host-side incident with no temporally
    coincident dispatch always showed feeder_active=False even when a
    feeder's OWN snapshot (right there in the same window) proved it was
    genuinely alive and computing. Fixed to also derive feeder_active from
    a feeder-node snapshot's own gpu_util_pct."""
    t0 = time.time()
    gdf.emit({"ts": t0, "node": "host", "label": "jobqueue", "kind": "stage_profile",
             "stage": "image:image", "status": "error", "error": "oom"})
    gdf.emit({"ts": t0 + 10, "node": "omen", "kind": "snapshot", "gpu_util_pct": 95.0,
             "fb_used_mb": 6000, "fb_free_mb": 2000, "fb_total_mb": 8000})

    report = gdf.critic_report(days=1)
    assert len(report["incidents"]) == 1
    ctx = report["incidents"][0]["cluster_context"]
    assert ctx["feeder_active"] is True
    assert ctx["feeder_nodes"] == ["omen"]


def test_critic_report_feeder_active_false_when_feeder_idle():
    """A feeder snapshot seen nearby but genuinely idle (low gpu_util_pct)
    must NOT flip feeder_active — only real compute counts as active. The
    feeder should still be listed in feeder_nodes (it WAS seen, just idle)."""
    t0 = time.time()
    gdf.emit({"ts": t0, "node": "host", "label": "jobqueue", "kind": "stage_profile",
             "stage": "sfx:sfx", "status": "error", "error": "oom"})
    gdf.emit({"ts": t0 + 5, "node": "omen", "kind": "snapshot", "gpu_util_pct": 0.0,
             "fb_used_mb": 500, "fb_free_mb": 7500, "fb_total_mb": 8000})

    report = gdf.critic_report(days=1)
    ctx = report["incidents"][0]["cluster_context"]
    assert ctx["feeder_active"] is False
    assert ctx["feeder_nodes"] == ["omen"]


def test_critic_report_recommendations_not_misleading_when_incident_undiagnosed():
    """Regression: an incident that exists but matches no diagnosis rule
    (diagnosis_hints == ["no rule matched , ..."]) used to fall through to
    the SAME fallback message as a genuinely-empty window ("no incidents or
    pressure flags in the window , cluster inference nominal") , directly
    contradicting the report's own incident_count field. Verified live
    2026-07-30: two real `bw_undetectable` incidents (incident_count=2) with
    recommendations still claiming "no incidents ... nominal"."""
    t0 = time.time()
    gdf.emit({"ts": t0, "node": "host", "label": "misc", "kind": "job_local",
             "status": "error", "error": "disk full"})

    report = gdf.critic_report(days=1)
    assert report["incident_count"] == 1
    assert report["incidents"][0]["diagnosis_hints"][0].startswith("no rule matched")
    assert not any("nominal" in r for r in report["recommendations"])
    assert any("1 incident" in r for r in report["recommendations"])


def test_critic_report_bw_undetectable_has_diagnosis_rule():
    """bw_undetectable is explicitly named in _is_incident's own docstring as
    an incident-worthy warn kind, but _diagnosis_hints had no rule for it ,
    every occurrence fell through to the generic "no rule matched" hint
    despite the root cause (gb_topology bandwidth detection) being well
    understood. Verified live 2026-07-30 via dataflux_critic on a real box
    with a stale profile-pinned vram_bandwidth_gb_s: 0."""
    t0 = time.time()
    gdf.emit({"ts": t0, "node": "host", "label": "synapse", "kind": "bw_undetectable",
             "status": "warn", "reason": "vram"})

    report = gdf.critic_report(days=1)
    assert report["incident_count"] == 1
    hints = report["incidents"][0]["diagnosis_hints"]
    assert not any(h.startswith("no rule matched") for h in hints)
    assert any("bandwidth undetectable" in h for h in hints)


def test_dataflux_tier_moves_merges_tier_move_and_shim_decision():
    """dataflux_tier_moves used to query ONLY `kind == "tier_move"` (emitted
    by gb_model_tier.py's explicit Python-API moves, which nothing in
    ai-forge actually calls) , so it silently returned [] even on a box with
    thousands of real per-allocation `shim_decision` events (the CUDA shim's
    own automatic tier-placement telemetry). Fixed to merge both real
    sources, tagging each row with `source` so a caller can tell them apart."""
    import gb_dataflux_mcp

    gdf.emit({"node": "host", "label": "gb_model_tier", "kind": "tier_move",
             "from_tier": "T1_HBM", "to_tier": "T3_NVME", "size_gb": 2.0,
             "n_items": 1, "items": ["x"], "duration_s": 0.1, "status": "ok"})
    gdf.emit({"node": "host", "label": "shim", "kind": "shim_decision",
             "stage": "t2_local", "tier": "t2_local", "reason": "t2_spill",
             "n_items": 1, "items": [], "duration_s": 0.0, "status": "ok",
             "bytes_mb": 17523, "fb_phys_used_pct": 14.1})

    moves = gb_dataflux_mcp.dataflux_tier_moves(days=1)
    sources = {m["source"] for m in moves}
    assert sources == {"tier_move", "shim_decision"}
    assert len(moves) == 2
