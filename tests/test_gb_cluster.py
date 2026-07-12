# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
pytest suite for gb_cluster.py's compute-orchestration API: cluster_map,
ClusterJobQueue, feeder_env, cluster_snapshot/_online_feeders_for_dispatch,
and the GB_MSG_FEEDER_STATUS wire parsing in gb_feeder_diag.py.

No CUDA, no /dev/greenboost, no running daemon/SSH needed. Feeder objects
are injected directly (the explicit `feeders=` param every dispatch
function accepts) or gb_cluster.feeders()/cluster_snapshot() is
monkeypatched , mirrors the mocked-wire-struct, no-network conventions in
tests/conftest.py.
"""
import struct
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_cluster as gc
import gb_dataflux as gdf
import gb_feeder_diag as fd

# Dataflux log isolation is handled by tests/conftest.py's autouse
# _isolate_dataflux_log_globally fixture (redirects GREENBOOST_DATAFLUX_LOG).


def _feeder(ip="10.0.0.2", online=True, gpu_util_pct=0, t1_free_mb=8000,
            link_mbps_ewma=0.0, **kw):
    return gc.Feeder(ip=ip, port=9740, hostname="feeder", ssh_user="u",
                     online=online, t1_free_mb=t1_free_mb, t1_total_mb=8192,
                     gpu_util_pct=gpu_util_pct, link_mbps_ewma=link_mbps_ewma,
                     **kw)


# ── cluster_map: zero-feeder / no-run_remote fallback ────────────────────────

def test_cluster_map_empty_items_returns_empty():
    assert gc.cluster_map([], lambda b: b) == []


def test_cluster_map_no_run_remote_respects_chunk_and_order():
    calls = []

    def run_local(batch):
        calls.append(list(batch))
        return [x * 2 for x in batch]

    result = gc.cluster_map(list(range(6)), run_local, chunk=2)
    assert result == [0, 2, 4, 6, 8, 10]
    assert calls == [[0, 1], [2, 3], [4, 5]]


def test_cluster_map_no_online_feeder_falls_back_to_single_local_call():
    def run_local(batch):
        return list(batch)

    def run_remote(feeder, batch):
        raise AssertionError("run_remote must not be called with feeders=[]")

    result = gc.cluster_map(list(range(5)), run_local, run_remote, feeders=[])
    assert result == list(range(5))


def test_cluster_map_auto_chunk_local_only_is_single_batch():
    calls = []

    def run_local(batch):
        calls.append(list(batch))
        return list(batch)

    gc.cluster_map(list(range(9)), run_local)
    assert calls == [list(range(9))]  # one batch call, no feeder to overlap with


# ── cluster_map: N-feeder dispatch ───────────────────────────────────────────

def test_cluster_map_dispatches_to_feeder_and_preserves_order():
    f = _feeder()
    local_batches, remote_batches = [], []

    def run_local(batch):
        local_batches.append(list(batch))
        time.sleep(0.01)
        return [("local", x) for x in batch]

    def run_remote(feeder, batch):
        remote_batches.append(list(batch))
        time.sleep(0.01)  # symmetric cost so both threads actually interleave
        return [("remote", x) for x in batch]

    items = list(range(20))
    result = gc.cluster_map(items, run_local, run_remote, feeders=[f], chunk=2)
    assert [r[1] for r in result] == items
    assert remote_batches, "feeder never picked up any chunk"
    assert local_batches, "host never picked up any chunk"


def test_cluster_map_multiple_feeders_only_known_ips_seen():
    f1, f2 = _feeder(ip="10.0.0.2"), _feeder(ip="10.0.0.3")
    seen = set()

    def run_local(batch):
        time.sleep(0.005)
        return list(batch)

    def run_remote(feeder, batch):
        seen.add(feeder.ip)
        return list(batch)

    items = list(range(40))
    result = gc.cluster_map(items, run_local, run_remote, feeders=[f1, f2], chunk=2)
    assert sorted(result) == items
    assert seen <= {"10.0.0.2", "10.0.0.3"}


# ── cluster_map: fault model ──────────────────────────────────────────────

def test_cluster_map_remote_failure_requeues_to_host():
    f = _feeder()
    remote_calls = {"n": 0}

    def run_local(batch):
        return [("local", x) for x in batch]

    def run_remote(feeder, batch):
        remote_calls["n"] += 1
        raise RuntimeError("feeder exploded")

    items = list(range(6))
    result = gc.cluster_map(items, run_local, run_remote, feeders=[f], chunk=2)
    assert [r[1] for r in result] == items
    assert all(r[0] == "local" for r in result)
    assert remote_calls["n"] >= 1


def test_cluster_map_run_local_exception_propagates():
    """Tested with no feeder so run_local is deterministically the ONLY
    path that touches items , with a feeder present this is a race (the
    faster side may finish first) that other tests exercise separately."""
    def run_local(batch):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        gc.cluster_map(list(range(4)), run_local, feeders=[])


# ── ClusterJobQueue ───────────────────────────────────────────────────────

def test_cluster_job_queue_no_feeder_runs_all_local():
    q = gc.ClusterJobQueue(run_local=lambda x: x + 1)
    futs = [q.submit(i) for i in range(10)]
    q.close(wait=True)
    assert [f.result(timeout=5) for f in futs] == list(range(1, 11))


def test_cluster_job_queue_dispatches_to_feeder():
    f = _feeder()

    def run_local(x):
        return ("local", x)

    def run_remote(feeder, x):
        return ("remote", x)

    q = gc.ClusterJobQueue(run_local, run_remote, feeders=[f])
    futs = [q.submit(i) for i in range(10)]
    q.close(wait=True)
    results = [fut.result(timeout=5) for fut in futs]
    assert sorted(r[1] for r in results) == list(range(10))


def test_cluster_job_queue_feeder_failure_falls_back_to_local():
    f = _feeder()

    def run_local(x):
        return ("local", x)

    def run_remote(feeder, x):
        raise RuntimeError("feeder gone")

    q = gc.ClusterJobQueue(run_local, run_remote, feeders=[f])
    futs = [q.submit(i) for i in range(5)]
    q.close(wait=True)
    results = [fut.result(timeout=5) for fut in futs]
    assert all(r[0] == "local" for r in results)
    assert sorted(r[1] for r in results) == list(range(5))


def test_cluster_job_queue_submit_after_close_raises():
    q = gc.ClusterJobQueue(run_local=lambda x: x)
    q.close(wait=True)
    with pytest.raises(RuntimeError):
        q.submit(1)


# ── telemetry-driven placement ───────────────────────────────────────────

def test_online_feeders_for_dispatch_filters_busy_and_offline(monkeypatch):
    busy = _feeder(ip="10.0.0.5", online=True, gpu_util_pct=95)
    idle = _feeder(ip="10.0.0.6", online=True, gpu_util_pct=10)
    offline = _feeder(ip="10.0.0.7", online=False, gpu_util_pct=0)

    def fake_snapshot(force=False, ssh_fallback_telemetry=True):
        return {"host": {}, "feeders": [asdict(busy), asdict(idle), asdict(offline)]}

    monkeypatch.setattr(gc, "cluster_snapshot", fake_snapshot)
    result = gc._online_feeders_for_dispatch()
    assert [f.ip for f in result] == ["10.0.0.6"]


def test_online_feeders_for_dispatch_filters_slow_link(monkeypatch):
    slow = _feeder(ip="10.0.0.8", online=True, link_mbps_ewma=0.5)
    fast = _feeder(ip="10.0.0.9", online=True, link_mbps_ewma=50.0)
    unmeasured = _feeder(ip="10.0.0.10", online=True, link_mbps_ewma=0.0)

    def fake_snapshot(force=False, ssh_fallback_telemetry=True):
        return {"host": {}, "feeders": [asdict(slow), asdict(fast), asdict(unmeasured)]}

    monkeypatch.setattr(gc, "cluster_snapshot", fake_snapshot)
    result = gc._online_feeders_for_dispatch()
    ips = {f.ip for f in result}
    assert ips == {"10.0.0.9", "10.0.0.10"}  # slow link excluded; unmeasured given a chance


def test_cluster_snapshot_cached(monkeypatch):
    gc._snapshot_cache["data"] = None
    gc._snapshot_cache["ts"] = 0.0
    calls = {"n": 0}

    def fake_feeders(probe=True, ssh_fallback_telemetry=False):
        calls["n"] += 1
        return []

    monkeypatch.setattr(gc, "feeders", fake_feeders)
    monkeypatch.setattr(gc, "_host_telemetry",
                        lambda: {"gpu_util_pct": None, "free_vram_mb": None, "temp_c": None})
    gc.cluster_snapshot()
    gc.cluster_snapshot()
    assert calls["n"] == 1
    gc.cluster_snapshot(force=True)
    assert calls["n"] == 2


# ── feeder_env: per-node gb-quant budget + TurboQuant propagation ───────────

def test_feeder_env_sets_per_node_quant_budget():
    f = _feeder(t1_free_mb=9216)  # 9 GiB free
    env = gc.feeder_env(f, workload="diffusion")
    assert env["GREENBOOST_CLUSTER"] == "0"
    budget = float(env["GB_QUANT_BUDGET_GB"])
    assert 7.5 < budget < 8.5  # (9216 - 1024) MB headroom, in GiB


def test_feeder_env_unknown_workload_raises():
    with pytest.raises(ValueError):
        gc.feeder_env(_feeder(), workload="bogus")


def test_feeder_env_extra_overrides():
    env = gc.feeder_env(_feeder(), extra={"GB_TQ_ATTN": "k4v3"})
    assert env["GB_TQ_ATTN"] == "k4v3"


def test_feeder_env_turboquant_propagates_from_local_env(monkeypatch):
    monkeypatch.setenv("GREENBOOST_TURBOQUANT", "1")
    env = gc.feeder_env(_feeder())
    assert env["GREENBOOST_TURBOQUANT"] == "1"


def test_feeder_env_llm_workload_keeps_cluster_on():
    env = gc.feeder_env(_feeder(), workload="llm")
    assert env["GREENBOOST_CLUSTER"] == "1"


# ── EWMA ──────────────────────────────────────────────────────────────────

def test_ewma_update_seeds_then_converges():
    store = {}
    v1 = gc._ewma_update(store, "k", 10.0)
    assert v1 == 10.0
    v2 = gc._ewma_update(store, "k", 0.0, alpha=0.5)
    assert v2 == 5.0


def test_cluster_map_updates_throughput_ewma_for_feeder_and_host():
    f = _feeder(ip="10.0.0.20")
    key = f"{f.ip}:{f.port}"
    gc._throughput_ewma.pop(key, None)
    gc._throughput_ewma.pop(gc._HOST_KEY, None)

    def run_local(batch):
        time.sleep(0.005)
        return list(batch)

    def run_remote(feeder, batch):
        time.sleep(0.005)
        return list(batch)

    gc.cluster_map(list(range(10)), run_local, run_remote, feeders=[f], chunk=2)
    assert gc._throughput_ewma.get(key, 0.0) > 0.0
    assert gc._throughput_ewma.get(gc._HOST_KEY, 0.0) > 0.0


# ── stage_bundle: rsync/mkdir wiring + link EWMA update ─────────────────────

def test_stage_bundle_pushes_and_updates_link_ewma(monkeypatch, tmp_path):
    calls = []

    class _Result:
        returncode = 0
        stderr = ""
        stdout = ""   # capture_output=True always yields stdout

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    f = _feeder(ip="10.0.0.30")
    key = f"{f.ip}:{f.port}"
    gc._link_mbps_ewma.pop(key, None)

    p = tmp_path / "afile.bin"
    p.write_bytes(b"x" * (1024 * 1024))
    gc.stage_bundle([str(p)], f)

    assert any(cmd[0] == "ssh" and "mkdir" in cmd[-1] for cmd in calls)
    assert any(cmd[0] == "rsync" for cmd in calls)
    assert gc._link_mbps_ewma.get(key, 0.0) >= 0.0


# ── ensure_feeder_model / feeder_has_model ──────────────────────────────────

def test_feeder_has_model_true_when_snapshot_dir_found(monkeypatch):
    calls = []

    class _Result:
        stdout = "/home/u/.cache/huggingface/hub/models--org--repo/snapshots/abc123\n"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    f = _feeder()
    assert gc.feeder_has_model(f, "org/repo") is True
    assert any("models--org--repo" in cmd[-1] for cmd in calls)


def test_feeder_has_model_false_when_no_snapshot(monkeypatch):
    class _Result:
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _Result())
    assert gc.feeder_has_model(_feeder(), "org/repo") is False


def test_feeder_has_model_false_on_timeout(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 15))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert gc.feeder_has_model(_feeder(), "org/repo") is False


def test_ensure_feeder_model_already_present_skips_push(monkeypatch, tmp_path):
    local_dir = tmp_path / "models--org--repo"
    local_dir.mkdir()
    monkeypatch.setattr(gc, "_hf_cache_root", lambda hf_home=None: tmp_path)
    monkeypatch.setattr(gc, "feeder_has_model", lambda *a, **kw: True)

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert gc.ensure_feeder_model(_feeder(), "org/repo") is True
    assert not calls  # no rsync/mkdir attempted — already present


def test_ensure_feeder_model_host_missing_cache_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(gc, "_hf_cache_root", lambda hf_home=None: tmp_path)  # empty dir
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert gc.ensure_feeder_model(_feeder(), "org/does-not-exist-locally") is False
    assert not calls


def test_ensure_feeder_model_pushes_when_missing(monkeypatch, tmp_path):
    local_dir = tmp_path / "models--org--repo"
    local_dir.mkdir()
    (local_dir / "snapshots").mkdir()
    monkeypatch.setattr(gc, "_hf_cache_root", lambda hf_home=None: tmp_path)
    monkeypatch.setattr(gc, "feeder_has_model", lambda *a, **kw: False)

    calls = []

    class _Result:
        returncode = 0
        stderr = ""
        stdout = ""   # capture_output=True always yields stdout

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert gc.ensure_feeder_model(_feeder(), "org/repo") is True
    assert any(cmd[0] == "ssh" and "mkdir" in cmd[-1] for cmd in calls)
    assert any(cmd[0] == "rsync" for cmd in calls)


def test_ensure_feeder_model_push_failure_returns_false(monkeypatch, tmp_path):
    local_dir = tmp_path / "models--org--repo"
    local_dir.mkdir()
    monkeypatch.setattr(gc, "_hf_cache_root", lambda hf_home=None: tmp_path)
    monkeypatch.setattr(gc, "feeder_has_model", lambda *a, **kw: False)

    class _MkdirOk:
        returncode = 0
        stderr = ""

    class _RsyncFail:
        returncode = 1
        stderr = "rsync: connection lost"

    results = iter([_MkdirOk(), _RsyncFail()])
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: next(results))
    assert gc.ensure_feeder_model(_feeder(), "org/repo") is False


def test_hf_repo_dirname_format():
    assert gc._hf_repo_dirname("black-forest-labs/FLUX.2-klein-9B") == \
        "models--black-forest-labs--FLUX.2-klein-9B"


# ── GB_MSG_FEEDER_STATUS wire parsing (gb_feeder_diag.query_feeder_status) ──

class _FakeSocket:
    """Minimal in-memory socket stand-in for gb_feeder_diag's send/recv."""

    def __init__(self, response_payload: bytes, msg_type=None):
        msg_type = fd.GB_MSG_RESPONSE if msg_type is None else msg_type
        hdr = struct.pack("<IHHII", fd.GB_NET_MAGIC, msg_type, fd.GB_NET_FLAG_RESPONSE,
                          len(response_payload), 0)
        self._buf = hdr + response_payload

    def sendall(self, data):
        pass

    def recv(self, n):
        chunk, self._buf = self._buf[:n], self._buf[n:]
        return chunk


def _pack_feeder_status_v30(**overrides):
    fields = dict(status=0, mps_sm_pct=0, t1_free=1_000_000, t1_total=2_000_000,
                  t2_free=3_000_000, t2_total=4_000_000, t3_free=5_000_000,
                  t3_total=6_000_000, kernel_dispatch_count=7, _pad=0)
    fields.update(overrides)
    order = ("status", "mps_sm_pct", "t1_free", "t1_total", "t2_free", "t2_total",
              "t3_free", "t3_total", "kernel_dispatch_count", "_pad")
    return struct.pack("<IIQQQQQQII", *[fields[k] for k in order])


def test_query_feeder_status_v30_short_reply_zeroes_telemetry():
    fd._reset_seq()
    sock = _FakeSocket(_pack_feeder_status_v30())
    result = fd.query_feeder_status(sock)
    assert result["kernel_dispatch_count"] == 7
    assert result["gpu_util_pct"] == 0
    assert result["gpu_temp_c"] == 0


def test_query_feeder_status_v31_parses_telemetry():
    fd._reset_seq()
    v30 = _pack_feeder_status_v30(kernel_dispatch_count=42)
    v31_ext = struct.pack("<HHIIII", 65, 120, 87, 3, 0x4, 0)
    sock = _FakeSocket(v30 + v31_ext)
    result = fd.query_feeder_status(sock)
    assert result["kernel_dispatch_count"] == 42
    assert result["gpu_temp_c"] == 65
    assert result["gpu_power_w"] == 120
    assert result["gpu_util_pct"] == 87
    assert result["ecc_dbe_count"] == 3
    assert result["throttle_reasons"] == 0x4


def test_query_feeder_status_too_short_returns_none():
    fd._reset_seq()
    sock = _FakeSocket(b"\x00" * 10)
    assert fd.query_feeder_status(sock) is None


# ── job shapes modeled on ai-forge image batches ────────────────────────────

def test_cluster_map_image_slug_batches_preserve_per_slug_seeds():
    """Mirrors ai-forge's gen_image_batch.py contract: run_local/run_remote
    receive a list of (slug, seed) pairs and must return one result per
    input, in order , regardless of whether the batch ran on host or feeder.
    """
    f = _feeder()
    slugs = [(f"card_{i}", 1000 + i) for i in range(12)]

    def run_local(batch):
        return [{"slug": s, "seed": seed, "node": "host"} for s, seed in batch]

    def run_remote(feeder, batch):
        return [{"slug": s, "seed": seed, "node": feeder.ip} for s, seed in batch]

    result = gc.cluster_map(slugs, run_local, run_remote, feeders=[f], chunk=3)
    assert [r["slug"] for r in result] == [s for s, _ in slugs]
    assert [r["seed"] for r in result] == [seed for _, seed in slugs]


# ── gb_dataflux integration: cluster_map/ClusterJobQueue emit events ────────

def test_cluster_map_emits_dataflux_events_local_and_remote():
    f = _feeder(ip="10.0.0.40")

    def run_local(batch):
        time.sleep(0.005)
        return list(batch)

    def run_remote(feeder, batch):
        time.sleep(0.005)
        return list(batch)

    gc.cluster_map(list(range(6)), run_local, run_remote, feeders=[f],
                   chunk=2, label="unit-test-label")
    events = gdf.read_events()
    assert events, "no dataflux events recorded"
    assert all(ev["label"] == "unit-test-label" for ev in events)
    nodes = {ev["node"] for ev in events}
    assert nodes == {"host", "feeder"}
    assert all(ev["status"] == "ok" for ev in events)


def test_cluster_map_emits_error_event_on_remote_failure():
    f = _feeder()

    def run_local(batch):
        return list(batch)

    def run_remote(feeder, batch):
        raise RuntimeError("boom")

    gc.cluster_map(list(range(2)), run_local, run_remote, feeders=[f], chunk=2)
    events = gdf.read_events()
    error_events = [e for e in events if e["status"] == "error"]
    assert error_events
    assert "boom" in error_events[0]["error"]


def test_cluster_job_queue_emits_dataflux_events():
    f = _feeder(ip="10.0.0.41")
    q = gc.ClusterJobQueue(run_local=lambda x: x, run_remote=lambda feeder, x: x,
                           feeders=[f], label="queue-test")
    futs = [q.submit(i) for i in range(4)]
    q.close(wait=True)
    [fut.result(timeout=5) for fut in futs]
    events = gdf.read_events()
    assert events
    assert all(ev["label"] == "queue-test" for ev in events)


def test_dataflux_summarize_aggregates_by_node():
    events = [
        {"ts": 1.0, "node": "host", "label": "x", "n_items": 2, "duration_s": 1.0, "status": "ok"},
        {"ts": 2.0, "node": "host", "label": "x", "n_items": 3, "duration_s": 2.0, "status": "ok"},
        {"ts": 3.0, "node": "feeder1", "label": "x", "n_items": 1, "duration_s": 0.5, "status": "error"},
    ]
    s = gdf.summarize(events)
    assert s["nodes"]["host"]["items"] == 5
    assert s["nodes"]["host"]["duration_s"] == 3.0
    assert s["nodes"]["feeder1"]["errors"] == 1
    assert s["errors"] == 1
    assert s["total_items"] == 6
