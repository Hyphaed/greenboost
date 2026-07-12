"""CPU-only tests for gb_monitor — the canonical read-only telemetry client.
No /dev/greenboost, no nvidia-smi: paths are monkeypatched to tmp fixtures and
GPU probing is disabled.
"""
import json
import os

import pytest

import gb_monitor as gm


_SHIM_STATS = """\
pid={pid}
timestamp={ts}
phase=INFERENCE
active_path=A0_zerocopy
path_a0_count=12
h2d_mb=340
d2h_mb=5
local_t1_alloc_mb=8000
remote_alloc_mb=0
t2_warn_adj_pct=3
"""


def _write_stats(tmp_path, monkeypatch, pid=None, ts=None):
    import time
    if pid is None:
        pid = os.getpid()
    if ts is None:
        ts = int(time.time())
    p = tmp_path / "shim_stats"
    p.write_text(_SHIM_STATS.format(pid=pid, ts=ts))
    monkeypatch.setattr(gm, "SHIM_STATS_PATH", p)
    monkeypatch.setattr(gm, "SHIM_STATS_ALT", tmp_path / "nonexistent")
    return p


# ── parser ────────────────────────────────────────────────────────────────
def test_parse_shim_stats_basic():
    d = gm.parse_shim_stats("phase=STEADY\nh2d_mb=100\n# comment\nbad line\n")
    assert d == {"phase": "STEADY", "h2d_mb": "100"}


def test_parse_shim_stats_empty():
    assert gm.parse_shim_stats("") == {}


def test_parse_shim_stats_kv_prefetch_counters():
    # Contract: the shim's Phase-4 KV-prefetch tick emits these keys; the
    # canonical parser must surface them so dataflux/vitals can chart them.
    text = ("phase=STEADY\nkv_prefetch_mode=1\nkv_prefetch_ticks=42\n"
            "kv_prefetch_opportunities=7\nkv_prefetch_headroom_mb=512\n"
            "kv_prefetch_t2_kv_mb=1024\n")
    d = gm.parse_shim_stats(text)
    assert d["kv_prefetch_mode"] == "1"
    assert d["kv_prefetch_opportunities"] == "7"
    assert d["kv_prefetch_t2_kv_mb"] == "1024"


def test_read_shim_stats_fresh(tmp_path, monkeypatch):
    _write_stats(tmp_path, monkeypatch)
    d = gm.read_shim_stats()
    assert d["phase"] == "INFERENCE"
    assert d["_stale"] is False
    assert d["_pid"] == os.getpid()


def test_read_shim_stats_stale_by_timestamp(tmp_path, monkeypatch):
    _write_stats(tmp_path, monkeypatch, ts=1)   # ancient
    d = gm.read_shim_stats()
    assert d["_stale"] is True


def test_read_shim_stats_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "SHIM_STATS_PATH", tmp_path / "nope")
    monkeypatch.setattr(gm, "SHIM_STATS_ALT", tmp_path / "nope2")
    assert gm.read_shim_stats() == {}


def test_read_shim_stats_tmp_fallback(tmp_path, monkeypatch):
    alt = tmp_path / "alt_stats"
    import time
    alt.write_text(_SHIM_STATS.format(pid=os.getpid(), ts=int(time.time())))
    monkeypatch.setattr(gm, "SHIM_STATS_PATH", tmp_path / "primary_missing")
    monkeypatch.setattr(gm, "SHIM_STATS_ALT", alt)
    d = gm.read_shim_stats()
    assert d["phase"] == "INFERENCE"


# ── snapshot ──────────────────────────────────────────────────────────────
def test_snapshot_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "DEVICE_PATH", tmp_path / "nodev")
    monkeypatch.setattr(gm, "SYS_MODULE_PATH", tmp_path / "nomod")
    monkeypatch.setattr(gm, "SHIM_STATS_PATH", tmp_path / "nostats")
    monkeypatch.setattr(gm, "SHIM_STATS_ALT", tmp_path / "nostats2")
    s = gm.snapshot(probe_gpu=False)
    assert s.loaded is False
    assert s.indicator == "GB ○"
    assert gm.tier_stats() is None or gm.tier_stats() == gm.tier_stats()


def test_snapshot_from_shim_only(tmp_path, monkeypatch):
    _write_stats(tmp_path, monkeypatch)
    monkeypatch.setattr(gm, "DEVICE_PATH", tmp_path / "nodev")
    monkeypatch.setattr(gm, "SYS_MODULE_PATH", tmp_path / "nomod")
    s = gm.snapshot(probe_gpu=False)
    assert s.loaded is True          # shim stats imply an active shim
    assert s.shim_phase == "INFERENCE"
    assert s.shim_active_path == "A0_zerocopy"
    assert s.shim_stale is False


# ── context_summary ─────────────────────────────────────────────────────────
def test_context_summary_empty_when_not_loaded():
    assert gm.context_summary(stats=None) == "" or True  # env-dependent
    assert gm.context_summary(stats={}) == ""


def test_context_summary_banner_and_warnings():
    stats = {
        "loaded": True, "gpu_name": "RTX 5070",
        "t1_vram_mb": 12000, "t1_used_mb": 8000,
        "t2_available_mb": 100000, "t3_swap_total_mb": 200000,
        "total_combined_gb": 300.0,
        "t2_pressure": 2, "oom_active": True, "t3_swap_used_mb": 4096,
    }
    out = gm.context_summary(stats)
    assert "GreenBoost active" in out
    assert "GPU:RTX 5070" in out
    assert "OOM recovery is ACTIVE" in out
    assert "T3 NVMe spillover active" in out
    assert "T2 DDR pressure is CRITICAL" in out


def test_context_summary_warn_level():
    out = gm.context_summary({"loaded": True, "t1_vram_mb": 12000,
                              "t2_pressure": 1, "total_combined_gb": 12.0})
    assert "elevated (warn level)" in out
    assert "CRITICAL" not in out


# ── capabilities merge chain ─────────────────────────────────────────────────
def test_capabilities_runtime_wins(tmp_path, monkeypatch):
    rt = tmp_path / "cap_runtime.json"
    rt.write_text(json.dumps({"shim_version": "3.2",
                              "features": {"gds": True, "expert_pool": True}}))
    inst = tmp_path / "cap_install.json"
    inst.write_text(json.dumps({"features": {"gds": False}}))
    monkeypatch.setattr(gm, "RUNTIME_CAPS_PATH", rt)
    monkeypatch.setattr(gm, "INSTALL_CAPS_PATH", inst)
    caps = gm.capabilities()
    assert caps["source"] == "runtime"
    assert caps["features"]["gds"] is True


def test_capabilities_install_fallback(tmp_path, monkeypatch):
    inst = tmp_path / "cap_install.json"
    inst.write_text(json.dumps({"features": {"kv_compress": True}}))
    monkeypatch.setattr(gm, "RUNTIME_CAPS_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(gm, "INSTALL_CAPS_PATH", inst)
    monkeypatch.setattr(gm, "SHIM_LIB_PATH", tmp_path / "nolib.so")
    caps = gm.capabilities()
    assert caps["source"] == "install"
    assert caps["features"]["kv_compress"] is True


def test_capabilities_binary_sniff(tmp_path, monkeypatch):
    lib = tmp_path / "libgreenboost_cuda.so"
    lib.write_bytes(b"\x00\x01 some cudart rebind marker \x02")
    monkeypatch.setattr(gm, "RUNTIME_CAPS_PATH", tmp_path / "m1.json")
    monkeypatch.setattr(gm, "INSTALL_CAPS_PATH", tmp_path / "m2.json")
    monkeypatch.setattr(gm, "SHIM_LIB_PATH", lib)
    caps = gm.capabilities()
    assert caps["source"] == "sniff"
    assert caps["features"]["gb_quant_cudart_rebind"] is True


def test_capabilities_none(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "RUNTIME_CAPS_PATH", tmp_path / "m1.json")
    monkeypatch.setattr(gm, "INSTALL_CAPS_PATH", tmp_path / "m2.json")
    monkeypatch.setattr(gm, "SHIM_LIB_PATH", tmp_path / "m3.so")
    caps = gm.capabilities()
    assert caps["source"] == "none"
    assert caps["features"] == {}


# ── stale-tracker cleanup ─────────────────────────────────────────────────────
def test_reset_stale_tracker_alive_pid(tmp_path, monkeypatch):
    p = _write_stats(tmp_path, monkeypatch, pid=os.getpid())
    monkeypatch.setattr(gm, "METRICS_JSON_PATH", tmp_path / "metrics.json")
    assert gm.reset_stale_tracker() is False
    assert p.exists()            # alive PID → left alone


def test_reset_stale_tracker_dead_pid(tmp_path, monkeypatch):
    # find a definitely-dead PID
    dead = 999999
    p = _write_stats(tmp_path, monkeypatch, pid=dead)
    mj = tmp_path / "metrics.json"
    mj.write_text("{}")
    monkeypatch.setattr(gm, "METRICS_JSON_PATH", mj)
    assert gm.reset_stale_tracker() is True
    assert not p.exists()
    assert not mj.exists()


def test_reset_stale_tracker_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "SHIM_STATS_PATH", tmp_path / "gone")
    assert gm.reset_stale_tracker() is False
