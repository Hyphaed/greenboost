#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for ReactiveOrchestrator , Loops A, C, E (B3 cluster, B4 health).

All filesystem/ioctl calls patched out. No CUDA, no daemon.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from gb_orchestrator import ReactiveOrchestrator, GpuHealth
from gb_telemetry import GpuMetrics, GbPoolInfo


# ── Helpers ──────────────────────────────────────────────────────────────────

class _Result:
    def __init__(self, reason="ok", applied=True):
        self.reason = reason
        self.applied = applied


class _MockGbControl:
    def __init__(self):
        self._last = {}
        self.calls = []   # (method_name, value, reason) , Loops O-S assertions

    def set_safety_reserve_gb(self, v, *, reason=""):
        self._last["safety_reserve_gb"] = (v, 0)
        return _Result(f"set={v}")

    def set_kv_reserve_mb(self, v, *, reason=""):
        self._last["kv_reserve_mb"] = (v, 0)
        return _Result(f"set={v}")

    def set_workstation_reserve_mb(self, v, *, reason=""):
        self._last["workstation_reserve_mb"] = (v, 0)
        return _Result(f"set={v}")

    def dump(self):
        return {}

    # ── continuous OS tuning levers (Loops O-S) ───────────────────────────────

    def set_cpu_governor(self, v, *, reason=""):
        self._last["cpu_governor"] = (v, 0)
        self.calls.append(("set_cpu_governor", v, reason))
        return _Result(f"set={v}")

    def set_energy_perf_pref(self, v, *, reason=""):
        self._last["energy_perf_pref"] = (v, 0)
        self.calls.append(("set_energy_perf_pref", v, reason))
        return _Result(f"set={v}")

    def set_numa_balancing(self, v, *, reason=""):
        self._last["numa_balancing"] = (v, 0)
        self.calls.append(("set_numa_balancing", v, reason))
        return _Result(f"set={v}")

    def set_swappiness(self, v, *, reason=""):
        self._last["swappiness"] = (v, 0)
        self.calls.append(("set_swappiness", v, reason))
        return _Result(f"set={v}")

    def set_dirty_background_ratio(self, v, *, reason=""):
        self._last["dirty_background_ratio"] = (v, 0)
        self.calls.append(("set_dirty_background_ratio", v, reason))
        return _Result(f"set={v}")

    def set_watermark_scale_factor(self, v, *, reason=""):
        self._last["watermark_scale_factor"] = (v, 0)
        self.calls.append(("set_watermark_scale_factor", v, reason))
        return _Result(f"set={v}")

    def set_gpu_persistence(self, v, *, reason=""):
        self._last["gpu_persistence"] = (v, 0)
        self.calls.append(("set_gpu_persistence", v, reason))
        return _Result(f"set={v}")

    def lock_gpu_clocks(self, lo, hi, *, reason=""):
        self._last["gpu_clocks_locked"] = ((lo, hi), 0)
        self.calls.append(("lock_gpu_clocks", (lo, hi), reason))
        return _Result(f"set={lo},{hi}")

    def reset_gpu_clocks(self, *, reason=""):
        self._last.pop("gpu_clocks_locked", None)
        self.calls.append(("reset_gpu_clocks", None, reason))
        return _Result("reset")

    def set_gpu_power_limit(self, v, *, reason=""):
        self._last["gpu_power_limit_w"] = (v, 0)
        self.calls.append(("set_gpu_power_limit", v, reason))
        return _Result(f"set={v}")

    def set_prefetch_throttle(self, v, *, reason=""):
        self._last["prefetch_throttle"] = (v, 0)
        self.calls.append(("set_prefetch_throttle", v, reason))
        return _Result(f"set={v}")

    def set_t3_cap_mb(self, v, *, reason=""):
        self._last["t3_cap_mb"] = (v, 0)
        self.calls.append(("set_t3_cap_mb", v, reason))
        return _Result(f"set={v}")

    def restore_baseline(self):
        self.calls.append(("restore_baseline", None, ""))
        self._last.clear()
        return {"cpu_governor": True}


def _make_metrics(fb_used_pct=0.0, fb_total_mb=12000,
                  ecc_dbe_volatile=0, health_ok=True,
                  kv_reserve_mb=0, kv_used_mb=0, kv_t2_mb=0,
                  temp_c=50.0, mem_copy_util_pct=0.0,
                  power_w=0.0, power_limit_w=0.0) -> GpuMetrics:
    m = GpuMetrics()
    m.fb_total_mb = fb_total_mb
    m.fb_used_pct = fb_used_pct
    m.fb_used_mb  = int(fb_total_mb * fb_used_pct / 100)
    m.fb_free_mb  = fb_total_mb - m.fb_used_mb
    m.ecc_dbe_volatile = ecc_dbe_volatile
    m.health_ok   = health_ok
    m.temp_c      = temp_c
    m.mem_copy_util_pct = mem_copy_util_pct
    m.power_w       = power_w
    m.power_limit_w = power_limit_w
    if kv_reserve_mb > 0:
        gb = GbPoolInfo()
        gb.kv_reserve_mb = kv_reserve_mb
        gb.kv_used_mb    = kv_used_mb
        gb.kv_t2_mb      = kv_t2_mb
        m.gb = gb
    return m


@pytest.fixture
def orch():
    ctrl = _MockGbControl()
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process")
        yield o


@pytest.fixture
def orch_with_tier():
    ctrl = _MockGbControl()
    tm = MagicMock()
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process", tier_manager=tm)
        yield o, tm


# ── Loop A , ECC ratchet ──────────────────────────────────────────────────────

def test_loop_a_ecc_ratchet_sets_degraded(orch):
    orch.on_metrics(_make_metrics(ecc_dbe_volatile=1))
    assert orch.ecc_degraded is True


def test_loop_a_ecc_ratchet_monotone(orch):
    orch.on_metrics(_make_metrics(ecc_dbe_volatile=3))
    first_seen = orch._ecc_seen
    # Lower count should not fire again (ratchet only goes up)
    orch.on_metrics(_make_metrics(ecc_dbe_volatile=1))
    assert orch._ecc_seen == first_seen   # no backward movement


def test_loop_a_raises_safety_reserve(orch):
    orch.on_metrics(_make_metrics(ecc_dbe_volatile=1))
    assert "safety_reserve_gb" in orch._ctrl._last


def test_loop_a_shrinks_kv_reserve(orch):
    orch.on_metrics(_make_metrics(ecc_dbe_volatile=1))
    assert "kv_reserve_mb" in orch._ctrl._last


# ── Loop C , predictive KV grow ───────────────────────────────────────────────

def test_loop_c_skipped_while_ecc_degraded(orch):
    """Even with sustained KV pressure (25 polls), Loop C must not grow KV while ECC degraded."""
    orch.on_metrics(_make_metrics(ecc_dbe_volatile=1))
    assert orch.ecc_degraded is True

    # Record KV reserve after ECC shrink
    ecc_kv = orch._ctrl._last.get("kv_reserve_mb", (0, 0))[0]

    # 25 polls at 93% KV pressure , enough to fire Loop C under normal conditions
    for _ in range(25):
        orch.on_metrics(_make_metrics(kv_reserve_mb=512, kv_used_mb=480))

    new_kv = orch._ctrl._last.get("kv_reserve_mb", (0, 0))[0]
    assert new_kv <= ecc_kv, "Loop C grew KV while ECC degraded , must not happen"


def test_loop_c_t1_headroom_clamp(orch):
    """When T1 headroom is too small, KV grow is clamped or skipped."""
    orch._total_vram_mb  = 12000
    orch._ws_reserve_mb  = 9000   # nearly saturated
    orch._ctrl._last["kv_reserve_mb"] = (0, 0)

    for _ in range(5):
        orch.on_metrics(_make_metrics(kv_reserve_mb=256, kv_used_mb=240))  # high pressure

    # Headroom = 12000 - 9000 - 0 - 2048 = 952 < KV_STEP_MB(512), but >= 0
    # so a clamped grow is allowed; but it should not exceed headroom
    kv_new = orch._ctrl._last.get("kv_reserve_mb", (0, 0))[0]
    headroom = orch._total_vram_mb - orch._ws_reserve_mb - 2048
    assert kv_new <= max(0, headroom), f"KV grew beyond T1 headroom: {kv_new} > {headroom}"


# ── Loop E , VRAM pressure → tier demotion ────────────────────────────────────

def test_loop_e_fires_auto_evict_after_sustained_pressure(orch_with_tier):
    o, tm = orch_with_tier
    # Need ~21 polls at 90% for EMA(alpha=1/6) to cross 87% + confirm=3
    # Use 30 polls for reliable margin
    for _ in range(30):
        o.on_metrics(_make_metrics(fb_used_pct=90.0))

    assert o._vram_pressure is True
    tm.auto_evict.assert_called_once()


def test_loop_e_does_not_fire_on_brief_spike(orch_with_tier):
    o, tm = orch_with_tier
    # Only 3 polls at 90% , EMA hasn't settled to 87% yet
    for _ in range(3):
        o.on_metrics(_make_metrics(fb_used_pct=90.0))
    assert o._vram_pressure is False
    tm.auto_evict.assert_not_called()


def test_loop_e_pressure_clears_when_vram_drops(orch_with_tier):
    o, tm = orch_with_tier
    # Enter pressure
    for _ in range(30):
        o.on_metrics(_make_metrics(fb_used_pct=90.0))
    assert o._vram_pressure is True

    # Feed 5 polls at 0% , EMA drops well below exit threshold (70%)
    for _ in range(5):
        o.on_metrics(_make_metrics(fb_used_pct=0.0))
    assert o._vram_pressure is False


def test_loop_e_no_crash_without_tier_manager(orch):
    """Loop E should not raise if no tier_manager is provided."""
    for _ in range(30):
        orch.on_metrics(_make_metrics(fb_used_pct=90.0))
    # Should complete without exception; vram_pressure flag is set
    assert orch._vram_pressure is True


# ── dump() ────────────────────────────────────────────────────────────────────

def test_dump_contains_vram_pressure_key(orch):
    d = orch.dump()
    assert "vram_pressure" in d


def test_dump_signals_include_fb_used_pct(orch):
    d = orch.dump()
    names = [s["name"] for s in d.get("signals", [])]
    assert "fb_used_pct" in names


def test_dump_signals_include_health_ok(orch):
    d = orch.dump()
    names = [s["name"] for s in d.get("signals", [])]
    assert "health_ok" in names


def test_dump_health_evict_armed_reflects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_ORCH_HEALTH_EVICT", "1")
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=_MockGbControl(), mode="process")
    assert o.dump()["health_evict_armed"] is True


# ── B3 , Cluster-aware Loop E ─────────────────────────────────────────────────

def test_b3_cluster_pressure_fires_auto_evict_without_local_threshold():
    """When cluster_tel.any_should_demote() is True, auto_evict fires even if
    local EMA is well below 87%."""
    ctrl = _MockGbControl()
    tm   = MagicMock()
    cluster_tel = MagicMock()
    cluster_tel.any_should_demote.return_value = True

    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(
            control=ctrl, mode="process",
            tier_manager=tm, cluster_tel=cluster_tel,
        )

    # Single poll at 50% local pressure (well below 87% threshold)
    o.on_metrics(_make_metrics(fb_used_pct=50.0))
    assert o._cluster_pressure is True
    tm.auto_evict.assert_called_once()


def test_b3_cluster_pressure_clears_when_feeders_ok():
    ctrl = _MockGbControl()
    tm   = MagicMock()
    cluster_tel = MagicMock()
    cluster_tel.any_should_demote.return_value = True

    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(
            control=ctrl, mode="process",
            tier_manager=tm, cluster_tel=cluster_tel,
        )

    o.on_metrics(_make_metrics(fb_used_pct=50.0))
    assert o._cluster_pressure is True

    cluster_tel.any_should_demote.return_value = False
    o.on_metrics(_make_metrics(fb_used_pct=50.0))
    assert o._cluster_pressure is False


def test_b3_no_cluster_tel_single_gpu_unchanged(orch_with_tier):
    """Without cluster_tel, behaviour is unchanged."""
    o, tm = orch_with_tier
    # 1 poll at 50% , no cluster_tel, no local threshold crossed
    o.on_metrics(_make_metrics(fb_used_pct=50.0))
    assert o._cluster_pressure is False
    tm.auto_evict.assert_not_called()


# ── B4 , Loop F DCGM health ──────────────────────────────────────────────────

def test_b4_health_advisory_without_env_flag():
    """When GB_ORCH_HEALTH_EVICT is not set, health degradation logs advisory
    but does NOT call auto_evict."""
    ctrl = _MockGbControl()
    tm   = MagicMock()

    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process", tier_manager=tm)

    # health_ok transitions from True → False over confirm=2 polls
    o.on_metrics(_make_metrics(health_ok=False))
    o.on_metrics(_make_metrics(health_ok=False))
    # After 2 confirmed polls, signal fires
    tm.auto_evict.assert_not_called()


def test_b4_health_evicts_when_armed(monkeypatch):
    """When GB_ORCH_HEALTH_EVICT=1, confirmed health degradation calls auto_evict."""
    monkeypatch.setenv("GB_ORCH_HEALTH_EVICT", "1")
    ctrl = _MockGbControl()
    tm   = MagicMock()

    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process", tier_manager=tm)

    assert o._health_evict_armed is True
    o.on_metrics(_make_metrics(health_ok=False))
    o.on_metrics(_make_metrics(health_ok=False))
    # confirm=2 → should have fired
    tm.auto_evict.assert_called_once()


def test_b4_dump_has_health_evict_armed(monkeypatch):
    monkeypatch.setenv("GB_ORCH_HEALTH_EVICT", "1")
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=_MockGbControl(), mode="process")
    d = o.dump()
    assert d["health_evict_armed"] is True
    assert "health_ok" in d


# ── Loop C , KV grow + shrink cycle ──────────────────────────────────────────

def test_do_kv_grow_sets_kv_reserve(orch):
    """_do_kv_grow calls set_kv_reserve_mb when T1 headroom allows."""
    orch._total_vram_mb  = 12000
    orch._ws_reserve_mb  = 0
    d = {}
    orch._do_kv_grow("test_reason", d)
    assert "kv_reserve_mb" in orch._ctrl._last
    new_kv = orch._ctrl._last["kv_reserve_mb"][0]
    assert new_kv == orch._kv_step_mb   # grew from 0 by one step


def test_do_kv_grow_skips_when_no_headroom(orch):
    """When T1 headroom is zero or negative, KV grow is skipped."""
    orch._total_vram_mb  = 12000
    orch._ws_reserve_mb  = 12000   # no room at all
    d = {}
    orch._do_kv_grow("test_reason", d)
    assert d.get("action") == "kv_grow_skipped_no_headroom"


def test_do_kv_grow_clamps_to_headroom(orch):
    """When headroom < kv_step, grow is clamped to available headroom."""
    orch._total_vram_mb  = 12000
    orch._ws_reserve_mb  = 9000    # headroom = 12000 - 9000 - 0 - weights_floor
    # headroom = 12000 - 9000 - weights_floor , less than kv_step (512)
    # but > 0 when weights_floor = 2048, so headroom = 952
    d = {}
    orch._do_kv_grow("test_reason", d)
    new_kv = orch._ctrl._last.get("kv_reserve_mb", (0, 0))[0]
    # Should have grown, but by at most 952 (headroom)
    headroom = orch._total_vram_mb - orch._ws_reserve_mb - orch._weights_floor_mb
    assert new_kv <= max(0, headroom)


def test_kv_shrink_on_pressure_ok(orch):
    """When pressure clears, KV reserve shrinks by one step."""
    # Pre-set KV at 3072 MB (was grown by 2 steps from 2048 baseline)
    orch._ctrl._last["kv_reserve_mb"] = (3072, 0)
    orch._on_kv_pressure_ok(0.30)
    new_kv = orch._ctrl._last["kv_reserve_mb"][0]
    assert new_kv < 3072, "KV should have shrunk on pressure clear"
    assert new_kv == 3072 - orch._kv_step_mb


def test_kv_shrink_respects_floor(orch):
    """KV shrink never drops below the topology baseline (kv_floor_mb)."""
    floor = orch._kv_floor_mb
    orch._ctrl._last["kv_reserve_mb"] = (floor, 0)  # already at floor
    orch._on_kv_pressure_ok(0.30)
    # No-op: kv_floor_mb already reached
    kv_after = orch._ctrl._last["kv_reserve_mb"][0]
    assert kv_after == floor, "KV must not shrink below topology baseline"


def test_kv_shrink_blocked_when_ecc_degraded(orch):
    """ECC degraded flag blocks the shrink path (ECC owns KV reserve in degraded state)."""
    orch._ctrl._last["kv_reserve_mb"] = (3072, 0)
    orch.ecc_degraded = True
    orch._on_kv_pressure_ok(0.30)
    kv_after = orch._ctrl._last["kv_reserve_mb"][0]
    assert kv_after == 3072, "KV must not shrink while ECC degraded"


def test_kv_shrink_blocked_when_no_ctrl():
    """No crash and no shrink when ctrl is None."""
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=None, mode="process")
    o._ctrl = None
    o._on_kv_pressure_ok(0.30)   # must not raise


def test_kv_grow_shrink_records_decision(orch):
    """Both grow and shrink record to _decisions."""
    orch._total_vram_mb = 12000
    orch._ws_reserve_mb = 0
    orch._ctrl._last["kv_reserve_mb"] = (3072, 0)

    initial_decisions = len(orch._decisions)
    orch._do_kv_grow("test", {})
    orch._on_kv_pressure_ok(0.30)
    assert len(orch._decisions) >= initial_decisions + 1  # at least shrink recorded


# ── Loop B , workstation governor ─────────────────────────────────────────────

def test_loop_b_raises_ws_reserve_when_non_gb_enters(orch):
    """Above enter threshold, ws_reserve climbs by WS_STEP_MB."""
    # Must first initialise total_vram so thresholds are computed
    orch.on_metrics(_make_metrics(fb_total_mb=12000, fb_used_pct=0))
    enter_mb = int(12000 * 0.10)   # = 1200

    # Need EMA(non_gb) > 1200.  With alpha=1/8, pump at 4000 MB for ~12 polls
    for _ in range(15):
        orch.feed_vram_state(non_gb_mb=4000.0, total_mb=12000)

    assert orch._ws_above is True
    assert orch._ws_reserve_mb > 0


def test_loop_b_lowers_ws_reserve_when_non_gb_exits(orch):
    """After enter, when EMA drops below exit threshold reserve falls."""
    orch.on_metrics(_make_metrics(fb_total_mb=12000, fb_used_pct=0))

    # Enter: pump high non-GB
    for _ in range(15):
        orch.feed_vram_state(non_gb_mb=4000.0, total_mb=12000)
    assert orch._ws_above is True
    ws_peak = orch._ws_reserve_mb

    # Exit: feed zero non-GB for several polls to drop EMA < 7% of 12000 = 840
    for _ in range(20):
        orch.feed_vram_state(non_gb_mb=0.0, total_mb=12000)

    assert orch._ws_above is False
    assert orch._ws_reserve_mb < ws_peak


def test_loop_b_no_action_before_total_vram_known(orch):
    """Before thresholds are set (total_mb=0), Loop B handler returns early."""
    # 5 polls with total_mb=0: _ws_enter_mb stays 0 → early return in handler fires
    for _ in range(5):
        orch.feed_vram_state(non_gb_mb=9999.0, total_mb=0)
    assert orch._ws_above is False  # no fire without thresholds


# ── Loop D , thermal governor ─────────────────────────────────────────────────

def test_loop_d_raises_safety_reserve_on_high_temp(orch):
    """When temp_c EMA crosses 83°C for confirm=3 polls, safety_reserve_gb is raised.

    With EWMA alpha=1/8 and confirm=3, need ~21 polls at 95°C to cross
    the 83°C enter threshold reliably. Use 25 for margin.
    """
    for _ in range(25):
        orch.on_metrics(_make_metrics(fb_used_pct=10.0, temp_c=95.0))

    assert "safety_reserve_gb" in orch._ctrl._last


def test_loop_d_direct_handler_raises_safety(orch):
    """Directly call _on_temp_high to verify the handler actuates."""
    initial = orch._ctrl._last.get("safety_reserve_gb", (0, 0))[0]
    orch._on_temp_high(90.0)
    new_val = orch._ctrl._last["safety_reserve_gb"][0]
    assert new_val > initial


def test_loop_d_no_crash_when_ctrl_none():
    """_on_temp_high must not raise when ctrl is None."""
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=None, mode="process")
    o._ctrl = None
    o._on_temp_high(95.0)   # must not raise


def test_loop_d_safety_capped_at_safety_max(orch):
    """safety_reserve_gb never exceeds topology-calibrated ceiling."""
    orch._ctrl._last["safety_reserve_gb"] = (100, 0)   # artificially high
    orch._on_temp_high(90.0)
    new_val = orch._ctrl._last["safety_reserve_gb"][0]
    assert new_val <= orch._safety_max_gb


# ── _t1_headroom_mb ───────────────────────────────────────────────────────────

def test_t1_headroom_mb_uses_instance_weights_floor(orch):
    """_t1_headroom_mb uses self._weights_floor_mb, not the module-level constant."""
    orch._total_vram_mb   = 12000
    orch._ws_reserve_mb   = 0
    orch._weights_floor_mb = 3000
    orch._ctrl._last["kv_reserve_mb"] = (0, 0)
    h = orch._t1_headroom_mb()
    assert h == 12000 - 3000   # 9000


def test_t1_headroom_mb_returns_zero_when_vram_unknown(orch):
    assert orch._t1_headroom_mb() == 0  # _total_vram_mb starts at 0


# ── Loop C , pressure signal path (full EMA drive) ────────────────────────────

def test_loop_c_grows_kv_on_sustained_pressure(orch):
    """With 25 polls at 93% KV pressure, Loop C fires and grows KV reserve.

    kv_pressure = 480/512 = 0.9375, EWMA alpha=1/8, confirm=2.
    EMA crosses 0.85 enter threshold after ~20 polls; 25 gives a safe margin.
    """
    orch._total_vram_mb = 12000
    orch._ws_reserve_mb = 0

    for _ in range(25):
        orch.on_metrics(_make_metrics(kv_reserve_mb=512, kv_used_mb=480))

    assert "kv_reserve_mb" in orch._ctrl._last
    new_kv = orch._ctrl._last["kv_reserve_mb"][0]
    assert new_kv > 0, "Loop C should have grown KV reserve"


def test_loop_c_grow_then_shrink_full_cycle(orch):
    """Grow under pressure, then shrink when pressure clears.

    Pre-seed kv_reserve_mb to 3072 MiB so the grow lands at 3584 MiB,
    well above the topology floor (2048), enabling the subsequent shrink.
    """
    orch._total_vram_mb = 12000
    orch._ws_reserve_mb = 0
    orch._ctrl._last["kv_reserve_mb"] = (3072, 0)

    # Phase 1: drive KV pressure high
    for _ in range(25):
        orch.on_metrics(_make_metrics(kv_reserve_mb=512, kv_used_mb=480))

    grown_kv = orch._ctrl._last.get("kv_reserve_mb", (0, 0))[0]
    assert grown_kv > 3072, "KV should have grown above pre-seeded value"

    # Phase 2: clear pressure (kv_used = 0 → pressure = 0 → exit at <0.60)
    # With alpha=1/8, EMA drops from ~0.9375 to <0.60 in ~4 polls at 0;
    # with confirm=2 the exit fires at phase2 poll 4.
    for _ in range(8):
        orch.on_metrics(_make_metrics(kv_reserve_mb=512, kv_used_mb=0))

    shrunk_kv = orch._ctrl._last.get("kv_reserve_mb", (0, 0))[0]
    assert shrunk_kv < grown_kv, "KV should shrink after pressure clears"
    assert shrunk_kv >= orch._kv_floor_mb, "KV must not shrink below topology baseline"


# ── Loop C , kv_spilled path ──────────────────────────────────────────────────

def test_kv_spilled_triggers_immediate_grow(orch):
    """When KV spills to T2, _on_kv_spilled fires an immediate grow."""
    orch._total_vram_mb = 12000
    orch._ws_reserve_mb = 0

    orch._on_kv_spilled(True, False)   # new=True means spill happened

    assert "kv_reserve_mb" in orch._ctrl._last
    new_kv = orch._ctrl._last["kv_reserve_mb"][0]
    assert new_kv > 0


def test_kv_spilled_blocked_when_ecc_degraded(orch):
    """ECC degraded blocks the kv_spilled grow path."""
    orch._total_vram_mb = 12000
    orch.ecc_degraded = True
    orch._on_kv_spilled(True, False)
    assert "kv_reserve_mb" not in orch._ctrl._last


def test_kv_spilled_false_no_action(orch):
    """new=False (spill cleared) must not trigger a KV grow."""
    orch._total_vram_mb = 12000
    orch._on_kv_spilled(False, True)
    assert "kv_reserve_mb" not in orch._ctrl._last


def test_kv_spilled_via_on_metrics(orch):
    """on_metrics with kv_t2_mb>0 triggers the kv_spilled signal path (line 241)."""
    orch._total_vram_mb = 12000
    orch._ws_reserve_mb = 0
    orch.on_metrics(_make_metrics(kv_reserve_mb=512, kv_used_mb=480, kv_t2_mb=100))
    # kv_spilled_sig.set(True) was called; _on_kv_spilled fires on confirm=1
    assert "kv_reserve_mb" in orch._ctrl._last


# ── Loop C , clamp path (0 < headroom < kv_step_mb) ──────────────────────────

def test_do_kv_grow_clamp_path_fires_when_headroom_partial(orch):
    """When 0 < headroom < kv_step_mb, grow is clamped to available headroom.

    Setup: total=12000, ws=6600, kv_last=3072, weights_floor=2048
    headroom = 12000-6600-3072-2048 = 280 MiB (< kv_step_mb=512, > 0) → clamp fires.
    """
    orch._total_vram_mb = 12000
    orch._ws_reserve_mb = 6600   # headroom = 12000-6600-3072-2048 = 280 MiB < 512
    # weights floor is now %-of-VRAM derived (18%) — pin it so the arithmetic
    # in this test stays machine-independent.
    orch._weights_floor_mb = 2048
    orch._ctrl._last["kv_reserve_mb"] = (3072, 0)

    for _ in range(25):
        orch.on_metrics(_make_metrics(kv_reserve_mb=512, kv_used_mb=480))

    kv_new = orch._ctrl._last.get("kv_reserve_mb", (0, 0))[0]
    # headroom = 280: clamped grow → new_kv = 3072 + 280 = 3352
    assert kv_new == 3352, f"Expected clamped grow to 3352 MiB; got {kv_new}"


# ── Loop D , temp_ok path ────────────────────────────────────────────────────

def test_loop_d_temp_ok_fires_after_high_temp_exits(orch):
    """After sustained high temp (Loop D enter), clearing it fires _on_temp_ok."""
    orch._total_vram_mb = 12000

    # Phase 1: drive EMA above 83°C (confirm=3, alpha=1/8 → ~25 polls)
    for _ in range(25):
        orch.on_metrics(_make_metrics(fb_used_pct=10.0, temp_c=95.0))

    assert orch._temp_sig.in_hysteresis is True

    # Phase 2: drop EMA below 75°C exit threshold (alpha=1/8, from ~90°C → ~10 polls)
    for _ in range(15):
        orch.on_metrics(_make_metrics(fb_used_pct=10.0, temp_c=60.0))

    assert orch._temp_sig.in_hysteresis is False


# ── B4 Loop F , health restored + no_tier_manager paths ──────────────────────

def test_b4_health_restored_logs_no_evict(monkeypatch):
    """After health degradation, restoring health_ok fires _on_health_change(True)."""
    monkeypatch.setenv("GB_ORCH_HEALTH_EVICT", "1")
    ctrl = _MockGbControl()
    tm   = MagicMock()
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process", tier_manager=tm)

    # Degrade health
    o.on_metrics(_make_metrics(health_ok=False))
    o.on_metrics(_make_metrics(health_ok=False))
    tm.auto_evict.assert_called_once()

    # Restore health , confirm=2 polls with True
    o.on_metrics(_make_metrics(health_ok=True))
    o.on_metrics(_make_metrics(health_ok=True))
    # No second evict on restoration
    tm.auto_evict.assert_called_once()


def test_b4_health_armed_no_tier_manager_records_no_tier_manager(monkeypatch):
    """When armed but no tier_manager, decision records 'no_tier_manager'."""
    monkeypatch.setenv("GB_ORCH_HEALTH_EVICT", "1")
    ctrl = _MockGbControl()
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process")  # no tier_manager

    o.on_metrics(_make_metrics(health_ok=False))
    o.on_metrics(_make_metrics(health_ok=False))

    decisions = [d for d in o._decisions if d.get("loop") == "F_health_degraded"]
    assert len(decisions) == 1
    assert decisions[0]["action"] == "no_tier_manager"


# ── Misc paths ────────────────────────────────────────────────────────────────

def test_stop_is_callable(orch):
    """stop() exists and returns None."""
    assert orch.stop() is None


def test_decisions_ring_buffer_trims_at_64(orch):
    """After 65 recorded decisions, _decisions stays at 64 (ring buffer)."""
    for i in range(65):
        orch._record_decision({"loop": "test", "ts": i})
    assert len(orch._decisions) == 64


# ── Edge case paths ───────────────────────────────────────────────────────────

def test_do_kv_grow_returns_early_when_ctrl_none(orch):
    """_do_kv_grow bails immediately when _ctrl is None (line 517)."""
    orch._ctrl = None
    orch._total_vram_mb = 12000
    orch._do_kv_grow("test", {})
    # No exception, nothing set


def test_b3_cluster_auto_evict_exception_is_recorded():
    """B3 cluster: auto_evict raises → decision records 'tier_auto_evict_failed' (271-272)."""
    ctrl = _MockGbControl()
    tm   = MagicMock()
    tm.auto_evict.side_effect = RuntimeError("feeder gone")
    cluster_tel = MagicMock()
    cluster_tel.any_should_demote.return_value = True

    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process",
                                  tier_manager=tm, cluster_tel=cluster_tel)

    o.on_metrics(_make_metrics(fb_used_pct=50.0))
    decisions = [d for d in o._decisions if d.get("loop") == "E_cluster_pressure"]
    assert any("tier_auto_evict_failed" in d.get("action", "") for d in decisions)


def test_b3_no_tier_manager_records_no_tier_manager():
    """B3 cluster with no tier_manager → decision records 'no_tier_manager' (line 274)."""
    ctrl = _MockGbControl()
    cluster_tel = MagicMock()
    cluster_tel.any_should_demote.return_value = True

    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process", cluster_tel=cluster_tel)

    o.on_metrics(_make_metrics(fb_used_pct=50.0))
    decisions = [d for d in o._decisions if d.get("loop") == "E_cluster_pressure"]
    assert len(decisions) == 1
    assert decisions[0]["action"] == "no_tier_manager"


def test_feed_vram_state_sets_total_vram_when_zero():
    """feed_vram_state initialises _total_vram_mb and thresholds if 0 (lines 290-292)."""
    ctrl = _MockGbControl()
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process")

    assert o._total_vram_mb == 0
    o.feed_vram_state(1024.0, 12000)
    assert o._total_vram_mb == 12000


def test_ecc_change_early_return_when_no_new_errors(orch):
    """_on_ecc_change returns early when new_errors <= 0 (line 301)."""
    orch._ecc_seen = 5
    orch.ecc_degraded = False   # must not flip to True
    orch._on_ecc_change(5, 3)   # new=5, _ecc_seen=5 → new_errors=0
    assert orch.ecc_degraded is False


def test_loop_e_auto_evict_exception_is_logged(orch_with_tier):
    """Loop E records 'tier_auto_evict_failed' when auto_evict raises (lines 465-467)."""
    o, tm = orch_with_tier
    tm.auto_evict.side_effect = RuntimeError("disk full")

    for _ in range(30):
        o.on_metrics(_make_metrics(fb_used_pct=90.0))

    decisions = [d for d in o._decisions if d.get("loop") == "E_vram_pressure"]
    assert any("tier_auto_evict_failed" in d.get("action", "") for d in decisions)


def test_loop_f_auto_evict_exception_is_logged(monkeypatch):
    """Loop F records 'tier_auto_evict_failed' when armed and auto_evict raises (501-503)."""
    monkeypatch.setenv("GB_ORCH_HEALTH_EVICT", "1")
    ctrl = _MockGbControl()
    tm   = MagicMock()
    tm.auto_evict.side_effect = RuntimeError("GPU gone")

    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process", tier_manager=tm)

    o.on_metrics(_make_metrics(health_ok=False))
    o.on_metrics(_make_metrics(health_ok=False))

    decisions = [d for d in o._decisions if d.get("loop") == "F_health_degraded"]
    assert any("tier_auto_evict_failed" in d.get("action", "") for d in decisions)


# ── Loop D → C thermal gate (new mechanic) ───────────────────────────────────

def _make_orch_no_io(**kw):
    ctrl = kw.pop("ctrl", _MockGbControl())
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        return ReactiveOrchestrator(control=ctrl, mode="process", **kw)


def test_thermal_stress_starts_false():
    o = _make_orch_no_io()
    assert o.thermal_stress is False


def test_thermal_stress_set_on_temp_high():
    """_on_temp_high must set thermal_stress=True."""
    o = _make_orch_no_io()
    assert o.thermal_stress is False
    o._on_temp_high(85.0)
    assert o.thermal_stress is True


def test_thermal_stress_cleared_on_temp_ok():
    """_on_temp_ok must clear thermal_stress back to False."""
    o = _make_orch_no_io()
    o._on_temp_high(85.0)
    assert o.thermal_stress is True
    o._on_temp_ok(72.0)
    assert o.thermal_stress is False


def test_loop_c_skipped_when_thermal_stress():
    """With thermal_stress=True, sustained KV pressure must NOT trigger a KV grow."""
    o = _make_orch_no_io()
    o._total_vram_mb = 12000
    o.thermal_stress = True
    o._ctrl._last["kv_reserve_mb"] = (3072, 0)

    # Feed sustained high KV pressure (enough to normally trigger grow)
    for _ in range(25):
        o.on_metrics(_make_metrics(kv_reserve_mb=512, kv_used_mb=480))

    # kv_reserve_mb should be unchanged (grow suppressed)
    kv = o._ctrl._last.get("kv_reserve_mb", (3072, 0))[0]
    assert kv == 3072, f"KV grew to {kv} despite thermal_stress=True"


def test_loop_c_resumes_after_thermal_stress_clears():
    """After thermal_stress is cleared, Loop C fires a KV grow on the next
    high-pressure entry.

    Flow:
      1. Suppress phase: kv stays at 3072 (grow blocked by thermal_stress).
      2. Low-pressure phase: hysteresis exits; _on_kv_pressure_ok shrinks kv to 2560.
      3. Resume phase: thermal_stress=False, high pressure re-entered → grow fires → 3072.
    The key assertion is that the grow DID fire (kv_after_resume > kv_after_shrink).
    """
    o = _make_orch_no_io()
    o._total_vram_mb = 12000
    o.thermal_stress = True
    o._ctrl._last["kv_reserve_mb"] = (3072, 0)

    # 1. Suppressed phase
    for _ in range(25):
        o.on_metrics(_make_metrics(kv_reserve_mb=512, kv_used_mb=480))
    assert o._ctrl._last.get("kv_reserve_mb", (3072, 0))[0] == 3072

    # 2. Low-pressure exit: hysteresis exits, kv shrinks by kv_step_mb
    o.thermal_stress = False
    for _ in range(15):
        o.on_metrics(_make_metrics(kv_reserve_mb=512, kv_used_mb=0))
    kv_after_shrink = o._ctrl._last.get("kv_reserve_mb", (3072, 0))[0]

    # 3. Re-enter high pressure , Loop C should grow from kv_after_shrink
    for _ in range(25):
        o.on_metrics(_make_metrics(kv_reserve_mb=512, kv_used_mb=480))
    kv_after_resume = o._ctrl._last.get("kv_reserve_mb", (kv_after_shrink, 0))[0]
    assert kv_after_resume > kv_after_shrink, \
        f"Loop C did not grow KV after thermal_stress cleared: " \
        f"{kv_after_shrink} → {kv_after_resume}"


def test_thermal_stress_in_dump():
    """dump() must expose thermal_stress flag."""
    o = _make_orch_no_io()
    o.thermal_stress = True
    d = o.dump()
    assert "thermal_stress" in d
    assert d["thermal_stress"] is True


def test_loop_d_temp_high_sets_thermal_stress_via_on_metrics():
    """Sustained high temp via on_metrics triggers _on_temp_high → thermal_stress=True.

    α=1/8, _TEMP_ENTER_C=83.0: EMA(n)=87*(1-(7/8)^n) > 83 needs ~22 polls.
    Plus confirm=3 → 30 polls is a safe margin.
    """
    o = _make_orch_no_io()
    for _ in range(30):
        o.on_metrics(_make_metrics(temp_c=87.0))
    assert o.thermal_stress is True


def test_loop_d_temp_ok_clears_thermal_stress_via_on_metrics():
    """After temp drops below _TEMP_EXIT_C=75.0, thermal_stress clears."""
    o = _make_orch_no_io()
    # First set thermal stress (30 polls to settle + confirm)
    for _ in range(30):
        o.on_metrics(_make_metrics(temp_c=87.0))
    assert o.thermal_stress is True

    # Cool down: EMA from ~87°C → below 75°C needs many polls at 60°C;
    # α=1/8 so EMA(n)=87*(7/8)^n + 60*(1-(7/8)^n). Need (87-60)*(7/8)^n < 75-60=15
    # → (7/8)^n < 15/27 → n > 12. Feed 30 polls to be safe.
    for _ in range(30):
        o.on_metrics(_make_metrics(temp_c=60.0))
    assert o.thermal_stress is False


# ── topology fallback + ctrl=None init paths ──────────────────────────────────

def test_topology_fallback_when_import_fails():
    """When gb_topology.get_topology raises, ReactiveOrchestrator uses constant fallbacks."""
    import gb_topology
    from gb_orchestrator import _KV_STEP_MB, _SAFETY_MAX_GB, _KV_FLOOR_MB
    with patch.object(gb_topology, "get_topology", side_effect=RuntimeError("no topo")), \
         patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=_MockGbControl(), mode="process")
    assert o._kv_step_mb == _KV_STEP_MB
    assert o._safety_max_gb == _SAFETY_MAX_GB
    assert o._kv_floor_mb == _KV_FLOOR_MB


def test_ctrl_none_when_gb_control_import_fails():
    """When GbControl cannot be imported, _ctrl is None (not an error)."""
    with patch.dict("sys.modules", {"gb_control": None}), \
         patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=None, mode="process")
    assert o._ctrl is None


# ── persist / restore ECC degraded ───────────────────────────────────────────

def test_persist_ecc_degraded_file_fallback():
    """_persist_ecc_degraded falls back to direct Path write when ECC_DBE_FLAG.write_text raises."""
    import gb_supervisor
    o = _make_orch_no_io()
    o._ecc_seen = 3

    mock_flag = MagicMock()
    mock_flag.write_text.side_effect = PermissionError("daemon-only file")
    with patch.object(gb_supervisor, "ECC_DBE_FLAG", mock_flag), \
         patch("gb_orchestrator.Path") as MockPath:
        mock_p = MagicMock()
        MockPath.return_value = mock_p
        o._persist_ecc_degraded()
    MockPath.assert_called_once_with("/run/greenboost/ecc_dbe_flag")
    mock_p.write_text.assert_called_once_with("3")


def test_persist_ecc_degraded_swallows_double_exception():
    """_persist_ecc_degraded swallows exceptions from both flag write and fallback Path write."""
    import gb_supervisor
    o = _make_orch_no_io()
    o._ecc_seen = 5

    mock_flag = MagicMock()
    mock_flag.write_text.side_effect = PermissionError("daemon-only file")
    with patch.object(gb_supervisor, "ECC_DBE_FLAG", mock_flag), \
         patch("gb_orchestrator.Path") as MockPath:
        mock_p = MagicMock()
        mock_p.write_text.side_effect = PermissionError("read-only fs")
        MockPath.return_value = mock_p
        o._persist_ecc_degraded()  # must not raise despite double failure


def test_restore_ecc_degraded_reads_flag_file(tmp_path):
    """_restore_ecc_degraded restores ecc_degraded=True from a flag file."""
    flag_path = tmp_path / "ecc_dbe_flag"
    flag_path.write_text("5")
    o = _make_orch_no_io()

    with patch("gb_orchestrator.Path") as MockPath:
        mock_p = MagicMock()
        mock_p.read_text.return_value = "5"
        MockPath.return_value = mock_p
        o._restore_ecc_degraded()

    assert o.ecc_degraded is True
    assert o._ecc_seen == 5


def test_restore_ecc_degraded_noop_when_zero(tmp_path):
    """_restore_ecc_degraded leaves ecc_degraded=False when stored value is 0."""
    o = _make_orch_no_io()
    with patch("gb_orchestrator.Path") as MockPath:
        mock_p = MagicMock()
        mock_p.read_text.return_value = "0"
        MockPath.return_value = mock_p
        o._restore_ecc_degraded()
    assert o.ecc_degraded is False


def test_restore_ecc_degraded_noop_on_exception():
    """_restore_ecc_degraded swallows any exception (file missing, parse error, etc.)."""
    o = _make_orch_no_io()
    with patch("gb_orchestrator.Path", side_effect=Exception("broken")):
        o._restore_ecc_degraded()   # must not raise
    assert o.ecc_degraded is False


def test_write_state_file_exception_is_swallowed():
    """_write_state_file must not propagate filesystem errors."""
    o = _make_orch_no_io()
    with patch("gb_orchestrator._ORCH_STATE_FILE") as mock_p:
        mock_p.with_suffix.return_value.write_text.side_effect = PermissionError("read-only")
        o._write_state_file()   # must not raise


# ── G1 , memory-bandwidth stress gate ────────────────────────────────────────

def test_mem_bw_stress_starts_false():
    o = _make_orch_no_io()
    assert o.mem_bw_stress is False


def test_mem_bw_stress_set_on_high_util():
    """_on_mem_bw_high sets mem_bw_stress=True."""
    o = _make_orch_no_io()
    o._on_mem_bw_high(90.0)
    assert o.mem_bw_stress is True


def test_mem_bw_stress_cleared_on_ok():
    """_on_mem_bw_ok clears mem_bw_stress=False."""
    o = _make_orch_no_io()
    o._on_mem_bw_high(90.0)
    o._on_mem_bw_ok(60.0)
    assert o.mem_bw_stress is False


def test_loop_c_skipped_when_mem_bw_stress():
    """When mem_bw_stress=True, _on_kv_pressure_high skips KV grow."""
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"] = (3072, 0)
    o = _make_orch_no_io(ctrl=ctrl)
    o._total_vram_mb = 12000
    o.mem_bw_stress = True
    o._ctrl = ctrl

    o._on_kv_pressure_high(0.90)
    # KV reserve must NOT have changed
    assert ctrl._last.get("kv_reserve_mb", (3072, 0))[0] == 3072


def test_loop_c_resumes_after_mem_bw_stress_clears():
    """After mem_bw_stress clears, Loop C KV grow fires again on next pressure signal."""
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"] = (3072, 0)
    o = _make_orch_no_io(ctrl=ctrl)
    o._total_vram_mb = 12000
    o.mem_bw_stress = True
    o._ctrl = ctrl

    o._on_kv_pressure_high(0.90)                 # suppressed
    assert ctrl._last["kv_reserve_mb"][0] == 3072

    o.mem_bw_stress = False
    o._on_kv_pressure_high(0.90)                 # should fire now
    assert ctrl._last["kv_reserve_mb"][0] > 3072


def test_mem_bw_stress_in_dump():
    """dump() exposes mem_bw_stress key."""
    o = _make_orch_no_io()
    d = o.dump()
    assert "mem_bw_stress" in d
    assert d["mem_bw_stress"] is False
    o.mem_bw_stress = True
    assert o.dump()["mem_bw_stress"] is True


def test_mem_bw_sig_in_dump_signals():
    """dump()['signals'] includes the mem_copy_util_pct signal."""
    o = _make_orch_no_io()
    names = [s["name"] for s in o.dump()["signals"]]
    assert "mem_copy_util_pct" in names


def test_loop_g_sets_mem_bw_stress_via_on_metrics():
    """Sustained high mem_copy_util_pct through on_metrics triggers mem_bw_stress."""
    o = _make_orch_no_io()
    # _MEM_BW_CONFIRM=2 confirm polls + EWMA settling to cross 85%.
    # Feed 95% for 20 polls: α=1/8 EWMA crosses 85% well within 20 polls.
    for _ in range(20):
        o.on_metrics(_make_metrics(mem_copy_util_pct=95.0))
    assert o.mem_bw_stress is True


def test_loop_g_clears_mem_bw_stress_via_on_metrics():
    """After cooling, sustained low mem_copy_util_pct clears mem_bw_stress."""
    o = _make_orch_no_io()
    for _ in range(20):
        o.on_metrics(_make_metrics(mem_copy_util_pct=95.0))
    assert o.mem_bw_stress is True

    for _ in range(20):
        o.on_metrics(_make_metrics(mem_copy_util_pct=20.0))
    assert o.mem_bw_stress is False


# ── G2 , power_near_limit gate on Loop C ─────────────────────────────────────

def test_power_near_limit_false_when_limit_unknown():
    """power_near_limit is False when power_limit_w is 0 (field not sampled)."""
    from gb_telemetry import GpuMetrics
    m = GpuMetrics()
    m.power_w = 300.0
    m.power_limit_w = 0.0
    assert m.power_near_limit is False


def test_power_near_limit_false_when_below_threshold():
    from gb_telemetry import GpuMetrics
    m = GpuMetrics()
    m.power_w = 200.0
    m.power_limit_w = 300.0   # 66% , well below 93%
    assert m.power_near_limit is False


def test_power_near_limit_true_when_above_threshold():
    from gb_telemetry import GpuMetrics
    m = GpuMetrics()
    m.power_w = 282.0
    m.power_limit_w = 300.0   # 94% , above 93%
    assert m.power_near_limit is True


def test_loop_c_skipped_when_power_near_limit():
    """Loop C KV grow is suppressed when last_metrics.power_near_limit is True."""
    from gb_telemetry import GpuMetrics
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"] = (3072, 0)
    o = _make_orch_no_io(ctrl=ctrl)
    o._total_vram_mb = 12000
    o._ctrl = ctrl

    # Inject a last_metrics snapshot with power near limit
    lm = _make_metrics(power_w=285.0, power_limit_w=300.0)
    o._last_metrics = lm

    o._on_kv_pressure_high(0.90)
    # KV reserve must NOT have been grown
    assert ctrl._last.get("kv_reserve_mb", (3072, 0))[0] == 3072


def test_loop_c_fires_when_power_ok():
    """Loop C KV grow fires normally when power is well below limit."""
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"] = (3072, 0)
    o = _make_orch_no_io(ctrl=ctrl)
    o._total_vram_mb = 12000
    o._ctrl = ctrl

    lm = _make_metrics(power_w=150.0, power_limit_w=300.0)  # 50%, well below 93%
    o._last_metrics = lm

    o._on_kv_pressure_high(0.90)
    assert ctrl._last["kv_reserve_mb"][0] > 3072


# ── H1 , ECC SBE early-warning ────────────────────────────────────────────────

def test_sbe_elevated_starts_false():
    o = _make_orch_no_io()
    assert o.sbe_elevated is False
    assert o._sbe_seen == 0


def test_sbe_elevated_set_on_sbe_change():
    """_on_sbe_change sets sbe_elevated=True when SBE count rises."""
    o = _make_orch_no_io()
    o._on_sbe_change(3, 0)
    assert o.sbe_elevated is True
    assert o._sbe_seen == 3


def test_sbe_change_noop_when_count_does_not_rise():
    """_on_sbe_change is a no-op when new <= _sbe_seen (no new errors)."""
    o = _make_orch_no_io()
    o._sbe_seen = 5
    o._on_sbe_change(5, 5)   # same value , no new errors
    assert o.sbe_elevated is False


def test_sbe_elevated_in_dump():
    """dump() exposes sbe_elevated and sbe_seen."""
    o = _make_orch_no_io()
    d = o.dump()
    assert "sbe_elevated" in d
    assert "sbe_seen" in d
    assert d["sbe_elevated"] is False
    assert d["sbe_seen"] == 0

    o.sbe_elevated = True
    o._sbe_seen = 7
    d = o.dump()
    assert d["sbe_elevated"] is True
    assert d["sbe_seen"] == 7


def test_sbe_sig_in_dump_signals():
    """dump()['signals'] includes the ecc_sbe signal (Loop H)."""
    o = _make_orch_no_io()
    names = [s["name"] for s in o.dump()["signals"]]
    assert "ecc_sbe" in names


def test_loop_h_fires_via_on_metrics():
    """Sustained rising ecc_sbe_volatile through on_metrics triggers sbe_elevated."""
    from gb_telemetry import GpuMetrics
    o = _make_orch_no_io()

    def _sbe_metrics(n):
        m = _make_metrics()
        m.ecc_sbe_volatile = n
        return m

    # Signal confirm=2: need 2 consecutive polls with the same rising value
    o.on_metrics(_sbe_metrics(2))   # first poll , sbe > _sbe_seen=0, sets signal
    o.on_metrics(_sbe_metrics(2))   # second poll , confirm fires
    assert o.sbe_elevated is True
    assert o._sbe_seen == 2


def test_loop_h_no_fire_without_confirm():
    """A single SBE poll (confirm=2) does not immediately set sbe_elevated."""
    from gb_telemetry import GpuMetrics
    o = _make_orch_no_io()

    m = _make_metrics()
    m.ecc_sbe_volatile = 1
    o.on_metrics(m)   # first poll only , confirm not reached
    assert o.sbe_elevated is False


def test_loop_h_does_not_actuate_levers():
    """SBE warning is advisory only , kv_reserve and safety_reserve are unchanged."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    o._on_sbe_change(5, 0)
    # No lever moves
    assert "kv_reserve_mb" not in ctrl._last
    assert "safety_reserve_gb" not in ctrl._last


# ── Loop I , SM clock throttle ────────────────────────────────────────────────

def _feed_clock(o, sm_clock_mhz, count=1):
    """Feed N polls of sm_clock_mhz through on_metrics."""
    from gb_telemetry import GpuMetrics
    for _ in range(count):
        m = _make_metrics()
        m.sm_clock_mhz = sm_clock_mhz
        o.on_metrics(m)


def test_loop_i_starts_false():
    """clock_throttled starts False."""
    o = _make_orch_no_io()
    assert o.clock_throttled is False


def test_loop_i_no_fire_before_warmup():
    """Loop I needs 3 warmup polls before tracking starts."""
    o = _make_orch_no_io()
    # Prime max with 2 polls at 2587 MHz (below warmup=3)
    _feed_clock(o, 2587, 2)
    # Now feed low clock , should not fire yet (warmup incomplete)
    _feed_clock(o, 2100, 2)
    assert o.clock_throttled is False


def test_loop_i_sets_clock_throttled_after_confirm():
    """After warmup + enough low-clock polls, clock_throttled=True.

    EWMA α=1/6 is deliberately slow. With drop_pct ≈ 18.8% (2100 of 2587 MHz),
    the EMA crosses the 12% entry threshold after ~6 post-warmup polls; 10 polls
    gives headroom against confirm=2 rounding.
    """
    o = _make_orch_no_io()
    _feed_clock(o, 2587, 3)   # warmup: poll 3 feeds drop=0.0 to signal
    _feed_clock(o, 2100, 10)  # EMA converges to ~15% after 10 polls at 18.8%
    assert o.clock_throttled is True


def test_loop_i_gates_kv_grow_when_throttled():
    """When clock_throttled, Loop C skips KV grow."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    o.clock_throttled = True   # directly set flag
    o._on_kv_pressure_high(0.90)
    assert "kv_reserve_mb" not in ctrl._last


def test_loop_i_allows_kv_grow_when_not_throttled():
    """When clock_throttled=False and no other gates, Loop C fires KV grow."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    o.clock_throttled = False
    # Seed kv_reserve so headroom calculation doesn't block
    ctrl._last["kv_reserve_mb"] = (512, 0.0)
    ctrl._last["safety_reserve_gb"] = (4, 0.0)
    o._total_vram_mb = 12288
    o._on_kv_pressure_high(0.90)
    assert "kv_reserve_mb" in ctrl._last


def test_loop_i_clock_ok_clears_flag():
    """_on_clock_ok clears clock_throttled."""
    o = _make_orch_no_io()
    o.clock_throttled = True
    o._on_clock_ok(3.0)
    assert o.clock_throttled is False


def test_loop_i_mild_zone_halves_kv_step():
    """When SM clock drop EMA is in [8, 12)%, _do_kv_grow uses half step.

    EWMA α=1/6 with input=9.5%: converges to ~9.4% after 30 polls.
    9.4% is in [_SM_CLOCK_DROP_MILD=8, _SM_CLOCK_DROP_ENTER=12) → half step.
    No clock_throttled risk because 9.5 never crosses the 12% enter threshold.
    """
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"]    = (512, 0.0)
    ctrl._last["safety_reserve_gb"] = (4, 0.0)
    o = _make_orch_no_io(ctrl=ctrl)
    o._total_vram_mb = 12288
    # Drive EMA to ~9.4% (safely above 8%, well below 12% hard block)
    for _ in range(30):
        o._clock_drop_sig.set(9.5)
    assert not o.clock_throttled, "mild zone test must not trigger hard block"
    decision = {"loop": "test", "ts": 0}
    o._do_kv_grow("test_mild", decision)
    applied_kv = ctrl._last["kv_reserve_mb"][0]
    # Full step: old_kv(512) + step(512) = 1024; half step: 512 + 256 = 768
    assert applied_kv < 512 + o._kv_step_mb  # strictly less than full step
    assert applied_kv >= 512 + 128            # at least the floor half-step


def test_loop_i_pid_off_by_default_is_none():
    """GB_SM_PID unset (the common case) — _sm_pid must be None so
    _do_kv_grow's discrete halve/full path (pre-DI-6 behavior) is used,
    never the continuous PID path."""
    o = _make_orch_no_io()
    assert o._sm_pid is None


def test_loop_i_pid_scales_step_continuously_when_enabled(monkeypatch):
    monkeypatch.setenv("GB_SM_PID", "1")
    monkeypatch.setenv("GB_PID_KP", "0.05")
    monkeypatch.setenv("GB_PID_KI", "0.0")
    monkeypatch.setenv("GB_PID_KD", "0.0")
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"] = (512, 0.0)
    ctrl._last["safety_reserve_gb"] = (4, 0.0)
    o = _make_orch_no_io(ctrl=ctrl)
    assert o._sm_pid is not None
    o._total_vram_mb = 12288
    for _ in range(30):
        o._clock_drop_sig.set(9.5)   # same mild-zone drop as the discrete test above
    decision = {"loop": "test", "ts": 0}
    o._do_kv_grow("test_pid", decision)
    applied_kv = ctrl._last["kv_reserve_mb"][0]
    # PID output is continuous, not the fixed 50% halve — just confirm it
    # reacted (some step taken, but not necessarily the full step) rather
    # than asserting an exact scale (depends on integral history/dt).
    assert 512 <= applied_kv <= 512 + o._kv_step_mb


def test_loop_i_in_dump():
    """dump() includes clock_throttled and sm_clock_max_mhz."""
    o = _make_orch_no_io()
    o.clock_throttled = True
    o._sm_clock_max = 2587
    d = o.dump()
    assert d["clock_throttled"] is True
    assert d["sm_clock_max_mhz"] == 2587


def test_loop_i_sig_in_signals():
    """dump()['signals'] includes the sm_clock_drop_pct signal."""
    o = _make_orch_no_io()
    names = [s["name"] for s in o.dump()["signals"]]
    assert "sm_clock_drop_pct" in names


def test_loop_i_state_hint_throttled():
    """dump() sets state='throttled' for sm_clock_drop_pct when clock_throttled=True."""
    o = _make_orch_no_io()
    o.clock_throttled = True
    d = o.dump()
    clk_sig = next(s for s in d["signals"] if s["name"] == "sm_clock_drop_pct")
    assert clk_sig["state"] == "throttled"


def test_loop_i_state_hint_ok():
    """dump() sets state='ok' for sm_clock_drop_pct when not throttled."""
    o = _make_orch_no_io()
    o.clock_throttled = False
    d = o.dump()
    clk_sig = next(s for s in d["signals"] if s["name"] == "sm_clock_drop_pct")
    assert clk_sig["state"] == "ok"


def test_loop_i_does_not_fire_when_sm_clock_zero():
    """When sm_clock_mhz=0 (NVML unavailable), Loop I does nothing."""
    o = _make_orch_no_io()
    m = _make_metrics()
    m.sm_clock_mhz = 0
    for _ in range(10):
        o.on_metrics(m)
    assert o.clock_throttled is False
    assert o._sm_clock_max == 0


# ── Phase gate , shim_phase blocks KV grows during idle/loading ──────────────

def test_phase_gate_blocks_kv_grow_during_model_load():
    """Loop C skips KV grow when shim_phase=MODEL_LOAD."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    m = _make_metrics()
    m.shim_phase = "MODEL_LOAD"
    o._last_metrics = m
    o._on_kv_pressure_high(0.90)
    assert "kv_reserve_mb" not in ctrl._last


def test_phase_gate_blocks_kv_grow_during_deep_idle():
    """Loop C skips KV grow when shim_phase=DEEP_IDLE."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    m = _make_metrics()
    m.shim_phase = "DEEP_IDLE"
    o._last_metrics = m
    o._on_kv_pressure_high(0.90)
    assert "kv_reserve_mb" not in ctrl._last


def test_phase_gate_blocks_kv_grow_during_idle():
    """Loop C skips KV grow when shim_phase=IDLE."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    m = _make_metrics()
    m.shim_phase = "IDLE"
    o._last_metrics = m
    o._on_kv_pressure_high(0.90)
    assert "kv_reserve_mb" not in ctrl._last


def test_phase_gate_allows_kv_grow_during_inference():
    """Loop C fires KV grow when shim_phase=INFERENCE."""
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"] = (512, 0.0)
    ctrl._last["safety_reserve_gb"] = (4, 0.0)
    o = _make_orch_no_io(ctrl=ctrl)
    o._total_vram_mb = 12288
    m = _make_metrics()
    m.shim_phase = "INFERENCE"
    o._last_metrics = m
    o._on_kv_pressure_high(0.90)
    assert "kv_reserve_mb" in ctrl._last


def test_phase_gate_allows_kv_grow_when_phase_unknown():
    """Loop C fires when shim_phase='' (shim absent , conservative allow)."""
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"] = (512, 0.0)
    ctrl._last["safety_reserve_gb"] = (4, 0.0)
    o = _make_orch_no_io(ctrl=ctrl)
    o._total_vram_mb = 12288
    m = _make_metrics()
    m.shim_phase = ""
    o._last_metrics = m
    o._on_kv_pressure_high(0.90)
    assert "kv_reserve_mb" in ctrl._last


def test_phase_gate_kv_spill_idle_skip():
    """_on_kv_spilled skips the grow when shim_phase=IDLE."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    m = _make_metrics()
    m.shim_phase = "IDLE"
    o._last_metrics = m
    o._on_kv_spilled(True, False)
    assert "kv_reserve_mb" not in ctrl._last


def test_phase_in_dump():
    """dump() exposes shim_phase from last_metrics."""
    o = _make_orch_no_io()
    m = _make_metrics()
    m.shim_phase = "INFERENCE"
    o._last_metrics = m
    assert o.dump()["shim_phase"] == "INFERENCE"


# ── Loop J , INFERENCE→IDLE phase-transition KV reclaim ─────────────────────

def test_loop_j_reclaims_kv_on_inference_to_idle():
    """INFERENCE→IDLE shrinks KV reserve to floor immediately."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    o._kv_floor_mb = 0  # pin floor to 0 regardless of topology profile
    ctrl._last["kv_reserve_mb"] = (4096, 0.0)  # well above floor
    # Two metrics: first INFERENCE, then IDLE
    m_inf = _make_metrics()
    m_inf.shim_phase = "INFERENCE"
    o.on_metrics(m_inf)

    m_idle = _make_metrics()
    m_idle.shim_phase = "IDLE"
    o.on_metrics(m_idle)

    kv_after = ctrl._last.get("kv_reserve_mb", (0, 0))[0]
    assert kv_after == 0, f"Loop J must reclaim KV to floor 0, got {kv_after}"


def test_loop_j_reclaims_on_inference_to_deep_idle():
    """INFERENCE→DEEP_IDLE also triggers Loop J reclaim."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    o._kv_floor_mb = 0
    ctrl._last["kv_reserve_mb"] = (4096, 0.0)
    m_inf = _make_metrics()
    m_inf.shim_phase = "INFERENCE"
    o.on_metrics(m_inf)

    m_deep = _make_metrics()
    m_deep.shim_phase = "DEEP_IDLE"
    o.on_metrics(m_deep)

    assert ctrl._last.get("kv_reserve_mb", (0, 0))[0] == 0


def test_loop_j_no_reclaim_when_already_at_floor():
    """No set_kv_reserve_mb call when KV is already at floor."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    o._kv_floor_mb = 0
    ctrl._last["kv_reserve_mb"] = (0, 0.0)  # already at floor
    m_inf = _make_metrics()
    m_inf.shim_phase = "INFERENCE"
    o.on_metrics(m_inf)

    m_idle = _make_metrics()
    m_idle.shim_phase = "IDLE"
    o.on_metrics(m_idle)

    # kv_reserve_mb must not have changed (was already at floor)
    assert ctrl._last.get("kv_reserve_mb", (0, 0))[0] == 0


def test_loop_j_no_reclaim_on_idle_to_idle():
    """Repeated IDLE→IDLE polls must not trigger Loop J."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    o._kv_floor_mb = 0
    ctrl._last["kv_reserve_mb"] = (4096, 0.0)
    m_idle = _make_metrics()
    m_idle.shim_phase = "IDLE"
    for _ in range(3):
        o.on_metrics(m_idle)

    # kv_reserve_mb should not have been touched (no INFERENCE→IDLE transition)
    assert ctrl._last.get("kv_reserve_mb", (0, 0))[0] == 4096


def test_loop_j_no_reclaim_when_ecc_degraded():
    """ECC degraded blocks Loop J reclaim."""
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    o._kv_floor_mb = 0
    o.ecc_degraded = True
    ctrl._last["kv_reserve_mb"] = (4096, 0.0)

    m_inf = _make_metrics()
    m_inf.shim_phase = "INFERENCE"
    o.on_metrics(m_inf)

    m_idle = _make_metrics()
    m_idle.shim_phase = "IDLE"
    o.on_metrics(m_idle)

    # ECC degraded must block the reclaim
    assert ctrl._last.get("kv_reserve_mb", (0, 0))[0] == 4096


def test_loop_j_last_phase_in_dump():
    """dump() exposes last_phase for Loop J observability."""
    o = _make_orch_no_io()
    m = _make_metrics()
    m.shim_phase = "INFERENCE"
    o.on_metrics(m)
    assert o.dump()["last_phase"] == "INFERENCE"


# ── Loop K , post-throttle KV restore ────────────────────────────────────────

def _feed_clock_and_pressure(o, mhz, pressure_ratio, count):
    """Feed clock + KV pressure polls to drive both Loop I and Loop C signals."""
    for _ in range(count):
        m = _make_metrics(kv_reserve_mb=512, kv_used_mb=int(512 * pressure_ratio))
        m.sm_clock_mhz = mhz
        m.shim_phase = "INFERENCE"
        o.on_metrics(m)


def test_loop_k_deferred_grow_after_clock_recovery():
    """After throttle clears with KV still in hysteresis, Loop K fires one KV grow."""
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"] = (512, 0.0)
    ctrl._last["safety_reserve_gb"] = (4, 0.0)
    o = _make_orch_no_io(ctrl=ctrl)
    o._total_vram_mb = 12288

    # Phase 1: warmup at full clock to establish sm_clock_max
    _feed_clock_and_pressure(o, 2850, 0.50, 5)

    # Phase 2: throttled + high KV pressure (but Clock I blocks the grow)
    _feed_clock_and_pressure(o, 2300, 0.95, 15)  # ~19% drop → clock_throttled=True

    assert o.clock_throttled is True

    # Record KV after throttle phase (grow was blocked)
    kv_throttled = ctrl._last.get("kv_reserve_mb", (0, 0))[0]

    # Phase 3: clock recovers but KV pressure still high → Loop K should fire
    # α=1/6 EWMA: from ~19% drop needs ~8-9 polls to converge below exit=6%.
    # confirm=2 means 2 additional polls after crossing. Use 14 for margin.
    m_ok = _make_metrics(kv_reserve_mb=512, kv_used_mb=int(512 * 0.95))
    m_ok.sm_clock_mhz = 2820  # back to near-max → drop_pct ≈ 1% → below exit=6%
    m_ok.shim_phase = "INFERENCE"
    for _ in range(14):
        o.on_metrics(m_ok)

    assert o.clock_throttled is False
    kv_after = ctrl._last.get("kv_reserve_mb", (0, 0))[0]
    assert kv_after > kv_throttled, "Loop K must grow KV when clock recovers with pressure still high"


def test_loop_k_no_grow_when_kv_pressure_not_in_hysteresis():
    """Loop K is a no-op when kv_pressure has already exited the high band."""
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"] = (512, 0.0)
    o = _make_orch_no_io(ctrl=ctrl)
    o._total_vram_mb = 12288

    # Warmup max clock
    _feed_clock_and_pressure(o, 2850, 0.20, 5)  # low pressure
    # Throttle
    _feed_clock_and_pressure(o, 2300, 0.20, 15)  # still low pressure
    assert o.clock_throttled is True

    kv_before_recovery = ctrl._last.get("kv_reserve_mb", (0, 0))[0]

    # Clock recovers, pressure still low → no Loop K grow
    m_ok = _make_metrics(kv_reserve_mb=512, kv_used_mb=100)  # low pressure
    m_ok.sm_clock_mhz = 2820
    m_ok.shim_phase = "INFERENCE"
    for _ in range(14):
        o.on_metrics(m_ok)

    assert o.clock_throttled is False
    kv_after = ctrl._last.get("kv_reserve_mb", (0, 0))[0]
    # kv_pressure not in hysteresis → Loop K must not grow
    assert not o._kv_pressure_sig.in_hysteresis
    assert kv_after == kv_before_recovery


def test_loop_k_blocked_during_idle_phase():
    """Loop K skips grow when shim_phase=IDLE even with pressure in hysteresis."""
    ctrl = _MockGbControl()
    ctrl._last["kv_reserve_mb"] = (512, 0.0)
    ctrl._last["safety_reserve_gb"] = (4, 0.0)
    o = _make_orch_no_io(ctrl=ctrl)
    o._total_vram_mb = 12288

    # Warmup + throttle with high KV pressure
    _feed_clock_and_pressure(o, 2850, 0.95, 5)
    _feed_clock_and_pressure(o, 2300, 0.95, 15)
    assert o.clock_throttled is True

    kv_after_throttle = ctrl._last.get("kv_reserve_mb", (0, 0))[0]

    # Clock recovers but we're now in IDLE , Loop K must not fire
    m_ok = _make_metrics(kv_reserve_mb=512, kv_used_mb=int(512 * 0.95))
    m_ok.sm_clock_mhz = 2820
    m_ok.shim_phase = "IDLE"
    for _ in range(14):
        o.on_metrics(m_ok)

    assert o.clock_throttled is False
    kv_after = ctrl._last.get("kv_reserve_mb", (0, 0))[0]
    assert kv_after == kv_after_throttle, "Loop K must not grow KV during IDLE phase"


def test_loop_j_no_crash_when_ctrl_none():
    """_reclaim_kv_for_idle returns safely when ctrl is None."""
    o = _make_orch_no_io()
    o._ctrl = None  # simulate no control interface
    # Must not raise , the null guard must fire and return early
    o._reclaim_kv_for_idle()


# ── Loop N , Adaptive telemetry poll rate ────────────────────────────────────

def _make_orch_with_tel():
    """Orchestrator with a mock TelemetryManager for Loop N tests."""
    ctrl = _MockGbControl()
    tel  = MagicMock()
    tel.poll_ms = 500
    def _set_poll(ms):
        tel.poll_ms = ms
    tel.set_poll_interval_ms.side_effect = _set_poll
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="process", tel_manager=tel)
    return o, tel


def _feed_phase(o, phase: str, gb_pool=None) -> None:
    """Push two metrics with given shim_phase so pairwise() fires a transition."""
    from gb_telemetry import GpuMetrics
    for _ in range(2):
        m = _make_metrics()
        m.shim_phase = phase
        if gb_pool is not None:
            m.gb = gb_pool
        o.on_metrics(m)


def test_loop_n_speeds_up_during_inference():
    """INFERENCE phase → poll_ms drops to 250."""
    o, tel = _make_orch_with_tel()
    _feed_phase(o, "INFERENCE")
    assert tel.poll_ms == 250


def test_loop_n_slows_down_during_idle():
    """IDLE phase → poll_ms rises to 1000."""
    o, tel = _make_orch_with_tel()
    _feed_phase(o, "INFERENCE")   # first go into inference
    _feed_phase(o, "IDLE")        # then transition to idle
    assert tel.poll_ms == 1000


def test_loop_n_slows_down_during_deep_idle():
    """DEEP_IDLE phase → poll_ms rises to 1000."""
    o, tel = _make_orch_with_tel()
    _feed_phase(o, "INFERENCE")
    _feed_phase(o, "DEEP_IDLE")
    assert tel.poll_ms == 1000


def test_loop_n_unknown_phase_uses_default():
    """MODEL_LOAD / unknown → poll_ms set to 500 (default)."""
    o, tel = _make_orch_with_tel()
    _feed_phase(o, "INFERENCE")   # sets 250
    _feed_phase(o, "MODEL_LOAD")  # should reset to 500
    assert tel.poll_ms == 500


def test_loop_n_no_op_without_tel_manager():
    """_adapt_poll_rate must not raise when tel_manager is None."""
    o = _make_orch_no_io()
    assert o._tel_manager is None
    # Feed a transition , must not raise
    _feed_phase(o, "INFERENCE")
    _feed_phase(o, "IDLE")


def test_loop_n_no_call_when_poll_ms_unchanged():
    """set_poll_interval_ms not called when poll_ms already at target."""
    o, tel = _make_orch_with_tel()
    tel.poll_ms = 250  # already at INFERENCE target
    tel.set_poll_interval_ms.reset_mock()
    _feed_phase(o, "INFERENCE")
    tel.set_poll_interval_ms.assert_not_called()


def test_loop_n_dump_reflects_poll_ms():
    """dump()['poll_ms'] matches the tel_manager's current poll_ms."""
    o, tel = _make_orch_with_tel()
    _feed_phase(o, "INFERENCE")
    assert o.dump()["poll_ms"] == 250
    _feed_phase(o, "IDLE")
    assert o.dump()["poll_ms"] == 1000


def test_loop_n_dump_default_without_tel():
    """dump()['poll_ms'] returns default constant when no tel_manager."""
    o = _make_orch_no_io()
    from gb_orchestrator import _POLL_MS_DEFAULT
    assert o.dump()["poll_ms"] == _POLL_MS_DEFAULT


# ── Continuous OS tuning , Loops O-S ─────────────────────────────────────────

def _make_supervisor_orch(monkeypatch, os_tune="1", **kw):
    """Construct a supervisor-mode orchestrator with GB_OS_TUNE set before
    __init__ reads the env (os_tune_enabled is computed once at construction)."""
    if os_tune is None:
        monkeypatch.delenv("GB_OS_TUNE", raising=False)
    else:
        monkeypatch.setenv("GB_OS_TUNE", os_tune)
    ctrl = kw.pop("ctrl", _MockGbControl())
    with patch.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         patch.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(control=ctrl, mode="supervisor", **kw)
    return o, ctrl


def _feed_phase_with_power(o, phase: str, power_limit_w=300.0, sm_clock_mhz=1800):
    """Like _feed_phase, but also carries power_limit_w/sm_clock_mhz so Loop O
    can compute a clock lock + power limit."""
    for _ in range(2):
        m = _make_metrics()
        m.shim_phase    = phase
        m.power_limit_w = power_limit_w
        m.sm_clock_mhz  = sm_clock_mhz
        o.on_metrics(m)


# ── gating ────────────────────────────────────────────────────────────────────

def test_os_tune_disabled_in_process_mode(monkeypatch):
    """os_tune_enabled requires mode=='supervisor' even with GB_OS_TUNE=1."""
    monkeypatch.setenv("GB_OS_TUNE", "1")
    o = _make_orch_no_io()
    assert o.os_tune_enabled is False
    assert o._os_tune_active() is False


def test_os_tune_disabled_without_env(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch, os_tune=None)
    assert o.os_tune_enabled is False
    assert o._os_tune_active() is False


def test_os_tune_active_when_supervisor_and_env(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    assert o.os_tune_enabled is True
    assert o._os_tune_active() is True


def test_os_tune_deferred_during_gaming_mode(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o.gaming_mode = True
    assert o._os_tune_active() is False


def test_gaming_mode_fed_from_gb_pool_info(orch):
    """on_metrics must mirror GbPoolInfo.gaming_mode onto orchestrator state."""
    gb = GbPoolInfo()
    gb.gaming_mode = True
    m = _make_metrics()
    m.gb = gb
    orch.on_metrics(m)
    assert orch.gaming_mode is True


# ── Loop O , performance envelope ────────────────────────────────────────────

def test_loop_o_engages_performance_envelope_on_inference(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    _feed_phase_with_power(o, "INFERENCE")
    names = [c[0] for c in ctrl.calls]
    for expected in ("set_cpu_governor", "set_energy_perf_pref", "set_numa_balancing",
                      "set_swappiness", "set_gpu_persistence", "lock_gpu_clocks",
                      "set_gpu_power_limit"):
        assert expected in names, f"{expected} not called: {names}"
    assert ctrl._last["cpu_governor"][0] == "performance"
    assert ctrl._last["numa_balancing"][0] is False


def test_loop_o_engages_on_model_load_and_steady(monkeypatch):
    for phase in ("MODEL_LOAD", "STEADY"):
        o, ctrl = _make_supervisor_orch(monkeypatch)
        _feed_phase_with_power(o, phase)
        assert any(c[0] == "set_cpu_governor" for c in ctrl.calls)


def test_loop_o_restores_baseline_on_deep_idle(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    _feed_phase_with_power(o, "INFERENCE")
    ctrl.calls.clear()
    _feed_phase_with_power(o, "DEEP_IDLE")
    assert any(c[0] == "restore_baseline" for c in ctrl.calls)


def test_loop_o_restores_baseline_on_idle(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    _feed_phase_with_power(o, "INFERENCE")
    ctrl.calls.clear()
    _feed_phase_with_power(o, "IDLE")
    assert any(c[0] == "restore_baseline" for c in ctrl.calls)


def test_loop_o_noop_without_os_tune_env(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch, os_tune=None)
    _feed_phase_with_power(o, "INFERENCE")
    assert ctrl.calls == []


def test_loop_o_noop_in_process_mode(monkeypatch):
    monkeypatch.setenv("GB_OS_TUNE", "1")
    ctrl = _MockGbControl()
    o = _make_orch_no_io(ctrl=ctrl)
    _feed_phase_with_power(o, "INFERENCE")
    assert ctrl.calls == []


def test_loop_o_deferred_during_gaming_mode(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o.gaming_mode = True
    _feed_phase_with_power(o, "INFERENCE")
    assert ctrl.calls == []


def test_loop_o_skips_clock_lock_without_sm_clock_max(monkeypatch):
    """lock_gpu_clocks must not fire when sm_clock_max is still 0 (no warmup)."""
    o, ctrl = _make_supervisor_orch(monkeypatch)
    for _ in range(2):
        m = _make_metrics()
        m.shim_phase = "INFERENCE"
        m.power_limit_w = 300.0
        m.sm_clock_mhz = 0  # never observed a clock yet
        o.on_metrics(m)
    assert not any(c[0] == "lock_gpu_clocks" for c in ctrl.calls)


# ── Loop P , thermal/throttle power cap ──────────────────────────────────────

def test_loop_p_thermal_high_steps_power_down(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._last_metrics = _make_metrics(power_limit_w=300.0)
    o._on_temp_high(85.0)
    assert ctrl._last["gpu_power_limit_w"][0] == 300 - 15


def test_loop_p_thermal_ok_steps_power_back_up(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._last_metrics = _make_metrics(power_limit_w=300.0)
    o._on_temp_high(85.0)
    o._on_temp_ok(72.0)
    assert ctrl._last["gpu_power_limit_w"][0] == 300


def test_loop_p_power_floored_at_half_tdp(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._last_metrics = _make_metrics(power_limit_w=300.0)
    for _ in range(40):
        o._on_temp_high(85.0)
    assert ctrl._last["gpu_power_limit_w"][0] >= 150


def test_loop_p_clock_throttled_steps_power_down(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._last_metrics = _make_metrics(power_limit_w=300.0)
    o._on_clock_throttled(15.0)
    assert ctrl._last["gpu_power_limit_w"][0] == 285


def test_loop_p_noop_in_process_mode(orch):
    orch._last_metrics = _make_metrics(power_limit_w=300.0)
    orch._on_temp_high(85.0)
    assert "gpu_power_limit_w" not in orch._ctrl._last


def test_loop_p_noop_without_power_limit_known(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._last_metrics = _make_metrics(power_limit_w=0.0)
    o._on_temp_high(85.0)
    assert "gpu_power_limit_w" not in ctrl._last


# ── Loop Q , host memory-pressure VM tune ────────────────────────────────────

def test_loop_q_mem_psi_high_tightens_vm(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._on_mem_psi_high(15.0)
    names = [c[0] for c in ctrl.calls]
    assert "set_watermark_scale_factor" in names
    assert "set_swappiness" in names
    assert "set_dirty_background_ratio" in names
    assert ctrl._last["watermark_scale_factor"][0] == 150


def test_loop_q_mem_psi_ok_relaxes_vm(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._on_mem_psi_high(15.0)
    ctrl.calls.clear()
    o._on_mem_psi_ok(2.0)
    names = [c[0] for c in ctrl.calls]
    assert "set_watermark_scale_factor" in names
    assert ctrl._last["watermark_scale_factor"][0] == 10


def test_loop_q_noop_in_process_mode(orch):
    orch._on_mem_psi_high(15.0)
    assert "watermark_scale_factor" not in orch._ctrl._last


def test_loop_q_fires_via_on_metrics_psi_feed(monkeypatch):
    """End-to-end: SystemMetrics.psi_mem_some_avg10 -> Signal hysteresis -> lever."""
    from gb_telemetry import SystemMetrics
    o, ctrl = _make_supervisor_orch(monkeypatch)
    for _ in range(25):
        m = _make_metrics()
        m.sys = SystemMetrics(psi_mem_some_avg10=15.0)
        o.on_metrics(m)
    assert any(c[0] == "set_watermark_scale_factor" for c in ctrl.calls)


# ── Loop R , host CPU/IO pressure assist ─────────────────────────────────────

def test_loop_r_cpu_psi_high_during_inference_reasserts_governor(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._last_metrics = _make_metrics()
    o._last_metrics.shim_phase = "INFERENCE"
    o._on_cpu_psi_high(25.0)
    assert ctrl._last["cpu_governor"][0] == "performance"


def test_loop_r_cpu_psi_high_outside_perf_phase_is_noop(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._last_metrics = _make_metrics()
    o._last_metrics.shim_phase = "IDLE"
    o._on_cpu_psi_high(25.0)
    assert "cpu_governor" not in ctrl._last


def test_loop_r_io_psi_high_with_zero_t3_telemetry_is_advisory(monkeypatch):
    """Edge case: t3_used_mb/t3_free_mb both 0 (no real capacity learned yet)
    -> baseline collapses to the floor immediately -> advisory, no lever call.
    Distinct from the DI-12 engaged-lever path below, which needs a real
    (non-zero) T3 capacity to step down from."""
    o, ctrl = _make_supervisor_orch(monkeypatch)
    gb = GbPoolInfo()
    gb.t3_pressure = 2
    o._last_metrics = _make_metrics()
    o._last_metrics.gb = gb
    o._on_io_psi_high(25.0)   # must not raise
    assert ctrl.calls == []


def test_loop_r_io_psi_high_without_t3_pressure_is_noop(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._last_metrics = _make_metrics()
    o._on_io_psi_high(25.0)
    assert ctrl.calls == []


# ── DI-12: reactive NVMe/T3 lever ────────────────────────────────────────────

def _make_t3_pressured_orch(monkeypatch, t3_used_mb=4096, t3_free_mb=6144):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    gb = GbPoolInfo()
    gb.t3_pressure = 2
    gb.t3_used_mb = t3_used_mb
    gb.t3_free_mb = t3_free_mb
    o._last_metrics = _make_metrics()
    o._last_metrics.gb = gb
    return o, ctrl


def test_di12_steps_t3_cap_down_on_real_pressure(monkeypatch):
    monkeypatch.delenv("GB_NVME_LEVER", raising=False)
    o, ctrl = _make_t3_pressured_orch(monkeypatch)
    o._on_io_psi_high(25.0)
    assert "t3_cap_mb" in ctrl._last
    new_cap = ctrl._last["t3_cap_mb"][0]
    baseline = 4096 + 6144
    assert new_cap < baseline   # stepped down from the learned baseline
    assert new_cap >= int(baseline * 0.10)   # never below the 10% floor
    assert o._t3_lever_active is True


def test_di12_escape_hatch_disables_lever(monkeypatch):
    monkeypatch.setenv("GB_NVME_LEVER", "0")
    o, ctrl = _make_t3_pressured_orch(monkeypatch)
    o._on_io_psi_high(25.0)
    assert ctrl.calls == []   # advisory only, exact pre-DI-12 behavior
    assert o._t3_lever_active is False


def test_di12_restores_cap_when_pressure_clears(monkeypatch):
    monkeypatch.delenv("GB_NVME_LEVER", raising=False)
    # Sustained pressure -> multiple step-downs, so ONE restore call can only
    # partially recover (a single down-step and up-step of the same
    # percentage always exactly cancel — this needs at least two step-downs
    # for a single restore to land strictly between the two).
    o, ctrl = _make_t3_pressured_orch(monkeypatch)
    o._on_io_psi_high(25.0)
    o._on_io_psi_high(25.0)
    stepped_down = ctrl._last["t3_cap_mb"][0]
    assert o._t3_lever_active is True
    o._on_io_psi_ok(5.0)
    restored = ctrl._last["t3_cap_mb"][0]
    assert restored > stepped_down   # geometric step back up
    assert restored < 10240          # but not fully restored to baseline yet


def test_di12_fully_restores_to_zero_cap_after_enough_recoveries(monkeypatch):
    monkeypatch.delenv("GB_NVME_LEVER", raising=False)
    o, ctrl = _make_t3_pressured_orch(monkeypatch)
    o._on_io_psi_high(25.0)
    assert o._t3_lever_active is True
    for _ in range(200):   # geometric 2%/step converges back to baseline
        o._on_io_psi_ok(5.0)
        if not o._t3_lever_active:
            break
    assert o._t3_lever_active is False
    assert ctrl._last["t3_cap_mb"][0] == 0   # 0 = disk-limited/no cap, fully restored


def test_di12_restore_noop_when_lever_never_engaged(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._on_io_psi_ok(5.0)   # must not raise, must not call set_t3_cap_mb
    assert "t3_cap_mb" not in ctrl._last


def test_loop_r_noop_in_process_mode(orch):
    orch._last_metrics = _make_metrics()
    orch._last_metrics.shim_phase = "INFERENCE"
    orch._on_cpu_psi_high(25.0)
    assert "cpu_governor" not in orch._ctrl._last


# ── Loop S , PCIe saturation ──────────────────────────────────────────────────

def test_loop_s_pcie_saturated_engages_prefetch_throttle(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._on_pcie_saturation_change(True, False)
    assert ctrl._last["prefetch_throttle"][0] is True


def test_loop_s_pcie_clears_prefetch_throttle(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o._on_pcie_saturation_change(True, False)
    o._on_pcie_saturation_change(False, True)
    assert ctrl._last["prefetch_throttle"][0] is False


def test_loop_s_noop_in_process_mode(orch):
    orch._on_pcie_saturation_change(True, False)
    assert "prefetch_throttle" not in orch._ctrl._last


def test_loop_s_fires_via_on_metrics_pcie_saturated(monkeypatch):
    """End-to-end: GpuMetrics.pcie_saturated property -> Signal -> lever."""
    from gb_telemetry import GpuTopology
    o, ctrl = _make_supervisor_orch(monkeypatch)
    topo = GpuTopology(pcie_gen_current=4, pcie_width_current=16)
    for _ in range(6):
        m = _make_metrics()
        m.topology = topo
        m.pcie_tx_mb_s = 1_000_000.0   # force pcie_saturated True
        m.pcie_rx_mb_s = 1_000_000.0
        o.on_metrics(m)
    assert any(c[0] == "set_prefetch_throttle" for c in ctrl.calls)


# ── orchestrator.stop() resets GPU clock lock ────────────────────────────────

def test_stop_resets_gpu_clocks(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o.stop()
    assert any(c[0] == "reset_gpu_clocks" for c in ctrl.calls)


def test_stop_noop_without_ctrl():
    o = ReactiveOrchestrator(control=None, mode="supervisor")
    o.stop()  # must not raise


# ── dump() surfaces OS-tune state ────────────────────────────────────────────

def test_dump_includes_os_tune_state(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    d = o.dump()
    assert d["os_tune_enabled"] is True
    assert d["gaming_mode"] is False
    assert any(s["name"] == "psi_mem_some" for s in d["signals"])
    assert any(s["name"] == "psi_cpu_some" for s in d["signals"])
    assert any(s["name"] == "psi_io_some" for s in d["signals"])
    assert any(s["name"] == "pcie_saturated" for s in d["signals"])


# ── GpuHealth (DI-6, MuxFlow SysMonitor port) ────────────────────────────────

def test_gpu_health_starts_init():
    h = GpuHealth()
    assert h.state == GpuHealth.INIT
    assert not h.admits_overcommit()


def test_gpu_health_no_stress_becomes_healthy():
    h = GpuHealth()
    new = h.feed(ecc_degraded=False, thermal_stress=False,
                clock_throttled=False, mem_bw_stress=False)
    assert new == GpuHealth.HEALTHY
    assert h.admits_overcommit()


def test_gpu_health_single_stress_is_unhealthy():
    h = GpuHealth()
    h.feed(ecc_degraded=False, thermal_stress=False, clock_throttled=False, mem_bw_stress=False)
    new = h.feed(ecc_degraded=False, thermal_stress=True, clock_throttled=False, mem_bw_stress=False)
    assert new == GpuHealth.UNHEALTHY
    assert not h.admits_overcommit()


def test_gpu_health_two_simultaneous_stresses_is_overlimit():
    h = GpuHealth()
    new = h.feed(ecc_degraded=False, thermal_stress=True, clock_throttled=True, mem_bw_stress=False)
    assert new == GpuHealth.OVERLIMIT
    assert not h.admits_overcommit()


def test_gpu_health_ecc_degraded_forces_disabled_even_with_no_other_stress():
    h = GpuHealth()
    new = h.feed(ecc_degraded=True, thermal_stress=False, clock_throttled=False, mem_bw_stress=False)
    assert new == GpuHealth.DISABLED
    assert not h.admits_overcommit()


def test_gpu_health_recovers_from_unhealthy_to_healthy():
    h = GpuHealth()
    h.feed(ecc_degraded=False, thermal_stress=True, clock_throttled=False, mem_bw_stress=False)
    assert h.state == GpuHealth.UNHEALTHY
    new = h.feed(ecc_degraded=False, thermal_stress=False, clock_throttled=False, mem_bw_stress=False)
    assert new == GpuHealth.HEALTHY
    assert h.admits_overcommit()


def test_gpu_health_feed_returns_none_when_state_unchanged():
    h = GpuHealth()
    h.feed(ecc_degraded=False, thermal_stress=False, clock_throttled=False, mem_bw_stress=False)
    same = h.feed(ecc_degraded=False, thermal_stress=False, clock_throttled=False, mem_bw_stress=False)
    assert same is None   # no transition -> no event, matches Signal's on_enter/on_exit convention


def test_dump_includes_di6_health_state(monkeypatch):
    o, ctrl = _make_supervisor_orch(monkeypatch)
    o.on_metrics(_make_metrics())
    d = o.dump()
    assert d["health_state"] in (GpuHealth.HEALTHY, GpuHealth.UNHEALTHY,
                                 GpuHealth.OVERLIMIT, GpuHealth.DISABLED)
    assert isinstance(d["admits_overcommit"], bool)
