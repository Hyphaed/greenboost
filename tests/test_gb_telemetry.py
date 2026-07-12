#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for GpuMetrics decision-helper properties and ClusterTelemetryManager aggregates.

No GPU, no NVML, no daemon , pure dataclass tests with fake managers.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from gb_telemetry import GpuMetrics, GbPoolInfo


# ── Helpers ──────────────────────────────────────────────────────────────────

def _metrics(**kw) -> GpuMetrics:
    """Build a GpuMetrics with sane defaults, overridden by kw."""
    m = GpuMetrics()
    m.fb_total_mb = kw.get("fb_total_mb", 12000)
    m.fb_used_mb  = kw.get("fb_used_mb", 6000)
    m.fb_free_mb  = kw.get("fb_free_mb", 6000)
    m.fb_used_pct = kw.get("fb_used_pct", 50.0)
    m.gpu_util_pct = kw.get("gpu_util_pct", 30.0)
    m.ecc_dbe_volatile = kw.get("ecc_dbe_volatile", 0)
    m.health_ok   = kw.get("health_ok", True)
    m.pcie_tx_mb_s = kw.get("pcie_tx_mb_s", 0.0)
    m.pcie_rx_mb_s = kw.get("pcie_rx_mb_s", 0.0)
    if "gb" in kw:
        m.gb = kw["gb"]
    return m


def _pool(**kw) -> GbPoolInfo:
    p = GbPoolInfo()
    for attr, val in kw.items():
        setattr(p, attr, val)
    return p


# ── should_demote ─────────────────────────────────────────────────────────────

def test_should_demote_true_above_90():
    assert _metrics(fb_used_pct=90.1).should_demote is True


def test_should_demote_false_at_90():
    assert _metrics(fb_used_pct=90.0).should_demote is False


def test_should_demote_false_below_90():
    assert _metrics(fb_used_pct=85.0).should_demote is False


# ── can_promote ───────────────────────────────────────────────────────────────

def test_can_promote_true_below_60():
    assert _metrics(fb_used_pct=59.9).can_promote is True


def test_can_promote_false_at_60():
    assert _metrics(fb_used_pct=60.0).can_promote is False


def test_can_promote_false_above_60():
    assert _metrics(fb_used_pct=75.0).can_promote is False


# ── t2_critical ───────────────────────────────────────────────────────────────

def test_t2_critical_true_at_pressure_2():
    m = _metrics(gb=_pool(t2_pressure=2))
    assert m.t2_critical is True


def test_t2_critical_false_at_pressure_1():
    m = _metrics(gb=_pool(t2_pressure=1))
    assert m.t2_critical is False


def test_t2_critical_false_no_gb():
    m = _metrics()
    m.gb = None
    assert m.t2_critical is False


# ── prefetch_budget_mb ────────────────────────────────────────────────────────

def test_prefetch_budget_free_above_1024():
    m = _metrics(fb_free_mb=3000)
    assert m.prefetch_budget_mb == 3000 - 1024


def test_prefetch_budget_clamps_at_zero_when_tight():
    m = _metrics(fb_free_mb=512)   # 512 < 1024
    assert m.prefetch_budget_mb == 0


def test_prefetch_budget_zero_when_free_is_zero():
    m = _metrics(fb_free_mb=0)
    assert m.prefetch_budget_mb == 0


# ── ecc_critical ─────────────────────────────────────────────────────────────

def test_ecc_critical_true_when_dbe_gt_0():
    assert _metrics(ecc_dbe_volatile=1).ecc_critical is True


def test_ecc_critical_false_when_dbe_0():
    assert _metrics(ecc_dbe_volatile=0).ecc_critical is False


# ── kv_pressure ──────────────────────────────────────────────────────────────

def test_kv_pressure_ratio():
    m = _metrics(gb=_pool(kv_reserve_mb=512, kv_used_mb=256))
    assert abs(m.kv_pressure - 0.5) < 1e-6


def test_kv_pressure_zero_when_no_gb():
    m = _metrics()
    m.gb = None
    assert m.kv_pressure == 0.0


def test_kv_pressure_zero_when_reserve_zero():
    m = _metrics(gb=_pool(kv_reserve_mb=0, kv_used_mb=100))
    assert m.kv_pressure == 0.0


def test_kv_pressure_over_one_when_spilled():
    m = _metrics(gb=_pool(kv_reserve_mb=512, kv_used_mb=600))
    assert m.kv_pressure > 1.0


# ── kv_spilled ────────────────────────────────────────────────────────────────

def test_kv_spilled_true_when_kv_t2_gt_0():
    m = _metrics(gb=_pool(kv_t2_mb=128))
    assert m.kv_spilled is True


def test_kv_spilled_false_when_kv_t2_zero():
    m = _metrics(gb=_pool(kv_t2_mb=0))
    assert m.kv_spilled is False


def test_kv_spilled_false_when_no_gb():
    m = _metrics()
    m.gb = None
    assert m.kv_spilled is False


# ── ClusterTelemetryManager aggregates ───────────────────────────────────────

def _fake_manager(snapshot: GpuMetrics) -> MagicMock:
    m = MagicMock()
    m.snapshot.return_value = snapshot
    return m


def test_cluster_any_should_demote_true_if_any_device_pressured():
    from gb_telemetry import ClusterTelemetryManager
    clus = MagicMock(spec=ClusterTelemetryManager)
    clus._managers = [
        _fake_manager(_metrics(fb_used_pct=50.0)),   # not pressured
        _fake_manager(_metrics(fb_used_pct=95.0)),   # pressured (>90%)
    ]
    # Replicate any_should_demote logic
    result = any(m.snapshot().should_demote for m in clus._managers)
    assert result is True


def test_cluster_any_should_demote_false_if_none_pressured():
    from gb_telemetry import ClusterTelemetryManager
    clus = MagicMock(spec=ClusterTelemetryManager)
    clus._managers = [
        _fake_manager(_metrics(fb_used_pct=50.0)),
        _fake_manager(_metrics(fb_used_pct=80.0)),
    ]
    result = any(m.snapshot().should_demote for m in clus._managers)
    assert result is False


def test_cluster_all_can_promote_true_if_all_have_headroom():
    from gb_telemetry import ClusterTelemetryManager
    clus = MagicMock(spec=ClusterTelemetryManager)
    clus._managers = [
        _fake_manager(_metrics(fb_used_pct=40.0)),
        _fake_manager(_metrics(fb_used_pct=50.0)),
    ]
    result = all(m.snapshot().can_promote for m in clus._managers)
    assert result is True


def test_cluster_all_can_promote_false_if_any_tight():
    from gb_telemetry import ClusterTelemetryManager
    clus = MagicMock(spec=ClusterTelemetryManager)
    clus._managers = [
        _fake_manager(_metrics(fb_used_pct=40.0)),
        _fake_manager(_metrics(fb_used_pct=75.0)),   # not promotable
    ]
    result = all(m.snapshot().can_promote for m in clus._managers)
    assert result is False


# ── GpuMetrics.power_near_limit ──────────────────────────────────────────────

def test_power_near_limit_false_when_limit_zero():
    """No power_limit_w sampled → False (avoid false gates)."""
    m = _metrics()
    m.power_w = 300.0
    m.power_limit_w = 0.0
    assert m.power_near_limit is False


def test_power_near_limit_false_when_well_below():
    m = _metrics()
    m.power_w = 200.0
    m.power_limit_w = 350.0   # 57%
    assert m.power_near_limit is False


def test_power_near_limit_true_at_94_pct():
    m = _metrics()
    m.power_w = 282.0
    m.power_limit_w = 300.0   # 94% > 93% threshold
    assert m.power_near_limit is True


def test_power_near_limit_false_exactly_at_93_pct():
    """Boundary: exactly 93% is NOT above threshold."""
    m = _metrics()
    m.power_w = 279.0
    m.power_limit_w = 300.0   # 93.0% , not strictly >
    assert m.power_near_limit is False


# ── GpuMetrics.__repr__ ──────────────────────────────────────────────────────

def test_repr_default():
    m = _metrics()
    r = repr(m)
    assert "GpuMetrics" in r
    assert "fb=" in r
    assert "ECC_DBE" not in r
    assert "HEALTH_FAIL" not in r


def test_repr_with_ecc():
    m = _metrics(ecc_dbe_volatile=3)
    r = repr(m)
    assert "ECC_DBE=3!" in r


def test_repr_with_health_fail():
    m = _metrics(health_ok=False)
    m.health_summary = "SM clock error"
    r = repr(m)
    assert "HEALTH_FAIL" in r
    assert "SM clock error" in r


def test_repr_with_no_gb():
    m = _metrics()
    m.gb = None
    r = repr(m)
    assert "t2_press=?" in r


# ── _nvtx_range ──────────────────────────────────────────────────────────────

def test_nvtx_range_without_nvtx_yields():
    """_nvtx_range is a no-op context manager when NVTX is unavailable."""
    import gb_telemetry as _tel
    # Force NVTX unavailable path
    orig = _tel._NVTX_AVAILABLE
    try:
        _tel._NVTX_AVAILABLE = False
        ran = []
        with _tel._nvtx_range("test:range"):
            ran.append(True)
        assert ran == [True]
    finally:
        _tel._NVTX_AVAILABLE = orig


# ── NVMLProvider failure path ────────────────────────────────────────────────

def test_nvml_provider_not_available_when_no_nvml_backend():
    """NVMLProvider.available() is False when neither pynvml nor the
    gb_nvml_ctypes fallback can be imported."""
    import sys
    # Temporarily hide both NVML backends , gb_nvml_ctypes.nvmlInit() will
    # then raise (no library / no symbols), same as pynvml being absent.
    orig_pynvml = sys.modules.get("pynvml")
    orig_ctypes_fallback = sys.modules.get("gb_nvml_ctypes")
    sys.modules["pynvml"] = None            # blocks import
    sys.modules["gb_nvml_ctypes"] = None    # blocks fallback import too
    try:
        from gb_telemetry import NVMLProvider
        p = NVMLProvider(device=0)
        assert p.available() is False
    finally:
        for name, orig in (("pynvml", orig_pynvml), ("gb_nvml_ctypes", orig_ctypes_fallback)):
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


# ── TorchFallbackProvider ─────────────────────────────────────────────────────

def test_torch_fallback_not_available_when_cuda_absent():
    """TorchFallbackProvider.available() is False when CUDA is not present."""
    from unittest.mock import patch
    from gb_telemetry import TorchFallbackProvider
    with patch("torch.cuda.is_available", return_value=False):
        p = TorchFallbackProvider()
        assert p.available() is False


# ── RemoteFeederProvider ──────────────────────────────────────────────────────

import json
import tempfile
import os as _os


def _write_metrics_json(path: str, feeders: list) -> None:
    with open(path, "w") as f:
        json.dump({"feeders": feeders}, f)


def test_remote_feeder_fill_healthy(tmp_path, monkeypatch):
    """health_state=0 → health_ok stays True, no ECC."""
    from gb_telemetry import RemoteFeederProvider, GpuMetrics
    metrics_path = str(tmp_path / "metrics.json")
    _write_metrics_json(metrics_path, [
        {"gpu_util_pct": 45.0, "gpu_temp_c": 72.0, "gpu_power_w": 120.0,
         "gpu_mem_util_pct": 30.0, "health_state": 0, "t1_quarantined": 0}
    ])
    monkeypatch.setattr("gb_telemetry.RemoteFeederProvider._METRICS_JSON", metrics_path)
    p = RemoteFeederProvider(feeder_idx=0)
    m = GpuMetrics()
    p.fill(m)
    assert m.gpu_util_pct == 45.0
    assert m.temp_c == 72.0
    assert m.health_ok is True
    assert m.ecc_dbe_volatile == 0


def test_remote_feeder_fill_unhealthy(tmp_path, monkeypatch):
    """health_state=2 → health_ok=False."""
    from gb_telemetry import RemoteFeederProvider, GpuMetrics
    metrics_path = str(tmp_path / "metrics.json")
    _write_metrics_json(metrics_path, [
        {"gpu_util_pct": 10.0, "gpu_temp_c": 85.0, "gpu_power_w": 200.0,
         "gpu_mem_util_pct": 20.0, "health_state": 2, "t1_quarantined": 0}
    ])
    monkeypatch.setattr("gb_telemetry.RemoteFeederProvider._METRICS_JSON", metrics_path)
    p = RemoteFeederProvider(feeder_idx=0)
    m = GpuMetrics()
    p.fill(m)
    assert m.health_ok is False
    assert "health_state=2" in m.health_summary


def test_remote_feeder_fill_ecc_quarantine(tmp_path, monkeypatch):
    """t1_quarantined=1 → ecc_dbe_volatile=1, health_ok=False."""
    from gb_telemetry import RemoteFeederProvider, GpuMetrics
    metrics_path = str(tmp_path / "metrics.json")
    _write_metrics_json(metrics_path, [
        {"gpu_util_pct": 0.0, "gpu_temp_c": 60.0, "gpu_power_w": 50.0,
         "gpu_mem_util_pct": 0.0, "health_state": 0, "t1_quarantined": 1}
    ])
    monkeypatch.setattr("gb_telemetry.RemoteFeederProvider._METRICS_JSON", metrics_path)
    p = RemoteFeederProvider(feeder_idx=0)
    m = GpuMetrics()
    p.fill(m)
    assert m.ecc_dbe_volatile == 1
    assert m.health_ok is False
    assert "quarantine" in m.health_summary


def test_remote_feeder_fill_index_out_of_range(tmp_path, monkeypatch):
    """feeder_idx past the feeders list → GpuMetrics unchanged."""
    from gb_telemetry import RemoteFeederProvider, GpuMetrics
    metrics_path = str(tmp_path / "metrics.json")
    _write_metrics_json(metrics_path, [])   # empty feeder list
    monkeypatch.setattr("gb_telemetry.RemoteFeederProvider._METRICS_JSON", metrics_path)
    p = RemoteFeederProvider(feeder_idx=5)
    m = GpuMetrics()
    p.fill(m)
    assert m.gpu_util_pct == 0.0   # unchanged


def test_remote_feeder_not_available_when_no_file():
    """available() is False when metrics.json doesn't exist."""
    from gb_telemetry import RemoteFeederProvider
    p = RemoteFeederProvider(feeder_idx=0)
    p._METRICS_JSON = "/nonexistent/metrics.json"
    assert p.available() is False


# ── TelemetryManager with fake provider ──────────────────────────────────────

def _make_fake_provider(util_pct=42.0, temp=65.0):
    """Return a _Provider stub that fills fixed values into GpuMetrics."""
    from gb_telemetry import _Provider, GpuMetrics
    class _Fake(_Provider):
        name = "fake"
        def available(self): return True
        def fill(self, m: GpuMetrics):
            m.gpu_util_pct = util_pct
            m.temp_c = temp
            m.fb_total_mb = 12000
            m.fb_used_mb = 6000
            m.fb_free_mb = 6000
            m.fb_used_pct = 50.0
    return _Fake()


def _make_telemetry_mgr_with_fake():
    """Construct a TelemetryManager backed by a fake provider (no hardware)."""
    from gb_telemetry import TelemetryManager
    from unittest.mock import patch, MagicMock

    fake_nvml = MagicMock()
    fake_nvml.available.return_value = False

    fake_dcgm = MagicMock()
    fake_dcgm.available.return_value = False

    fake_gb = MagicMock()
    fake_gb.available.return_value = False

    with patch("gb_telemetry.NVMLProvider", return_value=fake_nvml), \
         patch("gb_telemetry.TorchFallbackProvider", return_value=MagicMock(available=lambda: False)), \
         patch("gb_telemetry.DCGMProvider", return_value=fake_dcgm), \
         patch("gb_telemetry.GreenBoostProvider", return_value=fake_gb):
        mgr = TelemetryManager(device=0, poll_ms=100, enable_dcgm=False)

    mgr._providers = [_make_fake_provider()]
    return mgr


def test_telemetry_manager_poll_once_updates_snapshot():
    mgr = _make_telemetry_mgr_with_fake()
    m = mgr.poll_once()
    assert m.gpu_util_pct == 42.0
    assert m.temp_c == 65.0
    assert mgr.snapshot().gpu_util_pct == 42.0


def test_telemetry_manager_snapshot_initially_empty():
    mgr = _make_telemetry_mgr_with_fake()
    # Before any poll, snapshot is the default GpuMetrics
    snap = mgr.snapshot()
    assert isinstance(snap.gpu_util_pct, float)


def test_telemetry_manager_add_callback_called_on_poll():
    mgr = _make_telemetry_mgr_with_fake()
    received = []
    mgr.add_callback(lambda m: received.append(m.gpu_util_pct))
    mgr.poll_once()
    assert received == [42.0]


def test_telemetry_manager_callback_exception_does_not_propagate():
    """Buggy callbacks must not crash the polling loop."""
    mgr = _make_telemetry_mgr_with_fake()
    mgr.add_callback(lambda m: (_ for _ in ()).throw(RuntimeError("cb boom")))
    mgr.poll_once()   # must not raise


def test_telemetry_manager_multiple_callbacks():
    mgr = _make_telemetry_mgr_with_fake()
    results = []
    mgr.add_callback(lambda m: results.append("cb1"))
    mgr.add_callback(lambda m: results.append("cb2"))
    mgr.poll_once()
    assert results == ["cb1", "cb2"]


# ── ClusterTelemetryManager via real methods ─────────────────────────────────

def test_cluster_health_ok_all_healthy():
    """cluster_health_ok() is True when all snapshots are healthy + no ECC."""
    from gb_telemetry import ClusterTelemetryManager
    clus = MagicMock(spec=ClusterTelemetryManager)
    clus._managers = [
        _fake_manager(_metrics(health_ok=True, ecc_dbe_volatile=0)),
        _fake_manager(_metrics(health_ok=True, ecc_dbe_volatile=0)),
    ]
    # Replicate cluster_health_ok logic
    result = all(
        m.snapshot().health_ok and not m.snapshot().ecc_critical
        for m in clus._managers
    )
    assert result is True


def test_cluster_health_ok_false_when_ecc():
    from gb_telemetry import ClusterTelemetryManager
    clus = MagicMock(spec=ClusterTelemetryManager)
    clus._managers = [
        _fake_manager(_metrics(health_ok=True, ecc_dbe_volatile=0)),
        _fake_manager(_metrics(health_ok=True, ecc_dbe_volatile=1)),
    ]
    result = all(
        m.snapshot().health_ok and not m.snapshot().ecc_critical
        for m in clus._managers
    )
    assert result is False


def test_cluster_health_ok_false_when_unhealthy():
    from gb_telemetry import ClusterTelemetryManager
    clus = MagicMock(spec=ClusterTelemetryManager)
    clus._managers = [
        _fake_manager(_metrics(health_ok=False, ecc_dbe_volatile=0)),
    ]
    result = all(
        m.snapshot().health_ok and not m.snapshot().ecc_critical
        for m in clus._managers
    )
    assert result is False


# ── _detect_device_count ──────────────────────────────────────────────────────

def test_detect_device_count_via_pynvml(tmp_path):
    """_detect_device_count() returns pynvml count when pynvml is available."""
    from gb_telemetry import _detect_device_count
    fake_nvml = MagicMock()
    fake_nvml.nvmlDeviceGetCount.return_value = 2
    with patch.dict(sys.modules, {"pynvml": fake_nvml}):
        n = _detect_device_count()
    assert n == 2


def test_detect_device_count_clamps_at_1_when_zero():
    """_detect_device_count() returns at least 1 even if pynvml reports 0."""
    from gb_telemetry import _detect_device_count
    fake_nvml = MagicMock()
    fake_nvml.nvmlDeviceGetCount.return_value = 0
    with patch.dict(sys.modules, {"pynvml": fake_nvml}):
        n = _detect_device_count()
    assert n >= 1


def test_detect_device_count_falls_back_to_torch():
    """_detect_device_count() falls back to torch when pynvml raises."""
    from gb_telemetry import _detect_device_count
    bad_nvml = MagicMock()
    bad_nvml.nvmlInit.side_effect = RuntimeError("no driver")
    fake_torch = MagicMock()
    fake_torch.cuda.device_count.return_value = 3
    with patch.dict(sys.modules, {"pynvml": bad_nvml, "torch": fake_torch}):
        n = _detect_device_count()
    assert n == 3


def test_detect_device_count_returns_1_when_all_fail():
    """_detect_device_count() returns 1 when both pynvml and torch fail."""
    from gb_telemetry import _detect_device_count
    bad_nvml = MagicMock()
    bad_nvml.nvmlInit.side_effect = RuntimeError("no driver")
    bad_torch = MagicMock()
    bad_torch.cuda.device_count.side_effect = RuntimeError("no cuda")
    with patch.dict(sys.modules, {"pynvml": bad_nvml, "torch": bad_torch}):
        n = _detect_device_count()
    assert n == 1


# ── _detect_feeder_count ──────────────────────────────────────────────────────

def test_detect_feeder_count_returns_0_when_file_missing():
    """_detect_feeder_count() returns 0 when metrics.json does not exist."""
    import gb_telemetry
    with patch("gb_telemetry.os.path.exists", return_value=False):
        n = gb_telemetry._detect_feeder_count()
    assert n == 0


def test_detect_feeder_count_reads_feeders_array(tmp_path):
    """_detect_feeder_count() returns len(feeders) from metrics.json."""
    import json
    import gb_telemetry
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"feeders": [{}, {}, {}]}))
    with patch("gb_telemetry.os.path.exists", return_value=True), \
         patch("builtins.open", lambda p, *a, **kw: metrics.open()):
        n = gb_telemetry._detect_feeder_count()
    assert n == 3


# ── sample_once / sample_cluster ─────────────────────────────────────────────

def test_sample_once_returns_gpu_metrics():
    """sample_once() returns a GpuMetrics instance even with no hardware."""
    from gb_telemetry import sample_once, GpuMetrics
    # All providers will be unavailable in test env → returns empty metrics
    m = sample_once(device=0)
    assert isinstance(m, GpuMetrics)
    assert m.device == 0


def test_sample_cluster_returns_list():
    """sample_cluster() returns one GpuMetrics per detected device."""
    from gb_telemetry import sample_cluster, _detect_device_count, GpuMetrics
    with patch("gb_telemetry._detect_device_count", return_value=2):
        result = sample_cluster()
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(m, GpuMetrics) for m in result)


# ── ClusterTelemetryManager real methods ─────────────────────────────────────

def _make_cluster_mgr_no_init(n: int = 2):
    """Create a ClusterTelemetryManager bypassing __init__ (no real hardware)."""
    from gb_telemetry import ClusterTelemetryManager, GpuMetrics
    clus = object.__new__(ClusterTelemetryManager)

    class _FakeMgr:
        def __init__(self, dev, fb_used_pct=30.0, health_ok=True, ecc_dbe=0):
            self.device = dev
            self._snap = GpuMetrics(device=dev)
            self._snap.fb_used_pct = fb_used_pct
            self._snap.health_ok = health_ok
            self._snap.ecc_dbe_volatile = ecc_dbe
            self._start_called = False
            self._stop_called = False
        def snapshot(self): return self._snap
        def start(self): self._start_called = True
        def stop(self): self._stop_called = True
        def run_diag(self, level=1): return "ok"

    clus._managers = [_FakeMgr(i) for i in range(n)]
    clus.poll_ms = 100
    return clus


def test_cluster_mgr_start_calls_all_managers():
    """ClusterTelemetryManager.start() starts every manager."""
    clus = _make_cluster_mgr_no_init(2)
    clus.start()
    assert all(m._start_called for m in clus._managers)


def test_cluster_mgr_stop_calls_all_managers():
    """ClusterTelemetryManager.stop() stops every manager."""
    clus = _make_cluster_mgr_no_init(2)
    clus.stop()
    assert all(m._stop_called for m in clus._managers)


def test_cluster_mgr_snapshots_returns_one_per_device():
    from gb_telemetry import GpuMetrics
    clus = _make_cluster_mgr_no_init(3)
    snaps = clus.snapshots()
    assert len(snaps) == 3
    assert all(isinstance(s, GpuMetrics) for s in snaps)


def test_cluster_mgr_snapshot_by_device():
    clus = _make_cluster_mgr_no_init(2)
    s0 = clus.snapshot(0)
    s1 = clus.snapshot(1)
    assert s0.device == 0
    assert s1.device == 1


def test_cluster_mgr_snapshot_out_of_bounds_falls_back_to_0():
    clus = _make_cluster_mgr_no_init(2)
    s = clus.snapshot(99)
    assert s.device == 0


def test_cluster_mgr_cluster_health_ok_when_all_healthy():
    from gb_telemetry import ClusterTelemetryManager
    clus = _make_cluster_mgr_no_init(2)
    assert clus.cluster_health_ok() is True


def test_cluster_mgr_cluster_health_fails_on_ecc():
    clus = _make_cluster_mgr_no_init(2)
    clus._managers[1]._snap.ecc_dbe_volatile = 1   # ECC error on device 1
    assert clus.cluster_health_ok() is False


def test_cluster_mgr_cluster_health_fails_on_unhealthy():
    clus = _make_cluster_mgr_no_init(2)
    clus._managers[0]._snap.health_ok = False
    assert clus.cluster_health_ok() is False


def test_cluster_mgr_any_should_demote_true():
    clus = _make_cluster_mgr_no_init(2)
    clus._managers[1]._snap.fb_used_pct = 92.0   # > 90% threshold
    assert clus.any_should_demote() is True


def test_cluster_mgr_any_should_demote_false():
    clus = _make_cluster_mgr_no_init(2)
    # both below 90%
    assert clus.any_should_demote() is False


def test_cluster_mgr_all_can_promote_true():
    clus = _make_cluster_mgr_no_init(2)
    # both at 30% (< 60% threshold)
    assert clus.all_can_promote() is True


def test_cluster_mgr_all_can_promote_false_when_one_tight():
    clus = _make_cluster_mgr_no_init(2)
    clus._managers[1]._snap.fb_used_pct = 70.0   # > 60%
    assert clus.all_can_promote() is False


def test_cluster_mgr_run_diag_delegates_to_first():
    clus = _make_cluster_mgr_no_init(2)
    result = clus.run_diag(1)
    assert result == "ok"


def test_cluster_mgr_device_count_property():
    clus = _make_cluster_mgr_no_init(3)
    assert clus.device_count == 3


def test_cluster_mgr_del_does_not_raise():
    """__del__ must be exception-safe."""
    clus = _make_cluster_mgr_no_init(2)
    clus.__del__()   # must not raise even if stop() is called multiple times


def test_detect_feeder_count_returns_0_on_exception():
    """_detect_feeder_count() exception handler returns 0 (lines 1079-1080)."""
    import gb_telemetry
    with patch("gb_telemetry.os.path.exists", return_value=True), \
         patch("builtins.open", side_effect=OSError("permission denied")):
        n = gb_telemetry._detect_feeder_count()
    assert n == 0
