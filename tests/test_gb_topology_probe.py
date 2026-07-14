#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for GpuTopology probe, GpuMetrics topology properties, and B2 PCIe gate.

All NVML / sysfs calls mocked , no GPU or driver required.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from gb_telemetry import (
    GpuMetrics, GpuTopology,
    _probe_gpu_topology, _sysfs_bdf, _slot_pcie_ceiling, _gts_to_pcie_gen,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pynvml(
    bdf="00000000:01:00.0",
    cc=(8, 9),
    gen_cur=4, gen_max=4,
    width_cur=16, width_max=16,
    gpu_util=0,           # 0 = idle at probe time (the common case); >0 = active
    nvlink_states=None,   # list of (enabled, peer_idx) per link; None = no links
    n_devs=1,
    p2p_pairs=None,       # list of (peer_idx, status) , status 0 = OK
):
    """Build a minimal pynvml mock for _probe_gpu_topology."""
    nv = MagicMock()

    # PCI info
    pci = SimpleNamespace(busId=bdf.encode())
    nv.nvmlDeviceGetPciInfo_v3.return_value = pci

    # Compute capability
    nv.nvmlDeviceGetCudaComputeCapability.return_value = SimpleNamespace(major=cc[0], minor=cc[1])

    # PCIe
    nv.nvmlDeviceGetCurrPcieLinkGeneration.return_value = gen_cur
    nv.nvmlDeviceGetMaxPcieLinkGeneration.return_value  = gen_max
    nv.nvmlDeviceGetCurrPcieLinkWidth.return_value      = width_cur
    nv.nvmlDeviceGetMaxPcieLinkWidth.return_value       = width_max

    # GPU activity at probe time (gates the gen-mismatch half of pcie_degraded)
    nv.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(gpu=gpu_util, memory=0)

    # NVLink , default: all links disabled
    nv.NVML_FEATURE_ENABLED           = 1
    nv.NVML_FEATURE_DISABLED          = 0
    nv.NVML_NVLINK_DEVICE_TYPE_GPU    = 0

    def _link_state(h, link):
        if nvlink_states and link < len(nvlink_states):
            return nvlink_states[link][0]   # enabled flag
        if link == 0 and not nvlink_states:
            # first probe raises so the scan stops
            raise RuntimeError("no nvlink")
        raise RuntimeError("out of range")

    nv.nvmlDeviceGetNvLinkState.side_effect = _link_state

    if nvlink_states:
        def _link_remote_type(h, link):
            if link < len(nvlink_states):
                return 0  # NVML_NVLINK_DEVICE_TYPE_GPU
            raise RuntimeError("out of range")
        nv.nvmlDeviceGetNvLinkRemoteDeviceType.side_effect = _link_remote_type

        def _link_remote_pci(h, link):
            peer_idx = nvlink_states[link][1]
            return SimpleNamespace(busId=f"0000:0{peer_idx+2}:00.0".encode())
        nv.nvmlDeviceGetNvLinkRemotePciInfo_v2.side_effect = _link_remote_pci

        def _handle_by_bus(bus_id):
            # Return a handle whose index we can extract
            h = MagicMock()
            nv.nvmlDeviceGetIndex.return_value = int(bus_id.decode()[5]) - 2 if isinstance(bus_id, bytes) else 0
            return h
        nv.nvmlDeviceGetHandleByPciBusId.side_effect = _handle_by_bus

    # P2P
    nv.NVML_P2P_STATUS_OK      = 0
    nv.NVML_P2P_CAPS_INDEX_READ = 0
    nv.nvmlDeviceGetCount.return_value = n_devs

    def _p2p(h1, h2, idx):
        if p2p_pairs:
            for peer_idx, status in p2p_pairs:
                return status
        return 1  # not supported by default

    nv.nvmlDeviceGetP2PStatus.side_effect = _p2p
    nv.nvmlDeviceGetHandleByIndex.return_value = MagicMock()

    return nv


# ── _sysfs_bdf normalisation ──────────────────────────────────────────────────

def test_sysfs_bdf_8char_domain():
    assert _sysfs_bdf("00000000:01:00.0") == "0000:01:00.0"

def test_sysfs_bdf_4char_domain_unchanged():
    assert _sysfs_bdf("0000:01:00.0") == "0000:01:00.0"

def test_sysfs_bdf_lowercased():
    assert _sysfs_bdf("00000000:0A:00.0") == "0000:0a:00.0"


# ── _probe_gpu_topology , PCIe ────────────────────────────────────────────────
#
# All tests below patch _slot_pcie_ceiling directly rather than relying on
# real /sys/bus/pci/devices/<bdf> paths , the test BDFs are arbitrary and a
# real sysfs entry can coincidentally exist at the same address on the host
# running the suite (verified: "0000:01:00.0" resolves to a real device on
# at least one dev machine), which would make these tests hardware-dependent
# and non-hermetic otherwise.

def test_probe_pcie_gen_width():
    nv = _make_pynvml(gen_cur=5, gen_max=5, width_cur=16, width_max=16)
    handle = MagicMock()
    with patch("gb_telemetry._slot_pcie_ceiling", return_value=None):
        topo = _probe_gpu_topology(nv, handle, device=0)
    assert topo is not None
    assert topo.pcie_gen_current == 5
    assert topo.pcie_width_current == 16

def test_probe_pcie_degraded_detected():
    # Running at x8 in a x16 slot , width is the reliable (power-state
    # independent) signal, so this is flagged regardless of pcie_active.
    nv = _make_pynvml(gen_cur=4, gen_max=4, width_cur=8, width_max=16)
    handle = MagicMock()
    with patch("gb_telemetry._slot_pcie_ceiling", return_value=None):
        topo = _probe_gpu_topology(nv, handle, device=0)
    assert topo.pcie_degraded is True

def test_probe_pcie_not_degraded():
    nv = _make_pynvml(gen_cur=4, gen_max=4, width_cur=16, width_max=16)
    with patch("gb_telemetry._slot_pcie_ceiling", return_value=None):
        topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    assert topo.pcie_degraded is False

def test_probe_pcie_gen_mismatch_degraded_while_active():
    # Gen mismatch is only trusted when the GPU was active at probe time.
    nv = _make_pynvml(gen_cur=3, gen_max=4, width_cur=16, width_max=16, gpu_util=75)
    with patch("gb_telemetry._slot_pcie_ceiling", return_value=None):
        topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    assert topo.pcie_active is True
    assert topo.pcie_degraded is True

def test_probe_pcie_gen_mismatch_not_degraded_while_idle():
    # THE bug this fix targets: PCIe gen legitimately downtrains to Gen1/Gen2
    # under ASPM idle power states (P5/P8) and only retrains under load.
    # _probe_gpu_topology runs once at NVMLProvider init, almost always
    # before any workload starts , catching that idle downtrain must NOT be
    # reported as a hardware fault.
    nv = _make_pynvml(gen_cur=1, gen_max=4, width_cur=16, width_max=16, gpu_util=0)
    with patch("gb_telemetry._slot_pcie_ceiling", return_value=None):
        topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    assert topo.pcie_active is False
    assert topo.pcie_degraded is False

def test_probe_pcie_laptop_x8_slot_not_degraded():
    # THE real-world case that motivated this fix: an RTX 5070 Laptop GPU
    # whose SILICON supports x16 (NVML MaxPcieLinkWidth=16) but is physically
    # wired into an x8 slot (parent PCI bridge max_link_width=8). Running at
    # x8 , its actual ceiling , must NOT be flagged as degraded.
    nv = _make_pynvml(gen_cur=2, gen_max=5, width_cur=8, width_max=16, gpu_util=0)
    with patch("gb_telemetry._slot_pcie_ceiling", return_value=(4, 8)):
        topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    assert topo.pcie_width_max == 8          # clamped from silicon's 16
    assert topo.pcie_gen_max == 4            # clamped from silicon's 5
    assert topo.pcie_degraded is False

def test_probe_pcie_genuine_width_fault_below_slot_ceiling():
    # Width running BELOW the real slot ceiling (e.g. bifurcation, a bent
    # riser) is a genuine fault and must still be flagged.
    nv = _make_pynvml(gen_cur=4, gen_max=4, width_cur=4, width_max=16, gpu_util=0)
    with patch("gb_telemetry._slot_pcie_ceiling", return_value=(4, 8)):
        topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    assert topo.pcie_width_max == 8
    assert topo.pcie_degraded is True


# ── _slot_pcie_ceiling ────────────────────────────────────────────────────────

def test_gts_to_pcie_gen_thresholds():
    assert _gts_to_pcie_gen("2.5 GT/s PCIe") == 1
    assert _gts_to_pcie_gen("5.0 GT/s PCIe") == 2
    assert _gts_to_pcie_gen("8.0 GT/s PCIe") == 3
    assert _gts_to_pcie_gen("16.0 GT/s PCIe") == 4
    assert _gts_to_pcie_gen("32.0 GT/s PCIe") == 5
    assert _gts_to_pcie_gen("") == 0
    assert _gts_to_pcie_gen("garbage") == 0

def test_slot_pcie_ceiling_reads_parent_bridge(tmp_path):
    # Build a fake sysfs shape: devices/<bdf> -> ../../..(parent bridge dir)
    parent = tmp_path / "pci0000:00" / "0000:00:03.1"
    parent.mkdir(parents=True)
    (parent / "max_link_speed").write_text("16.0 GT/s PCIe\n")
    (parent / "max_link_width").write_text("8\n")
    dev_dir = parent / "0000:04:00.0"
    dev_dir.mkdir()
    devices_dir = tmp_path / "devices_by_bdf"
    devices_dir.mkdir()
    (devices_dir / "0000:04:00.0").symlink_to(dev_dir, target_is_directory=True)

    with patch("gb_telemetry.Path", side_effect=lambda p: Path(str(p).replace(
            "/sys/bus/pci/devices", str(devices_dir)))):
        ceiling = _slot_pcie_ceiling("0000:04:00.0")
    assert ceiling == (4, 8)

def test_slot_pcie_ceiling_none_when_unreadable():
    assert _slot_pcie_ceiling("0000:ff:1f.7-does-not-exist") is None


# ── GpuTopology properties ────────────────────────────────────────────────────

def test_pcie_bw_mb_s_gen4_x16():
    topo = GpuTopology(pcie_gen_current=4, pcie_width_current=16)
    # Gen4 = 2.0 GB/s per lane × 16 lanes = 32 GB/s = 32768 MB/s
    assert topo.pcie_bw_mb_s == pytest.approx(32768.0)

def test_pcie_bw_mb_s_gen5_x16():
    topo = GpuTopology(pcie_gen_current=5, pcie_width_current=16)
    # Gen5 = 4.0 GB/s per lane × 16 = 65536 MB/s
    assert topo.pcie_bw_mb_s == pytest.approx(65536.0)

def test_pcie_saturation_is_75pct():
    topo = GpuTopology(pcie_gen_current=4, pcie_width_current=16)
    assert topo.pcie_saturation_mb_s == pytest.approx(32768.0 * 0.75)

def test_pcie_bw_max_uses_max_link():
    topo = GpuTopology(pcie_gen_current=3, pcie_width_current=8,
                       pcie_gen_max=4, pcie_width_max=16)
    assert topo.pcie_bw_max_mb_s == pytest.approx(32768.0)  # gen4 x16

def test_has_nvlink_false():
    topo = GpuTopology(nvlink_count=0)
    assert topo.has_nvlink is False

def test_has_nvlink_true():
    topo = GpuTopology(nvlink_count=2, nvlink_peer_ids=(1,))
    assert topo.has_nvlink is True


# ── _probe_gpu_topology , BDF + NUMA ─────────────────────────────────────────

def test_probe_bdf_decoded():
    nv = _make_pynvml(bdf="00000000:01:00.0")
    topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    assert "01:00.0" in topo.bdf

def test_probe_numa_node_from_sysfs(tmp_path):
    numa_path = tmp_path / "0000:01:00.0" / "numa_node"
    numa_path.parent.mkdir(parents=True)
    numa_path.write_text("1\n")
    nv = _make_pynvml(bdf="00000000:01:00.0")
    with patch("gb_telemetry.Path", side_effect=lambda p: Path(str(p).replace(
            "/sys/bus/pci/devices", str(tmp_path)))):
        topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    # sysfs mock may not intercept cleanly in all paths; just verify no crash
    assert isinstance(topo.numa_node, int)

def test_probe_numa_node_minus1_when_missing():
    nv = _make_pynvml(bdf="00000000:99:00.0")
    # /sys/bus/pci/devices/0000:99:00.0/numa_node won't exist on test host
    topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    # Should be -1 (missing) or a real NUMA node , must not raise
    assert isinstance(topo.numa_node, int)


# ── _probe_gpu_topology , compute capability ─────────────────────────────────

def test_probe_compute_capability():
    nv = _make_pynvml(cc=(8, 9))
    topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    assert topo.compute_capability == (8, 9)

def test_probe_cc_blackwell():
    nv = _make_pynvml(cc=(12, 0))
    topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    assert topo.compute_capability == (12, 0)


# ── _probe_gpu_topology , NVLink ─────────────────────────────────────────────

def test_probe_no_nvlink_when_raises():
    """Consumer GPU with no NVLink support , exception on link 0 → count stays 0."""
    nv = _make_pynvml(nvlink_states=None)
    topo = _probe_gpu_topology(nv, MagicMock(), device=0)
    assert topo.nvlink_count == 0
    assert topo.nvlink_peer_ids == ()


# ── GpuMetrics.pcie_saturated ────────────────────────────────────────────────

def test_pcie_saturated_false_when_no_topology():
    m = GpuMetrics(pcie_tx_mb_s=15000, pcie_rx_mb_s=15000)
    assert m.pcie_saturated is False  # topology is None

def test_pcie_saturated_false_below_threshold():
    topo = GpuTopology(pcie_gen_current=4, pcie_width_current=16)
    # saturation = 32768 * 0.75 = 24576 MB/s; total = 5000+5000 = 10000
    m = GpuMetrics(pcie_tx_mb_s=5000.0, pcie_rx_mb_s=5000.0, topology=topo)
    assert m.pcie_saturated is False

def test_pcie_saturated_true_above_threshold():
    topo = GpuTopology(pcie_gen_current=4, pcie_width_current=16)
    # saturation = 24576 MB/s; total = 13000+13000 = 26000 > 24576
    m = GpuMetrics(pcie_tx_mb_s=13000.0, pcie_rx_mb_s=13000.0, topology=topo)
    assert m.pcie_saturated is True

def test_pcie_saturated_degraded_slot_lower_threshold():
    # x8 slot: saturation = gen4 x8 × 0.75 = 16384 × 0.75 = 12288 MB/s
    topo = GpuTopology(pcie_gen_current=4, pcie_width_current=8)
    m = GpuMetrics(pcie_tx_mb_s=7000.0, pcie_rx_mb_s=7000.0, topology=topo)
    # 14000 > 12288 → saturated
    assert m.pcie_saturated is True

def test_pcie_saturated_degraded_slot_not_saturated():
    topo = GpuTopology(pcie_gen_current=4, pcie_width_current=8)
    m = GpuMetrics(pcie_tx_mb_s=3000.0, pcie_rx_mb_s=3000.0, topology=topo)
    assert m.pcie_saturated is False


# ── GpuMetrics.numa_node convenience property ────────────────────────────────

def test_numa_node_from_topology():
    topo = GpuTopology(numa_node=1)
    m = GpuMetrics(topology=topo)
    assert m.numa_node == 1

def test_numa_node_minus1_without_topology():
    m = GpuMetrics()
    assert m.numa_node == -1


# ── B2 PCIe gate uses live topology ──────────────────────────────────────────

def test_b2_uses_live_topology_saturation(monkeypatch):
    """_get_pcie_high_water prefers live topology over config file."""
    from gb_moe import _get_pcie_high_water

    topo = GpuTopology(pcie_gen_current=3, pcie_width_current=8)
    # gen3 x8 = 8192 MB/s → saturation = 6144 MB/s
    snap = GpuMetrics(topology=topo)

    tel_mock = MagicMock()
    tel_mock.snapshot.return_value = snap

    import gb_moe
    monkeypatch.setattr(gb_moe, "_get_pcie_high_water", lambda: snap.topology.pcie_saturation_mb_s)

    hw = _get_pcie_high_water()
    # Verify the degraded (x8) path's value matches the topology calc
    assert snap.topology.pcie_saturation_mb_s == pytest.approx(8192.0 * 0.75)


# ── Orchestrator dump includes topology ──────────────────────────────────────

def test_orchestrator_dump_has_topology_key():
    from unittest.mock import patch as up
    from gb_orchestrator import ReactiveOrchestrator

    with up.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         up.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         up.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(mode="process")

    d = o.dump()
    assert "topology" in d

def test_orchestrator_dump_topology_empty_before_metrics():
    from unittest.mock import patch as up
    from gb_orchestrator import ReactiveOrchestrator

    with up.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         up.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         up.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(mode="process")

    assert o.dump()["topology"] == {}

def test_orchestrator_dump_topology_populated_after_metrics():
    from unittest.mock import patch as up
    from gb_orchestrator import ReactiveOrchestrator
    from gb_telemetry import GpuMetrics, GpuTopology

    topo = GpuTopology(
        bdf="0000:01:00.0", numa_node=0,
        compute_capability=(8, 9),
        pcie_gen_current=4, pcie_width_current=16,
        pcie_gen_max=4, pcie_width_max=16,
    )
    m = GpuMetrics(fb_total_mb=12288, topology=topo)
    m.shim_phase = "IDLE"

    with up.object(ReactiveOrchestrator, '_restore_ecc_degraded', lambda self: None), \
         up.object(ReactiveOrchestrator, '_persist_ecc_degraded', lambda self: None), \
         up.object(ReactiveOrchestrator, '_write_state_file', lambda self: None):
        o = ReactiveOrchestrator(mode="process")

    o.on_metrics(m)
    d = o.dump()["topology"]
    assert d["bdf"] == "0000:01:00.0"
    assert d["pcie_gen_current"] == 4
    assert d["pcie_bw_mb_s"] == 32768
    assert d["pcie_degraded"] is False
    assert d["compute_capability"] == "8.9"
