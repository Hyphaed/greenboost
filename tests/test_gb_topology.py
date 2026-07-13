#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_topology , profile parser, derived properties, singleton behavior.

No GPU, no daemon. Uses tmp_path for isolated profile files.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gb_topology import TopologyProfile, _parse_profile, get_topology, reload_topology


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_profile(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "default.md"
    p.write_text(content)
    return p


_FULL_PROFILE = """\
## CPU
logical_cpus: 32
p_cores: 8
e_cores: 16
p_core_cpus: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
e_core_cpus: "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
pcores_max_cpu: 15
golden_cpu_min: 4
golden_cpu_max: 7
l3_cache_mb: 36
numa_nodes: 1

## GPU
gpu_count: 1
gpu_model: "NVIDIA GeForce RTX 5070"
vram_gb: 11
compute_capability: "12.0"
pcie_gen: 4
pcie_lanes: 16
nvlink: false

## RAM
ram_total_gb: 61
ram_type: DDR4
ram_speed_mt: 3600

## GreenBoost Parameters
physical_vram_gb: 11
virtual_vram_gb: 42
safety_reserve_gb: 4
nvme_swap_gb: 31
kv_reserve_mb: 2048
ollama_num_ctx: 32768
"""


# ── Parser ─────────────────────────────────────────────────────────────────────

def test_parse_full_profile(tmp_path):
    p = _write_profile(tmp_path, _FULL_PROFILE)
    t = _parse_profile(p)

    assert t.vram_gb == 11
    assert t.pcie_gen == 4
    assert t.pcie_lanes == 16
    assert t.compute_capability == "12.0"
    assert t.ram_total_gb == 61
    assert t.ram_speed_mt == 3600
    assert t.virtual_vram_gb == 42
    assert t.safety_reserve_gb == 4
    assert t.kv_reserve_mb == 2048
    assert t.ollama_num_ctx == 32768
    assert t.golden_cpu_min == 4
    assert t.golden_cpu_max == 7
    assert t.l3_cache_mb == 36
    assert t.numa_nodes == 1


def test_parse_cpu_lists(tmp_path):
    p = _write_profile(tmp_path, _FULL_PROFILE)
    t = _parse_profile(p)

    assert len(t.p_core_cpus) == 16
    assert t.p_core_cpus[0] == 0
    assert t.p_core_cpus[-1] == 15
    assert len(t.e_core_cpus) == 16
    assert t.e_core_cpus[0] == 16


def test_parse_missing_file_returns_defaults(monkeypatch):
    # Hardware fields no longer inherit reference-box literals: missing file →
    # live detection → sentinel 0 (patched here so the test is machine-independent).
    import gb_topology as gt
    for det in ("_detect_vram_gb", "_detect_ram_total_gb", "_detect_pcie_gen",
                "_detect_ram_speed_mt", "_detect_virtual_vram_gb"):
        monkeypatch.setattr(gt, det, lambda: 0)
    monkeypatch.setattr(gt, "_detect_compute_capability", lambda: "")
    monkeypatch.delenv("GREENBOOST_KV_RESERVE_MB", raising=False)
    t = _parse_profile(Path("/nonexistent/path/profile.md"))
    assert t.vram_gb == 0            # sentinel, never the reference 12
    assert t.pcie_gen == 0           # sentinel
    assert t.pcie_lanes == 16
    assert t.kv_reserve_mb == 512    # %-derived floor (VRAM unknown)
    assert t.safety_reserve_gb == 2  # %-derived floor (RAM unknown)
    assert t.p_core_cpus == []
    assert t.golden_cpu_min == -1


def test_parse_partial_profile(tmp_path, monkeypatch):
    """Only GPU section present , CPU fields fall back to defaults."""
    monkeypatch.delenv("GREENBOOST_KV_RESERVE_MB", raising=False)
    p = _write_profile(tmp_path, "vram_gb: 8\npcie_gen: 3\npcie_lanes: 8\n")
    t = _parse_profile(p)

    assert t.vram_gb == 8
    assert t.pcie_gen == 3
    assert t.pcie_lanes == 8
    assert t.p_core_cpus == []  # not in file → default
    # kv reserve is %-derived from this profile's VRAM, not a 2048 literal
    assert t.kv_reserve_mb == max(512, 8 * 1024 // 6)


def test_parse_quoted_strings(tmp_path):
    p = _write_profile(tmp_path, 'compute_capability: "12.0"\n')
    t = _parse_profile(p)
    assert t.compute_capability == "12.0"


def test_parse_unquoted_strings(tmp_path):
    p = _write_profile(tmp_path, "compute_capability: 8.6\n")
    t = _parse_profile(p)
    assert t.compute_capability == "8.6"


def test_parse_ignores_comment_lines(tmp_path):
    content = "# This is a comment\nvram_gb: 16\n"
    p = _write_profile(tmp_path, content)
    t = _parse_profile(p)
    assert t.vram_gb == 16


def test_parse_ignores_section_headers(tmp_path):
    content = "## GPU Section\nvram_gb: 16\n"
    p = _write_profile(tmp_path, content)
    t = _parse_profile(p)
    assert t.vram_gb == 16


# ── CPU derived properties ────────────────────────────────────────────────────

def test_p_core_count(tmp_path):
    p = _write_profile(tmp_path, _FULL_PROFILE)
    t = _parse_profile(p)
    assert t.p_core_count == 16  # 0-15 inclusive


def test_e_core_count(tmp_path):
    p = _write_profile(tmp_path, _FULL_PROFILE)
    t = _parse_profile(p)
    assert t.e_core_count == 16  # 16-31 inclusive


def test_golden_cores_present(tmp_path):
    p = _write_profile(tmp_path, _FULL_PROFILE)
    t = _parse_profile(p)
    assert t.has_golden_cores is True
    assert t.golden_core_cpus == [4, 5, 6, 7]
    assert t.golden_core_count == 4


def test_golden_cores_absent_when_no_range():
    t = TopologyProfile(golden_cpu_min=-1, golden_cpu_max=-1)
    assert t.has_golden_cores is False
    assert t.golden_core_cpus == []
    assert t.golden_core_count == 0


def test_inference_cpus_prefers_golden(tmp_path):
    p = _write_profile(tmp_path, _FULL_PROFILE)
    t = _parse_profile(p)
    assert t.inference_cpus == [4, 5, 6, 7]


def test_inference_cpus_falls_back_to_pcores(tmp_path):
    content = 'p_core_cpus: "0,1,2,3"\n'
    p = _write_profile(tmp_path, content)
    t = _parse_profile(p)
    assert not t.has_golden_cores
    assert t.inference_cpus == [0, 1, 2, 3]


def test_inference_threads_golden(tmp_path):
    p = _write_profile(tmp_path, _FULL_PROFILE)
    t = _parse_profile(p)
    assert t.inference_threads == 4  # len([4,5,6,7])


def test_background_threads_ecores(tmp_path):
    p = _write_profile(tmp_path, _FULL_PROFILE)
    t = _parse_profile(p)
    # min(4, 16//2) = min(4, 8) = 4
    assert t.background_threads == 4


def test_background_threads_no_ecores():
    t = TopologyProfile(p_core_cpus=[0, 1, 2, 3, 4, 5, 6, 7])
    # No e_cores → p_core_count//4 = 8//4 = 2
    assert t.background_threads == 2


# ── GPU derived properties ────────────────────────────────────────────────────

def test_pcie_peak_bw_gen4_x16():
    t = TopologyProfile(pcie_gen=4, pcie_lanes=16)
    # 16 lanes × 2.0 GB/s × 1024 MB/GB = 32768 MB/s
    assert abs(t.pcie_peak_bw_mb_s - 32768.0) < 1.0


def test_pcie_peak_bw_gen3_x8():
    t = TopologyProfile(pcie_gen=3, pcie_lanes=8)
    # 8 × 1.0 GB/s × 1024 = 8192 MB/s
    assert abs(t.pcie_peak_bw_mb_s - 8192.0) < 1.0


def test_pcie_peak_bw_gen5_x16():
    t = TopologyProfile(pcie_gen=5, pcie_lanes=16)
    # 16 × 4.0 × 1024 = 65536 MB/s
    assert abs(t.pcie_peak_bw_mb_s - 65536.0) < 1.0


def test_pcie_saturation_gen4_x16():
    t = TopologyProfile(pcie_gen=4, pcie_lanes=16)
    # 32768 × 0.75 = 24576 MB/s (matches original _get_pcie_high_water formula)
    assert abs(t.pcie_saturation_mb_s - 24576.0) < 1.0


def test_pcie_saturation_gen3_x8():
    t = TopologyProfile(pcie_gen=3, pcie_lanes=8)
    # 8192 × 0.75 = 6144 MB/s
    assert abs(t.pcie_saturation_mb_s - 6144.0) < 1.0


def test_physical_vram_mb():
    t = TopologyProfile(vram_gb=11)
    assert t.physical_vram_mb == 11 * 1024


def test_virtual_vram_mb():
    t = TopologyProfile(virtual_vram_gb=42)
    assert t.virtual_vram_mb == 42 * 1024


def test_gpu_cc_major_blackwell():
    t = TopologyProfile(compute_capability="12.0")
    assert t.gpu_cc_major == 12
    assert t.is_blackwell is True


def test_gpu_cc_major_ampere():
    t = TopologyProfile(compute_capability="8.6")
    assert t.gpu_cc_major == 8
    assert t.is_blackwell is False


def test_gpu_cc_major_invalid_fallback():
    t = TopologyProfile(compute_capability="bad")
    assert t.gpu_cc_major == 8  # default fallback
    assert t.is_blackwell is False


# ── KV / safety derived properties ───────────────────────────────────────────

def test_kv_step_mb_quarter_of_reserve():
    t = TopologyProfile(kv_reserve_mb=2048)
    assert t.kv_step_mb == 512  # 2048 // 4


def test_kv_step_mb_minimum_128():
    t = TopologyProfile(kv_reserve_mb=256)
    assert t.kv_step_mb == 128  # max(128, 256//4=64) = 128


def test_kv_max_mb_double_reserve():
    t = TopologyProfile(kv_reserve_mb=2048)
    assert t.kv_max_mb == 4096  # min(8192, 2048*2)


def test_kv_max_mb_capped_at_8192():
    t = TopologyProfile(kv_reserve_mb=8192)
    assert t.kv_max_mb == 8192  # min(8192, 16384) = 8192


def test_safety_max_gb_profile_plus_two():
    t = TopologyProfile(safety_reserve_gb=4)
    assert t.safety_max_gb == 6  # 4 + 2


def test_safety_max_gb_capped_at_10():
    t = TopologyProfile(safety_reserve_gb=10)
    assert t.safety_max_gb == 10  # min(10, 12) = 10


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_get_topology_returns_same_instance():
    import gb_topology
    gb_topology._TOPOLOGY = None  # force fresh load
    t1 = get_topology()
    t2 = get_topology()
    assert t1 is t2


def test_reload_topology_returns_fresh_instance(tmp_path):
    import gb_topology
    p = _write_profile(tmp_path, "vram_gb: 24\n")
    gb_topology._TOPOLOGY = None

    t1 = get_topology()
    t2 = reload_topology(p)
    # After reload the singleton is updated to the new file
    assert get_topology() is t2
    assert t2.vram_gb == 24

    # Restore singleton for subsequent tests
    gb_topology._TOPOLOGY = None


def test_get_topology_after_reload_reflects_new_values(tmp_path):
    import gb_topology
    p = _write_profile(tmp_path, "pcie_gen: 5\npcie_lanes: 16\n")
    reload_topology(p)
    t = get_topology()
    assert t.pcie_gen == 5
    # Restore
    gb_topology._TOPOLOGY = None


def test_get_topology_on_this_machine():
    """Integration: the installed profile is readable and returns a valid object."""
    import gb_topology
    gb_topology._TOPOLOGY = None
    t = get_topology()
    # RTX 5070 / i9-14900KF profile asserts
    assert t.pcie_gen == 4
    assert t.pcie_lanes == 16
    assert t.vram_gb == 11
    assert t.is_blackwell is True
    assert t.inference_cpus == [4, 5, 6, 7]  # golden cores
    assert abs(t.pcie_saturation_mb_s - 24576.0) < 1.0
