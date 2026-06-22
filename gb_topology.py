#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Centralized topology singleton — reads /etc/greenboost/profiles/default.md
once per process and exposes typed constants to all consumer modules.

Replace ad-hoc profile parsing scattered across gb_moe._get_pcie_high_water()
and any llama-server thread logic with a single call to get_topology().

Environment override: GB_TOPOLOGY_PROFILE=/path/to/alt.md (useful in tests).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_PROFILE_PATH = Path(
    os.environ.get("GB_TOPOLOGY_PROFILE", "/etc/greenboost/profiles/default.md")
)

_PCIE_BW_PER_LANE_GBS = {3: 1.0, 4: 2.0, 5: 4.0, 6: 8.0}


@dataclass(frozen=True)
class TopologyProfile:
    # CPU
    p_core_cpus:   List[int] = field(default_factory=list)
    e_core_cpus:   List[int] = field(default_factory=list)
    golden_cpu_min: int = -1
    golden_cpu_max: int = -1
    pcores_max_cpu: int = 15
    l3_cache_mb:    int = 0
    numa_nodes:     int = 1

    # GPU
    vram_gb:             int = 12
    compute_capability:  str = "8.0"
    pcie_gen:            int = 4
    pcie_lanes:          int = 16

    # RAM
    ram_total_gb: int = 64
    ram_speed_mt: int = 3200

    # GreenBoost runtime params
    virtual_vram_gb:   int = 42
    safety_reserve_gb: int = 4
    nvme_swap_gb:      int = 0
    kv_reserve_mb:     int = 2048
    ollama_num_ctx:    int = 8192

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def p_core_count(self) -> int:
        return len(self.p_core_cpus)

    @property
    def e_core_count(self) -> int:
        return len(self.e_core_cpus)

    @property
    def has_golden_cores(self) -> bool:
        return self.golden_cpu_min >= 0 and self.golden_cpu_max >= self.golden_cpu_min

    @property
    def golden_core_cpus(self) -> List[int]:
        if not self.has_golden_cores:
            return []
        return list(range(self.golden_cpu_min, self.golden_cpu_max + 1))

    @property
    def golden_core_count(self) -> int:
        return len(self.golden_core_cpus)

    @property
    def inference_cpus(self) -> List[int]:
        """Best CPUs for latency-sensitive inference threads.

        Priority: golden cores → P-cores → empty (caller falls back to OS default).
        """
        if self.has_golden_cores:
            return self.golden_core_cpus
        return list(self.p_core_cpus)

    @property
    def inference_threads(self) -> int:
        """Recommended thread count for llama-server / llamacpp."""
        cpus = self.inference_cpus
        return max(1, len(cpus))

    @property
    def background_threads(self) -> int:
        """Thread count for prefetch / eviction workers on E-cores."""
        if self.e_core_cpus:
            return min(4, max(1, len(self.e_core_cpus) // 2))
        return max(1, self.p_core_count // 4)

    @property
    def pcie_peak_bw_mb_s(self) -> float:
        """Theoretical single-direction PCIe bandwidth in MB/s."""
        bw_gbs = _PCIE_BW_PER_LANE_GBS.get(self.pcie_gen, 2.0)
        return self.pcie_lanes * bw_gbs * 1024.0

    @property
    def pcie_saturation_mb_s(self) -> float:
        """Tx+Rx saturation threshold (75% of one-direction peak).

        When pcie_tx_mb_s + pcie_rx_mb_s exceeds this, the PCIe bus is the
        dominant bottleneck for T2→T1 DMA prefetch.
        """
        return self.pcie_peak_bw_mb_s * 0.75

    @property
    def physical_vram_mb(self) -> int:
        return self.vram_gb * 1024

    @property
    def virtual_vram_mb(self) -> int:
        return self.virtual_vram_gb * 1024

    @property
    def gpu_cc_major(self) -> int:
        try:
            return int(self.compute_capability.split(".")[0])
        except (ValueError, IndexError):
            return 8

    @property
    def is_blackwell(self) -> bool:
        """True for CUDA CC ≥ 12 (RTX 5xxx / Blackwell)."""
        return self.gpu_cc_major >= 12

    @property
    def kv_step_mb(self) -> int:
        """KV reserve growth step — one quarter of the configured base reserve."""
        return max(128, self.kv_reserve_mb // 4)

    @property
    def kv_max_mb(self) -> int:
        """KV reserve ceiling — 2× base reserve, hard cap at 8192 MB."""
        return min(8192, self.kv_reserve_mb * 2)

    @property
    def safety_max_gb(self) -> int:
        """Safety reserve ceiling under ECC/fault conditions."""
        return min(10, self.safety_reserve_gb + 2)


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_profile(path: Path) -> TopologyProfile:
    raw: dict = {}
    if path.exists():
        try:
            for line in path.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip().strip('"')
                if val:
                    raw[key] = val
        except OSError:
            pass

    def _int(k: str, default: int) -> int:
        try:
            return int(raw[k])
        except (KeyError, ValueError, TypeError):
            return default

    def _str(k: str, default: str) -> str:
        return str(raw.get(k, default))

    def _cpulist(k: str) -> List[int]:
        result: List[int] = []
        for part in raw.get(k, "").split(","):
            part = part.strip()
            if part.isdigit():
                result.append(int(part))
        return result

    return TopologyProfile(
        p_core_cpus=_cpulist("p_core_cpus"),
        e_core_cpus=_cpulist("e_core_cpus"),
        golden_cpu_min=_int("golden_cpu_min", -1),
        golden_cpu_max=_int("golden_cpu_max", -1),
        pcores_max_cpu=_int("pcores_max_cpu", 15),
        l3_cache_mb=_int("l3_cache_mb", 0),
        numa_nodes=_int("numa_nodes", 1),
        vram_gb=_int("vram_gb", 12),
        compute_capability=_str("compute_capability", "8.0"),
        pcie_gen=_int("pcie_gen", 4),
        pcie_lanes=_int("pcie_lanes", 16),
        ram_total_gb=_int("ram_total_gb", 64),
        ram_speed_mt=_int("ram_speed_mt", 3200),
        virtual_vram_gb=_int("virtual_vram_gb", 42),
        safety_reserve_gb=_int("safety_reserve_gb", 4),
        nvme_swap_gb=_int("nvme_swap_gb", 0),
        kv_reserve_mb=_int("kv_reserve_mb", 2048),
        ollama_num_ctx=_int("ollama_num_ctx", 8192),
    )


# ── Singleton ─────────────────────────────────────────────────────────────────

_TOPOLOGY: Optional[TopologyProfile] = None


def get_topology() -> TopologyProfile:
    """Return the cached topology singleton (parsed once per process)."""
    global _TOPOLOGY
    if _TOPOLOGY is None:
        _TOPOLOGY = _parse_profile(_PROFILE_PATH)
    return _TOPOLOGY


def reload_topology(path: Optional[Path] = None) -> TopologyProfile:
    """Force re-parse. Used in tests and after profile regeneration."""
    global _TOPOLOGY
    _TOPOLOGY = _parse_profile(path or _PROFILE_PATH)
    return _TOPOLOGY
