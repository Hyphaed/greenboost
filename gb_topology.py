#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Centralized topology singleton , reads /etc/greenboost/profiles/default.md
once per process and exposes typed constants to all consumer modules.

Replace ad-hoc profile parsing scattered across gb_moe._get_pcie_high_water()
and any llama-server thread logic with a single call to get_topology().

Environment override: GB_TOPOLOGY_PROFILE=/path/to/alt.md (useful in tests).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

_PROFILE_PATH = Path(
    os.environ.get("GB_TOPOLOGY_PROFILE", "/etc/greenboost/profiles/default.md")
)

_PCIE_BW_PER_LANE_GBS = {3: 1.0, 4: 2.0, 5: 4.0, 6: 8.0}


def _warn(msg: str) -> None:
    print(f"[gb_topology] WARNING: {msg}", file=sys.stderr, flush=True)


# ── Live hardware detection ───────────────────────────────────────────────────
# CLAUDE.md rule: hardware-shaped fields never inherit the reference box's
# numbers. Resolution order: profile value → live detection (NVML /
# /proc/meminfo) → sentinel 0/"" with a LOUD warning. Cached per process.

_DETECT_CACHE: dict = {}


def _cached(key: str, fn):
    if key not in _DETECT_CACHE:
        _DETECT_CACHE[key] = fn()
    return _DETECT_CACHE[key]


def _nvml():
    """(pynvml module, device-0 handle) or (None, None). Cached, never raises.
    Falls back to the vendored gb_nvml_ctypes binding (same as gb_telemetry)
    on nodes without the pynvml package."""
    def _init():
        try:
            try:
                import pynvml
            except ImportError:
                import gb_nvml_ctypes as pynvml
            pynvml.nvmlInit()
            return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return None, None
    return _cached("nvml", _init)


def _detect_vram_mb() -> int:
    def _go():
        nv, h = _nvml()
        if h is not None:
            try:
                return int(nv.nvmlDeviceGetMemoryInfo(h).total // (1024 * 1024))
            except Exception:
                pass
        try:  # pynvml missing — one nvidia-smi probe, cached for the process
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            return int(out.stdout.splitlines()[0].strip())
        except Exception:
            return 0
    return _cached("vram_mb", _go)


def _detect_vram_gb() -> int:
    mb = _detect_vram_mb()
    return int(round(mb / 1024)) if mb else 0


def _detect_compute_capability() -> str:
    def _go():
        nv, h = _nvml()
        if h is not None:
            try:
                major, minor = nv.nvmlDeviceGetCudaComputeCapability(h)
                return f"{major}.{minor}"
            except Exception:
                pass
        return ""
    return _cached("cc", _go)


def _detect_pcie_gen() -> int:
    def _go():
        nv, h = _nvml()
        if h is not None:
            try:
                return int(nv.nvmlDeviceGetMaxPcieLinkGeneration(h))
            except Exception:
                pass
        return 0
    return _cached("pcie_gen", _go)


def _detect_pcie_lanes() -> int:
    """Link width (lanes) via NVML — mirrors _detect_pcie_gen. A card riser'd
    to x8/x4 must not be assumed x16: pcie_peak_bw_mb_s (feeds synapse tok/s
    estimates and telemetry) is wrong by exactly that factor otherwise.

    Clamped against the PHYSICAL SLOT's ceiling (parent PCI bridge), not just
    the GPU silicon's own max , NVML's MaxPcieLinkWidth reports what the CHIP
    supports regardless of how many lanes the slot actually wires (e.g. a
    x16-capable laptop dGPU soldered into a x8 slot , the exact false
    "gen4x16" reading fixed in gb_telemetry.py 2026-07-14). Reuses
    gb_telemetry's _slot_pcie_ceiling/_sysfs_bdf , same sysfs read, ONE
    implementation, not a second copy. (MaxPcieLinkGeneration needs no such
    clamp: nvml.h documents it already accounts for the bus's link *speed*
    ceiling, only *width* has this gap.)"""
    def _go():
        nv, h = _nvml()
        if h is None:
            return 0
        try:
            width = int(nv.nvmlDeviceGetMaxPcieLinkWidth(h))
        except Exception:
            return 0
        try:
            from gb_telemetry import _slot_pcie_ceiling, _sysfs_bdf
            pci_fn = getattr(nv, "nvmlDeviceGetPciInfo_v3", None) or nv.nvmlDeviceGetPciInfo
            raw = pci_fn(h).busId
            bdf = raw.decode("ascii", errors="ignore").strip("\x00") if isinstance(raw, bytes) else str(raw)
            ceiling = _slot_pcie_ceiling(_sysfs_bdf(bdf))
            if ceiling is not None:
                width = min(width, ceiling[1])
        except Exception:
            pass
        return width
    return _cached("pcie_lanes", _go)


def _detect_ram_total_gb() -> int:
    def _go():
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(round(int(line.split()[1]) / (1024 * 1024)))
        except (OSError, ValueError, IndexError):
            pass
        return 0
    return _cached("ram_total_gb", _go)


def _detect_ram_speed_mt() -> int:
    def _go():
        try:  # dmidecode is root-only; unprivileged failure → sentinel 0
            out = subprocess.run(["dmidecode", "-t", "memory"],
                                 capture_output=True, text=True, timeout=10)
            for line in out.stdout.splitlines():
                line = line.strip()
                if line.startswith(("Configured Memory Speed:", "Speed:")):
                    tok = line.split(":", 1)[1].strip().split()
                    if tok and tok[0].isdigit():
                        return int(tok[0])
        except Exception:
            pass
        return 0
    return _cached("ram_speed_mt", _go)


def _detect_vram_bw_gb_s() -> float:
    """T1 VRAM bandwidth GB/s = mem clock × 2 (DDR) × bus width / 8. NVML first,
    nvidia-smi (clocks.max.memory, memory.bus_width) fallback. 0 when neither
    works (rule: no reference-box constant, caller keeps 0 sentinel)."""
    def _go():
        nv, h = _nvml()
        if h is not None:
            try:
                clk = nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_MEM)
                bus = nv.nvmlDeviceGetMemoryBusWidth(h)
                return round(clk * 2 * bus / 8 / 1000.0, 1)
            except Exception:
                pass
        try:  # pynvml missing — one nvidia-smi probe, cached for the process
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.max.memory,memory.bus_width",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            clk, bus = (float(x) for x in out.stdout.splitlines()[0].split(","))
            return round(clk * 2 * bus / 8 / 1000.0, 1)
        except Exception:
            return 0.0
    return _cached("vram_bw_gb_s", _go)


def _detect_net_link_mbps() -> int:
    """Primary NIC link speed in Mbps , the ethernet link that carries cluster
    transfers. Prefers the interface owning the default route (the real cluster
    path), else the fastest carrier-up physical interface. sysfs only, 0 when
    unknown (e.g. a virtual/wireless link with no `speed` attribute)."""
    def _read_speed(iface: str) -> int:
        try:
            v = int(Path(f"/sys/class/net/{iface}/speed").read_text().strip())
            return v if v > 0 else 0     # -1 / huge sentinel on down/virtual links
        except (OSError, ValueError):
            return 0

    def _go():
        # 1) interface of the default route
        try:
            for line in Path("/proc/net/route").read_text().splitlines()[1:]:
                f = line.split()
                if len(f) > 1 and f[1] == "00000000":       # destination 0.0.0.0
                    s = _read_speed(f[0])
                    if s:
                        return s
        except OSError:
            pass
        # 2) fastest carrier-up physical interface
        best = 0
        try:
            for p in Path("/sys/class/net").iterdir():
                if p.name == "lo" or (p / "device").exists() is False:
                    continue  # skip loopback + virtual (no backing device)
                try:
                    if (p / "carrier").read_text().strip() != "1":
                        continue
                except OSError:
                    continue
                best = max(best, _read_speed(p.name))
        except OSError:
            pass
        return best
    return _cached("net_link_mbps", _go)


def _detect_virtual_vram_gb() -> int:
    """Physical VRAM + kernel T2 pool (pool_brief) — the node's real virtual span."""
    def _go():
        vram = _detect_vram_gb()
        if not vram:
            return 0
        t2 = 0
        try:
            m = re.search(r"T2:\d+/(\d+)GB",
                          Path("/sys/class/greenboost/greenboost/pool_brief").read_text())
            if m:
                t2 = int(m.group(1))
        except OSError:
            pass
        return vram + t2
    return _cached("virtual_vram_gb", _go)


# ── Shared %-derived reserves (rule: budgets/reserves are %-of-topology) ──────

def compute_reserve_gb(physical_vram_mb: float) -> float:
    """Per-device compute/graph workspace reserve: max(0.75 GiB, 8% of VRAM).

    THE shared derivation for gb_synapse (--tensor-split / layer fitting) and
    gb_cluster.feeder_env (per-node gb-quant budgets) — one formula, so a new
    card size never needs a new literal anywhere."""
    return max(0.75, 0.08 * physical_vram_mb / 1024.0)


def hbm_headroom_mb(physical_vram_mb: "int | None" = None) -> int:
    """T1 activation/latent headroom: max(512, 10% of physical VRAM MB).

    Single derivation for gb_init and gb_model_tier (previously two divergent
    literals: 1500 and 1024 MB, both shaped by the reference card)."""
    if physical_vram_mb is None:
        physical_vram_mb = get_topology().physical_vram_mb or _detect_vram_mb()
    if physical_vram_mb <= 0:
        _warn("hbm_headroom_mb: VRAM undetected — falling back to 1024 MB")
        return 1024
    return max(512, int(physical_vram_mb * 0.10))


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

    # GPU — 0/"" sentinels: real values come from the profile or live
    # detection in _parse_profile (rule: never inherit reference-box numbers)
    vram_gb:             int = 0
    gpu_model:           str = ""
    compute_capability:  str = ""
    vram_bw_gb_s:        float = 0.0   # T1 VRAM bandwidth (mem clock × bus width)
    pcie_gen:            int = 0
    pcie_lanes:          int = 16

    # RAM — 0 sentinels, same rule as the GPU block
    ram_total_gb: int = 0              # T2 capacity
    ram_speed_mt: int = 0              # T2 speed (transfer rate MT/s)

    # Network — the ethernet link that carries cluster transfers (0 = unknown)
    net_link_mbps: int = 0             # primary NIC link speed (physical ceiling)

    # GreenBoost runtime params — reserves are %-derived in _parse_profile
    virtual_vram_gb:   int = 0
    safety_reserve_gb: int = 0
    nvme_swap_gb:      int = 0
    kv_reserve_mb:     int = 0
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
        """KV reserve growth step , one quarter of the configured base reserve."""
        return max(128, self.kv_reserve_mb // 4)

    @property
    def kv_max_mb(self) -> int:
        """KV reserve ceiling: 2× base reserve, capped at 25% of THIS card's
        VRAM (never a fixed 8192 that would let an 8 GB card reserve most of
        its VRAM for KV). Falls back to 8192 only when VRAM is undetected."""
        vram_cap = max(1024, self.physical_vram_mb // 4) if self.physical_vram_mb > 0 else 8192
        return min(vram_cap, self.kv_reserve_mb * 2)

    @property
    def safety_max_gb(self) -> int:
        """Safety reserve ceiling under ECC/fault conditions: base+2, capped at
        25% of THIS node's RAM (never a fixed 10 GB). Falls back to 10 when RAM
        is undetected."""
        ram_cap = max(4, self.ram_total_gb // 4) if self.ram_total_gb > 0 else 10
        return min(ram_cap, self.safety_reserve_gb + 2)


# ── Parser ────────────────────────────────────────────────────────────────────

def _profile_kv(text: str) -> dict:
    """Parse profile .md `key: value` lines into a raw string dict."""
    raw: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"')
        if val:
            raw[key] = val
    return raw


def _build_profile(raw: dict, *, live: bool, src: str) -> TopologyProfile:
    """Build a TopologyProfile from a parsed profile dict.

    live=True  → local node: missing hardware fields fall back to live NVML/
                 dmidecode detection (then a 0 sentinel, LOUD).
    live=False → REMOTE node's profile text: NO live detection (it would
                 describe the HOST, not the feeder) — missing fields stay 0.
    Reserve derivations are %-of-topology in both modes, driven by the parsed
    (remote or local) vram/ram — never inherited reference-box absolutes.
    """
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

    def _hw_int(k: str, detect) -> int:
        """Hardware-shaped field: profile → (live) detection → 0 sentinel, LOUD."""
        try:
            return int(raw[k])
        except (KeyError, ValueError, TypeError):
            pass
        v = detect() if live else 0
        if not v:
            _warn(f"{k}: missing from {src} and not detectable — sentinel 0")
        return v

    def _hw_float(k: str, detect) -> float:
        """Float variant of _hw_int (e.g. bandwidth GB/s)."""
        try:
            return float(raw[k])
        except (KeyError, ValueError, TypeError):
            pass
        v = detect() if live else 0.0
        if not v:
            _warn(f"{k}: missing from {src} and not detectable — sentinel 0")
        return v

    vram_gb = _hw_int("vram_gb", _detect_vram_gb)
    ram_total_gb = _hw_int("ram_total_gb", _detect_ram_total_gb)

    cc = _str("compute_capability", "")
    if not cc and live:
        cc = _detect_compute_capability()
    if not cc:
        _warn(f'compute_capability: missing from {src} and not detectable — sentinel ""')

    # Reserves are %-derived from THIS node's topology, never inherited absolutes:
    #   safety = max(2 GB, 6% of RAM); kv = profile → env (>0, local only) →
    #   max(512, VRAM/6) (VRAM/6 reproduces 2048 MB on the 12 GB reference card).
    safety_reserve_gb = _int("safety_reserve_gb", 0)
    if safety_reserve_gb <= 0:
        safety_reserve_gb = max(2, ram_total_gb * 6 // 100)
    kv_reserve_mb = _int("kv_reserve_mb", 0)
    if kv_reserve_mb <= 0 and live:
        try:  # kmod/env-provided reserve wins when present (and > 0) — local only
            kv_reserve_mb = int(os.environ.get("GREENBOOST_KV_RESERVE_MB", "0"))
        except ValueError:
            kv_reserve_mb = 0
    if kv_reserve_mb <= 0:
        kv_reserve_mb = max(512, vram_gb * 1024 // 6)

    return TopologyProfile(
        p_core_cpus=_cpulist("p_core_cpus"),
        e_core_cpus=_cpulist("e_core_cpus"),
        golden_cpu_min=_int("golden_cpu_min", -1),
        golden_cpu_max=_int("golden_cpu_max", -1),
        pcores_max_cpu=_int("pcores_max_cpu", 15),
        l3_cache_mb=_int("l3_cache_mb", 0),
        numa_nodes=_int("numa_nodes", 1),
        vram_gb=vram_gb,
        gpu_model=_str("gpu_model", ""),
        compute_capability=cc,
        vram_bw_gb_s=_hw_float("vram_bandwidth_gb_s", _detect_vram_bw_gb_s),
        pcie_gen=_hw_int("pcie_gen", _detect_pcie_gen),
        # 16 (not the usual 0 sentinel) is the final fallback here: pcie_lanes
        # is a divisor-like factor in pcie_peak_bw_mb_s downstream , 0 lanes
        # would zero out bandwidth estimates instead of just being imprecise.
        pcie_lanes=_hw_int("pcie_lanes", _detect_pcie_lanes) or 16,
        ram_total_gb=ram_total_gb,
        ram_speed_mt=_hw_int("ram_speed_mt", _detect_ram_speed_mt),
        net_link_mbps=_hw_int("net_link_mbps", _detect_net_link_mbps),
        virtual_vram_gb=_hw_int("virtual_vram_gb", _detect_virtual_vram_gb),
        safety_reserve_gb=safety_reserve_gb,
        nvme_swap_gb=_int("nvme_swap_gb", 0),
        kv_reserve_mb=kv_reserve_mb,
        ollama_num_ctx=_int("ollama_num_ctx", 8192),
    )


def parse_profile_text(text: str, src: str = "<remote>") -> TopologyProfile:
    """Pure parse of a profile .md's TEXT with NO live hardware detection.

    For a REMOTE node's profile fetched over the cluster fabric — live NVML/
    dmidecode detection here would describe the HOST, not the feeder, so missing
    hardware fields stay 0 sentinels. Reserve fields are still %-derived from the
    remote node's own parsed vram/ram."""
    return _build_profile(_profile_kv(text), live=False, src=src)


def _parse_profile(path: Path) -> TopologyProfile:
    text = ""
    if path.exists():
        try:
            text = path.read_text(errors="replace")
        except OSError:
            pass
    return _build_profile(_profile_kv(text), live=True, src=str(path))


def topology_dict(tp: TopologyProfile) -> dict:
    """Flat dict of a TopologyProfile + the derived properties consumers need
    (p_core_count, inference_threads, pcie_peak_bw_mb_s, is_blackwell). Used for
    the cluster topology JSON cache and MCP output."""
    d = asdict(tp)
    d.update(
        p_core_count=tp.p_core_count,
        e_core_count=tp.e_core_count,
        inference_threads=tp.inference_threads,
        pcie_peak_bw_mb_s=round(tp.pcie_peak_bw_mb_s, 1),
        physical_vram_mb=tp.physical_vram_mb,
        is_blackwell=tp.is_blackwell,
        gpu_cc_major=tp.gpu_cc_major,
    )
    return d


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
