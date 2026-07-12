#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_telemetry.py , GreenBoost GPU telemetry layer.

Provider-based architecture: each backend produces the same GpuMetrics
snapshot; the TelemetryManager assembles them and caches the result in a
lock-protected atomic slot.  Orchestrators read the CACHED snapshot at
stage boundaries , no blocking ioctl/NVML calls on the inference path.

Provider stack (in priority order):
  NVMLProvider       , pynvml; VRAM / clocks / power / temp / PCIe
  DCGMProvider       , embedded host engine (no daemon); ECC error detection,
                       instant power (FI 157), NVLink bw (FI 449), health-check
                       system (DCGM_HEALTH_WATCH_ALL), NVLink topology;
                       cluster-aware: one provider watches ALL GPUs in a group
  GreenBoostProvider , GB_IOCTL_GET_INFO; T2/T3 pool stats, pressure levels,
                       phase sequence, buffer counts , nothing NVML/DCGM exposes
  EbpfProvider       , /run/greenboost/ebpf_stats; T2↔T3 migration rates,
                       cold-evict/alloc/pin rates, UVM fault rates from the
                       greenboost-ebpf-trace daemon (ebpf/gb_trace.c).
                       Zero-cost when tracer absent (fields stay 0.0)
  TorchFallbackProvider , torch.cuda.memory_allocated(); last resort
  NVTXAnnotator      , pushes "gb:telemetry" ranges into the NVTX event log
                       (~/Dev/nvidia_dcgm/NVTX/python/src/) for Nsight capture

Background design:
  - Dedicated daemon thread polls every `poll_ms` ms (default 100 ms)
  - Each poll calls providers, merges results, stores in _snapshot (locked)
  - GpuMetrics.should_demote / can_promote are computed from the SNAPSHOT
  - The snapshot field is read-without-lock (Python GIL protects the object
    reference replacement; readers get either old or new, never corrupt)

DCGM , embedded mode (dcgmStartEmbedded, no dcgmd daemon required):
  Single GPU: ECC, instant power, health check, NVLink returns 0 (RTX 5070)
  Cluster (2+ GPUs): device group, per-device field watches, topology query,
    cluster-wide health check across all devices in the GreenBoost cluster.
  Field IDs used:
    100  DCGM_FI_DEV_SM_CLOCK          MHz
    150  DCGM_FI_DEV_GPU_TEMP          °C
    155  DCGM_FI_DEV_POWER_USAGE       W
    157  DCGM_FI_DEV_POWER_USAGE_INSTANT  W
    200  DCGM_FI_DEV_PCIE_TX_THROUGHPUT   KB/s
    201  DCGM_FI_DEV_PCIE_RX_THROUGHPUT   KB/s
    204  DCGM_FI_DEV_MEM_COPY_UTIL        %
    251  DCGM_FI_DEV_FB_FREE           MB
    252  DCGM_FI_DEV_FB_USED           MB
    254  DCGM_FI_DEV_FB_USED_PERCENT   %
    310  DCGM_FI_DEV_ECC_SBE_VOL_TOTAL    single-bit volatile ECC
    311  DCGM_FI_DEV_ECC_DBE_VOL_TOTAL    double-bit volatile ECC (critical)
    313  DCGM_FI_DEV_ECC_DBE_AGG_TOTAL    double-bit aggregate ECC
    449  DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL  MB/s (0 on RTX 5070)

Usage:
    # Single GPU
    tel = TelemetryManager(device=0, poll_ms=100)
    tel.start()
    m = tel.snapshot()       # always fast , reads cached value
    if m.ecc_dbe_volatile:
        raise RuntimeError(f"ECC double-bit error: {m.ecc_dbe_volatile}")
    if m.should_demote:
        tier_manager.auto_evict(m)
    tel.stop()

    # GreenBoost cluster (2+ GPUs)
    clus = ClusterTelemetryManager(poll_ms=100)
    clus.start()
    snapshots = clus.snapshots()  # list of GpuMetrics, one per device
    if not clus.cluster_health_ok():
        log.warning("Cluster health check failed")
"""
from __future__ import annotations

import contextlib
import ctypes
import fcntl
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── NVTX integration ──────────────────────────────────────────────────────────
_NVTX_AVAILABLE = False
_nvtx_annotate = None

def _init_nvtx():
    global _NVTX_AVAILABLE, _nvtx_annotate
    nvtx_python = os.path.expanduser("~/Dev/nvidia_dcgm/NVTX/python/src")
    if nvtx_python not in sys.path:
        sys.path.insert(0, nvtx_python)
    try:
        import nvtx
        _nvtx_annotate = nvtx.annotate
        _NVTX_AVAILABLE = True
    except Exception:
        pass

_init_nvtx()


@contextlib.contextmanager
def _nvtx_range(message: str, color: str = "green", domain: str = "GreenBoost"):
    """Push NVTX range visible in Nsight Systems timeline."""
    if _NVTX_AVAILABLE and _nvtx_annotate is not None:
        with _nvtx_annotate(message=message, color=color, domain=domain):
            yield
    else:
        yield


# ── GreenBoost ioctl ──────────────────────────────────────────────────────────
_GB_DEV = "/dev/greenboost"
_IOC_READ = 2

def _IOR(magic, nr, size):
    return (_IOC_READ << 30) | (size << 16) | (ord(magic) << 8) | nr

class _GbInfo(ctypes.Structure):
    _fields_ = [
        ("vram_physical_mb",     ctypes.c_uint64),
        ("total_ram_mb",         ctypes.c_uint64),
        ("free_ram_mb",          ctypes.c_uint64),
        ("allocated_mb",         ctypes.c_uint64),
        ("max_pool_mb",          ctypes.c_uint64),
        ("safety_reserve_mb",    ctypes.c_uint64),
        ("available_mb",         ctypes.c_uint64),
        ("active_buffers",       ctypes.c_uint32),
        ("oom_active",           ctypes.c_uint32),
        ("nvme_swap_total_mb",   ctypes.c_uint64),
        ("nvme_swap_used_mb",    ctypes.c_uint64),
        ("nvme_swap_free_mb",    ctypes.c_uint64),
        ("nvme_t3_allocated_mb", ctypes.c_uint64),
        ("swap_pressure",        ctypes.c_uint32),
        ("kv_reserve_mb",        ctypes.c_uint32),
        ("kv_used_mb",           ctypes.c_uint32),
        ("kv_t2_mb",             ctypes.c_uint32),
        ("t2_pressure",          ctypes.c_uint32),
        ("phase_reset_seq",      ctypes.c_uint32),
        ("gaming_mode",          ctypes.c_uint32),
        ("_pad",                 ctypes.c_uint32),
        ("total_combined_mb",    ctypes.c_uint64),
    ]

_GB_IOCTL_GET_INFO_CMD = _IOR('G', 2, ctypes.sizeof(_GbInfo))


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class GbPoolInfo:
    """GreenBoost three-tier pool snapshot (from GB_IOCTL_GET_INFO)."""
    t1_vram_mb: int      = 0
    t2_allocated_mb: int = 0
    t2_available_mb: int = 0
    t2_max_mb: int       = 0
    t3_used_mb: int      = 0
    t3_free_mb: int      = 0
    t2_pressure: int     = 0   # 0=ok 1=warn 2=critical
    t3_pressure: int     = 0
    oom_active: bool     = False
    active_buffers: int  = 0
    phase_reset_seq: int = 0
    # Extended fields , previously discarded from GB_IOCTL_GET_INFO
    gaming_mode: bool    = False
    kv_reserve_mb: int   = 0
    kv_used_mb: int      = 0
    kv_t2_mb: int        = 0   # KV bytes spilled to T2 DDR
    safety_reserve_mb: int = 0
    total_combined_mb: int = 0
    total_ram_mb: int    = 0
    free_ram_mb: int     = 0
    nvme_t3_allocated_mb: int = 0


# PCIe encoding efficiency per generation (GB/s per lane, one direction).
_PCIE_BW_GBPS_PER_LANE: Dict[int, float] = {3: 1.0, 4: 2.0, 5: 4.0, 6: 8.0}

# Maximum NVLink ports to probe (NVML_NVLINK_MAX_LINKS = 18 as of vR595).
_NVLINK_MAX_LINKS = 18


@dataclass(frozen=True)
class GpuTopology:
    """
    Static hardware topology for one GPU , probed once at NVML init.

    Fields survive for the lifetime of the process; never re-probed unless
    the TelemetryManager is torn down and rebuilt.  Attached to every
    GpuMetrics snapshot so orchestrators can make bandwidth-aware decisions
    without repeating sysfs/NVML calls on the hot path.
    """
    device: int               = 0

    # PCIe BDF ("0000:01:00.0") , used to look up sysfs NUMA node.
    bdf: str                  = ""

    # NUMA node that the GPU PCIe root port hangs off (-1 = unknown/UMA).
    numa_node: int            = -1

    # CUDA compute capability (major, minor).
    compute_capability: Tuple[int, int] = (0, 0)

    # Negotiated PCIe link (what the slot actually runs at).
    pcie_gen_current: int     = 4
    pcie_width_current: int   = 16

    # Device-maximum PCIe link (what the GPU silicon supports).
    pcie_gen_max: int         = 4
    pcie_width_max: int       = 16

    # Active NVLink connections.
    nvlink_count: int         = 0
    # Device indices of NVLink peers (GPU-to-GPU direct links only).
    nvlink_peer_ids: Tuple[int, ...] = ()

    # Device indices with P2P read access (NVML_P2P_CAPS_INDEX_READ).
    p2p_device_ids: Tuple[int, ...] = ()

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def pcie_bw_mb_s(self) -> float:
        """One-direction theoretical bandwidth (MB/s) at current link speed."""
        return (
            self.pcie_width_current
            * _PCIE_BW_GBPS_PER_LANE.get(self.pcie_gen_current, 2.0)
            * 1024.0
        )

    @property
    def pcie_bw_max_mb_s(self) -> float:
        """One-direction theoretical bandwidth at device-max link speed."""
        return (
            self.pcie_width_max
            * _PCIE_BW_GBPS_PER_LANE.get(self.pcie_gen_max, 2.0)
            * 1024.0
        )

    @property
    def pcie_saturation_mb_s(self) -> float:
        """75% of current one-direction peak , TX+RX combined threshold.

        When pcie_tx_mb_s + pcie_rx_mb_s exceeds this the PCIe bus is the
        dominant bottleneck for T2→T1 DMA prefetch (B2 gate).
        """
        return self.pcie_bw_mb_s * 0.75

    @property
    def pcie_degraded(self) -> bool:
        """True when the slot is running below device maximum.

        Common causes: x8 slot, PCIe gen mismatch, or bifurcation.
        Logged as a one-time advisory; does not block inference.
        """
        return (
            self.pcie_gen_current < self.pcie_gen_max
            or self.pcie_width_current < self.pcie_width_max
        )

    @property
    def has_nvlink(self) -> bool:
        return self.nvlink_count > 0


def _sysfs_bdf(bdf: str) -> str:
    """Normalise NVML busId to the 4-digit-domain format sysfs uses.

    NVML returns "00000000:01:00.0" (8-char domain); sysfs expects
    "0000:01:00.0" (4-char domain) on most x86 systems.
    """
    bdf = bdf.strip().lower()
    m = re.match(r'([0-9a-f]{8}):([0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])', bdf)
    if m:
        # Strip leading 4 hex chars from the 8-char domain
        return m.group(1)[4:] + ":" + m.group(2)
    return bdf


def _probe_gpu_topology(pynvml_mod: object, handle: object, device: int) -> Optional[GpuTopology]:
    """Probe static GPU topology via NVML.  Called once at NVMLProvider init.

    Every call is guarded individually so a missing capability (consumer GPU
    has no NVLink, container has no sysfs) never prevents the rest from
    populating.
    """
    nv = pynvml_mod

    # ── BDF ──────────────────────────────────────────────────────────────────
    bdf = ""
    try:
        pci = nv.nvmlDeviceGetPciInfo_v3(handle)
        raw = pci.busId
        bdf = raw.decode("ascii", errors="ignore").strip("\x00") if isinstance(raw, bytes) else str(raw)
    except Exception:
        try:
            pci = nv.nvmlDeviceGetPciInfo(handle)
            raw = pci.busId
            bdf = raw.decode("ascii", errors="ignore").strip("\x00") if isinstance(raw, bytes) else str(raw)
        except Exception:
            pass

    # ── NUMA node ─────────────────────────────────────────────────────────────
    numa_node = -1
    if bdf:
        for candidate in (_sysfs_bdf(bdf), bdf.lower()):
            p = Path(f"/sys/bus/pci/devices/{candidate}/numa_node")
            try:
                numa_node = int(p.read_text().strip())
                break
            except Exception:
                pass

    # ── Compute capability ────────────────────────────────────────────────────
    cc: Tuple[int, int] = (0, 0)
    try:
        r = nv.nvmlDeviceGetCudaComputeCapability(handle)
        cc = (int(r.major), int(r.minor))
    except AttributeError:
        try:
            cc = tuple(nv.nvmlDeviceGetCudaComputeCapability(handle))[:2]  # type: ignore[assignment]
        except Exception:
            pass
    except Exception:
        pass

    # ── PCIe link ─────────────────────────────────────────────────────────────
    gen_cur, gen_max, width_cur, width_max = 4, 4, 16, 16
    try:
        gen_cur = int(nv.nvmlDeviceGetCurrPcieLinkGeneration(handle))
    except Exception:
        pass
    try:
        gen_max = int(nv.nvmlDeviceGetMaxPcieLinkGeneration(handle))
    except Exception:
        gen_max = gen_cur
    try:
        width_cur = int(nv.nvmlDeviceGetCurrPcieLinkWidth(handle))
    except Exception:
        pass
    try:
        width_max = int(nv.nvmlDeviceGetMaxPcieLinkWidth(handle))
    except Exception:
        width_max = width_cur

    # ── NVLink ───────────────────────────────────────────────────────────────
    nvlink_count = 0
    nvlink_peer_ids: List[int] = []
    try:
        enabled = getattr(nv, "NVML_FEATURE_ENABLED", 1)
        gpu_type = getattr(nv, "NVML_NVLINK_DEVICE_TYPE_GPU", 0)
        for link in range(_NVLINK_MAX_LINKS):
            try:
                state = nv.nvmlDeviceGetNvLinkState(handle, link)
                if state != enabled:
                    continue
                nvlink_count += 1
                try:
                    rtype = nv.nvmlDeviceGetNvLinkRemoteDeviceType(handle, link)
                    if rtype != gpu_type:
                        continue
                    try:
                        rinfo = nv.nvmlDeviceGetNvLinkRemotePciInfo_v2(handle, link)
                        peer_h = nv.nvmlDeviceGetHandleByPciBusId(rinfo.busId)
                        peer_idx = int(nv.nvmlDeviceGetIndex(peer_h))
                        if peer_idx not in nvlink_peer_ids:
                            nvlink_peer_ids.append(peer_idx)
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                break  # link index out of range , stop scanning
    except Exception:
        pass

    # ── P2P access ───────────────────────────────────────────────────────────
    p2p_device_ids: List[int] = []
    try:
        p2p_ok = getattr(nv, "NVML_P2P_STATUS_OK", 0)
        read_idx = getattr(nv, "NVML_P2P_CAPS_INDEX_READ", 0)
        n_devs = int(nv.nvmlDeviceGetCount())
        for peer_idx in range(n_devs):
            if peer_idx == device:
                continue
            try:
                peer_h = nv.nvmlDeviceGetHandleByIndex(peer_idx)
                status = nv.nvmlDeviceGetP2PStatus(handle, peer_h, read_idx)
                if int(status) == p2p_ok:
                    p2p_device_ids.append(peer_idx)
            except Exception:
                pass
    except Exception:
        pass

    return GpuTopology(
        device=device,
        bdf=bdf,
        numa_node=numa_node,
        compute_capability=cc,
        pcie_gen_current=gen_cur,
        pcie_gen_max=gen_max,
        pcie_width_current=width_cur,
        pcie_width_max=width_max,
        nvlink_count=nvlink_count,
        nvlink_peer_ids=tuple(nvlink_peer_ids),
        p2p_device_ids=tuple(p2p_device_ids),
    )


@dataclass
class SystemMetrics:
    """
    Host-level OS pressure snapshot , SystemProvider only (no NVML/DCGM
    equivalent).  These are the signals a continuous OS tuner needs that pure
    GPU telemetry cannot see: CPU saturation, memory reclaim stalls, IO waits.
    All fields read from /proc; PSI fields are 0.0 on kernels without
    CONFIG_PSI (pre-4.20) or when /proc/pressure is absent (containers).
    """
    cpu_util_pct: float       = 0.0   # 1 - idle delta over the poll interval
    loadavg_1: float          = 0.0
    loadavg_5: float          = 0.0
    loadavg_15: float         = 0.0
    nr_running: int           = 0     # runnable tasks (from /proc/loadavg "r/n")
    # PSI "some" = at least one task stalled; "full" = ALL tasks stalled.
    # avg10 = 10s trailing average % of time stalled.
    psi_cpu_some_avg10: float = 0.0
    psi_mem_some_avg10: float = 0.0
    psi_mem_full_avg10: float = 0.0
    psi_io_some_avg10: float  = 0.0
    mem_available_mb: int     = 0     # /proc/meminfo MemAvailable
    swap_used_mb: int         = 0


@dataclass
class GpuMetrics:
    """
    Point-in-time GPU metrics snapshot , assembled from all providers.
    Immutable once published; readers share it safely under the GIL.
    """
    timestamp: float       = field(default_factory=time.monotonic)
    device: int            = 0
    # VRAM (NVML / DCGM FI 251/252/254)
    fb_used_mb: int        = 0
    fb_free_mb: int        = 0
    fb_total_mb: int       = 0
    fb_used_pct: float     = 0.0
    # Compute (DCGM FI 100)
    sm_clock_mhz: int      = 0
    gpu_util_pct: float    = 0.0   # SM compute utilization 0-100% (NVML util.gpu / heartbeat)
    # Memory subsystem (DCGM FI 204)
    mem_copy_util_pct: float = 0.0
    # Thermal / power (DCGM FI 150/155/157)
    temp_c: float          = 0.0
    power_w: float         = 0.0
    power_limit_w: float   = 0.0   # TDP / power management limit from NVML
    power_instant_w: float = 0.0
    # PCIe (DCGM FI 200/201), KB/s → MB/s
    pcie_tx_mb_s: float    = 0.0
    pcie_rx_mb_s: float    = 0.0
    # NVLink total bandwidth (DCGM FI 449; 0 on RTX 5070 / no NVLink)
    nvlink_bw_mb_s: float  = 0.0
    # ECC error counters (DCGM FI 310/311/313) , production-critical
    ecc_sbe_volatile: int  = 0   # single-bit errors (correctable)
    ecc_dbe_volatile: int  = 0   # double-bit errors (uncorrectable , hardware risk)
    ecc_dbe_aggregate: int = 0   # persistent aggregate DBE (lifetime counter)
    # DCGM health check result
    health_ok: bool        = True
    health_summary: str    = ""  # human-readable DCGM health status
    # GreenBoost T2/T3 pool (GreenBoostProvider , not in NVML/DCGM)
    gb: Optional[GbPoolInfo] = None
    # Shim inference phase (from /run/greenboost/phase; empty when shim absent)
    shim_phase: str          = ""

    # Static hardware topology , probed once at NVMLProvider init, attached
    # to every snapshot.  None in TorchFallback / container environments.
    topology: Optional[GpuTopology] = None

    # Host OS pressure (SystemProvider , CPU/mem/IO; not exposed by NVML/DCGM)
    sys: Optional[SystemMetrics] = None

    # eBPF tier-migration telemetry (EbpfProvider , /run/greenboost/ebpf_stats)
    # All rates are per-second averages over a 5-second sliding window.
    # Fields stay at 0.0 / 0 when the eBPF tracer is not running.
    t3_evict_rate:    float = 0.0   # T2 → T3 evictions / s
    t3_promote_rate:  float = 0.0   # T3 → T2 promotions / s
    cold_evict_rate:  float = 0.0   # cold-sweep trigger events / s
    t3_bytes_out_s:   float = 0.0   # bytes evicted to T3 per second
    t3_bytes_in_s:    float = 0.0   # bytes promoted from T3 per second
    alloc_rate:       float = 0.0   # T2 DDR page-pool alloc / s
    pin_rate:         float = 0.0   # DMA-BUF / pinned alloc / s
    uvm_fault_rate:   float = 0.0   # UVM GPU faults / s (0 when no managed memory)
    uvm_pages_in:     int   = 0     # cumulative UVM pages migrated to GPU
    uvm_pages_out:    int   = 0     # cumulative UVM pages migrated from GPU

    # ── decision helpers ──────────────────────────────────────────────────────

    @property
    def should_demote(self) -> bool:
        """VRAM > 90% , trigger ModelTierManager demotion."""
        return self.fb_used_pct > 90.0

    @property
    def can_promote(self) -> bool:
        """VRAM < 60% , safe to promote a model from CPU."""
        return self.fb_used_pct < 60.0

    @property
    def t2_critical(self) -> bool:
        """T2 DDR pool at critical pressure , consider T3 eviction."""
        return bool(self.gb and self.gb.t2_pressure >= 2)

    @property
    def prefetch_budget_mb(self) -> int:
        """Estimated free VRAM available for async prefetch (conservative)."""
        return max(0, self.fb_free_mb - 1024)   # keep 1 GB buffer

    @property
    def ecc_critical(self) -> bool:
        """Double-bit ECC error detected , hardware-level memory corruption risk."""
        return self.ecc_dbe_volatile > 0

    @property
    def kv_pressure(self) -> float:
        """KV cache pressure: used / reserve ratio (0..1+).  Signal source for Loop C."""
        if not self.gb or self.gb.kv_reserve_mb <= 0:
            return 0.0
        return self.gb.kv_used_mb / self.gb.kv_reserve_mb

    @property
    def kv_spilled(self) -> bool:
        """True when KV cache has overflowed T1 VRAM into T2 DDR.  Immediate Loop C trigger."""
        return bool(self.gb and self.gb.kv_t2_mb > 0)

    @property
    def power_near_limit(self) -> bool:
        """True when GPU is drawing >93% of its TDP , power-limit throttle imminent.

        Used as an additional Loop C gate: growing the KV reserve when near the
        power limit increases memory bandwidth consumption, which raises power draw
        further, which triggers clock throttling and hurts inference latency.
        Returns False when power_limit_w is unknown (0) to avoid false gates.
        """
        return (
            self.power_limit_w > 0
            and self.power_w > 0
            and (self.power_w / self.power_limit_w) > 0.93
        )

    @property
    def pcie_saturated(self) -> bool:
        """True when combined TX+RX throughput exceeds 75% of current link bandwidth.

        Uses live GpuTopology (actual negotiated gen×width) rather than the
        config-file profile, so slot degradation (x8 slot, gen mismatch) is
        automatically accounted for.  Returns False when topology is unavailable
        to avoid false blocking in container / TorchFallback environments.
        """
        if self.topology is None or self.topology.pcie_bw_mb_s <= 0:
            return False
        return (self.pcie_tx_mb_s + self.pcie_rx_mb_s) > self.topology.pcie_saturation_mb_s

    @property
    def numa_node(self) -> int:
        """NUMA node the GPU is attached to (-1 = unknown/UMA).

        Convenience accessor , delegates to topology so callers don't need
        to guard `topology is not None`.
        """
        return self.topology.numa_node if self.topology else -1

    @property
    def cpu_saturated(self) -> bool:
        """True when CPU PSI "some" pressure exceeds 20% over 10s.

        Signals the host CPU is the bottleneck (scheduling delay), not the
        GPU , Loop R uses this to assert governor=performance even when GPU
        util alone wouldn't justify it (e.g. heavy tokenization/dispatch).
        """
        return bool(self.sys and self.sys.psi_cpu_some_avg10 > 20.0)

    @property
    def mem_pressure_high(self) -> bool:
        """True when memory reclaim is stalling tasks (PSI mem "some" > 10%).

        Distinct from gb.t2_pressure (GreenBoost's own T2 pool accounting) ,
        this reflects whole-system reclaim stalls, including non-GreenBoost
        memory users.  Loop Q uses this to tighten vm.* tunables.
        """
        return bool(self.sys and self.sys.psi_mem_some_avg10 > 10.0)

    @property
    def io_pressure_high(self) -> bool:
        """True when IO is stalling tasks (PSI io "some" > 20% over 10s).

        Loop R uses this during T3 NVMe spill to justify reactive NVMe
        read_ahead/nr_requests adjustment.
        """
        return bool(self.sys and self.sys.psi_io_some_avg10 > 20.0)

    @property
    def migration_active(self) -> bool:
        """True when T3 tier-migration is occurring (evictions or promotions above noise).

        Used by downstream consumers (e.g. Loop S prefetch throttle, vitals) to
        detect active T2↔T3 I/O pressure.  Threshold of 0.1 events/s filters
        out isolated single evictions that would otherwise trigger false positives.
        Requires the eBPF tracer to be running; returns False when tracer absent.
        """
        return self.t3_evict_rate > 0.1 or self.t3_promote_rate > 0.1

    def __repr__(self) -> str:
        ecc = f" ECC_DBE={self.ecc_dbe_volatile}!" if self.ecc_dbe_volatile else ""
        health = "" if self.health_ok else f" HEALTH_FAIL={self.health_summary!r}"
        mig = f" mig_evict={self.t3_evict_rate:.1f}/s" if self.t3_evict_rate > 0 else ""
        return (
            f"GpuMetrics(dev={self.device} "
            f"fb={self.fb_used_mb}/{self.fb_total_mb}MB {self.fb_used_pct:.0f}% "
            f"sm={self.sm_clock_mhz}MHz pwr={self.power_w:.0f}W "
            f"temp={self.temp_c:.0f}°C "
            f"t2_press={self.gb.t2_pressure if self.gb else '?'}"
            f"{mig}{ecc}{health})"
        )


# ── Providers ─────────────────────────────────────────────────────────────────

class _Provider:
    """Base provider , fill fields it knows into a GpuMetrics instance."""
    name: str = "base"

    def available(self) -> bool:
        return False

    def fill(self, m: GpuMetrics) -> None:
        pass

    def close(self) -> None:
        pass


def _log_topology(t: GpuTopology) -> None:
    """One-shot startup log , summarises the probed topology."""
    degraded = " DEGRADED(slot)" if t.pcie_degraded else ""
    nvlink = f" NVLink×{t.nvlink_count}(peers={list(t.nvlink_peer_ids)})" if t.has_nvlink else ""
    p2p = f" P2P→{list(t.p2p_device_ids)}" if t.p2p_device_ids else ""
    numa = f" NUMA={t.numa_node}" if t.numa_node >= 0 else ""
    print(
        f"[gb_telemetry] GPU{t.device} topology:"
        f" PCIe gen{t.pcie_gen_current}x{t.pcie_width_current}"
        f"(max gen{t.pcie_gen_max}x{t.pcie_width_max})"
        f" {t.pcie_bw_mb_s/1024:.1f}GB/s{degraded}"
        f" CC={t.compute_capability[0]}.{t.compute_capability[1]}"
        f" BDF={t.bdf}{numa}{nvlink}{p2p}",
        flush=True,
    )


class NVMLProvider(_Provider):
    """
    pynvml , VRAM, clocks, utilization, power, temperature, PCIe.
    Always attempted first; available on any NVIDIA driver install.
    """
    name = "nvml"

    def __init__(self, device: int = 0):
        self._device = device
        self._handle = None
        self._ok = False
        self._topology: Optional[GpuTopology] = None
        try:
            try:
                import pynvml
            except ImportError:
                # ctypes fallback over libnvidia-ml.so.1 , same API surface,
                # no pip dependency (gb_nvml_ctypes.py ships with GreenBoost).
                import gb_nvml_ctypes as pynvml
            self._pynvml = pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            self._ok = True
            self._topology = _probe_gpu_topology(pynvml, self._handle, device)
            if self._topology:
                _log_topology(self._topology)
        except Exception:
            pass

    def available(self) -> bool:
        return self._ok

    def fill(self, m: GpuMetrics) -> None:
        if not self._ok:
            return
        pynvml = self._pynvml
        h = self._handle
        if self._topology is not None:
            m.topology = self._topology
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            m.fb_total_mb = mem.total // (1024 * 1024)
            m.fb_used_mb  = mem.used  // (1024 * 1024)
            m.fb_free_mb  = mem.free  // (1024 * 1024)
            m.fb_used_pct = 100.0 * m.fb_used_mb / (m.fb_total_mb or 1)
        except Exception:
            pass
        try:
            m.sm_clock_mhz = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        except Exception:
            pass
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            m.gpu_util_pct      = float(util.gpu)
            m.mem_copy_util_pct = float(util.memory)
        except Exception:
            pass
        try:
            m.temp_c = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        try:
            m.power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            pass
        try:
            m.power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        except Exception:
            pass
        try:
            tx = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            rx = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            m.pcie_tx_mb_s = tx / 1024.0
            m.pcie_rx_mb_s = rx / 1024.0
        except Exception:
            pass

    def close(self) -> None:
        if self._ok:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self._ok = False


class DCGMProvider(_Provider):
    """
    DCGM embedded host engine , no daemon required (dcgmStartEmbedded).

    Single GPU: supplements NVMLProvider with instant power (FI 157), ECC
    error counters (FI 310/311/313), and DCGM_HEALTH_WATCH_ALL pre-inference
    sanity check.

    GreenBoost cluster (2+ GPUs): uses dcgmGroupCreate / dcgmGroupAddDevice to
    build a device group over ALL detected GPUs.  Field watches run per-device;
    health check spans the entire group.  NVLink topology is queried ,
    RTX 5070 returns 0 for NVLink bandwidth (no physical NVLink), which is
    handled gracefully.

    Initialisation is attempted at construction; silently disabled if DCGM
    libraries are absent (AI inference continues normally).
    """
    name = "dcgm"

    # DCGM field IDs
    _FI_POWER_INSTANT  = 157
    _FI_ECC_SBE_VOL    = 310
    _FI_ECC_DBE_VOL    = 311
    _FI_ECC_DBE_AGG    = 313
    _FI_NVLINK_BW      = 449

    _WATCH_FIELDS = [
        _FI_POWER_INSTANT,
        _FI_ECC_SBE_VOL,
        _FI_ECC_DBE_VOL,
        _FI_ECC_DBE_AGG,
        _FI_NVLINK_BW,
    ]

    # DCGM health systems to watch on every GPU
    _HEALTH_SYSTEMS = (
        0x01   # DCGM_HEALTH_WATCH_PCIE
        | 0x04   # DCGM_HEALTH_WATCH_PMU
        | 0x10   # DCGM_HEALTH_WATCH_MEM
        | 0x20   # DCGM_HEALTH_WATCH_SM
        | 0x40   # DCGM_HEALTH_WATCH_INFOROM
        | 0x80   # DCGM_HEALTH_WATCH_THERMAL
        | 0x100  # DCGM_HEALTH_WATCH_POWER
        | 0x200  # DCGM_HEALTH_WATCH_DRIVER
        # NVLink (0x02) and NVSwitch (0x400/0x800) omitted , unsupported on RTX 5070
        # ConnectX (0x1000) omitted , only relevant in InfiniBand clusters
    )

    def __init__(self, device: int = 0):
        self._device = device
        self._ok = False
        self._handle = None
        self._agent = None
        self._structs = None
        self._gpu_ids: List[int] = []
        self._group_id = None
        self._nvlink_active = False   # True when cluster GPUs have NVLink
        self._health_check_counter = 0
        self._init()

    def _init(self):
        dcgm_py = os.path.expanduser("~/Dev/nvidia_dcgm/DCGM/testing/python3")
        if dcgm_py not in sys.path:
            sys.path.insert(0, dcgm_py)
        try:
            import dcgm_structs
            import dcgm_agent

            # dcgmInit() was removed in DCGM ≥ 3.x (init is implicit).
            if hasattr(dcgm_structs, "dcgmInit"):
                dcgm_structs.dcgmInit()
            self._structs = dcgm_structs
            self._agent   = dcgm_agent

            # Embedded host engine , no dcgmd daemon required
            handle = dcgm_agent.dcgmStartEmbedded(
                dcgm_structs.DCGM_OPERATION_MODE_AUTO
            )
            self._handle = handle

            # Discover all GPUs in the system
            all_ids = list(dcgm_agent.dcgmGetAllSupportedDevices(handle))
            if not all_ids:
                raise RuntimeError("DCGM: no supported GPUs found")
            self._gpu_ids = all_ids

            # Build a device group spanning all GPUs (cluster-aware)
            group_id = dcgm_agent.dcgmGroupCreate(
                handle,
                dcgm_structs.DCGM_GROUP_DEFAULT,   # empty group, type=custom
                b"gb_cluster",
            )
            for gid in all_ids:
                dcgm_agent.dcgmGroupAddDevice(handle, group_id, gid)
            self._group_id = group_id

            # Watch supplemental fields on every GPU
            for gid in all_ids:
                for fi in self._WATCH_FIELDS:
                    try:
                        dcgm_agent.dcgmWatchField(
                            handle, gid, fi,
                            updateFreq=100_000,   # 100 ms in microseconds
                            maxKeepAge=30.0,
                            maxKeepSamples=10,
                        )
                    except Exception:
                        pass   # field unsupported on this GPU , skip gracefully

            dcgm_agent.dcgmUpdateAllFields(handle, waitForUpdate=True)

            # Enable DCGM health watch on the group
            try:
                dcgm_agent.dcgmHealthSet_v2(
                    handle, group_id, self._HEALTH_SYSTEMS,
                    updateInterval=30_000_000,  # 30 s in microseconds
                    maxKeepAge=300.0,
                )
            except Exception:
                # Fallback to v1 API if v2 unavailable
                try:
                    dcgm_agent.dcgmHealthSet(handle, group_id, self._HEALTH_SYSTEMS)
                except Exception:
                    pass

            # Query NVLink topology , detect cluster NVLink connectivity
            if len(all_ids) > 1:
                try:
                    topo = dcgm_agent.dcgmGetDeviceTopology(handle, all_ids[0])
                    # Any NVLink topology bit set → active NVLink
                    NVLINK_MASK = 0x0000FF00
                    if topo and (topo.gpuPaths[0].localNvLinkIds & NVLINK_MASK):
                        self._nvlink_active = True
                except Exception:
                    pass

            self._ok = True
            _gpu_count = len(all_ids)
            _mode = "cluster" if _gpu_count > 1 else "single-GPU"
            print(
                f"[gb_telemetry] DCGM embedded ({_mode}, {_gpu_count} GPU(s))"
                f"  NVLink={'active' if self._nvlink_active else 'none'}",
                flush=True,
            )
        except Exception as exc:
            # DCGM unavailable , inference continues without it
            print(f"[gb_telemetry] DCGM unavailable ({exc.__class__.__name__}): {exc}",
                  flush=True)

    # ------------------------------------------------------------------
    def available(self) -> bool:
        return self._ok

    def _get_field_value(self, gpu_id: int, fi: int) -> Optional[float]:
        """Return the latest numeric value for a watched field, or None."""
        try:
            vals = self._agent.dcgmGetLatestValues(self._handle, gpu_id, [fi])
            if vals:
                v = vals[0].value
                # Try int64 first, then double
                try:
                    return float(v.i64)
                except Exception:
                    return float(v.dbl)
        except Exception:
            return None

    def fill(self, m: GpuMetrics) -> None:
        if not self._ok:
            return
        try:
            self._agent.dcgmUpdateAllFields(self._handle, waitForUpdate=False)
        except Exception:
            return

        # Find our target GPU index within the DCGM device list
        gpu_id = (
            self._gpu_ids[self._device]
            if self._device < len(self._gpu_ids)
            else self._gpu_ids[0]
        )

        # Instant power , more accurate than NVML averaged power
        v = self._get_field_value(gpu_id, self._FI_POWER_INSTANT)
        if v is not None and v > 0:
            m.power_instant_w = v / 1000.0

        # ECC counters , production-critical (HBM bit-flip detection)
        v = self._get_field_value(gpu_id, self._FI_ECC_SBE_VOL)
        if v is not None:
            m.ecc_sbe_volatile = max(0, int(v))

        v = self._get_field_value(gpu_id, self._FI_ECC_DBE_VOL)
        if v is not None:
            m.ecc_dbe_volatile = max(0, int(v))

        v = self._get_field_value(gpu_id, self._FI_ECC_DBE_AGG)
        if v is not None:
            m.ecc_dbe_aggregate = max(0, int(v))

        # NVLink bandwidth (0 on RTX 5070 , no NVLink hardware)
        v = self._get_field_value(gpu_id, self._FI_NVLINK_BW)
        if v is not None and v >= 0:
            m.nvlink_bw_mb_s = v

        # DCGM health check , run every 60 polls (~30 s at 500 ms poll interval)
        # to avoid interfering with Triton kernel launches and inference steps.
        self._health_check_counter += 1
        if self._health_check_counter >= 60:
            self._health_check_counter = 0
            self._run_health_check(m)
        # Between health checks, carry last known health state
        elif not m.health_ok:
            m.health_ok = False

    def _run_health_check(self, m: GpuMetrics) -> None:
        """Run DCGM health check across the device group and record result."""
        try:
            import dcgm_structs as ds
            result = self._agent.dcgmHealthCheck(
                self._handle,
                self._group_id,
                version=ds.dcgmHealthResponse_version5,
            )
            if result.overallHealth == 0:   # DCGM_HEALTH_RESULT_PASS
                m.health_ok = True
                m.health_summary = "PASS"
            else:
                m.health_ok = False
                # Extract incident details from the response
                incidents = []
                try:
                    for i in range(result.incidentCount):
                        inc = result.incidents[i]
                        incidents.append(
                            f"gpu{inc.entityInfo.entityId}:"
                            f"sys{inc.system:#x}:{inc.health}"
                        )
                except Exception:
                    incidents = [f"health={result.overallHealth}"]
                m.health_summary = " ".join(incidents)
        except Exception:
            pass   # health check API unavailable , leave health_ok unchanged

    def cluster_device_count(self) -> int:
        """Number of GPUs in the DCGM device group."""
        return len(self._gpu_ids)

    def nvlink_active(self) -> bool:
        """True when cluster GPUs have active NVLink connections."""
        return self._nvlink_active

    def run_diag(self, level: int = 1) -> str:
        """
        Run DCGM diagnostics synchronously.  level: 1=quick, 2=medium, 3=long.
        Returns a human-readable result string.  Call from pre-inference sanity
        checks , NOT from the telemetry background thread.
        """
        if not self._ok:
            return "DCGM unavailable"
        try:
            from DcgmDiag import DcgmDiag
            diag = DcgmDiag(
                gpuIds=self._gpu_ids,
                testNamesStr=f"level {level}",
                verbose=False,
            )
            response = diag.Execute(self._handle)
            return f"DCGM diag level {level}: passed={response.numTestsPassed} "  \
                   f"failed={response.numTestsFailed}"
        except Exception as exc:
            return f"DCGM diag error: {exc}"

    def close(self) -> None:
        if self._ok:
            try:
                self._agent.dcgmStopEmbedded(self._handle)
                self._structs.dcgmShutdown()
            except Exception:
                pass
        self._ok = False


class GreenBoostProvider(_Provider):
    """
    GreenBoost ioctl provider , T2/T3 pool info not exposed by NVML or DCGM.
    Reads GB_IOCTL_GET_INFO (cmd 2) from /dev/greenboost.
    """
    name = "greenboost"

    def available(self) -> bool:
        return os.path.exists(_GB_DEV)

    def fill(self, m: GpuMetrics) -> None:
        try:
            with open(_GB_DEV, "rb") as f:
                info = _GbInfo()
                fcntl.ioctl(f.fileno(), _GB_IOCTL_GET_INFO_CMD, info)
                m.gb = GbPoolInfo(
                    t1_vram_mb          = info.vram_physical_mb,
                    t2_allocated_mb     = info.allocated_mb,
                    t2_available_mb     = info.available_mb,
                    t2_max_mb           = info.max_pool_mb,
                    t3_used_mb          = info.nvme_swap_used_mb,
                    t3_free_mb          = info.nvme_swap_free_mb,
                    t2_pressure         = info.t2_pressure,
                    t3_pressure         = info.swap_pressure,
                    oom_active          = bool(info.oom_active),
                    active_buffers      = info.active_buffers,
                    phase_reset_seq     = info.phase_reset_seq,
                    # Extended , previously discarded
                    gaming_mode         = bool(info.gaming_mode),
                    kv_reserve_mb       = info.kv_reserve_mb,
                    kv_used_mb          = info.kv_used_mb,
                    kv_t2_mb            = info.kv_t2_mb,
                    safety_reserve_mb   = info.safety_reserve_mb,
                    total_combined_mb   = info.total_combined_mb,
                    total_ram_mb        = info.total_ram_mb,
                    free_ram_mb         = info.free_ram_mb,
                    nvme_t3_allocated_mb= info.nvme_t3_allocated_mb,
                )
        except Exception:
            pass
        # Read inference phase from /run/greenboost/phase (written by shim on
        # every phase transition; absent when shim is not loaded → leave empty)
        try:
            phase_path = "/run/greenboost/phase"
            if os.path.exists(phase_path):
                for _line in open(phase_path):
                    if _line.startswith("phase="):
                        m.shim_phase = _line.split("=", 1)[1].strip()
                        break
        except Exception:
            pass
        # Shim-side stats (/run/greenboost/shim_stats , key=value per line,
        # rewritten continuously during load/decode). Only trust it while
        # fresh (<=30s old) so a stale file from a dead/replaced process
        # never clobbers live ioctl data. Fills the phase (more current than
        # /run/greenboost/phase on some builds) and backstops kv/t2 fields the
        # kmod ioctl reports as 0 (e.g. when GB_IOCTL_GET_INFO doesn't track
        # shim-local counters).
        try:
            stats_path = "/run/greenboost/shim_stats"
            if os.path.exists(stats_path):
                kv: dict[str, str] = {}
                with open(stats_path) as _f:
                    for _line in _f:
                        if "=" in _line:
                            _k, _, _v = _line.strip().partition("=")
                            kv[_k.strip()] = _v.strip()

                def _sf(key: str) -> float:
                    try:
                        return float(kv.get(key, "0"))
                    except ValueError:
                        return 0.0

                fresh = (time.time() - _sf("timestamp")) <= 30.0
                if fresh:
                    if kv.get("phase"):
                        m.shim_phase = kv["phase"]
                    if m.gb is not None:
                        if not m.gb.kv_reserve_mb:
                            m.gb.kv_reserve_mb = int(_sf("kv_reserve_effective_mb"))
                        if not m.gb.kv_used_mb:
                            m.gb.kv_used_mb = int(_sf("kv_t1_tracked_mb"))
                        if not m.gb.t2_allocated_mb:
                            m.gb.t2_allocated_mb = int(_sf("tier_t2_local_cur_mb"))
        except Exception:
            pass


class EbpfProvider(_Provider):
    """
    eBPF tier-migration telemetry provider.

    Reads /run/greenboost/ebpf_stats written every 500 ms by the
    greenboost-ebpf-trace daemon (ebpf/gb_trace.c).  Fills the eBPF
    migration fields of GpuMetrics (t3_evict_rate, t3_promote_rate, etc.).

    When the tracer is not running all fields stay at their default 0.0 /
    0 values , callers must not treat absence as an error.

    Format of ebpf_stats: one "key=value\\n" pair per line.  Same
    convention as /run/greenboost/shim_stats written by the CUDA shim.
    """
    name = "ebpf"
    _STATS_PATH = "/run/greenboost/ebpf_stats"

    def available(self) -> bool:
        return os.path.exists(self._STATS_PATH)

    def fill(self, m: GpuMetrics) -> None:
        try:
            kv: dict[str, str] = {}
            with open(self._STATS_PATH) as _f:
                for _line in _f:
                    if "=" in _line:
                        _k, _, _v = _line.strip().partition("=")
                        kv[_k.strip()] = _v.strip()

            def _f32(key: str) -> float:
                try:
                    return float(kv.get(key, "0"))
                except ValueError:
                    return 0.0

            def _int(key: str) -> int:
                try:
                    return int(kv.get(key, "0"))
                except ValueError:
                    return 0

            m.t3_evict_rate   = _f32("t3_evict_rate")
            m.t3_promote_rate = _f32("t3_promote_rate")
            m.cold_evict_rate = _f32("cold_evict_rate")
            m.t3_bytes_out_s  = _f32("t3_bytes_out_s")
            m.t3_bytes_in_s   = _f32("t3_bytes_in_s")
            m.alloc_rate      = _f32("alloc_rate")
            m.pin_rate        = _f32("pin_rate")
            m.uvm_fault_rate  = _f32("uvm_fault_rate")
            m.uvm_pages_in    = _int("uvm_pages_in")
            m.uvm_pages_out   = _int("uvm_pages_out")
        except Exception:
            pass


class SystemProvider(_Provider):
    """
    Host OS pressure provider , /proc only, no NVML/DCGM equivalent.

    Fills GpuMetrics.sys with CPU/memory/IO pressure signals the continuous
    OS tuner (gb_orchestrator Loops O-S) needs to decide when to flip
    governor/clocks/vm.* tunables.  Stdlib-only, cheap (a few small file
    reads per poll); never blocks the inference path.

    PSI (/proc/pressure/*) requires CONFIG_PSI (kernel >= 4.20, most distros
    since ~2019) and is absent in some containers , guarded individually so
    a missing file never breaks the rest of the snapshot.
    """
    name = "system"

    def __init__(self):
        self._prev_idle: Optional[float] = None
        self._prev_total: Optional[float] = None

    def available(self) -> bool:
        return os.path.exists("/proc/stat")

    def fill(self, m: GpuMetrics) -> None:
        sys_m = SystemMetrics()

        # ── CPU utilization (delta of aggregate /proc/stat "cpu" line) ──────
        try:
            with open("/proc/stat") as f:
                first = f.readline()
            fields = [float(x) for x in first.split()[1:]]
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)  # idle + iowait
            total = sum(fields)
            if self._prev_total is not None and total > self._prev_total:
                d_total = total - self._prev_total
                d_idle = idle - self._prev_idle
                sys_m.cpu_util_pct = max(0.0, min(100.0, 100.0 * (1.0 - d_idle / d_total)))
            self._prev_idle, self._prev_total = idle, total
        except Exception:
            pass

        # ── load average + runnable-task count ───────────────────────────────
        try:
            with open("/proc/loadavg") as f:
                parts = f.read().split()
            sys_m.loadavg_1, sys_m.loadavg_5, sys_m.loadavg_15 = (
                float(parts[0]), float(parts[1]), float(parts[2])
            )
            sys_m.nr_running = int(parts[3].split("/")[0])
        except Exception:
            pass

        # ── PSI pressure (avg10 field of each /proc/pressure/* file) ─────────
        def _psi_avg10(path: str, line_prefix: str) -> float:
            try:
                with open(path) as f:
                    for line in f:
                        if line.startswith(line_prefix):
                            for tok in line.split():
                                if tok.startswith("avg10="):
                                    return float(tok.split("=", 1)[1])
            except Exception:
                pass
            return 0.0

        sys_m.psi_cpu_some_avg10 = _psi_avg10("/proc/pressure/cpu", "some")
        sys_m.psi_mem_some_avg10 = _psi_avg10("/proc/pressure/memory", "some")
        sys_m.psi_mem_full_avg10 = _psi_avg10("/proc/pressure/memory", "full")
        sys_m.psi_io_some_avg10  = _psi_avg10("/proc/pressure/io", "some")

        # ── MemAvailable / swap used ──────────────────────────────────────────
        try:
            mem_total_kb = mem_avail_kb = swap_total_kb = swap_free_kb = 0
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        mem_avail_kb = int(line.split()[1])
                    elif line.startswith("SwapTotal:"):
                        swap_total_kb = int(line.split()[1])
                    elif line.startswith("SwapFree:"):
                        swap_free_kb = int(line.split()[1])
            sys_m.mem_available_mb = mem_avail_kb // 1024
            sys_m.swap_used_mb = max(0, swap_total_kb - swap_free_kb) // 1024
        except Exception:
            pass

        m.sys = sys_m


class TorchFallbackProvider(_Provider):
    """
    Pure torch.cuda fallback , used when pynvml is not installed.
    Limited to: VRAM used/free.
    """
    name = "torch"

    def available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def fill(self, m: GpuMetrics) -> None:
        try:
            import torch
            d = torch.device(f"cuda:{m.device}")
            props = torch.cuda.get_device_properties(d)
            m.fb_total_mb = props.total_memory // (1024 * 1024)
            m.fb_used_mb  = torch.cuda.memory_allocated(d) // (1024 * 1024)
            m.fb_free_mb  = m.fb_total_mb - m.fb_used_mb
            m.fb_used_pct = 100.0 * m.fb_used_mb / (m.fb_total_mb or 1)
        except Exception:
            pass


class RemoteFeederProvider(_Provider):
    """
    Reads feeder GPU metrics from the shim's ``/run/greenboost/metrics.json``.

    The JSON is written every 250 ms by ``gb_write_stats()`` in the CUDA shim
    and includes temp, power, utilization, ECC and health for every connected
    feeder (populated from the heartbeat stream netd→netc).  No network I/O
    here , we just read the file the running shim already maintains.

    Parameters
    ----------
    feeder_idx : int
        Index into the ``feeders`` array in metrics.json (0-based).
    """
    name = "remote_feeder"
    _METRICS_JSON = "/run/greenboost/metrics.json"

    def __init__(self, feeder_idx: int = 0):
        self._idx = feeder_idx

    def available(self) -> bool:
        return os.path.exists(self._METRICS_JSON)

    def fill(self, m: GpuMetrics) -> None:
        try:
            import json
            with open(self._METRICS_JSON) as _f:
                data = json.load(_f)
            feeders = data.get("feeders", [])
            if self._idx >= len(feeders):
                return
            fd = feeders[self._idx]
            m.gpu_util_pct      = float(fd.get("gpu_util_pct", 0))
            m.mem_copy_util_pct = float(fd.get("gpu_mem_util_pct", 0))
            m.temp_c            = float(fd.get("gpu_temp_c", 0))
            m.power_w           = float(fd.get("gpu_power_w", 0))
            # Health: health_state 0=HEALTHY, >0=degraded/quarantined/disabled
            health_st = int(fd.get("health_state", 0))
            if health_st >= 2:          # UNHEALTHY or worse
                m.health_ok = False
                m.health_summary = f"feeder health_state={health_st}"
            # ECC: t1_quarantined is set when SBE elevated or any DBE found
            if int(fd.get("t1_quarantined", 0)):
                m.ecc_dbe_volatile = 1  # trigger ecc_critical + _ecc_guard
                m.health_ok = False
                m.health_summary = "feeder ECC quarantine"
        except Exception:
            pass


# ── TelemetryManager ─────────────────────────────────────────────────────────

class TelemetryManager:
    """
    Assembles providers, runs a background polling thread, and exposes an
    atomic cached GpuMetrics snapshot for zero-latency reads on the
    inference critical path.

    Parameters
    ----------
    device : int
        GPU device index.
    poll_ms : int
        Background poll interval in milliseconds (50–250 recommended).
    enable_dcgm : bool
        Attempt DCGMProvider (embedded mode, no daemon).
    callbacks : list of callables
        Called with GpuMetrics after each poll (optional).
    """

    def __init__(
        self,
        device: int = 0,
        poll_ms: int = 100,
        enable_dcgm: bool = True,
        callbacks: Optional[list] = None,
    ):
        self.device   = device
        self.poll_ms  = poll_ms
        self._cbs     = list(callbacks or [])
        self._lock    = threading.Lock()
        self._snapshot: GpuMetrics = GpuMetrics(device=device)
        self._thread: Optional[threading.Thread] = None
        self._stop    = threading.Event()
        self._dcgm: Optional[DCGMProvider] = None

        # Build provider chain
        self._providers: List[_Provider] = []

        nvml = NVMLProvider(device)
        if nvml.available():
            self._providers.append(nvml)
        else:
            tf = TorchFallbackProvider()
            if tf.available():
                self._providers.append(tf)

        if enable_dcgm:
            dcgm = DCGMProvider(device)
            if dcgm.available():
                self._providers.append(dcgm)
                self._dcgm = dcgm

        gb = GreenBoostProvider()
        if gb.available():
            self._providers.append(gb)

        ebpf = EbpfProvider()
        if ebpf.available():
            self._providers.append(ebpf)

        sysp = SystemProvider()
        if sysp.available():
            self._providers.append(sysp)

        active = [p.name for p in self._providers]
        print(f"[gb_telemetry] device={device} providers={active}", flush=True)

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name=f"gb-telemetry-{self.device}",
        )
        self._thread.start()

    def stop(self):
        """Stop background polling thread and release provider resources."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.poll_ms / 500.0))
        for p in self._providers:
            p.close()

    def snapshot(self) -> GpuMetrics:
        """Return the latest cached GpuMetrics. Always fast (no I/O)."""
        return self._snapshot

    def add_callback(self, fn):
        """Register a callback called with GpuMetrics after each poll."""
        self._cbs.append(fn)

    def poll_once(self) -> GpuMetrics:
        """Synchronously poll all providers and update the cache."""
        return self._do_poll()

    def run_diag(self, level: int = 1) -> str:
        """Run DCGM pre-inference diagnostics. Blocks until complete."""
        if self._dcgm:
            return self._dcgm.run_diag(level)
        return "DCGM not available"

    @property
    def dcgm_cluster_devices(self) -> int:
        """Number of GPUs visible to DCGM (1 for single GPU mode)."""
        return self._dcgm.cluster_device_count() if self._dcgm else 1

    @property
    def nvlink_active(self) -> bool:
        """True when DCGM detected active NVLink between cluster GPUs."""
        return self._dcgm.nvlink_active() if self._dcgm else False

    # ── internals ─────────────────────────────────────────────────────────────

    def set_poll_interval_ms(self, ms: int) -> None:
        """
        Adjust the background poll interval at runtime (clamped 50–2000 ms).
        The change takes effect on the next sleep expiry , at most one old
        interval passes before the new rate is active.

        Used by the orchestrator to speed up polling during INFERENCE (250 ms)
        and slow it during IDLE/DEEP_IDLE (1000 ms), saving CPU between tokens.
        """
        self.poll_ms = max(50, min(2000, int(ms)))

    def _poll_loop(self):
        # Pin this background thread to E-cores, keeping P/golden cores free for
        # tensor compute. sched_setaffinity(0, ...) targets the calling thread.
        try:
            from gb_topology import get_topology
            e_cpus = get_topology().e_core_cpus
            if e_cpus:
                os.sched_setaffinity(0, set(e_cpus))
        except Exception:
            pass
        # Read poll_ms on each iteration so set_poll_interval_ms() takes effect
        # within one cycle (no restart needed).
        while not self._stop.wait(timeout=self.poll_ms / 1000.0):
            try:
                self._do_poll()
            except Exception:
                pass

    def _do_poll(self) -> GpuMetrics:
        with _nvtx_range("gb:telemetry:poll", color="gray"):
            m = GpuMetrics(device=self.device)
            for p in self._providers:
                with _nvtx_range(f"gb:telemetry:{p.name}", color="gray"):
                    p.fill(m)
            # Atomic snapshot replacement (GIL ensures reference store is atomic)
            self._snapshot = m
            for cb in self._cbs:
                try:
                    cb(m)
                except Exception:
                    pass
        return m

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


# ── ClusterTelemetryManager ──────────────────────────────────────────────────

class ClusterTelemetryManager:
    """
    Multi-GPU telemetry for GreenBoost cluster mode (2+ devices working together).

    Creates one TelemetryManager per GPU device, sharing a single DCGMProvider
    that watches the entire device group (DCGM handles multi-GPU natively in
    embedded mode , no dcgmd cluster daemon required).

    Usage:
        clus = ClusterTelemetryManager(poll_ms=100)
        clus.start()
        snaps = clus.snapshots()           # list[GpuMetrics], one per device
        ok = clus.cluster_health_ok()      # True if all devices healthy
        hottest = max(snaps, key=lambda m: m.temp_c)
        clus.stop()
    """

    def __init__(
        self,
        poll_ms: int = 100,
        enable_dcgm: bool = True,
        callbacks: Optional[list] = None,
    ):
        self.poll_ms = poll_ms
        self._managers: List[TelemetryManager] = []

        # ── local GPU managers ────────────────────────────────────────────────
        n_devices = _detect_device_count()
        for dev in range(n_devices):
            mgr = TelemetryManager(
                device=dev,
                poll_ms=poll_ms,
                enable_dcgm=(enable_dcgm and dev == 0),  # DCGM on dev 0 watches all
                callbacks=callbacks,
            )
            self._managers.append(mgr)

        if not self._managers:
            # Fallback: at least one manager for device 0
            self._managers.append(TelemetryManager(
                device=0, poll_ms=poll_ms,
                enable_dcgm=enable_dcgm, callbacks=callbacks,
            ))

        # ── feeder GPU managers (Phase 3c) ────────────────────────────────────
        # Enumerate connected feeders from the shim's metrics.json; each feeder
        # gets a lightweight TelemetryManager backed by RemoteFeederProvider so
        # cluster_health_ok(), any_should_demote(), and snapshots() include the
        # laptop's GPU alongside the workstation's GPU.
        n_feeders = _detect_feeder_count()
        for fi in range(n_feeders):
            virtual_dev = n_devices + fi  # logical device index for this feeder
            mgr = TelemetryManager(
                device=virtual_dev,
                poll_ms=poll_ms,
                enable_dcgm=False,       # feeder telemetry comes from metrics.json
                callbacks=callbacks,
            )
            # Replace the provider stack with just the remote feeder provider
            mgr._providers = [RemoteFeederProvider(feeder_idx=fi)]
            self._managers.append(mgr)

        print(
            f"[gb_telemetry] ClusterTelemetryManager: "
            f"{len(self._managers)} device(s) "
            f"({n_devices} local + {n_feeders} feeder)",
            flush=True,
        )

    def start(self):
        """Start background polling on all devices."""
        for m in self._managers:
            m.start()

    def stop(self):
        """Stop all polling threads and release resources."""
        for m in self._managers:
            m.stop()

    def snapshots(self) -> List[GpuMetrics]:
        """Return cached GpuMetrics for each device. Always fast."""
        return [m.snapshot() for m in self._managers]

    def snapshot(self, device: int = 0) -> GpuMetrics:
        """Return cached GpuMetrics for a specific device."""
        if device < len(self._managers):
            return self._managers[device].snapshot()
        return self._managers[0].snapshot()

    def cluster_health_ok(self) -> bool:
        """True if all device snapshots report health_ok and no ECC DBE errors."""
        return all(
            m.health_ok and not m.ecc_critical
            for m in (mgr.snapshot() for mgr in self._managers)
        )

    def any_should_demote(self) -> bool:
        """True if ANY device in the cluster is VRAM-pressured."""
        return any(mgr.snapshot().should_demote for mgr in self._managers)

    def all_can_promote(self) -> bool:
        """True if ALL devices have sufficient VRAM headroom."""
        return all(mgr.snapshot().can_promote for mgr in self._managers)

    def run_diag(self, level: int = 1) -> str:
        """Run DCGM diagnostics on the primary device (covers cluster group)."""
        return self._managers[0].run_diag(level)

    @property
    def device_count(self) -> int:
        return len(self._managers)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_device_count() -> int:
    """Return number of local CUDA-capable GPUs. Tries pynvml, then torch."""
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return max(1, n)
    except Exception:
        pass
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def _detect_feeder_count() -> int:
    """Return number of connected feeder GPUs from the shim's metrics.json.

    The shim writes the active feeder list to ``/run/greenboost/metrics.json``
    every 250 ms; each entry in the ``feeders`` array is one remote GPU.
    Returns 0 if the file is absent or no shim is running.
    """
    try:
        import json
        _mpath = "/run/greenboost/metrics.json"
        if not os.path.exists(_mpath):
            return 0
        with open(_mpath) as _f:
            data = json.load(_f)
        return len(data.get("feeders", []))
    except Exception:
        return 0


def sample_once(device: int = 0) -> GpuMetrics:
    """Take a single synchronous telemetry snapshot. Lightweight , no thread."""
    m = GpuMetrics(device=device)
    for Cls in (NVMLProvider, TorchFallbackProvider, GreenBoostProvider, SystemProvider):
        p = Cls(device) if Cls is NVMLProvider else Cls()
        if p.available():
            p.fill(m)
    return m


def sample_cluster() -> List[GpuMetrics]:
    """Single synchronous snapshot for every detected GPU."""
    n = _detect_device_count()
    return [sample_once(dev) for dev in range(n)]
