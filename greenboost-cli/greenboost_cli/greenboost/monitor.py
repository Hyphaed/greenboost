"""GreenBoost kernel module monitor.

Detects and queries the GreenBoost VRAM extension kernel module.
GreenBoost extends GPU VRAM → System DDR RAM → NVMe in a 3-tier hierarchy,
allowing large models (e.g. 51 GB+) to run on 12 GB VRAM GPUs.

Query chain:
  1. /dev/greenboost  (ioctl GB_IOCTL_GET_INFO — richest, real-time stats)
  2. /sys/module/greenboost  (sysfs parameters — lighter)
  3. dkms status  (module installed but not loaded)
  4. /proc/modules  (last resort detection)
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── GreenBoost IOCTL interface (mirrors greenboost_ioctl.h) ──────────────────

_GB_IOCTL_MAGIC = ord('G')

def _IOR(type_: int, nr: int, size: int) -> int:
    return (2 << 30) | (size << 16) | (type_ << 8) | nr

def _IOW(type_: int, nr: int, size: int) -> int:
    return (1 << 30) | (size << 16) | (type_ << 8) | nr

def _IOWR(type_: int, nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (type_ << 8) | nr

class _GbInfo(ctypes.Structure):
    """Mirrors struct gb_info from greenboost_ioctl.h (v2.6).

    Previously carried three phantom fields (kv_compressed_mb,
    kv_compression_bits, kv_compression_sessions — not part of the real
    kernel struct, TurboQuant stats are tracked client-side only, see
    GreenBoostMonitor.set_turboquant) plus an anonymous _pad2 in place of
    the real gaming_mode field. That made this struct 144 bytes against the
    kernel's 136 (greenboost_ioctl.h's struct gb_info) — _IOR() encodes the
    struct size into the ioctl command number, so every GB_IOCTL_GET_INFO
    call with the wrong size returned ENOTTY and _try_ioctl() below silently
    returned False 100% of the time, permanently falling back to the much
    less reliable sysfs-regex path (confirmed live: this box's
    /dev/greenboost ioctl always failed before this fix)."""
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
        ("total_combined_mb",    ctypes.c_uint64),
    ]

_GB_IOCTL_GET_INFO = _IOR(_GB_IOCTL_MAGIC, 2, ctypes.sizeof(_GbInfo))


class _GbKvReserveReq(ctypes.Structure):
    _fields_ = [("reserve_mb", ctypes.c_uint32), ("_pad", ctypes.c_uint32)]

_GB_IOCTL_SET_KV_RESERVE = _IOW(_GB_IOCTL_MAGIC, 9, ctypes.sizeof(_GbKvReserveReq))


class _GbTurboQuantReq(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_uint32),
        ("bits",    ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("seed",    ctypes.c_uint32),
    ]

_GB_IOCTL_SET_TURBOQUANT = _IOW(_GB_IOCTL_MAGIC, 10, ctypes.sizeof(_GbTurboQuantReq))


class _GbPoolCapReq(ctypes.Structure):
    _fields_ = [("cap_mb", ctypes.c_uint64), ("prev_mb", ctypes.c_uint64)]

_GB_IOCTL_SET_POOL_CAP = _IOWR(_GB_IOCTL_MAGIC, 11, ctypes.sizeof(_GbPoolCapReq))


_DEBUG_VITALS_FLAG = Path("/etc/greenboost/debug_vitals.enabled")
_SHIM_STATS_PATH   = Path("/run/greenboost/shim_stats")
_SHIM_STATS_ALT    = Path("/tmp/greenboost_shim_stats")
_NVTX_LOG_PATH     = Path("/run/greenboost/nvtx_events.log")
_PHASE_PATH        = Path("/run/greenboost/phase")


@dataclass
class GreenBoostStatus:
    loaded: bool = False
    version: str = ""
    gpu_name: str = ""
    vram_physical_mb: float = 0.0
    vram_managed_gb: float = 0.0
    ram_pool_mb: float = 0.0
    ram_allocated_mb: float = 0.0
    ram_available_mb: float = 0.0
    active_buffers: int = 0
    oom_active: bool = False
    nvme_swap_total_mb: float = 0.0
    nvme_swap_used_mb: float = 0.0
    nvme_t3_allocated_mb: float = 0.0
    swap_pressure: int = 0
    t2_pressure: int = 0
    kv_reserve_mb: int = 2048
    kv_used_mb: int = 0
    kv_t2_mb: int = 0
    # TurboQuant KV-compression stats — tracked CLIENT-SIDE only (set by
    # set_turboquant() below), never read from the kernel ioctl: the real
    # struct gb_info has no such fields.
    kv_compressed_mb: int = 0
    kv_compression_bits: int = 0
    kv_compression_sessions: int = 0
    gaming_mode: bool = False
    total_combined_mb: float = 0.0
    system_ram_gb: float = 0.0
    nvme_cache_gb: float = 0.0
    error: str = ""
    last_checked: datetime = field(default_factory=datetime.now)
    # ── Shim stats (from /run/greenboost/shim_stats) ──────────────────────────
    shim_phase: str = ""
    shim_active_path: str = ""
    shim_path_a0: int = 0
    shim_path_a: int = 0
    shim_path_b: int = 0
    shim_path_c: int = 0
    shim_kernel_dispatch: int = 0
    shim_h2d_mb: int = 0
    shim_d2h_mb: int = 0
    shim_headroom_mb: int = 0
    shim_frag_pct: int = 0
    shim_cold_evicts: int = 0
    shim_kv_dedup: int = 0
    shim_kv_frag_mb: int = 0
    shim_remote_alloc_count: int = 0
    shim_remote_alloc_mb: int = 0
    shim_local_t1_mb: int = 0
    shim_stale: bool = True
    # ── GPU extended (from nvidia-smi extended query) ─────────────────────────
    gpu_temp_c: int = 0
    gpu_power_w: float = 0.0
    gpu_power_limit_w: float = 0.0
    gpu_util_pct: int = 0
    gpu_mem_util_pct: int = 0
    gpu_sm_clock_mhz: int = 0
    gpu_mem_clock_mhz: int = 0
    gpu_mem_used_mb: int = 0   # actual VRAM in use (nvidia-smi memory.used)
    gpu_mem_total_mb: int = 0  # physical VRAM total (nvidia-smi memory.total)
    # ── NVTX event log tail ───────────────────────────────────────────────────
    nvtx_tail: list = field(default_factory=list)
    # ── Debug vitals flag ─────────────────────────────────────────────────────
    debug_vitals_enabled: bool = False
    # ── Derived / guarded ─────────────────────────────────────────────────────
    _kv_t3_stale: bool = False  # kernel kv_t3 counter is stale (T3 alloc=0)

    @property
    def indicator(self) -> str:
        if self.error:
            return "GB ✗"
        if self.loaded:
            gpu = f" {self.gpu_name}" if self.gpu_name else ""
            return f"GB ✓{gpu}"
        return "GB ○"

    @property
    def color(self) -> str:
        if self.error:
            return "#e06c75"
        if self.loaded:
            return "#98c379"
        return "#abb2bf"

    @property
    def pressure_label(self) -> str:
        return {0: "ok", 1: "warn", 2: "critical"}.get(self.swap_pressure, "?")

    @property
    def t2_pressure_label(self) -> str:
        return {0: "ok", 1: "warn", 2: "critical"}.get(self.t2_pressure, "?")

    @property
    def total_combined_gb(self) -> float:
        return round(self.total_combined_mb / 1024, 1)

    def as_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "version": self.version,
            "gpu_name": self.gpu_name,
            "vram_physical_mb": self.vram_physical_mb,
            "ram_pool_mb": self.ram_pool_mb,
            "ram_allocated_mb": self.ram_allocated_mb,
            "ram_available_mb": self.ram_available_mb,
            "active_buffers": self.active_buffers,
            "oom_active": self.oom_active,
            "nvme_swap_total_mb": self.nvme_swap_total_mb,
            "nvme_swap_used_mb": self.nvme_swap_used_mb,
            "nvme_t3_allocated_mb": self.nvme_t3_allocated_mb,
            "swap_pressure": self.swap_pressure,
            "t2_pressure": self.t2_pressure,
            "kv_reserve_mb": self.kv_reserve_mb,
            "kv_compressed_mb": self.kv_compressed_mb,
            "kv_compression_bits": self.kv_compression_bits,
            "kv_compression_sessions": self.kv_compression_sessions,
            "kv_used_mb": self.kv_used_mb,
            "gaming_mode": self.gaming_mode,
            "total_combined_mb": self.total_combined_mb,
            "kv_t3_stale": self._kv_t3_stale,
            "error": self.error,
            "indicator": self.indicator,
            "color": self.color,
            "last_checked": self.last_checked.isoformat(),
        }


class GreenBoostMonitor:
    """Polls GreenBoost kernel module status via ioctl or sysfs."""

    DEVICE_PATH = Path("/dev/greenboost")
    SYS_MODULE_PATH = Path("/sys/module/greenboost")
    SYSFS_CLASS_PATH = Path("/sys/class/greenboost/greenboost/status")
    DKMS_STATUS_CMD = ["dkms", "status", "greenboost"]

    def __init__(self) -> None:
        self._status = GreenBoostStatus()

    @property
    def status(self) -> GreenBoostStatus:
        return self._status

    def refresh(self) -> GreenBoostStatus:
        self._status = self._detect()
        return self._status

    def set_kv_reserve(self, reserve_mb: int) -> bool:
        if not self.DEVICE_PATH.exists():
            return False
        try:
            import fcntl
            fd = os.open(str(self.DEVICE_PATH), os.O_RDWR)
            try:
                req = _GbKvReserveReq(reserve_mb=reserve_mb)
                fcntl.ioctl(fd, _GB_IOCTL_SET_KV_RESERVE, req)
                self._status.kv_reserve_mb = reserve_mb
                return True
            finally:
                os.close(fd)
        except OSError:
            return False

    def set_turboquant(self, enabled: bool, bits: int = 3, head_dim: int = 0, seed: int = 42) -> bool:
        if not self.DEVICE_PATH.exists():
            return False
        try:
            import fcntl
            fd = os.open(str(self.DEVICE_PATH), os.O_RDWR)
            try:
                req = _GbTurboQuantReq(
                    enabled=int(enabled), bits=bits, head_dim=head_dim, seed=seed,
                )
                fcntl.ioctl(fd, _GB_IOCTL_SET_TURBOQUANT, req)
                self._status.kv_compression_bits = bits if enabled else 0
                return True
            finally:
                os.close(fd)
        except OSError:
            return False

    def set_pool_cap(self, cap_mb: int) -> tuple[bool, int]:
        if not self.DEVICE_PATH.exists():
            return False, 0
        try:
            import fcntl
            fd = os.open(str(self.DEVICE_PATH), os.O_RDWR)
            try:
                req = _GbPoolCapReq(cap_mb=cap_mb, prev_mb=0)
                fcntl.ioctl(fd, _GB_IOCTL_SET_POOL_CAP, req)
                self._status.ram_pool_mb = float(cap_mb)
                return True, cap_mb
            finally:
                os.close(fd)
        except OSError:
            return False, 0

    @staticmethod
    def compute_dynamic_t2_mb(safety_reserve_gb: int = 9, target_pct: float = 0.80) -> int:
        import ctypes as _ct

        class _SysInfo(_ct.Structure):
            _fields_ = [
                ("uptime",    _ct.c_long),
                ("loads",     _ct.c_ulong * 3),
                ("totalram",  _ct.c_ulong),
                ("freeram",   _ct.c_ulong),
                ("sharedram", _ct.c_ulong),
                ("bufferram", _ct.c_ulong),
                ("totalswap", _ct.c_ulong),
                ("freeswap",  _ct.c_ulong),
                ("procs",     _ct.c_ushort),
                ("_pad",      _ct.c_ubyte * 22),
                ("mem_unit",  _ct.c_uint),
            ]

        try:
            libc = _ct.CDLL("libc.so.6", use_errno=True)
            si = _SysInfo()
            libc.sysinfo(_ct.byref(si))
            total_mb = int((si.totalram * si.mem_unit) >> 20)
            avail_mb = int(((si.freeram + si.bufferram) * si.mem_unit) >> 20)
            safety_mb = safety_reserve_gb * 1024
            cap_mb = int(min(avail_mb * target_pct, total_mb - safety_mb))
            return max(cap_mb, 4096)
        except Exception:
            return 4096

    def apply_dynamic_pool_cap(self, safety_reserve_gb: int = 9, target_pct: float = 0.80) -> tuple[bool, int]:
        if not self._status.loaded:
            return False, 0
        cap_mb = self.compute_dynamic_t2_mb(safety_reserve_gb, target_pct)
        return self.set_pool_cap(cap_mb)

    def _detect(self) -> GreenBoostStatus:
        s = GreenBoostStatus(last_checked=datetime.now())

        if self._try_ioctl(s):
            self._detect_gpu(s)
            self._read_version(s)
            self._fill_from_sysfs_status(s)
            self._read_shim_stats(s)
            self._read_gpu_extended(s)
            self._read_nvtx_tail(s)
            self._read_debug_vitals_flag(s)
            return s

        if self.SYS_MODULE_PATH.exists():
            s.loaded = True
            self._read_version(s)
            self._read_sysfs_stats(s)
            self._read_sysfs_class_status(s)
            self._fill_from_sysfs_status(s)
            self._detect_gpu(s)
            self._read_shim_stats(s)
            self._read_gpu_extended(s)
            self._read_nvtx_tail(s)
            self._read_debug_vitals_flag(s)
            return s

        try:
            result = subprocess.run(
                self.DKMS_STATUS_CMD, capture_output=True, text=True, timeout=5,
            )
            output = result.stdout.lower()
            if "installed" in output or "built" in output:
                s.loaded = False
                s.error = "GreenBoost installed but not loaded — run: sudo modprobe greenboost"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            modules = Path("/proc/modules").read_text()
            if "greenboost" in modules:
                s.loaded = True
                self._detect_gpu(s)
        except OSError:
            pass

        return s

    def _try_ioctl(self, s: GreenBoostStatus) -> bool:
        if not self.DEVICE_PATH.exists():
            return False
        try:
            fd = os.open(str(self.DEVICE_PATH), os.O_RDWR)
        except OSError:
            return False

        try:
            info = _GbInfo()
            import fcntl
            try:
                fcntl.ioctl(fd, _GB_IOCTL_GET_INFO, info)
            except OSError:
                return False

            s.loaded = True
            s.vram_physical_mb = info.vram_physical_mb
            s.vram_managed_gb = round(info.vram_physical_mb / 1024, 1)
            s.ram_pool_mb = info.max_pool_mb
            s.ram_allocated_mb = info.allocated_mb
            s.ram_available_mb = info.available_mb
            s.active_buffers = info.active_buffers
            s.oom_active = bool(info.oom_active)
            s.nvme_swap_total_mb = info.nvme_swap_total_mb
            s.nvme_swap_used_mb = info.nvme_swap_used_mb
            s.nvme_t3_allocated_mb = info.nvme_t3_allocated_mb
            s.swap_pressure = info.swap_pressure
            s.t2_pressure = info.t2_pressure
            s.kv_reserve_mb = info.kv_reserve_mb
            s.kv_used_mb = info.kv_used_mb
            s.kv_t2_mb = info.kv_t2_mb
            # kv_compressed_mb/kv_compression_bits/kv_compression_sessions are
            # NOT in the kernel struct — client-tracked only, set by
            # set_turboquant(); leave s's current values untouched here.
            s.gaming_mode = bool(info.gaming_mode)
            s.total_combined_mb = info.total_combined_mb
            s.system_ram_gb = round(info.total_ram_mb / 1024, 1)
            s.nvme_cache_gb = round(info.nvme_swap_total_mb / 1024, 1)
            return True
        except Exception:
            return False
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def _read_version(self, s: GreenBoostStatus) -> None:
        version_file = self.SYS_MODULE_PATH / "version"
        if version_file.exists():
            try:
                s.version = version_file.read_text().strip()
            except OSError:
                pass

    def _read_sysfs_stats(self, s: GreenBoostStatus) -> None:
        base = self.SYS_MODULE_PATH / "parameters"
        if not base.exists():
            return
        for attr, key in [
            ("vram_managed_gb", "vram_managed_gb"),
            ("system_ram_gb", "system_ram_gb"),
            ("nvme_cache_gb", "nvme_cache_gb"),
        ]:
            p = base / key
            if p.exists():
                try:
                    setattr(s, attr, float(p.read_text().strip()))
                except ValueError:
                    pass

    def _read_sysfs_class_status(self, s: GreenBoostStatus) -> None:
        if not self.SYSFS_CLASS_PATH.exists():
            return
        try:
            data: dict[str, int] = {}
            for line in self.SYSFS_CLASS_PATH.read_text().splitlines():
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                v = v.strip()
                if v.lstrip("-").isdigit():
                    data[k.strip()] = int(v)
            for field_name in ("kv_t2_mb", "kv_used_mb", "kv_reserve_mb",
                               "swap_pressure", "t2_pressure"):
                if field_name in data:
                    setattr(s, field_name, data[field_name])
            if "allocated_mb" in data:
                s.ram_allocated_mb = float(data["allocated_mb"])
        except Exception:
            pass

    def _fill_from_sysfs_status(self, s: GreenBoostStatus) -> None:
        """Supplement ioctl data with sysfs status for T2/T3 pool sizes.

        The ioctl returns max_pool_mb=0 when no explicit pool cap is set via
        GB_IOCTL_SET_POOL_CAP. The sysfs status file always has the live values.
        """
        if not self.SYSFS_CLASS_PATH.exists():
            return
        try:
            import re as _re
            text = self.SYSFS_CLASS_PATH.read_text()
            if s.ram_pool_mb == 0:
                m = _re.search(r'System RAM pool\s*:\s*(\d+)\s*GB', text)
                if m:
                    s.ram_pool_mb = int(m.group(1)) * 1024
            if s.ram_allocated_mb == 0:
                m = _re.search(r'T2 allocated\s*:\s*(\d+)\s*MB', text)
                if m:
                    s.ram_allocated_mb = float(m.group(1))
            m = _re.search(r'T2 available\s*:\s*(\d+)\s*MB', text)
            if m:
                s.ram_available_mb = float(m.group(1))
            if s.nvme_swap_total_mb == 0:
                m = _re.search(r'T3 backing file\s*:\s*(\d+)\s*GB', text)
                if m:
                    s.nvme_swap_total_mb = int(m.group(1)) * 1024
            if s.nvme_swap_used_mb == 0:
                # The kernel module (greenboost.c) prints "T3 allocated : N MB",
                # never "Swap used" — that string doesn't exist anywhere in the
                # kmod source, so this regex matched nothing and T3 usage was
                # structurally pinned to 0 regardless of real T3 activity
                # (confirmed: greenboost.c:2418 is the actual line format).
                m = _re.search(r'T3 allocated\s*:\s*(\d+)\s*MB', text)
                if m:
                    s.nvme_swap_used_mb = float(m.group(1))
            # KV T3: trust the kernel counter only when T3 is actually allocated.
            # If nvme_t3_allocated_mb==0 but kv_t3 counter is non-zero, the counter
            # is stale (not reset on process exit) — suppress the false positive.
            if s.nvme_t3_allocated_mb == 0 and s.kv_used_mb > 0:
                m = _re.search(r'KV in T3[^:]*:\s*(\d+)\s*MB', text)
                if m:
                    kv_t3_raw = int(m.group(1))
                    m3 = _re.search(r'T3 allocated\s*:\s*(\d+)\s*MB', text)
                    t3_alloc = int(m3.group(1)) if m3 else 0
                    if kv_t3_raw > 0 and t3_alloc == 0:
                        # stale counter — zero it so alerts don't fire falsely
                        s._kv_t3_stale = True
        except Exception:
            pass

    def _read_shim_stats(self, s: GreenBoostStatus) -> None:
        stats_path = _SHIM_STATS_PATH if _SHIM_STATS_PATH.exists() else _SHIM_STATS_ALT
        if not stats_path.exists():
            return
        try:
            content = stats_path.read_text(errors="replace")
            import time as _time
            ts_str = self._parse_field(content, "timestamp")
            try:
                stale = (int(ts_str) == 0) or ((_time.time() - int(ts_str)) > 30)
            except (ValueError, TypeError):
                stale = True
            s.shim_stale = stale
            s.shim_phase           = self._parse_field(content, "phase")
            s.shim_active_path     = self._parse_field(content, "active_path")
            s.shim_path_a0         = self._parse_int(content, "path_a0_count")
            s.shim_path_a          = self._parse_int(content, "path_a_count")
            s.shim_path_b          = self._parse_int(content, "path_b_count")
            s.shim_path_c          = self._parse_int(content, "path_c_count")
            s.shim_kernel_dispatch = self._parse_int(content, "kernel_dispatch_count")
            s.shim_h2d_mb          = self._parse_int(content, "h2d_mb")
            s.shim_d2h_mb          = self._parse_int(content, "d2h_mb")
            s.shim_headroom_mb     = self._parse_int(content, "vram_headroom_mb")
            s.shim_frag_pct        = self._parse_int(content, "t2_warn_adj_pct")
            s.shim_cold_evicts     = self._parse_int(content, "cold_epoch_evict_count")
            s.shim_kv_dedup        = self._parse_int(content, "kv_dedup_hits")
            s.shim_kv_frag_mb      = self._parse_int(content, "kv_internal_frag_mb")
            s.shim_remote_alloc_count = self._parse_int(content, "remote_alloc_count")
            s.shim_remote_alloc_mb = self._parse_int(content, "remote_alloc_mb")
            s.shim_local_t1_mb     = self._parse_int(content, "local_t1_alloc_mb")
        except Exception:
            pass

    @staticmethod
    def _parse_field(content: str, key: str) -> str:
        import re
        m = re.search(rf"(?m)^{re.escape(key)}=(\S+)", content)
        return m.group(1) if m else ""

    @staticmethod
    def _parse_int(content: str, key: str) -> int:
        import re
        m = re.search(rf"(?m)^{re.escape(key)}=(\d+)", content)
        try:
            return int(m.group(1)) if m else 0
        except (ValueError, AttributeError):
            return 0

    def _read_gpu_extended(self, s: GreenBoostStatus) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw,power.limit,"
                    "utilization.gpu,utilization.memory,"
                    "clocks.current.sm,clocks.current.memory,"
                    "memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            def _toint(v: str) -> int:
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    return 0
            def _tofloat(v: str) -> float:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return 0.0
            if len(parts) >= 7:
                s.gpu_temp_c        = _toint(parts[0])
                s.gpu_power_w       = _tofloat(parts[1])
                s.gpu_power_limit_w = _tofloat(parts[2])
                s.gpu_util_pct      = _toint(parts[3])
                s.gpu_mem_util_pct  = _toint(parts[4])
                s.gpu_sm_clock_mhz  = _toint(parts[5])
                s.gpu_mem_clock_mhz = _toint(parts[6])
            if len(parts) >= 9:
                s.gpu_mem_used_mb  = _toint(parts[7])
                s.gpu_mem_total_mb = _toint(parts[8])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _read_nvtx_tail(self, s: GreenBoostStatus, n: int = 10) -> None:
        if not _NVTX_LOG_PATH.exists():
            return
        try:
            lines = _NVTX_LOG_PATH.read_text(errors="replace").splitlines()
            s.nvtx_tail = [l for l in lines if l.strip()][-n:]
        except OSError:
            pass

    def _read_debug_vitals_flag(self, s: GreenBoostStatus) -> None:
        s.debug_vitals_enabled = _DEBUG_VITALS_FLAG.exists()

    def _detect_gpu(self, s: GreenBoostStatus) -> None:
        if s.gpu_name:
            return
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if parts:
                    s.gpu_name = parts[0]
                if len(parts) > 1 and not s.vram_physical_mb:
                    mem_str = parts[1].replace("MiB", "").strip()
                    try:
                        s.vram_physical_mb = int(mem_str)
                        s.vram_managed_gb = round(int(mem_str) / 1024, 1)
                    except ValueError:
                        pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass


# Singleton
_monitor: Optional[GreenBoostMonitor] = None


def get_monitor() -> GreenBoostMonitor:
    global _monitor
    if _monitor is None:
        _monitor = GreenBoostMonitor()
        _monitor.refresh()
    return _monitor


# ── Convenience functions for context_builder + statusline ────────────────────

def _canonical_monitor():
    """The canonical read-only client gb_monitor in the greenboost repo. The
    _GbInfo mirror above drifted from greenboost_ioctl.h (phantom kv_compression
    fields); gb_monitor tracks the header exactly. Prefer it for read paths and
    keep this module's control methods (set_kv_reserve etc.) as the actuation
    surface. None when the greenboost checkout isn't importable."""
    try:
        from greenboost_cli.gb_paths import gb_module
        return gb_module("gb_monitor")
    except Exception:
        return None


def get_tier_stats() -> dict | None:
    """Return tier stats dict, or None if GreenBoost is not available.
    Sources from the canonical gb_monitor client when present (its ioctl struct
    matches the header); falls back to this module's local reader otherwise."""
    _canon = _canonical_monitor()
    if _canon is not None:
        try:
            ts = _canon.tier_stats()
            if ts is not None:
                return ts
        except Exception:
            pass
    try:
        m = get_monitor()
        s = m.status
        if not s.loaded:
            return None
        # t1_vram_mb: physical VRAM capacity (from kernel module / nvidia-smi total)
        # t1_used_mb: actual VRAM in use right now (nvidia-smi memory.used — ground truth)
        t1_total = s.gpu_mem_total_mb or int(s.vram_physical_mb)
        t1_used  = s.gpu_mem_used_mb
        return {
            "loaded": s.loaded,
            "gpu_name": s.gpu_name,
            "t1_vram_mb": t1_total,
            "t1_used_mb": t1_used,
            "t2_pool_mb": s.ram_pool_mb,
            "t2_allocated_mb": s.ram_allocated_mb,
            "t2_available_mb": s.ram_available_mb,
            "t3_swap_total_mb": s.nvme_swap_total_mb,
            "t3_swap_used_mb": s.nvme_swap_used_mb,
            "total_combined_gb": s.total_combined_gb,
            "t2_pressure": s.t2_pressure,
            "swap_pressure": s.swap_pressure,
            "kv_compression_bits": s.kv_compression_bits,
            "oom_active": s.oom_active,
        }
    except Exception:
        return None


def get_banner_line(stats: dict | None = None) -> str:
    """Return a one-line GreenBoost status for system prompt injection."""
    if stats is None:
        stats = get_tier_stats()
    if not stats:
        return ""
    t1_used  = round(stats.get("t1_used_mb",    0) / 1024, 1)
    t1_total = round(stats.get("t1_vram_mb",    0) / 1024, 1)
    t2 = round(stats.get("t2_available_mb", 0) / 1024, 1)
    t3 = round(stats.get("t3_swap_total_mb", 0) / 1024, 1)
    total = stats.get("total_combined_gb", 0)
    gpu = stats.get("gpu_name", "")
    tq = stats.get("kv_compression_bits", 0)
    tq_str = f" TurboQuant:{tq}b" if tq else ""
    pressure = stats.get("t2_pressure", 0)
    pressure_str = " T2:warn" if pressure == 1 else (" T2:critical" if pressure == 2 else "")
    t1_str = f"{t1_used}/{t1_total}GB" if t1_used else f"{t1_total}GB"

    line = (f"\n- GreenBoost active: T1={t1_str} VRAM  T2={t2}GB RAM  T3={t3}GB NVMe"
            f"  total={total}GB{tq_str}{pressure_str}")
    if gpu:
        line += f"  GPU:{gpu}"
    return line + "\n"
