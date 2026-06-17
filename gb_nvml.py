#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_nvml.py — unified pynvml singleton for GreenBoost.

Replaces four divergent pynvml impls:
  NVMLProvider     (gb_telemetry.py)   — hot-path 500 ms poller
  _NVMLSampler     (gb_supervisor.py)  — supervisor tick queries
  inline pynvml    (gb_vitals_helper.py) — superset queries incl power_limit

get_nvml(device=0) returns a per-process NvmlHandle singleton.
atexit shutdown is registered once; all callers share the same handle.
Each query method is try/except → safe default (0 / '' / False).
"""
from __future__ import annotations
import atexit

_handles: "dict[int, NvmlHandle]" = {}
_atexit_registered = False


class NvmlHandle:
    """
    Lazy init-once NVML handle for device `device`.
    Covers the superset of queries used by all prior call sites:
    mem / util / temp / power / power_limit / clocks / ECC DBE+SBE volatile+agg / name.
    """

    def __init__(self, device: int = 0) -> None:
        self._device = device
        self._ok     = False
        self._h      = None
        self._nvml   = None
        try:
            import pynvml
            self._nvml = pynvml
            pynvml.nvmlInit()
            self._h  = pynvml.nvmlDeviceGetHandleByIndex(device)
            self._ok = True
        except Exception:
            pass

    @property
    def ok(self) -> bool:
        return self._ok

    def device_name(self) -> str:
        if not self._ok:
            return ""
        try:
            n = self._nvml.nvmlDeviceGetName(self._h)
            if isinstance(n, bytes):
                n = n.decode("utf-8", errors="replace")
            return n.strip()
        except Exception:
            return ""

    def mem(self) -> "tuple[int, int, int, float]":
        """(used_mb, free_mb, total_mb, used_pct)"""
        if not self._ok:
            return 0, 0, 0, 0.0
        try:
            m     = self._nvml.nvmlDeviceGetMemoryInfo(self._h)
            used  = m.used  // (1 << 20)
            free  = m.free  // (1 << 20)
            total = m.total // (1 << 20)
            pct   = 100.0 * used / (total or 1)
            return used, free, total, pct
        except Exception:
            return 0, 0, 0, 0.0

    def total_mb(self) -> int:
        if not self._ok:
            return 0
        try:
            m = self._nvml.nvmlDeviceGetMemoryInfo(self._h)
            return m.total // (1 << 20)
        except Exception:
            return 0

    def util(self) -> "tuple[float, float]":
        """(gpu_util_pct, mem_copy_util_pct)"""
        if not self._ok:
            return 0.0, 0.0
        try:
            u = self._nvml.nvmlDeviceGetUtilizationRates(self._h)
            return float(u.gpu), float(u.memory)
        except Exception:
            return 0.0, 0.0

    def temp_c(self) -> float:
        if not self._ok:
            return 0.0
        try:
            return float(self._nvml.nvmlDeviceGetTemperature(
                self._h, self._nvml.NVML_TEMPERATURE_GPU))
        except Exception:
            return 0.0

    def power_w(self) -> float:
        if not self._ok:
            return 0.0
        try:
            return self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
        except Exception:
            return 0.0

    def power_limit_w(self) -> float:
        if not self._ok:
            return 0.0
        try:
            return self._nvml.nvmlDeviceGetPowerManagementLimit(self._h) / 1000.0
        except Exception:
            return 0.0

    def clocks_mhz(self) -> "tuple[int, int]":
        """(sm_clock_mhz, mem_clock_mhz)"""
        if not self._ok:
            return 0, 0
        sm = mem = 0
        try:
            sm  = int(self._nvml.nvmlDeviceGetClockInfo(self._h, self._nvml.NVML_CLOCK_SM))
        except Exception:
            pass
        try:
            mem = int(self._nvml.nvmlDeviceGetClockInfo(self._h, self._nvml.NVML_CLOCK_MEM))
        except Exception:
            pass
        return sm, mem

    def ecc_dbe_volatile(self) -> int:
        if not self._ok:
            return 0
        try:
            return int(self._nvml.nvmlDeviceGetTotalEccErrors(
                self._h,
                self._nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                self._nvml.NVML_VOLATILE_ECC,
            ))
        except Exception:
            return 0

    def ecc_dbe_aggregate(self) -> int:
        if not self._ok:
            return 0
        try:
            return int(self._nvml.nvmlDeviceGetTotalEccErrors(
                self._h,
                self._nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                self._nvml.NVML_AGGREGATE_ECC,
            ))
        except Exception:
            return 0

    def ecc_sbe_volatile(self) -> int:
        if not self._ok:
            return 0
        try:
            return int(self._nvml.nvmlDeviceGetTotalEccErrors(
                self._h,
                self._nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                self._nvml.NVML_VOLATILE_ECC,
            ))
        except Exception:
            return 0

    def pcie_mb_s(self) -> "tuple[float, float]":
        """(tx_mb_s, rx_mb_s)"""
        if not self._ok:
            return 0.0, 0.0
        try:
            tx = self._nvml.nvmlDeviceGetPcieThroughput(
                self._h, self._nvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0
            rx = self._nvml.nvmlDeviceGetPcieThroughput(
                self._h, self._nvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0
            return tx, rx
        except Exception:
            return 0.0, 0.0

    def sm_clock_mhz(self) -> int:
        """Convenience: SM clock only."""
        return self.clocks_mhz()[0]

    def close(self) -> None:
        """Shutdown NVML for this handle.  Idempotent."""
        if self._ok and self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._ok = False


def _shutdown_all() -> None:
    """Called once at process exit — close every open handle."""
    for h in list(_handles.values()):
        h.close()
    _handles.clear()


def get_nvml(device: int = 0) -> NvmlHandle:
    """
    Return the per-process NvmlHandle singleton for *device*.
    atexit shutdown registered on first call.  Subsequent calls return
    the cached handle — no re-init.  Thread-safe under the GIL.
    """
    global _atexit_registered
    if device not in _handles:
        _handles[device] = NvmlHandle(device)
        if not _atexit_registered:
            atexit.register(_shutdown_all)
            _atexit_registered = True
    return _handles[device]
