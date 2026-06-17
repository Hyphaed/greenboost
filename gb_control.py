#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_control.py — unified GreenBoost runtime actuator layer.

The ONE place that mutates GreenBoost runtime behavior.

Three backends (tried in order per lever):
  sysfs   — /sys/module/greenboost/parameters/<param>  (0644, re-read ~500ms)
  ioctl   — /dev/greenboost  (requires CAP_SYS_ADMIN for most cmds)
  control — /run/greenboost/control  (KEY=VALUE; shim reads every ~2 s)

Design principles (mirror g_t2_warn_adj controller in greenboost_cuda_shim.c):
  clamp        — every value hard-clamped before any write
  step         — bounded incremental; no set-point jumps
  idempotence  — skip write when value unchanged
  rate-limit   — min_interval_s (default 10 s) prevents thrashing
  dry-run gate — when dry_run=True or GB_ORCH_ACTUATE != '1': compute + log only

Privilege model:
  GbControl._priv = (os.geteuid() == 0)
  sysfs/ioctl writes no-op-with-warning when unprivileged.
  Control-file writes work for any user who can write /run/greenboost.

Usage:
    ctrl = GbControl(dry_run=True)           # observe mode; no lever moves
    ctrl = GbControl()                        # actuate when GB_ORCH_ACTUATE=1
    r = ctrl.set_kv_reserve_mb(1024, "kv_approaching_reserve")
    # ActuatorResult(lever='kv_reserve_mb', applied=True, old=512, new=1024, ...)
"""
from __future__ import annotations

import ctypes
import fcntl
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("gb_control")

# ── ioctl helpers (mirrors gb_telemetry.py:115-116) ──────────────────────────
_IOC_READ  = 2
_IOC_WRITE = 1

def _IOR(magic: str, nr: int, size: int) -> int:
    return (_IOC_READ  << 30) | (size << 16) | (ord(magic) << 8) | nr

def _IOW(magic: str, nr: int, size: int) -> int:
    return (_IOC_WRITE << 30) | (size << 16) | (ord(magic) << 8) | nr

def _IOWR(magic: str, nr: int, size: int) -> int:
    return ((_IOC_READ | _IOC_WRITE) << 30) | (size << 16) | (ord(magic) << 8) | nr

def _IO(magic: str, nr: int) -> int:
    return (ord(magic) << 8) | nr


# ── ioctl req structs (from greenboost_ioctl.h) ───────────────────────────────

class _GbKvReserveReq(ctypes.Structure):
    _fields_ = [("reserve_mb", ctypes.c_uint32), ("_pad", ctypes.c_uint32)]

class _GbEvictReq(ctypes.Structure):
    _fields_ = [("buf_id", ctypes.c_int32), ("_pad", ctypes.c_uint32)]

class _GbPoolCapReq(ctypes.Structure):
    _fields_ = [("cap_mb", ctypes.c_uint64), ("prev_mb", ctypes.c_uint64)]

class _GbT3CapReq(ctypes.Structure):
    _fields_ = [("cap_mb", ctypes.c_uint64), ("prev_mb", ctypes.c_uint64)]

class _GbGamingReq(ctypes.Structure):
    _fields_ = [("active", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]

class _GbReleasePidReq(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint32), ("_pad", ctypes.c_uint32)]

_GB_DEV = "/dev/greenboost"
_GB_IOCTL_MAGIC = "G"

_CMD_RESET         = _IO (_GB_IOCTL_MAGIC, 3)
_CMD_EVICT         = _IOW(_GB_IOCTL_MAGIC, 5,  ctypes.sizeof(_GbEvictReq))
_CMD_SET_KV_RESERVE= _IOW(_GB_IOCTL_MAGIC, 9,  ctypes.sizeof(_GbKvReserveReq))
_CMD_SET_POOL_CAP  = _IOWR(_GB_IOCTL_MAGIC, 11, ctypes.sizeof(_GbPoolCapReq))
_CMD_RESET_PHASE   = _IO (_GB_IOCTL_MAGIC, 15)
_CMD_RELEASE_PID   = _IOW(_GB_IOCTL_MAGIC, 16, ctypes.sizeof(_GbReleasePidReq))
_CMD_SET_T3_CAP    = _IOWR(_GB_IOCTL_MAGIC, 17, ctypes.sizeof(_GbT3CapReq))
_CMD_GAMING_MODE   = _IOW(_GB_IOCTL_MAGIC, 20, ctypes.sizeof(_GbGamingReq))

# ── sysfs paths ───────────────────────────────────────────────────────────────
_SYSFS_PREFIX = "/sys/module/greenboost/parameters"
_SYSFS_SAFETY_RESERVE_GB = f"{_SYSFS_PREFIX}/safety_reserve_gb"
_SYSFS_VIRTUAL_VRAM_GB   = f"{_SYSFS_PREFIX}/virtual_vram_gb"
_SYSFS_GAMING_MODE       = f"{_SYSFS_PREFIX}/gaming_mode"
_SYSFS_KV_RESERVE_MB     = f"{_SYSFS_PREFIX}/kv_reserve_mb"
_SYSFS_IDLE_CLEANUP_SEC  = f"{_SYSFS_PREFIX}/idle_cleanup_sec"
_SYSFS_DEBUG_MODE        = f"{_SYSFS_PREFIX}/debug_mode"

# ── control file path ─────────────────────────────────────────────────────────
_CONTROL_FILE = Path("/run/greenboost/control")
_CONTROL_TMP  = Path("/run/greenboost/.control.tmp")


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class ActuatorResult:
    lever:   str
    applied: bool
    old:     Any
    new:     Any
    reason:  str


# ── main class ────────────────────────────────────────────────────────────────

class GbControl:
    """
    Unified GreenBoost runtime actuator.  One instance per supervisor/process.
    Create with dry_run=True for observe/plan-only mode (no lever moves).
    GB_ORCH_ACTUATE=1 additionally gates per-call application.
    """

    def __init__(
        self,
        dry_run: bool = False,
        min_interval_s: float = 10.0,
    ) -> None:
        # Actuate only when explicitly enabled AND not in dry-run mode
        _env_actuate = os.environ.get("GB_ORCH_ACTUATE", "0") == "1"
        self._actuate      = _env_actuate and not dry_run
        self._dry_run      = dry_run
        self.min_interval_s = min_interval_s
        self._priv         = (os.geteuid() == 0)
        # lever state: name → (last_value, last_change_monotonic)
        self._last: dict[str, tuple[Any, float]] = {}

    # ── backends ──────────────────────────────────────────────────────────────

    def _write_sysfs(self, path: str, value: str) -> bool:
        if not self._priv:
            log.warning("[gb_control] no privilege for sysfs write %s", path)
            return False
        try:
            with open(path, "w") as f:
                f.write(value + "\n")
            return True
        except Exception as exc:
            log.warning("[gb_control] sysfs write %s failed: %s", path, exc)
            return False

    def _ioctl(self, cmd: int, req: Optional[ctypes.Structure] = None) -> bool:
        if not self._priv and cmd not in (_CMD_GAMING_MODE,):
            log.warning("[gb_control] no privilege for ioctl 0x%x", cmd)
            return False
        if not os.path.exists(_GB_DEV):
            log.debug("[gb_control] %s absent — skipping ioctl 0x%x", _GB_DEV, cmd)
            return False
        try:
            fd = os.open(_GB_DEV, os.O_RDWR)
            try:
                if req is not None:
                    fcntl.ioctl(fd, cmd, req)
                else:
                    fcntl.ioctl(fd, cmd)
            finally:
                os.close(fd)
            return True
        except Exception as exc:
            log.warning("[gb_control] ioctl 0x%x failed: %s", cmd, exc)
            return False

    def _write_control(self, **kv: Any) -> bool:
        """
        Merge kv pairs into /run/greenboost/control atomically via temp+rename.
        Pairs with the shim's mtime gate — only re-parses when mtime changes.
        """
        existing: dict[str, str] = {}
        try:
            if _CONTROL_FILE.exists():
                for line in _CONTROL_FILE.read_text().splitlines():
                    if "=" in line:
                        k, _, v = line.partition("=")
                        existing[k.strip()] = v.strip()
        except Exception:
            pass

        for k, v in kv.items():
            existing[k] = str(v)

        content = "".join(f"{k}={v}\n" for k, v in sorted(existing.items()))
        try:
            _CONTROL_TMP.parent.mkdir(parents=True, exist_ok=True)
            _CONTROL_TMP.write_text(content)
            _CONTROL_TMP.rename(_CONTROL_FILE)
            return True
        except Exception as exc:
            log.warning("[gb_control] control file write failed: %s", exc)
            return False

    # ── bounded lever helper ──────────────────────────────────────────────────

    def _lever(
        self,
        name: str,
        new: Any,
        lo: Any,
        hi: Any,
        apply_fn: "Any",
        reason: str,
        *,
        step: Optional[Any] = None,
    ) -> ActuatorResult:
        """
        Bounded incremental lever (g_t2_warn_adj discipline):
          clamp → bounded step → idempotence → rate-limit → log → dry-run → apply.
        """
        try:
            new = max(lo, min(hi, new))
        except TypeError:
            pass  # non-numeric (bool)

        old, t = self._last.get(name, (None, 0.0))

        # Bounded incremental: no large set-point jumps
        if step is not None and old is not None:
            try:
                delta = new - old
                delta = max(-step, min(step, delta))
                new = old + delta
                try:
                    new = max(lo, min(hi, new))
                except TypeError:
                    pass
            except TypeError:
                pass

        # Idempotence
        if new == old:
            return ActuatorResult(name, False, old, new, "unchanged")

        # Rate-limit / hysteresis
        now = time.monotonic()
        if now - t < self.min_interval_s:
            return ActuatorResult(name, False, old, new, "rate_limited")

        if self._dry_run or not self._actuate:
            log.info(
                "[gb_control] DRY-RUN lever=%s %s→%s reason=%s", name, old, new, reason
            )
            # Still record in _last so dry-run reports sensible state
            self._last[name] = (new, now)
            return ActuatorResult(name, False, old, new, f"dry_run:{reason}")

        ok = apply_fn(new)
        if ok:
            self._last[name] = (new, now)
        log.info(
            "[gb_control] lever=%s %s→%s applied=%s reason=%s", name, old, new, ok, reason
        )
        return ActuatorResult(name, ok, old, new, reason if ok else "apply_failed")

    # ── public lever functions ────────────────────────────────────────────────

    def set_kv_reserve_mb(self, mb: int, reason: str = "") -> ActuatorResult:
        """
        Set T1 KV reserve.  Preferred: ioctl GB_IOCTL_SET_KV_RESERVE (kernel-authoritative).
        Fallback: control file kv_reserve_mb (shim-local; effective when module absent).
        """
        def _apply(v: int) -> bool:
            req = _GbKvReserveReq(reserve_mb=int(v), _pad=0)
            ok = self._ioctl(_CMD_SET_KV_RESERVE, req)
            # Also write to control file as shim-local fallback
            self._write_control(kv_reserve_mb=v)
            return ok or True  # control-file write always succeeds; log ioctl status above

        return self._lever("kv_reserve_mb", mb, 0, 65536, _apply, reason, step=512)

    def set_safety_reserve_gb(self, gb: int, reason: str = "") -> ActuatorResult:
        def _apply(v: int) -> bool:
            return self._write_sysfs(_SYSFS_SAFETY_RESERVE_GB, str(v))
        return self._lever("safety_reserve_gb", gb, 1, 64, _apply, reason, step=1)

    def set_workstation_reserve_mb(self, mb: int, reason: str = "") -> ActuatorResult:
        """Non-root usable (only writes control file; no sysfs/ioctl)."""
        def _apply(v: int) -> bool:
            return self._write_control(workstation_reserve_mb=v)
        return self._lever("workstation_reserve_mb", mb, 0, 8192, _apply, reason, step=256)

    def set_virtual_vram_gb(self, gb: int, reason: str = "") -> ActuatorResult:
        def _apply(v: int) -> bool:
            return self._write_sysfs(_SYSFS_VIRTUAL_VRAM_GB, str(v))
        return self._lever("virtual_vram_gb", gb, 0, 512, _apply, reason, step=2)

    def set_gaming_mode(self, on: bool, reason: str = "") -> ActuatorResult:
        def _apply(v: bool) -> bool:
            req = _GbGamingReq(active=int(v), reserved=0)
            ok = self._ioctl(_CMD_GAMING_MODE, req)
            self._write_sysfs(_SYSFS_GAMING_MODE, "1" if v else "0")
            return ok
        return self._lever("gaming_mode", on, False, True, _apply, reason)

    def set_pool_cap_mb(self, mb: int, reason: str = "") -> ActuatorResult:
        def _apply(v: int) -> bool:
            req = _GbPoolCapReq(cap_mb=int(v), prev_mb=0)
            return self._ioctl(_CMD_SET_POOL_CAP, req)
        return self._lever("pool_cap_mb", mb, 4096, 2**20, _apply, reason, step=2048)

    def set_t3_cap_mb(self, mb: int, reason: str = "") -> ActuatorResult:
        """0 = disk-limited (no cap)."""
        def _apply(v: int) -> bool:
            req = _GbT3CapReq(cap_mb=int(v), prev_mb=0)
            return self._ioctl(_CMD_SET_T3_CAP, req)
        return self._lever("t3_cap_mb", mb, 0, 2**20, _apply, reason, step=4096)

    def set_idle_cleanup_sec(self, s: int, reason: str = "") -> ActuatorResult:
        def _apply(v: int) -> bool:
            return self._write_sysfs(_SYSFS_IDLE_CLEANUP_SEC, str(v))
        return self._lever("idle_cleanup_sec", s, 5, 3600, _apply, reason, step=30)

    def set_debug_mode(self, on: bool, reason: str = "") -> ActuatorResult:
        def _apply(v: bool) -> bool:
            return self._write_sysfs(_SYSFS_DEBUG_MODE, "1" if v else "0")
        return self._lever("debug_mode", on, False, True, _apply, reason)

    def evict_cold(self, buf_id: int = -1, reason: str = "") -> ActuatorResult:
        """Evict cold T2 buffers.  buf_id=-1 → kernel picks LRU cold candidate."""
        name = "evict_cold"
        old, t = self._last.get(name, (None, 0.0))
        now = time.monotonic()
        rate_limit = self.min_interval_s * 3  # evict is aggressive; triple rate-limit
        if now - t < rate_limit:
            return ActuatorResult(name, False, old, buf_id, "rate_limited")
        if self._dry_run or not self._actuate:
            log.info("[gb_control] DRY-RUN evict_cold buf_id=%d reason=%s", buf_id, reason)
            self._last[name] = (buf_id, now)
            return ActuatorResult(name, False, old, buf_id, f"dry_run:{reason}")
        req = _GbEvictReq(buf_id=buf_id, _pad=0)
        ok = self._ioctl(_CMD_EVICT, req)
        if ok:
            self._last[name] = (buf_id, now)
        log.info("[gb_control] evict_cold buf_id=%d applied=%s reason=%s", buf_id, ok, reason)
        return ActuatorResult(name, ok, old, buf_id, reason if ok else "apply_failed")

    def request_phase_reset(self, reason: str = "") -> ActuatorResult:
        """Increment phase_reset_seq; shim resets alloc phase on next 64-alloc boundary."""
        name = "phase_reset"
        old, t = self._last.get(name, (None, 0.0))
        now = time.monotonic()
        if now - t < self.min_interval_s * 3:
            return ActuatorResult(name, False, old, 1, "rate_limited")
        if self._dry_run or not self._actuate:
            log.info("[gb_control] DRY-RUN phase_reset reason=%s", reason)
            self._last[name] = (1, now)
            return ActuatorResult(name, False, old, 1, f"dry_run:{reason}")
        ok = self._ioctl(_CMD_RESET_PHASE)
        if ok:
            self._last[name] = (1, now)
        log.info("[gb_control] phase_reset applied=%s reason=%s", ok, reason)
        return ActuatorResult(name, ok, old, 1, reason if ok else "apply_failed")

    def reset_oom_guard(self, reason: str = "") -> ActuatorResult:
        """Clear kernel OOM guard (GB_IOCTL_RESET). Same call as supervisor:274."""
        name = "oom_reset"
        old, t = self._last.get(name, (None, 0.0))
        now = time.monotonic()
        if now - t < self.min_interval_s:
            return ActuatorResult(name, False, old, 1, "rate_limited")
        if self._dry_run or not self._actuate:
            log.info("[gb_control] DRY-RUN oom_reset reason=%s", reason)
            self._last[name] = (1, now)
            return ActuatorResult(name, False, old, 1, f"dry_run:{reason}")
        ok = self._ioctl(_CMD_RESET)
        if ok:
            self._last[name] = (1, now)
        log.info("[gb_control] oom_reset applied=%s reason=%s", ok, reason)
        return ActuatorResult(name, ok, old, 1, reason if ok else "apply_failed")

    def release_pid(self, pid: int = 0, reason: str = "") -> ActuatorResult:
        """Release all T2/T3 kernel buffers for pid (0=caller). Requires CAP_SYS_ADMIN for other PIDs."""
        name = f"release_pid_{pid}"
        old, t = self._last.get(name, (None, 0.0))
        now = time.monotonic()
        if now - t < self.min_interval_s:
            return ActuatorResult(name, False, old, pid, "rate_limited")
        if self._dry_run or not self._actuate:
            log.info("[gb_control] DRY-RUN release_pid pid=%d reason=%s", pid, reason)
            self._last[name] = (pid, now)
            return ActuatorResult(name, False, old, pid, f"dry_run:{reason}")
        req = _GbReleasePidReq(pid=pid, _pad=0)
        ok = self._ioctl(_CMD_RELEASE_PID, req)
        if ok:
            self._last[name] = (pid, now)
        log.info("[gb_control] release_pid=%d applied=%s reason=%s", pid, ok, reason)
        return ActuatorResult(name, ok, old, pid, reason if ok else "apply_failed")

    def set_kv_size_threshold_mb(self, mb: int, reason: str = "") -> ActuatorResult:
        """Shim-side KV size threshold (control file only)."""
        def _apply(v: int) -> bool:
            return self._write_control(kv_size_threshold_mb=v)
        return self._lever("kv_size_threshold_mb", mb, 1, 4096, _apply, reason, step=16)

    def set_swa_window_mb(self, mb: int, reason: str = "") -> ActuatorResult:
        """Shim-side SWA sliding-window KV size (control file; 0=disable)."""
        def _apply(v: int) -> bool:
            return self._write_control(swa_window_mb=v)
        return self._lever("swa_window_mb", mb, 0, 65536, _apply, reason, step=1024)

    def set_phase_detect(self, on: bool, reason: str = "") -> ActuatorResult:
        """Enable/disable shim phase detection (control file)."""
        def _apply(v: bool) -> bool:
            return self._write_control(phase_detect=int(v))
        return self._lever("phase_detect", on, False, True, _apply, reason)

    # ── state dump ───────────────────────────────────────────────────────────

    def dump(self) -> dict:
        """Current lever states for debug_dump() / observability."""
        return {
            name: {"value": val, "last_change_ts": ts}
            for name, (val, ts) in self._last.items()
        }
