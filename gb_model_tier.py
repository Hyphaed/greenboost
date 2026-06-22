#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_model_tier.py — GreenBoost model paging: HBM → Pinned RAM → NVMe.

Integrates with the existing GreenBoost T1/T2/T3 infrastructure:
  T1 = GPU VRAM (HBM, ~12 GB on RTX 5070)
  T2 = System DDR RAM (pinned, GreenBoost managed, up to ~80% of RAM)
  T3 = NVMe swap (kernel-managed via /var/lib/greenboost/t3_store)

Uses GreenBoost ioctl advise hints to help the kernel's LRU decide eviction
order: GB_MADVISE_HOT (keep in T2 / don't evict), GB_MADVISE_COLD (evict
sooner), GB_MADVISE_FREEZE (pin).

Also uses GB_IOCTL_SESSION_ACTIVE / SESSION_IDLE for pipeline stage transitions.

Usage:
    from gb_model_tier import ModelTierManager

    tm = ModelTierManager(hbm_headroom_mb=1024)
    tm.register("unet",      pipe.unet)
    tm.register("vae",       pipe.vae)
    tm.register("encoder",   pipe.text_encoder)

    # Before denoising: promote denoiser, demote encoders
    tm.promote("unet")
    tm.demote("encoder")

    # When VRAM pressure > threshold: auto-evict cold models
    tm.auto_evict(tel.sample())
"""
from __future__ import annotations

import ctypes
import fcntl
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


# ── GreenBoost ioctl constants ────────────────────────────────────────────────
_GB_DEV = "/dev/greenboost"

_IOC_WRITE = 1
_IOC_READ  = 2

def _IOW(magic, nr, size):
    return (_IOC_WRITE << 30) | (size << 16) | (ord(magic) << 8) | nr

def _IOR(magic, nr, size):
    return (_IOC_READ << 30) | (size << 16) | (ord(magic) << 8) | nr

class _GbMadviseReq(ctypes.Structure):
    _fields_ = [("buf_id", ctypes.c_int32), ("advise", ctypes.c_uint32)]

class _GbSessionReq(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]

_GB_MADVISE_COLD    = 0
_GB_MADVISE_HOT     = 1
_GB_MADVISE_FREEZE  = 2

_GB_IOCTL_MADVISE       = _IOW('G', 4, ctypes.sizeof(_GbMadviseReq))
_GB_IOCTL_SESSION_IDLE  = _IOW('G', 18, ctypes.sizeof(_GbSessionReq))
_GB_IOCTL_SESSION_ACTIVE = _IOW('G', 19, ctypes.sizeof(_GbSessionReq))


def _gb_session(active: bool):
    """Signal GreenBoost kernel that this PID is active/idle."""
    if not os.path.exists(_GB_DEV):
        return
    try:
        with open(_GB_DEV, "rb") as f:
            req = _GbSessionReq(pid=0, reserved=0)
            cmd = _GB_IOCTL_SESSION_ACTIVE if active else _GB_IOCTL_SESSION_IDLE
            fcntl.ioctl(f.fileno(), cmd, req)
    except Exception:
        pass


# ── Tier definitions ──────────────────────────────────────────────────────────
class Tier:
    T1 = "T1_HBM"    # GPU VRAM
    T2 = "T2_DDR"    # Pinned RAM (GreenBoost managed)
    T3 = "T3_NVME"   # NVMe (saved to disk via torch.save)


@dataclass
class _ModelEntry:
    name: str
    module: torch.nn.Module
    tier: str = Tier.T1
    last_used: float = field(default_factory=time.time)
    t3_path: Optional[str] = None   # path when evicted to NVMe


class ModelTierManager:
    """
    Manage model component placement across T1 (GPU) / T2 (CPU pinned) / T3 (NVMe).

    Designed for Diffusers-style pipelines where multiple large sub-models
    (UNet, VAE, text encoders, ControlNets) must share a 12 GB GPU.

    Parameters
    ----------
    hbm_headroom_mb : int
        Keep at least this many MB free in T1 before promoting.
        Default: 1024 (1 GB headroom for activations/latents).
    t3_dir : str
        Directory for NVMe-evicted model checkpoints.
    async_transfers : bool
        Use gb_stream_sched transfer stream for non-blocking H2D/D2H.
    """

    def __init__(
        self,
        hbm_headroom_mb: int = 1024,
        t3_dir: str = "/var/lib/greenboost/model_pages",
        async_transfers: bool = True,
    ):
        self.hbm_headroom_mb = hbm_headroom_mb
        self.t3_dir = t3_dir
        self.async_transfers = async_transfers
        self._entries: Dict[str, _ModelEntry] = {}
        os.makedirs(t3_dir, exist_ok=True)
        _gb_session(active=True)

    def register(self, name: str, module: torch.nn.Module, tier: str = Tier.T1):
        """Register a model component for tier management."""
        self._entries[name] = _ModelEntry(name=name, module=module, tier=tier)

    # ── tier transitions ──────────────────────────────────────────────────────

    def promote(self, name: str):
        """
        Move model to T1 (GPU). If currently at T3 (NVMe), loads from disk first.
        If VRAM headroom is insufficient, demotes LRU candidates first.
        """
        e = self._entries[name]
        if e.tier == Tier.T1:
            e.last_used = time.time()
            return

        if e.tier == Tier.T3:
            self._load_from_t3(e)

        # Ensure headroom before promoting
        self._evict_for_headroom()

        if self.async_transfers:
            import gb_stream_sched as gs
            with gs.on("transfer"):
                e.module.to("cuda", non_blocking=True)
            gs.wait_for("transfer", on="gemm")
        else:
            e.module.to("cuda")

        e.tier = Tier.T1
        e.last_used = time.time()
        print(f"[gb_tier] promoted '{name}' → T1 (GPU)", flush=True)

    def demote(self, name: str):
        """
        Move model to T2 (CPU pinned RAM). Frees T1 VRAM immediately.
        """
        e = self._entries[name]
        if e.tier == Tier.T2:
            return

        if self.async_transfers:
            import gb_stream_sched as gs
            with gs.on("transfer"):
                e.module.to("cpu", non_blocking=True)
            gs.wait_for("transfer", on="gemm")
        else:
            e.module.to("cpu")

        e.tier = Tier.T2
        e.last_used = time.time()
        print(f"[gb_tier] demoted '{name}' → T2 (CPU)", flush=True)

    def evict(self, name: str):
        """
        Move model to T3 (NVMe). Saves state_dict to disk and frees RAM.
        Use when T2 is under pressure (t2_pressure == 2).
        """
        e = self._entries[name]
        if e.tier == Tier.T3:
            return

        if e.tier == Tier.T1:
            self.demote(name)

        t3_path = os.path.join(self.t3_dir, f"{name}.pt")
        torch.save(e.module.state_dict(), t3_path)
        # Move to meta device: frees all parameter memory while preserving shapes
        # so _load_from_t3 can use to_empty()+load_state_dict(assign=True) later.
        e.module.to("meta")
        e.tier = Tier.T3
        e.t3_path = t3_path
        print(f"[gb_tier] evicted '{name}' → T3 (NVMe) at {t3_path}", flush=True)

    def _load_from_t3(self, e: _ModelEntry):
        if not e.t3_path or not os.path.exists(e.t3_path):
            raise FileNotFoundError(f"T3 checkpoint for '{e.name}' not found: {e.t3_path}")
        state = torch.load(e.t3_path, map_location="cpu", weights_only=True)
        # Re-allocate parameter storage (evict moved module to meta device) then
        # fill from saved state.  assign=True avoids in-place copy size checks.
        e.module.to_empty(device="cpu")
        e.module.load_state_dict(state, assign=True)
        e.tier = Tier.T2
        print(f"[gb_tier] restored '{e.name}' from T3 → T2", flush=True)

    # ── automatic eviction ────────────────────────────────────────────────────

    def auto_evict(self, metrics: "GpuMetrics"):
        """
        Called with a telemetry snapshot. Demotes / evicts based on pressure:
          fb_used_pct > 90%  → demote LRU T1 model
          t2_pressure == 2   → evict LRU T2 model to T3
        """
        if metrics.fb_used_pct > 90.0:
            lru = self._lru_in_tier(Tier.T1)
            if lru:
                self.demote(lru)

        if metrics.gb and metrics.gb.t2_pressure >= 2:
            lru = self._lru_in_tier(Tier.T2)
            if lru:
                self.evict(lru)

    def _lru_in_tier(self, tier: str) -> Optional[str]:
        candidates = [(e.last_used, e.name) for e in self._entries.values() if e.tier == tier]
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def _evict_for_headroom(self):
        """Demote T1 models until we have hbm_headroom_mb free."""
        if not torch.cuda.is_available():
            return
        props = torch.cuda.get_device_properties(0)
        total_mb = props.total_memory // (1024 * 1024)
        while True:
            used_mb = torch.cuda.memory_allocated() // (1024 * 1024)
            if (total_mb - used_mb) >= self.hbm_headroom_mb:
                break
            lru = self._lru_in_tier(Tier.T1)
            if not lru:
                break
            self.demote(lru)

    # ── session lifecycle ─────────────────────────────────────────────────────

    def session_idle(self):
        """Signal GreenBoost kernel that inference is idle (lower priority)."""
        _gb_session(active=False)

    def session_active(self):
        """Signal GreenBoost kernel that inference is active (higher priority)."""
        _gb_session(active=True)

    def close(self):
        """Demote all T1 models and signal session idle."""
        for name, e in self._entries.items():
            if e.tier == Tier.T1:
                self.demote(name)
        _gb_session(active=False)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
