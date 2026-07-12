#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_model_tier.py , GreenBoost model paging: HBM → Pinned RAM → NVMe.

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


def _module_size_gb(module) -> float:
    """Approximate resident size of a module's parameters+buffers, in GiB."""
    try:
        total = sum(p.numel() * p.element_size() for p in module.parameters())
        total += sum(b.numel() * b.element_size() for b in module.buffers())
        return total / 2**30
    except Exception:
        return 0.0


def _df_emit_tier(name: str, from_tier: str, to_tier: str, size_gb: float,
                  duration_s: float, extra: "Optional[dict]" = None) -> None:
    """Record a T1/T2/T3 model placement move to the dataflux log , works
    standalone on a single host, no cluster/feeder required. Best-effort,
    never raises."""
    try:
        import gb_dataflux
        evt = {
            "node": "host", "label": "gb_model_tier", "kind": "tier_move",
            "n_items": 1, "items": [name], "duration_s": round(duration_s, 3),
            "status": "ok", "from_tier": from_tier, "to_tier": to_tier,
            "size_gb": round(size_gb, 3),
        }
        if extra:
            evt.update(extra)
        gb_dataflux.emit(evt)
    except Exception:
        pass


# ── T3 transparent zstd compression ───────────────────────────────────────────
# T3 is the cold tier by definition, so decompression cost is dwarfed by the
# NVMe read + module rematerialization it rides on; trading a little CPU for a
# smaller on-disk footprint is always worth it there.  Never a hard dependency:
# probe python-zstandard, then the `zstd` CLI, then fall back to plain torch.save
# (matching the repo's vendored-deps convention).  Opt out with GB_T3_COMPRESS=0.
_T3_ZSTD_BACKEND = None   # None (unprobed) | "py" | "cli" | "none"


def _t3_zstd_backend() -> str:
    """Return the available compression backend: 'py', 'cli', or 'none'."""
    global _T3_ZSTD_BACKEND
    if _T3_ZSTD_BACKEND is not None:
        return _T3_ZSTD_BACKEND
    if os.environ.get("GB_T3_COMPRESS", "1") == "0":
        _T3_ZSTD_BACKEND = "none"
        return _T3_ZSTD_BACKEND
    try:
        import zstandard  # noqa: F401
        _T3_ZSTD_BACKEND = "py"
    except Exception:
        import shutil
        _T3_ZSTD_BACKEND = "cli" if shutil.which("zstd") else "none"
    return _T3_ZSTD_BACKEND


def _t3_save(state: dict, base_path: str) -> "tuple[str, int]":
    """Save `state` to T3, compressing when a backend is available.  `base_path`
    ends in '.pt'; the compressed variant appends '.zst'.  Returns
    (actual_path, bytes_on_disk)."""
    backend = _t3_zstd_backend()
    if backend == "py":
        import zstandard
        zpath = base_path + ".zst"
        cctx = zstandard.ZstdCompressor(level=3)
        with open(zpath, "wb") as fh, cctx.stream_writer(fh) as comp:
            torch.save(state, comp)   # streamed, no full in-memory copy
        return zpath, os.path.getsize(zpath)
    if backend == "cli":
        import subprocess
        zpath = base_path + ".zst"
        torch.save(state, base_path)
        try:
            subprocess.run(["zstd", "-q", "-3", "-f", "-o", zpath, base_path],
                           check=True)
            os.remove(base_path)
            return zpath, os.path.getsize(zpath)
        except Exception:
            # CLI failed mid-flight: keep the plain file we already wrote.
            if os.path.exists(zpath):
                os.remove(zpath)
            return base_path, os.path.getsize(base_path)
    torch.save(state, base_path)
    return base_path, os.path.getsize(base_path)


def _t3_load(path: str) -> dict:
    """Load a T3 checkpoint written by `_t3_save` (or a legacy plain '.pt').

    A '.zst' on disk MUST be decompressed regardless of the current
    GB_T3_COMPRESS setting, so decompression is capability-driven (try the
    python lib, then the CLI) rather than gated on the cached backend flag."""
    if not path.endswith(".zst"):
        return torch.load(path, map_location="cpu", weights_only=True)
    try:
        import zstandard
        import io
        buf = io.BytesIO()
        with open(path, "rb") as fh:
            zstandard.ZstdDecompressor().copy_stream(fh, buf)   # streamed
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=True)
    except ImportError:
        pass
    import subprocess
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".pt", dir=os.path.dirname(path))
    os.close(fd)
    try:
        subprocess.run(["zstd", "-d", "-q", "-f", "-o", tmp, path], check=True)
        return torch.load(tmp, map_location="cpu", weights_only=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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
        t0 = time.time()
        from_tier = e.tier

        if e.tier == Tier.T3:
            self._load_from_t3(e)

        # Ensure headroom before promoting
        self._evict_for_headroom()

        if self.async_transfers:
            import gb_stream_sched as gs
            # gs.wait_for(..., on="gemm") only syncs the named "gemm"
            # stream, which is current only inside an explicit
            # `with gs.on("gemm")` block - callers here run on torch's
            # ambient current stream instead, so that wait was a no-op
            # and the GPU read could race the H2D copy. Wait directly
            # on the stream that will actually run the next forward.
            with gs.on("transfer"):
                e.module.to("cuda", non_blocking=True)
            ev = torch.cuda.Event()
            gs.stream("transfer").record_event(ev)
            torch.cuda.current_stream().wait_event(ev)
        else:
            e.module.to("cuda")

        e.tier = Tier.T1
        e.last_used = time.time()
        print(f"[gb_tier] promoted '{name}' → T1 (GPU)", flush=True)
        _df_emit_tier(name, from_tier, Tier.T1, _module_size_gb(e.module), time.time() - t0)

    def demote(self, name: str):
        """
        Move model to T2 (CPU pinned RAM). Frees T1 VRAM immediately.
        """
        e = self._entries[name]
        if e.tier == Tier.T2:
            return
        t0 = time.time()
        from_tier = e.tier

        if self.async_transfers:
            import gb_stream_sched as gs
            with gs.on("transfer"):
                e.module.to("cpu", non_blocking=True)
            # The consumer here is the host (callers read the CPU tensor
            # directly, e.g. evict()'s state_dict() save right after) -
            # a device-side stream wait doesn't block the host, so use
            # an event sync instead of gs.wait_for(..., on="gemm")
            # (which was a no-op for the same reason as in promote()).
            ev = torch.cuda.Event()
            gs.stream("transfer").record_event(ev)
            ev.synchronize()
        else:
            e.module.to("cpu")

        e.tier = Tier.T2
        e.last_used = time.time()
        print(f"[gb_tier] demoted '{name}' → T2 (CPU)", flush=True)
        _df_emit_tier(name, from_tier, Tier.T2, _module_size_gb(e.module), time.time() - t0)

    def evict(self, name: str):
        """
        Move model to T3 (NVMe). Saves state_dict to disk and frees RAM.
        Use when T2 is under pressure (t2_pressure == 2).
        """
        e = self._entries[name]
        if e.tier == Tier.T3:
            return
        t0 = time.time()
        from_tier = e.tier

        if e.tier == Tier.T1:
            self.demote(name)

        size_gb = _module_size_gb(e.module)
        base_path = os.path.join(self.t3_dir, f"{name}.pt")
        t3_path, disk_bytes = _t3_save(e.module.state_dict(), base_path)
        # Move to meta device: frees all parameter memory while preserving shapes
        # so _load_from_t3 can use to_empty()+load_state_dict(assign=True) later.
        e.module.to("meta")
        e.tier = Tier.T3
        e.t3_path = t3_path
        compressed = t3_path.endswith(".zst")
        ratio = (size_gb * 2**30 / disk_bytes) if compressed and disk_bytes else 1.0
        print(f"[gb_tier] evicted '{name}' → T3 (NVMe) at {t3_path}"
              f"{f' (zstd {ratio:.2f}x)' if compressed else ''}", flush=True)
        _df_emit_tier(name, from_tier, Tier.T3, size_gb, time.time() - t0,
                      extra={"compressed": compressed,
                             "disk_mb": round(disk_bytes / 2**20, 1),
                             "compress_ratio": round(ratio, 2)})

    def _load_from_t3(self, e: _ModelEntry):
        if not e.t3_path or not os.path.exists(e.t3_path):
            raise FileNotFoundError(f"T3 checkpoint for '{e.name}' not found: {e.t3_path}")
        state = _t3_load(e.t3_path)
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
