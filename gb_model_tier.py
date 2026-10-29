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

    tm = ModelTierManager()   # headroom auto-derived: 10% of VRAM (gb_topology)
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
from gb_evict_policy import (  # noqa: E402
    PRIORITY_DEFAULT as _PRIORITY_DEFAULT,
    PerTierEvictionController,
    clamp_priority,
)


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
    # KV-cache tier-serde opt-in (missing_features.md item (e)). None
    # (default) , byte-identical to pre-KV-compression behavior. Only set
    # these for a component whose state_dict holds attention cache tensors;
    # passing kv_bits for a weight-bearing module WILL degrade the weights ,
    # see gb_tier_kv.py's module docstring for why this is opt-in only, no
    # name/shape heuristics.
    kv_bits: Optional[int] = None
    kv_keys: Optional[tuple] = None
    kv_codec: Optional[str] = None
    # Eviction priority (gb_evict_policy: 0..100, LOWER is evicted FIRST).
    # Every entry defaults to PRIORITY_DEFAULT, which makes prioritized-LRU
    # degenerate to the plain LRU this manager used before - so leaving it
    # unset changes nothing.
    priority: int = _PRIORITY_DEFAULT


def _module_size_gb(module) -> float:
    """Approximate resident size of a module's parameters+buffers, in GiB."""
    try:
        total = sum(p.numel() * p.element_size() for p in module.parameters())
        total += sum(b.numel() * b.element_size() for b in module.buffers())
        return total / 2**30
    except Exception:
        return 0.0


# Pinned host pages can't be swapped or reclaimed, so pinning must leave the host RAM
# safety floor free. Same 10% policy as the CUDA shim's floor; GREENBOOST_HOST_RAM_SAFETY_MB
# overrides it (in MiB), exactly as it does for the shim.
_PIN_HEADROOM_FRACTION = 0.10


def _meminfo_kib(field: str) -> int:
    """One /proc/meminfo field in KiB, 0 when it can't be read."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _pinned_budget_ok(nbytes: int) -> bool:
    """True when pinning `nbytes` still leaves the host RAM safety floor available.
    Unknown memory figures count as "no": never pin blind."""
    total_kib, avail_kib = _meminfo_kib("MemTotal"), _meminfo_kib("MemAvailable")
    if not total_kib or not avail_kib:
        return False
    override = os.environ.get("GREENBOOST_HOST_RAM_SAFETY_MB", "")
    floor_kib = int(override) * 1024 if override.isdigit() else int(total_kib * _PIN_HEADROOM_FRACTION)
    return (avail_kib - floor_kib) * 1024 >= nbytes


def _demote_to_pinned(t: "torch.Tensor") -> "torch.Tensor":
    """CUDA tensor -> pinned host tensor, copied asynchronously. A pageable destination
    turns a "non_blocking" device-to-host copy into a staged, effectively synchronous
    one, and makes the later promote copy staged as well. Anything that isn't a dense
    CUDA tensor takes the ordinary path."""
    if not t.is_cuda or t.layout != torch.strided:
        return t.to("cpu")
    dst = torch.empty_like(t, device="cpu", pin_memory=True)
    dst.copy_(t, non_blocking=True)
    return dst


# Process-wide tier-thrashing detector (gb_thrash). Observation only: it
# counts T1<->T2 ping-pong and reports, and nothing reads its verdict to make
# a placement decision. Hooked into _df_emit_tier rather than into each
# transition so it sees exactly the moves the flight recorder sees, and cannot
# drift out of step with them.
_THRASH = None


def thrash_detector():
    """The process-wide ThrashDetector, created on first use."""
    global _THRASH
    if _THRASH is None:
        from gb_thrash import ThrashDetector
        _THRASH = ThrashDetector()
    return _THRASH


def _df_emit_tier(name: str, from_tier: str, to_tier: str, size_gb: float,
                  duration_s: float, extra: "Optional[dict]" = None) -> None:
    """Record a T1/T2/T3 model placement move to the dataflux log , works
    standalone on a single host, no cluster/feeder required. Best-effort,
    never raises."""
    thrash_extra = {}
    try:
        st = thrash_detector().record_move(name, from_tier, to_tier)
        thrash_extra = {
            "thrash_events": st.num_thrashing_events,
            "thrash_active": thrash_detector().is_thrashing(name),
        }
    except Exception:  # noqa: BLE001  never let observation break a tier move
        pass
    try:
        import gb_dataflux
        evt = {
            "node": "host", "label": "gb_model_tier", "kind": "tier_move",
            "n_items": 1, "items": [name], "duration_s": round(duration_s, 3),
            "status": "ok", "from_tier": from_tier, "to_tier": to_tier,
            "size_gb": round(size_gb, 3),
        }
        evt.update(thrash_extra)
        # Was this spill ever undone? Only the mover knows, and only now:
        # after the fact "still resident in T2" and "evicted from T2" are
        # both just "no promotion followed". Hooked here for the same reason
        # gb_thrash is , it sees exactly the moves the recorder sees.
        evt.update(gb_dataflux.spill_outcome(name, from_tier, to_tier, size_gb))
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


def _t3_save(state: dict, base_path: str, kv_bits: "Optional[int]" = None,
            kv_keys: "Optional[tuple]" = None, kv_codec: "Optional[str]" = None,
            device: str = "cpu") -> "tuple[str, int, dict]":
    """Save `state` to T3, compressing when a backend is available.  `base_path`
    ends in '.pt'; the compressed variant appends '.zst'.  Returns
    (actual_path, bytes_on_disk, kv_stats).

    kv_bits=None (default) , byte-identical to the pre-KV-compression
    behavior, kv_stats is empty. kv_bits set , KV-cache-shaped tensors in
    `state` are quantized FIRST (gb_tier_kv.encode_state), THEN the result
    goes through the existing zstd path below unchanged , order matters:
    zstd on packed uint8 indices still earns a little (norms + framing),
    and non-KV tensors in the same state_dict stay untouched and bit-exact
    either way (missing_features.md item (e))."""
    kv_stats: dict = {}
    if kv_bits is not None:
        import gb_tier_kv
        state, kv_stats = gb_tier_kv.encode_state(
            state, kv_bits, keys=kv_keys, codec=kv_codec, device=device)
    backend = _t3_zstd_backend()
    if backend == "py":
        import zstandard
        zpath = base_path + ".zst"
        cctx = zstandard.ZstdCompressor(level=3)
        with open(zpath, "wb") as fh, cctx.stream_writer(fh) as comp:
            torch.save(state, comp)   # streamed, no full in-memory copy
        return zpath, os.path.getsize(zpath), kv_stats
    if backend == "cli":
        import subprocess
        zpath = base_path + ".zst"
        torch.save(state, base_path)
        try:
            subprocess.run(["zstd", "-q", "-3", "-f", "-o", zpath, base_path],
                           check=True)
            os.remove(base_path)
            return zpath, os.path.getsize(zpath), kv_stats
        except Exception:
            # CLI failed mid-flight: keep the plain file we already wrote.
            if os.path.exists(zpath):
                os.remove(zpath)
            return base_path, os.path.getsize(base_path), kv_stats
    torch.save(state, base_path)
    return base_path, os.path.getsize(base_path), kv_stats


def _t3_load_raw(path: str) -> dict:
    """Load a T3 checkpoint written by `_t3_save` (or a legacy plain '.pt'),
    WITHOUT reversing any KV-cache tier-serde encoding , see `_t3_load` for
    the public entry point that also does that.

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


def _t3_load(path: str) -> dict:
    """Load a T3 checkpoint written by `_t3_save`, reversing KV-cache
    tier-serde encoding when present (gb_tier_kv.decode_state is a no-op ,
    returns its argument by identity , for every checkpoint that doesn't
    carry a "__gb_kv__" manifest, i.e. every legacy checkpoint on disk
    today)."""
    raw = _t3_load_raw(path)
    import gb_tier_kv
    return gb_tier_kv.decode_state(raw)


class ModelTierManager:
    """
    Manage model component placement across T1 (GPU) / T2 (CPU pinned) / T3 (NVMe).

    Designed for Diffusers-style pipelines where multiple large sub-models
    (UNet, VAE, text encoders, ControlNets) must share a 12 GB GPU.

    Parameters
    ----------
    hbm_headroom_mb : int | None
        Keep at least this many MB free in T1 before promoting.
        Default None: derived from this card's VRAM — max(512, 10%) via
        gb_topology.hbm_headroom_mb (rule: no absolute reference literal).
    t3_dir : str
        Directory for NVMe-evicted model checkpoints.
    async_transfers : bool
        Use gb_stream_sched transfer stream for non-blocking H2D/D2H.
    pin_host : bool | None
        Demote into pinned host memory (budget-checked). None reads GB_TIER_PIN_HOST.
    """

    def __init__(
        self,
        hbm_headroom_mb: "int | None" = None,
        t3_dir: str = "/var/lib/greenboost/model_pages",
        async_transfers: bool = True,
        pin_host: "bool | None" = None,
    ):
        if hbm_headroom_mb is None:
            # %-derived default (10% of VRAM, floor 512) shared with gb_init
            # via gb_topology — replaces the flat 1024/1500 MB literals (rule).
            try:
                from gb_topology import hbm_headroom_mb as _derive_headroom_mb
                hbm_headroom_mb = _derive_headroom_mb()
            except Exception:
                hbm_headroom_mb = 1024   # last resort: gb_topology unavailable
        self.hbm_headroom_mb = hbm_headroom_mb
        self.t3_dir = t3_dir
        self.async_transfers = async_transfers
        # Opt-in (GB_TIER_PIN_HOST=1): demote into pinned host memory so the copy is truly
        # asynchronous and promote() streams from pinned pages. Off by default until it has
        # been measured on a real diffusion pipeline.
        if pin_host is None:
            pin_host = os.environ.get("GB_TIER_PIN_HOST", "") in ("1", "on", "true", "yes")
        self.pin_host = pin_host
        self._entries: Dict[str, _ModelEntry] = {}
        # Per-tier prioritized-LRU eviction (gb_evict_policy). Tracks the same
        # entries as _entries, bucketed by tier and priority, so victim
        # selection stops being a full re-sort of every entry on every call.
        self._evict_ctl = PerTierEvictionController()
        os.makedirs(t3_dir, exist_ok=True)
        _gb_session(active=True)

    def register(self, name: str, module: torch.nn.Module, tier: str = Tier.T1,
                *, kv_bits: "Optional[int]" = None,
                kv_keys: "Optional[tuple]" = None,
                kv_codec: "Optional[str]" = None,
                priority: "Optional[int]" = None):
        """Register a model component for tier management.

        kv_bits: opt in to KV-cache tier-serde compression (T3 spill only ,
            missing_features.md item (e)) for THIS component. None (default)
            leaves behavior byte-identical to before this feature existed.
            Only pass it for a module whose state_dict holds attention cache
            tensors , passing it for a weight-bearing module WILL degrade
            the weights (gb_tier_kv has no name/shape heuristics; this
            kwarg IS the classification). kv_keys restricts encoding to
            state_dict keys starting with one of these prefixes (e.g.
            ("k_cache", "v_cache") on a module that also holds non-cache
            buffers); None encodes every eligible tensor in the module.
            kv_codec: "polarquant" | "turboquant" | None (env-resolved via
            GB_TIER_KV_PRESET, see gb_tier_kv.resolve_codec)."""
        e = _ModelEntry(
            name=name, module=module, tier=tier,
            kv_bits=kv_bits, kv_keys=kv_keys, kv_codec=kv_codec,
            priority=clamp_priority(priority))
        prev = self._entries.get(name)
        if prev is not None:
            self._evict_ctl.forget(prev)
        self._entries[name] = e
        self._evict_ctl.schedule_for_eviction(e)

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

        self._evict_ctl.retier(e, e.tier, Tier.T1)
        e.tier = Tier.T1
        e.last_used = time.time()
        print(f"[gb_tier] promoted '{name}' → T1 (GPU)", flush=True)
        _df_emit_tier(name, from_tier, Tier.T1, _module_size_gb(e.module), time.time() - t0)

    def _module_to_host(self, module, *, non_blocking: bool) -> None:
        """Move a module's tensors to host RAM. With pinning on and the host RAM floor
        respected, the destinations are pinned; otherwise the plain module.to("cpu")."""
        if self.pin_host and _pinned_budget_ok(int(_module_size_gb(module) * 2**30)):
            module._apply(_demote_to_pinned)
            if not non_blocking:
                torch.cuda.current_stream().synchronize()
        else:
            module.to("cpu", non_blocking=non_blocking)

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
                self._module_to_host(e.module, non_blocking=True)
            # The consumer here is the host (callers read the CPU tensor
            # directly, e.g. evict()'s state_dict() save right after) -
            # a device-side stream wait doesn't block the host, so use
            # an event sync instead of gs.wait_for(..., on="gemm")
            # (which was a no-op for the same reason as in promote()).
            ev = torch.cuda.Event()
            gs.stream("transfer").record_event(ev)
            ev.synchronize()
        else:
            self._module_to_host(e.module, non_blocking=False)

        self._evict_ctl.retier(e, e.tier, Tier.T2)
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
        t3_path, disk_bytes, kv_stats = _t3_save(
            e.module.state_dict(), base_path, kv_bits=e.kv_bits,
            kv_keys=e.kv_keys, kv_codec=e.kv_codec)
        # Move to meta device: frees all parameter memory while preserving shapes
        # so _load_from_t3 can use to_empty()+load_state_dict(assign=True) later.
        e.module.to("meta")
        self._evict_ctl.retier(e, e.tier, Tier.T3)
        e.tier = Tier.T3
        e.t3_path = t3_path
        compressed = t3_path.endswith(".zst")
        ratio = (size_gb * 2**30 / disk_bytes) if compressed and disk_bytes else 1.0
        kv_note = ""
        extra = {"compressed": compressed,
                "disk_mb": round(disk_bytes / 2**20, 1),
                "compress_ratio": round(ratio, 2)}
        if kv_stats:
            kv_pre = kv_stats.get("pre_bytes", 0)
            kv_post = kv_stats.get("post_bytes", 0)
            kv_ratio = round(kv_pre / kv_post, 2) if kv_post else 0.0
            extra.update({
                "kv_codec": kv_stats.get("codec"), "kv_bits": kv_stats.get("bits"),
                "kv_tensors": kv_stats.get("tensors", 0),
                "kv_tensors_skipped": kv_stats.get("tensors_skipped", 0),
                "kv_skip_reasons": kv_stats.get("skip_reasons", {}),
                "kv_ratio": kv_ratio,
            })
            if kv_stats.get("tensors"):
                kv_note = f" (kv {kv_stats['tensors']}x {kv_stats['codec']}@{kv_stats['bits']}b {kv_ratio:.2f}x)"
        print(f"[gb_tier] evicted '{name}' → T3 (NVMe) at {t3_path}"
              f"{f' (zstd {ratio:.2f}x)' if compressed else ''}{kv_note}", flush=True)
        _df_emit_tier(name, from_tier, Tier.T3, size_gb, time.time() - t0,
                      extra=extra)

    def _load_from_t3(self, e: _ModelEntry):
        if not e.t3_path or not os.path.exists(e.t3_path):
            raise FileNotFoundError(f"T3 checkpoint for '{e.name}' not found: {e.t3_path}")
        state = _t3_load(e.t3_path)
        # Re-allocate parameter storage (evict moved module to meta device) then
        # fill from saved state.  assign=True avoids in-place copy size checks.
        e.module.to_empty(device="cpu")
        e.module.load_state_dict(state, assign=True)
        self._evict_ctl.retier(e, e.tier, Tier.T2)
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
        """Name of the next entry that would be evicted from `tier`, or None.

        Delegates to the per-tier prioritized-LRU controller. With every entry
        at the default priority this returns exactly what the previous
        `sorted(last_used)[0]` scan returned; priorities only start to matter
        once a caller passes `register(..., priority=...)`.
        """
        victim = self._evict_ctl.victim(tier)
        return victim.name if victim is not None else None

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
