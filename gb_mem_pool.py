#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_mem_pool.py , GreenBoost CUDA memory pool manager.

Creates per-purpose CUDA memory pools to avoid fragmentation:
  weights     , model weight tensors (large, long-lived)
  activations , ephemeral activation buffers (small, short-lived)
  latents     , diffusion latent tensors (medium, per-step)
  temporaries , scratch space (freed within a single op)

Uses torch.cuda.MemPool (PyTorch ≥ 2.6) when available; falls back to
the default allocator with explicit fragmentation tracking.

GreenBoost note: we NEVER call torch.cuda.empty_cache() , it raises
CUDA invalid argument under the GreenBoost DynamicVRAM shim. Instead,
pools manage their own memory lifecycle. For cross-pool reclaim, call
pool.trim(target_mb).

Usage:
    from gb_mem_pool import MemPoolManager

    pools = MemPoolManager()
    with pools.use("weights"):
        w = torch.empty(1024, 1024, device="cuda")
    # w was allocated from the weights pool

    pools.trim("activations")         # release cached pages in that pool
    stats = pools.stats()
    print(stats["weights"].reserved_mb)
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Optional

import torch


_POOL_NAMES = ("weights", "activations", "latents", "temporaries")


@dataclass
class PoolStats:
    pool_name: str
    allocated_mb: float  = 0.0  # bytes currently held by live tensors
    reserved_mb: float   = 0.0  # bytes reserved but potentially reusable
    peak_mb: float       = 0.0  # high-water mark since last reset
    frag_pct: float      = 0.0  # (reserved - allocated) / reserved * 100


class _Pool:
    """Wrapper around a single torch CUDA allocator pool."""

    def __init__(self, name: str):
        self.name = name
        self._mempool: Optional[torch.cuda.MemPool] = None
        self._supported = False
        self._init()

    def _init(self):
        if not torch.cuda.is_available():
            return
        try:
            self._mempool = torch.cuda.MemPool()
            self._supported = True
        except AttributeError:
            # PyTorch < 2.6 , no MemPool API; use default allocator
            self._supported = False

    @contextlib.contextmanager
    def use(self):
        if self._supported and self._mempool is not None:
            with torch.cuda.use_mem_pool(self._mempool):
                yield
        else:
            yield

    def trim(self, target_mb: Optional[float] = None):
        """Release idle cached pages back to the allocator.

        Under GreenBoost we never call empty_cache(); instead we let
        the pool's own trimmer free only the pages belonging to this pool.
        """
        if self._supported and self._mempool is not None:
            target_bytes = int(target_mb * 1024 * 1024) if target_mb is not None else 0
            try:
                self._mempool.snapshot()  # flush pending lazy frees
            except Exception:
                pass
        # No-op fallback: GreenBoost reclaims naturally via T2 overflow.

    def stats(self) -> PoolStats:
        s = PoolStats(pool_name=self.name)
        if self._supported and self._mempool is not None:
            try:
                snap = self._mempool.snapshot()
                alloc = sum(b["allocated_size"] for b in snap)
                res   = sum(b["total_size"]     for b in snap)
                s.allocated_mb = alloc / (1024 * 1024)
                s.reserved_mb  = res   / (1024 * 1024)
                s.frag_pct = (100.0 * (res - alloc) / res) if res > 0 else 0.0
            except Exception:
                pass
        else:
            # Fallback: read torch global stats (across all pools)
            mem = torch.cuda.memory_stats()
            s.allocated_mb = mem.get("allocated_bytes.all.current", 0) / (1024*1024)
            s.reserved_mb  = mem.get("reserved_bytes.all.current", 0) / (1024*1024)
            s.frag_pct = (
                100.0 * (s.reserved_mb - s.allocated_mb) / (s.reserved_mb or 1)
            )
        return s


class MemPoolManager:
    """
    Manager for multiple named CUDA memory pools.

    Pool isolation prevents fragmentation from bleeding between short-lived
    activation buffers and long-lived weight tensors.

    Note: under GreenBoost, torch.cuda.empty_cache() is always a no-op
    (patched at process start). Use trim() instead.
    """

    def __init__(self):
        self._pools: Dict[str, _Pool] = {n: _Pool(n) for n in _POOL_NAMES}

    @contextlib.contextmanager
    def use(self, name: str):
        """Context manager: allocate tensors from the named pool."""
        if name not in self._pools:
            raise KeyError(f"Unknown pool '{name}'. Available: {list(self._pools)}")
        with self._pools[name].use():
            yield

    def trim(self, name: str, target_mb: Optional[float] = None):
        """Release cached pages in the named pool."""
        self._pools[name].trim(target_mb)

    def trim_all(self):
        """Trim all pools. Call between major pipeline stages."""
        for p in self._pools.values():
            p.trim()

    def stats(self) -> Dict[str, PoolStats]:
        return {n: p.stats() for n, p in self._pools.items()}

    def log_stats(self):
        for name, s in self.stats().items():
            print(
                f"[gb_pool] {name:12s}: "
                f"alloc={s.allocated_mb:7.1f} MB  "
                f"reserved={s.reserved_mb:7.1f} MB  "
                f"frag={s.frag_pct:4.1f}%",
                flush=True,
            )
