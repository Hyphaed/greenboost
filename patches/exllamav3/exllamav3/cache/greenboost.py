"""
GreenBoost KV cache layer for ExLlamaV3.
Copyright (C) 2026 Ferran Duarri. GPL v2 / Commercial - see LICENSE.

Routes KV cache allocations through GreenBoost Tier 2 (System DDR DMA-BUF pinned pages)
instead of GPU VRAM. Enables inference with contexts that exceed VRAM capacity.

Memory layout:
  - K and V each get their own DMA-BUF from /dev/greenboost
  - DMA-BUF pages are pinned System DDR (alloc_pages GFP_KERNEL|__GFP_COMP in kernel)
  - mmap'd into userspace and wrapped as zero-copy numpy→PyTorch CPU tensors
  - GPU attention: ExLlamaV3 calls .to(device) per-layer - PCIe 4.0 x16 ~32 GB/s

Bandwidth math for glm-4.7-flash:q8_0 (48 layers, 64 KV heads, 128 dim):
  Per-layer KV at 32K ctx  = 2 × 64 × 128 × 32768 × 2B = 1.07 GB → ~33 ms/layer
  Per-layer KV at 8K ctx   = 2 × 64 × 128 × 8192  × 2B = 268 MB  → ~8 ms/layer
  Best fit: use GreenBoost cache for layers that overflow VRAM (layers 18-48).

Fallback: if /dev/greenboost unavailable → standard CPU tensor (system RAM).
"""

from __future__ import annotations

import os
import fcntl
import mmap
import math
import struct
import logging
from typing import Optional, TYPE_CHECKING
from typing_extensions import override

import torch
import numpy as np

from ..constants import PAGE_SIZE
from ..model import Config
from .cache import CacheLayer

if TYPE_CHECKING:
    from ..modules import Attention

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GreenBoost IOCTL constants (must match greenboost_ioctl.h)
# ---------------------------------------------------------------------------

_GB_IOCTL_MAGIC     = ord('G')

# struct gb_alloc_req { gb_u64 size; gb_s32 fd; gb_u32 flags; }
# size = 8+4+4 = 16 bytes
_GB_ALLOC_REQ_FMT   = "=QiI"           # native byte order, explicit sizes
_GB_ALLOC_REQ_SIZE  = struct.calcsize(_GB_ALLOC_REQ_FMT)   # 16

# _IOWR('G', 1, 24) = 0xC0000000 | (24<<16) | ('G'<<8) | 1
_GB_IOCTL_ALLOC     = (
    (3 << 30) |
    (_GB_ALLOC_REQ_SIZE << 16) |
    (_GB_IOCTL_MAGIC << 8) |
    1
)

# Allocation flags
_GB_ALLOC_KV_CACHE  = 1 << 1           # hint kernel: this is a KV cache buffer

# ---------------------------------------------------------------------------
# Low-level GreenBoost allocator
# ---------------------------------------------------------------------------

def _gb_open() -> int:
    """Open /dev/greenboost and return fd, or -1 if unavailable."""
    try:
        return os.open("/dev/greenboost", os.O_RDWR | os.O_CLOEXEC)
    except OSError as e:
        log.debug("[GreenBoost cache] /dev/greenboost unavailable: %s", e)
        return -1


def _gb_alloc(gb_fd: int, size_bytes: int) -> tuple[int, mmap.mmap | None]:
    """
    Allocate `size_bytes` bytes from GreenBoost Tier 2 (System DDR).

    Returns (dma_buf_fd, mmap_obj) on success, (-1, None) on failure.
    The caller owns both the fd and the mmap and must close them.
    """
    if gb_fd < 0 or size_bytes <= 0:
        return -1, None

    req = struct.pack(_GB_ALLOC_REQ_FMT, size_bytes, -1, _GB_ALLOC_KV_CACHE)
    try:
        res = fcntl.ioctl(gb_fd, _GB_IOCTL_ALLOC, req)
    except OSError as e:
        log.debug("[GreenBoost cache] IOCTL_ALLOC failed: %s", e)
        return -1, None

    _, dma_fd, flags = struct.unpack(_GB_ALLOC_REQ_FMT, res)
    if dma_fd < 0:
        log.debug("[GreenBoost cache] alloc failed: dma_fd=%d flags=0x%x", dma_fd, flags)
        return -1, None

    try:
        mm = mmap.mmap(dma_fd, size_bytes, mmap.MAP_SHARED,
                       mmap.PROT_READ | mmap.PROT_WRITE)
    except OSError as e:
        log.debug("[GreenBoost cache] mmap failed: %s", e)
        os.close(dma_fd)
        return -1, None

    return dma_fd, mm


def _mm_to_tensor(mm: mmap.mmap, shape: tuple, dtype=np.float16) -> torch.Tensor:
    """
    Zero-copy wrap of a mmap object as a PyTorch CPU tensor.

    Path: mmap → numpy (shares buffer protocol) → torch.from_numpy (zero-copy).
    The returned tensor's storage IS the mmap region - no copy made.
    Lifetime: tensor keeps the numpy array alive; caller keeps mmap alive.
    """
    n = math.prod(shape)
    arr = np.frombuffer(mm, dtype=dtype, count=n).reshape(shape)
    t = torch.from_numpy(arr)   # zero-copy: shares numpy/mmap backing memory
    return t


# ---------------------------------------------------------------------------
# CacheLayer_greenboost
# ---------------------------------------------------------------------------

class CacheLayer_greenboost(CacheLayer):
    """
    ExLlamaV3 KV cache layer backed by GreenBoost Tier 2 System DDR DMA-BUF pages.

    Drop-in replacement for CacheLayer_fp16 for layers that overflow VRAM.
    K and V tensors live in pinned System DDR as zero-copy CPU tensors.

    Usage (ExLlamaV3 Cache):
        from exllamav3.cache.greenboost import CacheLayer_greenboost
        cache = Cache(model, max_num_tokens=32768, layer_type=CacheLayer_greenboost)

    Environment:
        GREENBOOST_CACHE_VERBOSE=1   - enable per-layer allocation logging
    """

    _verbose: bool = os.environ.get("GREENBOOST_CACHE_VERBOSE", "0") == "1"

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        gb_fd: int = -1,
        **kwargs,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of PAGE_SIZE ({PAGE_SIZE})."

        # Tensor shape: (num_pages, page_size, num_kv_heads, head_dim)
        self.shape: tuple | None = (
            (max_num_tokens // PAGE_SIZE, PAGE_SIZE,
             attention.num_kv_heads, attention.head_dim)
            if attention else None
        )

        # Shared /dev/greenboost fd - if not supplied, open our own
        self._own_gb_fd: bool = gb_fd < 0
        self._gb_fd: int = gb_fd if gb_fd >= 0 else _gb_open()

        # Allocated resources (filled by alloc())
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.device: torch.device | None = None

        # DMA-BUF bookkeeping
        self._k_fd: int = -1
        self._v_fd: int = -1
        self._k_mm: mmap.mmap | None = None
        self._v_mm: mmap.mmap | None = None
        self._using_greenboost: bool = False

    # -----------------------------------------------------------------------
    # CacheLayer interface
    # -----------------------------------------------------------------------

    @override
    def alloc(self, device: torch.device):
        """Allocate K and V tensors, preferring GreenBoost System DDR."""
        self.device = device

        if self.shape is None:
            return  # recurrent / non-attention layer

        elem_bytes = 2  # torch.half / float16
        size_bytes = math.prod(self.shape) * elem_bytes

        # ── Try GreenBoost DMA-BUF allocation ──────────────────────────────
        if self._gb_fd >= 0:
            self._k_fd, self._k_mm = _gb_alloc(self._gb_fd, size_bytes)
            try:
                self._v_fd, self._v_mm = _gb_alloc(self._gb_fd, size_bytes)
            except Exception:
                if self._k_fd >= 0:
                    try:
                        os.close(self._k_fd)
                    except Exception:
                        pass
                    self._k_fd = -1
                if self._k_mm is not None:
                    try:
                        self._k_mm.close()
                    except Exception:
                        pass
                    self._k_mm = None
                raise

        if self._k_mm is not None and self._v_mm is not None:
            # Zero-copy mmap → numpy → torch (CPU tensor, System DDR-backed)
            self.k = _mm_to_tensor(self._k_mm, self.shape, np.float16)
            self.v = _mm_to_tensor(self._v_mm, self.shape, np.float16)
            self._using_greenboost = True
            if self._verbose:
                mb = size_bytes / 1024 / 1024
                log.info(
                    "[GreenBoost cache] layer cache_id=%s  K+V = 2×%.0f MB  "
                    "→ Tier 2 System DDR (fd k=%d v=%d)",
                    self.cache_id, mb, self._k_fd, self._v_fd
                )
        else:
            # ── Fallback: standard CPU RAM ──────────────────────────────────
            self._cleanup_dma_bufs()
            self.k = torch.zeros(self.shape, dtype=torch.half)
            self.v = torch.zeros(self.shape, dtype=torch.half)
            self._using_greenboost = False
            if self._verbose:
                log.warning(
                    "[GreenBoost cache] layer cache_id=%s  GreenBoost unavailable "
                    "- falling back to CPU RAM", self.cache_id
                )

    @override
    def free(self):
        """Release K, V tensors and GreenBoost DMA-BUF resources."""
        self.k = None
        self.v = None
        self.device = None
        self._cleanup_dma_bufs()
        self._using_greenboost = False

    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor) -> tuple:
        """
        Return (k, v) tensors.

        If k/v are CPU tensors (GreenBoost System DDR), ExLlamaV3's attention module
        will call .to(device) before running GPU attention.
        """
        return self.k, self.v

    @override
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int,
    ):
        """Update handled by ExLlamaV3 attention kernels writing directly into tensors."""
        pass

    @override
    def copy_page(
        self,
        source: CacheLayer_greenboost,
        from_page: int,
        to_page: int,
        num_tokens: int,
    ):
        assert self.shape == source.shape
        self.k[to_page, :num_tokens].copy_(source.k[from_page, :num_tokens], non_blocking=False)
        self.v[to_page, :num_tokens].copy_(source.v[from_page, :num_tokens], non_blocking=False)

    @override
    def get_tensors(self):
        return [t for t in (self.k, self.v) if t is not None]

    @override
    def storage_size(self):
        if self.shape is None:
            return 0
        return 2 * math.prod(self.shape) * torch.half.itemsize  # K + V

    @override
    def overhead_size(self):
        return 0

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_greenboost,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "gb_fd": -1,   # worker opens its own fd
            },
        }

    @override
    def get_kv_alloc_placeholder(self):
        return None

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def is_greenboost(self) -> bool:
        """True if K/V are backed by GreenBoost System DDR DMA-BUF pages."""
        return self._using_greenboost

    def get_dma_fds(self) -> tuple[int, int]:
        """Return (k_dma_fd, v_dma_fd) - useful for cudaImportExternalMemory."""
        return self._k_fd, self._v_fd

    def prefetch(self):
        """Hint OS to prefetch DMA-BUF pages into CPU cache (MADV_WILLNEED)."""
        import ctypes
        MADV_WILLNEED = 3
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        for mm in (self._k_mm, self._v_mm):
            if mm is not None:
                addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                libc.madvise(ctypes.c_void_p(addr), len(mm), MADV_WILLNEED)

    def _cleanup_dma_bufs(self):
        for mm in (self._k_mm, self._v_mm):
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
        for fd in (self._k_fd, self._v_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except Exception:
                    pass
        self._k_mm = self._v_mm = None
        self._k_fd = self._v_fd = -1

    def __del__(self):
        self._cleanup_dma_bufs()
        if self._own_gb_fd and self._gb_fd >= 0:
            try:
                os.close(self._gb_fd)
            except Exception:
                pass
        self._gb_fd = -1


# ---------------------------------------------------------------------------
# Convenience: GreenBoost Cache factory
# ---------------------------------------------------------------------------

class GreenBoostCache:
    """
    Convenience wrapper: create an ExLlamaV3 Cache with GreenBoost Tier 2 KV storage.

    Usage:
        from exllamav3 import ExLlamaV3, Cache
        from exllamav3.cache.greenboost import GreenBoostCache

        model = ExLlamaV3(config)
        cache = GreenBoostCache(model, max_num_tokens=32768)
        # cache is a standard ExLlamaV3 Cache backed by System DDR

    The /dev/greenboost fd is opened once and shared across all layers (efficiency).
    If GreenBoost is unavailable, falls back transparently to CPU RAM.
    """

    def __init__(self, model, max_num_tokens: int, **kwargs):
        import os
        from .cache import Cache

        # F-L5-05: ExLlamaV3 uses a custom CUDA attention kernel, not
        # F.scaled_dot_product_attention, so gb_attn.py's SDPA intercept never
        # fires on this path. TurboQuant K/V quantisation is NOT applied here.
        if os.environ.get("GREENBOOST_TURBOQUANT", "0") != "0":
            log.warning(
                "[GreenBoost cache] GREENBOOST_TURBOQUANT=1 is set but "
                "ExLlamaV3 bypasses F.scaled_dot_product_attention - "
                "TurboQuant K/V compression is NOT active on this path. "
                "T2 DDR bandwidth cost is 3-4x higher than the documented "
                "architecture predicts. See patches/exllamav3/README."
            )

        # Open one shared gb_fd for all layers
        shared_fd = _gb_open()
        if shared_fd >= 0:
            log.info("[GreenBoost cache] opened /dev/greenboost (fd=%d)", shared_fd)
        else:
            log.warning("[GreenBoost cache] /dev/greenboost unavailable - using CPU RAM fallback")

        # Monkey-patch: inject shared gb_fd into every layer's kwargs
        def _layer_factory(config, attention, cache_id, max_num_tokens, **kw):
            return CacheLayer_greenboost(
                config, attention, cache_id, max_num_tokens,
                gb_fd=shared_fd, **kw
            )

        # Temporarily override layer_type with our factory
        self._cache = Cache(model, max_num_tokens, layer_type=_layer_factory, **kwargs)
        self._shared_fd = shared_fd

    def __getattr__(self, name):
        if name == '_cache':
            raise AttributeError(name)
        return getattr(self._cache, name)

    def __del__(self):
        if hasattr(self, "_shared_fd") and self._shared_fd >= 0:
            try:
                os.close(self._shared_fd)
            except Exception:
                pass
