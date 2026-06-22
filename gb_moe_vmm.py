#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_moe_vmm.py — GbExpertPool VMM-backed tiering for batched-*Experts MoE.

Track 4 of enhancements.md: per-expert physical chunk management via CUDA VMM
(cuMemCreate DEVICE + cuMemMap/Unmap) replaces Track 3's copy-based
(param.data[i].zero_() / .copy_()) tiering with true VRAM release.

Structural gap solved
---------------------
Current transformers 5.x stores all expert weights as a single 3D parameter
[num_experts, out_features, in_features].  GreenBoost's whole-buffer tiering
can't independently tier individual expert rows within that allocation.

GbExpertVMMPool wraps each per-param expert-tensor with a VMM virtual address
range of the same total size, where each expert occupies a separately
mappable DEVICE physical chunk (real VRAM).  Cold experts have their chunk
unmapped and released — zero VRAM footprint, unlike Track 3 which zeros bytes
but keeps the allocation.

Why DEVICE chunks, not HOST_NUMA (like T2):
  The model forward pass (batched expert GEMM) runs on-device.  Experts must
  be DEVICE-resident to be accessed at full VRAM bandwidth.  Cold experts are
  stored in caller-managed CPU pinned buffers (same as Track 3) and copied
  back on promotion.  A future improvement could keep cold data in
  HOST_NUMA_CURRENT DMA-BUF chunks (T2 path, no CPU copy) but that requires
  the GEMM to tolerate PCIe-bandwidth access for cold paths.

Key invariants
--------------
- The VA base returned by gb_expert_pool_va_base() is constant for the life of
  the pool.  PyTorch sees a tensor at that VA; only physical backing changes.
- Accessing an unmapped expert VA generates CUDA_ERROR_ILLEGAL_ADDRESS.
  The gate-hook correctness gate (inherited from gb_moe.py) ensures no cold
  expert is accessed without first calling promote().
- Pools limited to 64 experts by the residency bitmask (uint64_t); raise
  GB_EXPERT_POOL_MAX_EXPERTS in the shim for larger models.

Usage
-----
    import gb_moe_vmm

    mgr = gb_moe_vmm.GbMoEVMMManager(model, hot_threshold=0.05,
                                      cold_bits=4, prefetch_topn=2)
    n = mgr.attach()        # replaces experts_mod.w1/w2/w3 with VMM tensors
    ...run inference...
    print(mgr.status())
    mgr.detach()            # restores all cold experts, removes hooks
"""
from __future__ import annotations

import ctypes
import os
import struct
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from gb_moe import (
    GbMoEManager,
    _BatchedBlockState,
    _find_moe_blocks_batched,
    _infer_top_k,
    _EMA_DECAY,
    _REBALANCE_EVERY,
    _quant_int8_slice,
    _dequant_int8_slice,
)


# ── shim library loading ────────────────────────────────────────────────────

def _load_shim() -> Optional[ctypes.CDLL]:
    """Load the GreenBoost CUDA shim and resolve expert pool symbols."""
    candidates = [
        os.environ.get("GREENBOOST_SHIM_PATH", ""),
        "/usr/local/lib/libgreenboost_cuda.so",
        "/usr/lib/libgreenboost_cuda.so",
    ]
    lib = None
    for path in candidates:
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD)
            break
        except OSError:
            try:
                lib = ctypes.CDLL(path)
                break
            except OSError:
                continue
    if lib is None:
        return None

    # Resolve and type the expert pool API.
    def _sym(name, restype, *argtypes):
        try:
            fn = getattr(lib, name)
            fn.restype  = restype
            fn.argtypes = list(argtypes)
            return fn
        except AttributeError:
            return None

    lib.gb_expert_pool_create   = _sym("gb_expert_pool_create",
                                       ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_size_t)
    lib.gb_expert_pool_destroy  = _sym("gb_expert_pool_destroy",
                                       None, ctypes.c_void_p)
    lib.gb_expert_pool_promote  = _sym("gb_expert_pool_promote",
                                       ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_size_t)
    lib.gb_expert_pool_demote   = _sym("gb_expert_pool_demote",
                                       ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_size_t)
    lib.gb_expert_pool_residency = _sym("gb_expert_pool_residency",
                                        ctypes.c_uint64, ctypes.c_void_p)
    lib.gb_expert_pool_va_base  = _sym("gb_expert_pool_va_base",
                                       ctypes.c_uint64, ctypes.c_void_p)
    lib.gb_expert_pool_stride   = _sym("gb_expert_pool_stride",
                                       ctypes.c_size_t, ctypes.c_void_p)

    if lib.gb_expert_pool_create is None:
        return None  # shim present but doesn't have the expert pool API
    return lib


_SHIM: Optional[ctypes.CDLL] = None


def _get_shim() -> Optional[ctypes.CDLL]:
    global _SHIM
    if _SHIM is None:
        _SHIM = _load_shim()
    return _SHIM


# ── __cuda_array_interface__ wrapper ────────────────────────────────────────

class _VMMTensorView:
    """Adapts a raw CUDA GPU pointer to PyTorch via __cuda_array_interface__.

    PyTorch's torch.as_tensor() accepts objects implementing this protocol
    (same mechanism CuPy uses for zero-copy interop).  The resulting tensor is
    non-owning — PyTorch will not free the underlying memory on GC.

    Args:
        ptr:    CUdeviceptr (integer) of the GPU virtual address.
        shape:  tuple matching the full parameter shape.
        dtype:  torch.dtype of the underlying data.
        dim0_stride_bytes: physical byte stride between experts (shape[0]).
            GbExpertPool pads each expert's chunk up to its allocation
            granularity (e.g. 2 MiB) - this is almost always larger than
            the tightly-packed C-contiguous stride (shape[1]*shape[2]*elem),
            so it must override the natural stride for dim 0, or every
            expert past index 0 reads from the wrong physical offset.
    """
    def __init__(self, ptr: int, shape: tuple, dtype: torch.dtype,
                 dim0_stride_bytes: "int | None" = None):
        elem = torch.tensor([], dtype=dtype).element_size()
        numel = 1
        for s in shape:
            numel *= s
        # Compute C-contiguous strides in BYTES (required by the protocol v3),
        # then override dim 0 with the real per-expert physical stride.
        strides_elems: list = []
        acc = 1
        for s in reversed(shape):
            strides_elems.insert(0, acc)
            acc *= s
        strides_bytes = tuple(s * elem for s in strides_elems)
        if dim0_stride_bytes is not None and len(strides_bytes) > 0:
            strides_bytes = (dim0_stride_bytes,) + strides_bytes[1:]
        try:
            typestr = torch.empty([], dtype=dtype).numpy().dtype.str
        except TypeError:
            # NumPy has no native mapping for this dtype (e.g. bfloat16 - the
            # standard MoE expert weight dtype elsewhere in this codebase).
            # Expose the raw bytes as a same-width unsigned int instead;
            # init_from_param()/promote() callers must .view(dtype) the
            # resulting tensor back (metadata-only, zero-copy - the whole
            # point of this class is aliasing VMM memory without a copy).
            typestr = {1: "|u1", 2: "<u2", 4: "<u4", 8: "<u8"}[elem]
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape":   shape,
            "data":    (ptr, False),   # (ptr, read_only)
            "typestr": typestr,
            "strides": strides_bytes,
        }


def _as_tensor_real_dtype(view: "_VMMTensorView", dtype: torch.dtype) -> torch.Tensor:
    """torch.as_tensor(view), then reinterpret back to `dtype` if the array
    interface had to substitute a raw-bytes typestr (see _VMMTensorView).
    .view(dtype) is metadata-only - the VMM memory is never copied."""
    t = torch.as_tensor(view)
    if t.dtype != dtype:
        t = t.view(dtype)
    return t


# ── per-pool VMM state ──────────────────────────────────────────────────────

class GbExpertVMMPool:
    """One VMM pool per 3D expert parameter tensor.

    A batched-experts module has N parameter tensors (e.g., w1, w2, w3).
    One GbExpertVMMPool manages the per-expert physical chunks for a single
    parameter.  GbMoEVMMManager creates num_params pools per block.

    After __init__ + init_from_param(), the associated nn.Parameter's
    .data.data_ptr() == self.va_base (the VMM VA range).  All experts start
    T1-resident (DEVICE physical chunk mapped, data copied from the original
    cudaMalloc tensor).
    """

    def __init__(self, shim: ctypes.CDLL, num_experts: int, expert_bytes: int):
        self._shim    = shim
        self._pool    = shim.gb_expert_pool_create(num_experts, expert_bytes)
        if not self._pool:
            raise RuntimeError("gb_expert_pool_create returned NULL — "
                               "shim not loaded or cuMem VMM unsupported")
        self.num_experts  = num_experts
        self.expert_bytes = expert_bytes
        self.va_base: int = int(shim.gb_expert_pool_va_base(self._pool))
        self.stride: int  = int(shim.gb_expert_pool_stride(self._pool))
        # CPU pinned buffers for cold experts: {expert_idx: bytes-like}
        self._cpu_bufs: Dict[int, bytes] = {}
        # Quantized variants if cold_bits <= 4:
        #   {expert_idx: (int8_tensor, scale_tensor)} or bare tensor
        self._cpu_quant: Dict[int, object] = {}

    def init_from_param(self, param_tensor: torch.Tensor, cold_bits: int) -> torch.Tensor:
        """Copy expert data from the original CUDA tensor into per-expert VMM chunks.

        Returns a non-owning PyTorch tensor backed by the VMM VA range.
        The caller should replace nn.Parameter(...) with nn.Parameter(returned).
        """
        shim = self._shim
        shape = tuple(param_tensor.shape)
        dtype = param_tensor.dtype
        elem_size = param_tensor.element_size()

        for i in range(self.num_experts):
            # Materialise the expert slice as a contiguous CPU tensor.
            slice_cpu = param_tensor[i].detach().contiguous().cpu()
            src_ptr   = slice_cpu.data_ptr()
            nbytes    = slice_cpu.numel() * elem_size
            ret = int(shim.gb_expert_pool_promote(self._pool, i,
                                                  ctypes.c_void_p(src_ptr),
                                                  ctypes.c_size_t(nbytes)))
            if ret != 0:
                raise RuntimeError(f"gb_expert_pool_promote failed CUresult={ret} expert={i}")

        view = _VMMTensorView(self.va_base, shape, dtype, dim0_stride_bytes=self.stride)
        return _as_tensor_real_dtype(view, dtype)

    def promote(self, expert_idx: int, cold_bits: int) -> None:
        """Move a cold expert back to T1 (DEVICE VRAM)."""
        stored = self._cpu_quant.get(expert_idx)
        if stored is None:
            stored = self._cpu_bufs.get(expert_idx)
        if stored is None:
            return  # nothing to restore

        # Reconstruct float tensor for the H2D copy.
        if isinstance(stored, tuple):
            q, scale = stored
            data = _dequant_int8_slice(q, scale, torch.bfloat16)
        else:
            data = stored

        contiguous = data.contiguous()
        nbytes = contiguous.numel() * contiguous.element_size()
        ret = int(self._shim.gb_expert_pool_promote(
            self._pool, expert_idx,
            ctypes.c_void_p(contiguous.data_ptr()),
            ctypes.c_size_t(nbytes),
        ))
        if ret != 0:
            raise RuntimeError(f"VMM promote failed CUresult={ret} expert={expert_idx}")
        self._cpu_bufs.pop(expert_idx, None)
        self._cpu_quant.pop(expert_idx, None)

    def demote(self, expert_idx: int, orig_shape: tuple, orig_dtype: torch.dtype,
               cold_bits: int) -> None:
        """Unmap a hot expert from VRAM, save data to CPU pinned buffer."""
        # Allocate a CPU buffer of the right size.
        numel  = 1
        for s in orig_shape: numel *= s
        elem   = torch.tensor([], dtype=orig_dtype).element_size()
        nbytes = numel * elem

        cpu_buf = torch.empty(orig_shape, dtype=orig_dtype, pin_memory=True)
        ret = int(self._shim.gb_expert_pool_demote(
            self._pool, expert_idx,
            ctypes.c_void_p(cpu_buf.data_ptr()),
            ctypes.c_size_t(nbytes),
        ))
        if ret != 0:
            raise RuntimeError(f"VMM demote failed CUresult={ret} expert={expert_idx}")

        if cold_bits <= 4:
            flat  = cpu_buf.reshape(-1, orig_shape[-1]).float()
            q, sc = _quant_int8_slice(flat)
            self._cpu_quant[expert_idx] = (q.reshape(orig_shape), sc.reshape(orig_shape[:-1]))
        else:
            self._cpu_bufs[expert_idx] = cpu_buf

    def residency(self) -> int:
        """Return 64-bit bitmask of VRAM-resident experts."""
        return int(self._shim.gb_expert_pool_residency(self._pool))

    def destroy(self) -> None:
        if self._pool:
            self._shim.gb_expert_pool_destroy(self._pool)
            self._pool = None


# ── VMM-aware batched block state ───────────────────────────────────────────

class _VMMBatchedState:
    """Augments _BatchedBlockState with per-param VMM pool references."""

    def __init__(self, bst: _BatchedBlockState,
                 pools: Dict[str, GbExpertVMMPool],
                 orig_shapes: Dict[str, tuple],
                 orig_dtypes: Dict[str, torch.dtype]):
        self.bst         = bst
        self.pools       = pools          # {param_name: GbExpertVMMPool}
        self.orig_shapes = orig_shapes    # {param_name: shape_tuple}
        self.orig_dtypes = orig_dtypes    # {param_name: torch.dtype}


# ── GbMoEVMMManager ─────────────────────────────────────────────────────────

class GbMoEVMMManager(GbMoEManager):
    """Drop-in replacement for GbMoEManager that uses cuMem VMM physical chunk
    management for batched-*Experts blocks instead of copy-based tiering.

    Requires the GreenBoost CUDA shim (libgreenboost_cuda.so) to be loaded in
    the process (LD_PRELOAD or explicit dlopen).  Falls back to the base class
    copy-based tiering if the shim is unavailable.

    Parameters match GbMoEManager exactly.  Attach/detach/status API is identical.
    """

    def __init__(self, model: nn.Module, hot_threshold: float = 0.05,
                 cold_bits: int = 4, prefetch_topn: int = 2,
                 tier_manager=None):
        super().__init__(model, hot_threshold=hot_threshold, cold_bits=cold_bits,
                         prefetch_topn=prefetch_topn, tier_manager=tier_manager)
        self._shim: Optional[ctypes.CDLL] = _get_shim()
        self._vmm_states: List[_VMMBatchedState] = []

    # ── lifecycle ────────────────────────────────────────────────────────────

    def attach(self) -> int:
        n = super().attach()
        if self._shim is None:
            return n  # shim not available; base class handles everything

        # For each batched block that the base class registered, replace
        # its experts' parameters with VMM-backed tensors.
        new_states = []
        for bst in self._batched_blocks:
            pools:        Dict[str, GbExpertVMMPool] = {}
            orig_shapes:  Dict[str, tuple]            = {}
            orig_dtypes:  Dict[str, torch.dtype]      = {}
            failed = False
            for pname in bst.param_names:
                param = getattr(bst.experts_mod, pname)
                shape = tuple(param.shape)
                dtype = param.dtype
                # Per-expert bytes: elements per expert * element size
                expert_elems = 1
                for s in shape[1:]: expert_elems *= s
                expert_bytes = expert_elems * param.element_size()
                try:
                    pool = GbExpertVMMPool(self._shim, bst.num_experts, expert_bytes)
                    vmm_tensor = pool.init_from_param(param, self.cold_bits)
                    # Replace the parameter with the VMM-backed tensor.
                    setattr(bst.experts_mod, pname,
                            nn.Parameter(vmm_tensor, requires_grad=False))
                    pools[pname]       = pool
                    orig_shapes[pname] = shape
                    orig_dtypes[pname] = dtype
                except Exception as exc:
                    import sys
                    print(f"[gb_moe_vmm] VMM init failed for {bst.block_name}.{pname}: {exc}; "
                          "falling back to copy-based tiering", file=sys.stderr)
                    # Destroy partially-created pools for this block.
                    for p in pools.values(): p.destroy()
                    pools = {}
                    failed = True
                    break
            if not failed and pools:
                new_states.append(_VMMBatchedState(bst, pools, orig_shapes, orig_dtypes))
        self._vmm_states = new_states
        return n

    def detach(self) -> None:
        # Restore all cold VMM experts before the base class detach removes hooks.
        for vs in self._vmm_states:
            for i in range(vs.bst.num_experts):
                if str(i) not in vs.bst.hot:
                    for pname, pool in vs.pools.items():
                        pool.promote(i, self.cold_bits)
                    vs.bst.hot.add(str(i))
            # Restore the original parameter tensors so the model is left in
            # a usable state without the VMM VA range (which we're about to free).
            for pname, pool in vs.pools.items():
                param = getattr(vs.bst.experts_mod, pname)
                # Recover data from the VMM tensor into a plain CUDA tensor.
                data = param.data.clone()
                setattr(vs.bst.experts_mod, pname,
                        nn.Parameter(data, requires_grad=False))
                pool.destroy()
        self._vmm_states.clear()
        super().detach()

    # ── override expert slice management for VMM-backed blocks ───────────────

    def _restore_expert_slice(self, bst: _BatchedBlockState, expert_idx: int) -> None:
        vs = self._vmm_state_for(bst)
        if vs is None:
            return super()._restore_expert_slice(bst, expert_idx)
        for pname, pool in vs.pools.items():
            pool.promote(expert_idx, self.cold_bits)
        bst.hot.add(str(expert_idx))
        bst.cpu_bufs.pop(expert_idx, None)

    def _demote_expert_slice(self, bst: _BatchedBlockState, expert_idx: int) -> None:
        vs = self._vmm_state_for(bst)
        if vs is None:
            return super()._demote_expert_slice(bst, expert_idx)
        for pname, pool in vs.pools.items():
            shape = vs.orig_shapes[pname]
            dtype = vs.orig_dtypes[pname]
            pool.demote(expert_idx, shape[1:], dtype, self.cold_bits)
        bst.hot.discard(str(expert_idx))

    def _vmm_state_for(self, bst: _BatchedBlockState) -> Optional[_VMMBatchedState]:
        for vs in self._vmm_states:
            if vs.bst is bst:
                return vs
        return None

    # ── status ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        out = super().status()
        for vs in self._vmm_states:
            key = vs.bst.block_name
            if key in out:
                out[key]["vmm"] = True
                mask = 0
                for pool in vs.pools.values():
                    mask |= pool.residency()
                out[key]["vram_resident_mask"] = hex(mask)
        return out
