#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_moe.py , GreenBoost MoE expert-routing-aware memory management.

WHY: Mixture-of-Experts models (Mixtral, Qwen-MoE, DeepSeek-MoE-style HF
implementations) only execute a handful of experts per token per layer, but
naive loading keeps every expert resident in VRAM. This module observes
real routing decisions (a near-zero-cost forward hook on each layer's
gate/router , it inspects logits already computed during the normal
forward pass, no extra tensor compute) to:

  1. build a per-layer, per-expert frequency histogram (CPU-side bookkeeping
     only , no tensor compute on CPU, per GreenBoost's immutable design rule),
  2. keep cold experts compressed (gb_quant.quantize_module, int4) and
     DDR-resident (ModelTierManager, T2) instead of full-precision T1 VRAM,
  3. statistically prefetch the next layer's historically-hottest experts
     to T1 on the transfer stream while the current layer computes, so a
     demoted expert that becomes hot again isn't promoted synchronously on
     the critical path.

This is Phase 1 of the MoE design in enhancements.md Track 3: pure Python +
existing GreenBoost APIs (ModelTierManager, gb_quant, gb_stream_sched), no
kernel module changes. DDR-floor protection (cold experts never silently
evicted T2→T3 by *another* component's pressure-triggered auto_evict) is
achieved structurally here: experts are managed by this module's own
rebalance loop, not registered with any shared auto_evict() telemetry
callback, so they are never candidates for that path's T2→T3 eviction. A
true kernel-level floor hint (GB_IOCTL_SET_FLOOR_TIER) is deferred to
Phase 2 per the enhancements.md risk note , it would only matter if experts
were *also* registered with a shared auto_evict pool, which they are not.

Limitation (by design, not yet validated against a real model): cross-layer
prefetch prediction is a historical-frequency heuristic, not a learned or
exact router lookahead , layer i+1's actual routing depends on layer i's
output, which doesn't exist until layer i finishes. The prefetch therefore
warms the statistically likely experts, with a synchronous promote fallback
on misprediction; it does not (and cannot) guarantee a hit.

Usage:
    import gb_moe

    mgr = gb_moe.GbMoEManager(model, hot_threshold=0.05,
                               cold_bits=4, prefetch_topn=2)
    mgr.attach()
    ...run inference...
    print(mgr.status())
    mgr.detach()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from gb_topology import get_topology


def _get_pcie_high_water() -> float:
    """Return PCIe saturation threshold (MB/s).

    Priority:
    1. Live GpuTopology from the current telemetry snapshot , uses actual
       negotiated gen×width so a degraded slot (x8, gen mismatch) automatically
       lowers the threshold to the real limit.
    2. Config-file TopologyProfile (gb_topology.get_topology()) , theoretical
       from pcie_gen / pcie_lanes in /etc/greenboost/profiles/default.md.
    3. Hard fallback (PCIe 4.0 x16 @75% = 24576 MB/s).
    """
    try:
        from gb_init import get_telemetry
        tel = get_telemetry()
        if tel is not None:
            snap = tel.snapshot()
            if snap.topology and snap.topology.pcie_bw_mb_s > 0:
                return snap.topology.pcie_saturation_mb_s
    except Exception:
        pass
    try:
        return get_topology().pcie_saturation_mb_s
    except Exception:
        return 24576.0  # PCIe 4.0 x16 @ 75%


def _vram_budget_ok(needed_mb: float = 0.0) -> bool:
    """Return False when free VRAM headroom is too tight for prefetch.

    Reads the live GpuMetrics snapshot from the gb_init telemetry singleton.
    Falls back to True (optimistic) when telemetry is unavailable so MoE
    operation is never blocked in environments without the shim.

    Also returns False when:
      - GPU SM utilization > 95%: saturated compute, deferring PCIe I/O avoids
        adding latency to ongoing kernels.
      - PCIe TX+RX throughput > pcie_high_water (B2): T2→T1 DMA already
        saturated; additional prefetch I/O would contend with active transfers.
    """
    try:
        from gb_init import get_telemetry
        tel = get_telemetry()
        if tel is None:
            return True
        snap = tel.snapshot()
        if snap.gpu_util_pct > 95.0:
            return False
        # B2: PCIe direct gate , check actual bandwidth, not just SM utilization
        pcie_total = snap.pcie_tx_mb_s + snap.pcie_rx_mb_s
        if pcie_total > _get_pcie_high_water():
            return False
        return snap.prefetch_budget_mb >= max(needed_mb, 64.0)
    except Exception:
        return True


def _find_moe_blocks(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module, torch.nn.Module, "torch.nn.ModuleList | torch.nn.ModuleDict"]]:
    """Generic MoE-block detector: any submodule that has both a gate/router
    attribute and an `experts` ModuleList/ModuleDict attribute. Covers the
    common HF convention (Mixtral's `block_sparse_moe`, Qwen-MoE/DeepSeek-MoE
    style `mlp` blocks) without hardcoding a model family.

    Returns list of (block_name, block_module, gate_module, experts_container).
    """
    found = []
    for name, mod in model.named_modules():
        experts = getattr(mod, "experts", None)
        if not isinstance(experts, (torch.nn.ModuleList, torch.nn.ModuleDict)):
            continue
        gate = getattr(mod, "gate", None) or getattr(mod, "router", None)
        if gate is None or not isinstance(gate, torch.nn.Module):
            continue
        found.append((name, mod, gate, experts))
    return found


def _expert_items(experts) -> List[Tuple[str, torch.nn.Module]]:
    if isinstance(experts, torch.nn.ModuleDict):
        return list(experts.items())
    return [(str(i), m) for i, m in enumerate(experts)]


# ── batched-*Experts convention (transformers 5.x) ───────────────────────────

def _find_moe_blocks_batched(
    model: torch.nn.Module,
) -> "List[Tuple[str, torch.nn.Module, torch.nn.Module, torch.nn.Module, List[str], int]]":
    """Detect MoE blocks using the batched-*Experts convention from current
    transformers (Mixtral, Qwen3-MoE, OLMoE, Phi-MoE, GLM-MoE, etc.):

    A single ``*Experts`` sub-module owns all expert weights as 3D parameters
    with shape [num_experts, out_features, in_features], replacing the
    per-expert nn.ModuleList used in transformers <=4.50.

    Returns list of (block_name, block, gate, experts_mod, param_names, num_experts).
    Skips any block already covered by _find_moe_blocks (ModuleList/Dict).
    """
    found = []
    seen_gate_ids: set = set()
    for name, mod in model.named_modules():
        gate = getattr(mod, "gate", None) or getattr(mod, "router", None)
        if gate is None or not isinstance(gate, torch.nn.Module):
            continue
        if id(gate) in seen_gate_ids:
            continue
        experts_mod = getattr(mod, "experts", None)
        if experts_mod is None or isinstance(experts_mod, (torch.nn.ModuleList, torch.nn.ModuleDict)):
            continue
        if not isinstance(experts_mod, torch.nn.Module):
            continue
        params_3d = [
            (pn, p)
            for pn, p in experts_mod.named_parameters(recurse=False)
            if p.dim() == 3
        ]
        if not params_3d:
            continue
        num_experts = params_3d[0][1].shape[0]
        if not all(p.shape[0] == num_experts for _, p in params_3d):
            continue
        seen_gate_ids.add(id(gate))
        found.append((name, mod, gate, experts_mod, [pn for pn, _ in params_3d], num_experts))
    return found


def _quant_int8_slice(t: torch.Tensor) -> "Tuple[torch.Tensor, torch.Tensor]":
    """Per-row absmax int8 quantization of a 2D CPU tensor.
    scale = absmax / 127 so that q * scale reconstructs the original.
    Returns (int8_tensor, float32_scale_per_row).
    Provides ~2× memory reduction vs bf16 for cold expert slices."""
    f = t.reshape(-1, t.shape[-1]).float()
    scale = f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = (f / scale).clamp(-128, 127).round().to(torch.int8)
    return q.reshape(t.shape), scale.squeeze(-1)


def _dequant_int8_slice(
    q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    flat_q = q.reshape(-1, q.shape[-1]).float()
    flat_s = scale.reshape(-1, 1).float()
    return (flat_q * flat_s).reshape(q.shape).to(dtype)


@dataclass
class _BatchedBlockState:
    """State for one MoE block using the batched-*Experts 3D-param convention.

    Expert slices are managed as individual CPU-side buffers rather than via
    ModelTierManager (which requires nn.Module objects, not raw parameter views).
    cpu_bufs[expert_idx][param_name] holds either:
      - a bare CPU tensor (full-precision copy, cold_bits >= 8)
      - a (int8_tensor, scale_tensor) tuple (per-row absmax, cold_bits <= 4)
    When an expert is cold its GPU slice in experts_mod.{param_name}.data[i]
    is zeroed; restore copies the CPU buf back and refills the GPU slice.
    """
    block_name: str
    block: torch.nn.Module
    gate: torch.nn.Module
    experts_mod: torch.nn.Module
    param_names: List[str]
    num_experts: int
    top_k: int
    freq: "torch.Tensor"                # EMA histogram, CPU, len == num_experts
    hot: set = field(default_factory=set)   # str(i) for GPU-resident experts
    cpu_bufs: dict = field(default_factory=dict)  # {int: {str: tensor|tuple}}
    gate_handle: object = None
    calls: int = 0
    misses: int = 0             # B1: cold-expert restores in current window (not prefetched)
    last_miss_rate: float = 0.0 # B1: miss rate from last rebalance window (for status/adaptation)


@dataclass
class _BlockState:
    block_name: str
    block: torch.nn.Module
    gate: torch.nn.Module
    experts: List[Tuple[str, torch.nn.Module]]
    top_k: int
    freq: "torch.Tensor" = field(default=None)  # EMA histogram, len == num_experts
    hot: set = field(default_factory=set)        # currently-hot expert keys
    gate_handle: object = None
    pre_hook_handles: list = field(default_factory=list)
    calls: int = 0                                # this block's own hook-fire count


_EMA_DECAY = 0.98
_REBALANCE_EVERY = 64   # gate-hook firings between rebalance passes (per block)


def _infer_top_k(block: torch.nn.Module, num_experts: int) -> int:
    for attr in ("top_k", "num_experts_per_tok", "k"):
        v = getattr(block, attr, None)
        if isinstance(v, int) and 0 < v <= num_experts:
            return v
    return min(2, num_experts)  # common default (Mixtral uses 2)


class GbMoEManager:
    """Attach to an HF-style MoE model to track routing and rebalance
    expert placement/precision between T1 (hot, full precision) and
    T2 (cold, quantized) based on observed activation frequency.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model (or a submodule containing the MoE decoder layers).
    hot_threshold : float
        Minimum EMA-normalized frequency for an expert to be considered hot
        (kept/promoted to T1, full precision). Default 0.05 (5% of tokens).
    cold_bits : int | str
        Precision for cold experts via gb_quant.quantize_module. Default 4.
    prefetch_topn : int
        Number of next-block historically-hottest experts to prefetch async
        while the current block computes. Default 2.
    tier_manager : ModelTierManager | None
        Reuse an existing one (e.g. the diffusion/LLM orchestrator's), or a
        dedicated instance is created.
    """

    def __init__(self, model: torch.nn.Module, hot_threshold: float = 0.05,
                 cold_bits: "int | str" = 4, prefetch_topn: int = 2, tier_manager=None):
        self.model = model
        self.hot_threshold = hot_threshold
        self.cold_bits = cold_bits
        self.prefetch_topn = prefetch_topn

        from gb_model_tier import ModelTierManager
        self.tm = tier_manager or ModelTierManager()

        self._blocks: List[_BlockState] = []
        self._block_order: Dict[str, int] = {}
        self._batched_blocks: List[_BatchedBlockState] = []
        self._batched_order: Dict[str, int] = {}
        self._attached = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def attach(self) -> int:
        """Register gate forward hooks on every detected MoE block. Returns
        the number of blocks found (0 if the model has no MoE layers , safe
        no-op for dense models)."""
        if self._attached:
            return len(self._blocks)

        found = _find_moe_blocks(self.model)
        for idx, (name, block, gate, experts) in enumerate(found):
            items = _expert_items(experts)
            num_experts = len(items)
            if num_experts == 0:
                continue
            top_k = _infer_top_k(block, num_experts)
            st = _BlockState(
                block_name=name, block=block, gate=gate, experts=items,
                top_k=top_k,
                freq=torch.zeros(num_experts, dtype=torch.float32),
            )
            for key, mod in items:
                entry_name = f"moe::{name}::{key}"
                self.tm.register(entry_name, mod, tier="T1_HBM")
                st.hot.add(key)  # everything starts hot/full-precision
                # Correctness safety net: real per-token MoE routing can
                # select ANY expert regardless of its historical frequency
                # , a demoted/cold expert WILL be invoked sometimes, not
                # just in theory. Without this, _rebalance()'s demote() can
                # leave an expert on CPU while the model's own dispatch code
                # feeds it a CUDA tensor, crashing with a device-mismatch
                # error (confirmed by running this against a real Mixtral
                # forward pass). This pre-hook synchronously promotes a
                # cold expert back to T1 the instant it's actually about to
                # run, guaranteeing correctness; the predictive prefetch in
                # _prefetch_next is what keeps this path cold in practice.
                handle = mod.register_forward_pre_hook(self._make_pre_hook(entry_name))
                st.pre_hook_handles.append(handle)

            handle = gate.register_forward_hook(self._make_hook(st))
            st.gate_handle = handle
            self._block_order[name] = idx
            self._blocks.append(st)

        # ── batched-*Experts blocks (transformers 5.x) ───────────────────────
        for idx, (name, block, gate, experts_mod, param_names, num_experts) in enumerate(
            _find_moe_blocks_batched(self.model)
        ):
            if num_experts == 0:
                continue
            top_k = _infer_top_k(block, num_experts)
            bst = _BatchedBlockState(
                block_name=name, block=block, gate=gate,
                experts_mod=experts_mod, param_names=param_names,
                num_experts=num_experts, top_k=top_k,
                freq=torch.zeros(num_experts, dtype=torch.float32),
                hot=set(str(i) for i in range(num_experts)),
            )
            bst.gate_handle = gate.register_forward_hook(self._make_batched_hook(bst))
            self._batched_order[name] = idx
            self._batched_blocks.append(bst)

        self._attached = True
        return len(self._blocks) + len(self._batched_blocks)

    def _make_pre_hook(self, entry_name: str):
        def _pre_hook(module, inputs):
            entry = self.tm._entries.get(entry_name)
            if entry is not None and entry.tier != "T1_HBM":
                self.tm.promote(entry_name)
        return _pre_hook

    # ── batched-*Experts hooks and slice management ───────────────────────────

    def _make_batched_hook(self, bst: _BatchedBlockState):
        """Gate post-hook for batched-experts blocks.

        Fires immediately after gate.forward() returns routing logits, before
        the batched expert GEMM reads the 3D param tensors. This gives us time
        to synchronously restore any cold expert's GPU slice that was zeroed by
        _demote_expert_slice , the correctness guarantee for batched experts,
        equivalent to _make_pre_hook on the ModuleList path.
        """
        def _hook(module, inputs, output):
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(logits) or logits.numel() == 0:
                return
            with torch.no_grad():
                flat = logits.reshape(-1, logits.shape[-1]).detach()
                k = min(bst.top_k, flat.shape[-1])
                top_idx = torch.topk(flat.float(), k=k, dim=-1).indices.reshape(-1)
                counts = torch.bincount(top_idx, minlength=bst.num_experts).float()
                counts = counts / counts.sum().clamp_min(1.0)
                bst.freq.mul_(_EMA_DECAY).add_(counts.to(bst.freq.device), alpha=1.0 - _EMA_DECAY)
                # Correctness: restore GPU slices for any cold expert about to
                # participate in the batched GEMM.
                for i in top_idx.unique().tolist():
                    if str(i) not in bst.hot:
                        bst.misses += 1  # B1: prefetch miss , expert was cold at compute time
                        self._restore_expert_slice(bst, i)
            bst.calls += 1
            if bst.calls % _REBALANCE_EVERY == 0:
                self._rebalance_batched(bst)
            self._prefetch_next_batched(bst)
        return _hook

    def _restore_expert_slice(self, bst: _BatchedBlockState, expert_idx: int) -> None:
        """Synchronously copy a cold expert's CPU buffer back into its 3D-param
        slice on the GPU. Clears the cpu_bufs entry and marks the expert hot."""
        buf = bst.cpu_bufs.pop(expert_idx, None)
        if buf is None:
            return
        for pname in bst.param_names:
            param = getattr(bst.experts_mod, pname)
            stored = buf.get(pname)
            if stored is None:
                continue
            if isinstance(stored, tuple):
                q, scale = stored
                data = _dequant_int8_slice(q, scale, param.dtype)
            else:
                data = stored.to(param.dtype)
            with torch.no_grad():
                param.data[expert_idx].copy_(data.to(param.device), non_blocking=False)
        bst.hot.add(str(expert_idx))

    def _demote_expert_slice(self, bst: _BatchedBlockState, expert_idx: int) -> None:
        """Copy expert slice to a CPU-side buffer then zero the GPU slice so
        cold-expert memory is freed from VRAM. When cold_bits <= 4 the CPU
        copy is int8-quantized (per-row absmax, ~2× vs bf16)."""
        buf: dict = {}
        for pname in bst.param_names:
            param = getattr(bst.experts_mod, pname)
            slice_cpu = param.data[expert_idx].detach().to("cpu", non_blocking=False)
            if self.cold_bits <= 4:
                # Reshape to 2D for per-row absmax int8 quantization.
                orig_shape = slice_cpu.shape
                flat = slice_cpu.reshape(-1, orig_shape[-1])
                q, scale = _quant_int8_slice(flat)
                buf[pname] = (q.reshape(orig_shape), scale.reshape(orig_shape[:-1]))
            else:
                buf[pname] = slice_cpu
            with torch.no_grad():
                param.data[expert_idx].zero_()
        bst.cpu_bufs[expert_idx] = buf
        bst.hot.discard(str(expert_idx))

    def detach(self):
        for st in self._blocks:
            if st.gate_handle is not None:
                st.gate_handle.remove()
            for h in st.pre_hook_handles:
                h.remove()
        self._blocks.clear()
        self._block_order.clear()
        # Restore all cold batched-expert slices so the model is left in a
        # valid state (no zeroed slices) after detach.
        for bst in self._batched_blocks:
            for i in list(bst.cpu_bufs.keys()):
                self._restore_expert_slice(bst, i)
            if bst.gate_handle is not None:
                bst.gate_handle.remove()
        self._batched_blocks.clear()
        self._batched_order.clear()
        self._attached = False

    # ── routing observation ──────────────────────────────────────────────────

    def _make_hook(self, st: _BlockState):
        def _hook(module, inputs, output):
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(logits) or logits.numel() == 0:
                return
            with torch.no_grad():
                flat = logits.reshape(-1, logits.shape[-1]).detach()
                k = min(st.top_k, flat.shape[-1])
                top_idx = torch.topk(flat.float(), k=k, dim=-1).indices.reshape(-1)
                counts = torch.bincount(top_idx, minlength=st.freq.numel()).float()
                counts = counts / counts.sum().clamp_min(1.0)
                # freq is intentionally CPU-resident (cheap bookkeeping, not
                # tensor compute) , the model may run on CUDA, so move this
                # tiny per-block histogram update to CPU rather than freq to GPU.
                st.freq.mul_(_EMA_DECAY).add_(counts.to(st.freq.device), alpha=1.0 - _EMA_DECAY)

            st.calls += 1
            if st.calls % _REBALANCE_EVERY == 0:
                self._rebalance(st)
            self._prefetch_next(st)
        return _hook

    # ── compression / tier rebalance ─────────────────────────────────────────

    def _rebalance(self, st: _BlockState):
        import gb_quant

        total = st.freq.sum().clamp_min(1e-8)
        norm = st.freq / total
        for i, (key, mod) in enumerate(st.experts):
            is_hot = float(norm[i]) >= self.hot_threshold
            was_hot = key in st.hot
            entry_name = f"moe::{st.block_name}::{key}"
            if is_hot and not was_hot:
                self.tm.promote(entry_name)
                st.hot.add(key)
            elif not is_hot and was_hot:
                gb_quant.quantize_module(mod, bits=self.cold_bits)
                self.tm.demote(entry_name)
                st.hot.discard(key)

    # ── batched rebalance + prefetch ─────────────────────────────────────────

    def _rebalance_batched(self, bst: _BatchedBlockState) -> None:
        # B1: adapt prefetch_topn from per-window miss rate
        bst.last_miss_rate = bst.misses / _REBALANCE_EVERY
        if bst.last_miss_rate > 0.30:
            # Many cold-expert restores: prefetch more ahead-of-time
            self.prefetch_topn = min(self.prefetch_topn + 1, max(1, bst.num_experts // 2))
        elif bst.last_miss_rate < 0.05 and self.prefetch_topn > 1:
            # Very few misses: prefetching more than needed, save VRAM bandwidth
            self.prefetch_topn = max(1, self.prefetch_topn - 1)
        bst.misses = 0  # reset window for next _REBALANCE_EVERY calls

        # B1+: adapt hot_threshold from VRAM pressure + miss rate.
        # When VRAM is tight, raise the bar for "hot" so more experts are
        # demoted, freeing T1.  When VRAM is OK but misses are high, lower
        # the bar so more experts stay resident and avoid restore overhead.
        vram_ok = _vram_budget_ok()
        if not vram_ok:
            # VRAM pressure: be aggressive , evict more experts to T2
            self.hot_threshold = min(0.20, self.hot_threshold * 1.10)
        elif bst.last_miss_rate > 0.25:
            # VRAM fine but many cold restores: widen the hot set
            self.hot_threshold = max(0.01, self.hot_threshold * 0.90)

        total = bst.freq.sum().clamp_min(1e-8)
        norm = bst.freq / total
        for i in range(bst.num_experts):
            is_hot = float(norm[i]) >= self.hot_threshold
            was_hot = str(i) in bst.hot
            if is_hot and not was_hot:
                self._restore_expert_slice(bst, i)
            elif not is_hot and was_hot:
                self._demote_expert_slice(bst, i)

    def _prefetch_next_batched(self, bst: _BatchedBlockState) -> None:
        """Copy the historically-hottest cold expert slices of the NEXT batched
        block back to GPU before that block's GEMM runs.

        Copy is intentionally blocking (non_blocking=False): we mark the expert
        hot immediately after, so the next block's gate hook sees it as hot and
        skips the synchronous restore path. Making it truly async would require
        a CUDA event fence between the transfer and compute streams; the current
        conservative approach still avoids cold-path synchronous restores for
        correctly-predicted experts.

        Budget gate: skipped entirely when free VRAM headroom is below 64 MB or
        GPU SM utilization is saturated (>95%) , competing prefetch I/O during
        peak compute adds latency without benefit. The synchronous restore in
        _make_batched_hook remains as a correctness fallback for cold experts
        that become hot despite skipped prefetch.
        """
        idx = self._batched_order.get(bst.block_name)
        if idx is None or idx + 1 >= len(self._batched_blocks):
            return
        nxt = self._batched_blocks[idx + 1]
        if nxt.freq.sum() <= 0 or not nxt.cpu_bufs:
            return

        # Estimate cold expert size (MB) from first cpu_buf entry for budget gate
        estimated_mb: float = 0.0
        for buf in nxt.cpu_bufs.values():
            for stored in buf.values():
                t = stored[0] if isinstance(stored, tuple) else stored
                estimated_mb += t.numel() * t.element_size() / (1024 * 1024)
            break  # one expert is enough for the estimate

        if not _vram_budget_ok(estimated_mb * self.prefetch_topn):
            return

        top = torch.topk(
            nxt.freq, k=min(self.prefetch_topn, nxt.num_experts)
        ).indices.tolist()
        for i in top:
            if str(i) in nxt.hot or i not in nxt.cpu_bufs:
                continue
            self._restore_expert_slice(nxt, i)

    # ── statistical predictive prefetch ──────────────────────────────────────

    def _prefetch_next(self, st: _BlockState):
        """Prefetch the historically-hottest experts of the *next* MoE block
        on the transfer stream while this block's gate/expert compute runs.
        Heuristic only (see module docstring) , never blocks, never required
        for correctness (promote() on the critical path is the fallback)."""
        if not _vram_budget_ok():
            return
        idx = self._block_order.get(st.block_name)
        if idx is None or idx + 1 >= len(self._blocks):
            return
        nxt = self._blocks[idx + 1]
        if nxt.freq.sum() <= 0:
            return
        top = torch.topk(nxt.freq, k=min(self.prefetch_topn, nxt.freq.numel())).indices.tolist()

        import gb_stream_sched as gs
        for i in top:
            key, mod = nxt.experts[i]
            entry_name = f"moe::{nxt.block_name}::{key}"
            entry = self.tm._entries.get(entry_name)
            if entry is None or entry.tier == "T1_HBM":
                continue
            with gs.on("transfer"):
                mod.to("cuda", non_blocking=True)
            entry.tier = "T1_HBM"
            entry.last_used = time.time()

    # ── introspection ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        out = {}
        for st in self._blocks:
            total = float(st.freq.sum().clamp_min(1e-8))
            norm = (st.freq / total).tolist() if total > 0 else [0.0] * st.freq.numel()
            out[st.block_name] = {
                "convention": "modlist",
                "num_experts": len(st.experts),
                "top_k": st.top_k,
                "hot": sorted(st.hot),
                "freq": norm,
            }
        for bst in self._batched_blocks:
            total = float(bst.freq.sum().clamp_min(1e-8))
            norm = (bst.freq / total).tolist() if total > 0 else [0.0] * bst.num_experts
            out[bst.block_name] = {
                "convention": "batched_3d",
                "num_experts": bst.num_experts,
                "top_k": bst.top_k,
                "hot": sorted(bst.hot),
                "cold_on_cpu": sorted(bst.cpu_bufs.keys()),
                "freq": norm,
                "miss_rate": round(bst.last_miss_rate, 3),  # B1: last window miss rate
                "prefetch_topn": self.prefetch_topn,          # B1: current adapted value
                "hot_threshold": round(self.hot_threshold, 4), # B1+: adapted threshold
            }
        return out
