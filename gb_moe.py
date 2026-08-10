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
  2. keep cold experts LOSSLESSLY compressed (CPU-side, bit-exact ,
     _demote_expert_module/_restore_expert_module, see their docstrings)
     and DDR-resident (ModelTierManager, T2) instead of full-precision T1
     VRAM. 2026-08-10: replaced the original gb_quant.quantize_module(int4)
     demote path, which was irreversible , promote() never un-quantized,
     so an expert that went cold and came back hot returned permanently
     precision-degraded, contradicting this module's own "T1 = full
     precision" contract. compress_ratio tracking on this path also feeds
     entropy-aware placement (item 3 below): a well-compressing expert is a
     cheaper eviction than an equally-cold peer that doesn't compress, so
     it's held to a stricter hot-bar , see _entropy_adjusted_threshold.
  3. statistically (optionally MTP-predictively, see mtp_oracle on
     GbMoEManager.__init__) prefetch the next layer's likely-hot experts to
     T1 on the transfer stream while the current layer computes, so a
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
    # Optional, for models with native MTP heads (see mtp_oracle's
    # docstring on GbMoEManager.__init__) , omit entirely for pure-history
    # prefetch, unchanged from before this parameter existed:
    #   mgr = gb_moe.GbMoEManager(model, mtp_oracle=my_mtp_router_lookahead)
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


# ── lossless CPU-side compression for cold-expert demotion ─────────────────
#
# 2026-08-10: replaces gb_quant.quantize_module()'s use as the non-batched
# _rebalance() demote path. That call was IRREVERSIBLE — it replaces a
# module's Linear layers in place with lower-precision GemLite/TurboQuant
# kernels, and nothing on the promote side ever undoes it (ModelTierManager.
# promote() is a plain `module.to("cuda")`; it has no un-quantize). An
# expert that went cold and later became hot again came back permanently
# precision-degraded, silently, contradicting this module's own docstring
# ("T1, hot, full precision"). Confirmed by reading gb_model_tier.py's
# promote()/demote(): both are bare device-transfer calls, so
# quantize_module() was doing double duty as BOTH the DDR-footprint
# reduction AND an (irreversible) precision cut. _demote_expert_module /
# _restore_expert_module below split those two concerns: ModelTierManager
# still owns the device transfer (unchanged), and footprint reduction now
# comes from a genuinely lossless, genuinely reversible compression step
# instead of a lossy one.
#
# zlib (Python stdlib, zero new runtime dependency) is deliberately not
# nvCOMP/DFloat11-class — those need a real GPU codec runtime that isn't
# vendored on this box yet (see docs/research/spark-parity-survey.md's
# empirical codec gate, and patches/custom/0020-dma-buf-compressed-
# descriptor.patch, in ~/Dev/kernel_inference, which this demote path is
# the intended first consumer of once a GPU codec lands). zlib buys real,
# working, bit-exact compression today with no new dependency; swapping the
# two functions below for a GPU-side codec is a drop-in replacement for
# every caller once one is vendored and benchmarked — don't do that swap
# blind, benchmark it first per the survey's step 2.1.
#
# `.view(torch.uint8)` is what makes this dtype-agnostic (works for
# bfloat16 too, which plain `.numpy()` cannot handle — numpy has no native
# bfloat16 type): it reinterprets the same underlying bytes without
# copying or converting any value, so compression operates on the exact
# on-the-wire representation and decompression reconstructs it bit-exact.

def _lossless_compress_tensor(t: torch.Tensor) -> "Tuple[bool, bytes, torch.Size, torch.dtype]":
    """Bit-exact CPU compression of a tensor's raw bytes. Returns
    (is_compressed, payload, original_shape, original_dtype).

    Near-random data (e.g. a freshly torch.randn-initialized tensor —
    confirmed by this module's own tests) can have no exploitable
    redundancy at all, in which case zlib's own framing overhead makes
    its output LARGER than the input. Real model weights compress better
    in practice (see docs/research/spark-parity-survey.md's "LLM weight
    effective entropy is 2-10x lower than stored bitwidth" finding), but
    this function must never assume that — is_compressed=False signals
    the caller to store the raw bytes as-is instead of a "compressed"
    blob that would waste more DDR than a plain copy."""
    import zlib
    cpu_t = t.detach().to("cpu", non_blocking=False).contiguous()
    byte_view = cpu_t.view(torch.uint8)
    raw_bytes = byte_view.numpy().tobytes()
    compressed = zlib.compress(raw_bytes, level=6)
    if len(compressed) < len(raw_bytes):
        return True, compressed, cpu_t.shape, cpu_t.dtype
    return False, raw_bytes, cpu_t.shape, cpu_t.dtype


def _lossless_decompress_tensor(
    is_compressed: bool, payload: bytes, shape: "torch.Size", dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of _lossless_compress_tensor — bit-exact, not an
    approximation. Returns a CPU tensor; caller moves it to the target
    device (ModelTierManager.promote()'s subsequent `.to("cuda")` does
    this for the non-batched path)."""
    import zlib
    raw = zlib.decompress(payload) if is_compressed else payload
    byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return byte_tensor.view(dtype).reshape(shape).clone()


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
    # {expert_key: {param_name: (compressed_bytes, shape, dtype)}} for cold
    # experts — see _demote_expert_module/_restore_expert_module. A
    # parameter's real data is swapped out to here (replaced on-module with
    # a 1-element placeholder) while cold; restoring pops the entry and
    # writes the decompressed bytes back, bit-exact.
    lossless_bufs: dict = field(default_factory=dict)
    # {expert_key: last observed compressed_size / uncompressed_size} — the
    # entropy-aware placement signal: an expert that compresses better is a
    # cheaper eviction (less to re-fetch/recompute on promote) than an
    # equally-cold peer that doesn't, so demotion order can prefer it. Only
    # populated after an expert has been demoted at least once.
    compress_ratio: dict = field(default_factory=dict)


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
    hysteresis_margin : float
        Relative dead-zone around hot_threshold an expert's normalized
        frequency must clear before its resident state flips (promote needs
        norm >= hot_threshold*(1+margin); demote needs norm <
        hot_threshold*(1-margin); anything in between keeps its current
        state). Without this, an expert whose frequency sits near the
        threshold flaps promote/demote every _REBALANCE_EVERY window , each
        flip does real work (quantize/dequantize + a tier move), and
        hot_threshold itself is adapted every batched-rebalance pass
        (_rebalance_batched), which pushes MORE experts toward that boundary
        over time. Same anti-thrashing principle as colibri's
        tier_pick_lfru margin (25%+fixed-floor hysteresis on its heat
        score) , adapted here to a per-expert relative-frequency threshold
        rather than colibri's bounded-slot swap-gain comparison, since this
        manager tracks an unbounded hot SET rather than a fixed number of
        pinned slots. Default 0.25 (colibri's own default margin, `fc>>2`).
    tier_manager : ModelTierManager | None
        Reuse an existing one (e.g. the diffusion/LLM orchestrator's), or a
        dedicated instance is created.
    mtp_oracle : Callable[[str, torch.Tensor], "Iterable[int] | None"] | None
        Optional predictive-prefetch oracle for models with native
        multi-token-prediction (MTP) heads (e.g. this workstation's default
        Qwen3.6 model — see docs/research/spark-parity-survey.md's Track
        2.2 in ~/Dev/kernel_inference). The module docstring's own
        limitation note is exactly the gap this closes: "layer i+1's actual
        routing depends on layer i's output, which doesn't exist until
        layer i finishes" — true for the MAIN forward pass, but an MTP
        head predicts several tokens ahead using information already
        available, so its predicted routing can stand in for the
        historical-frequency heuristic the prefetch otherwise relies on
        alone. Called as `mtp_oracle(block_name, gate_logits)` right after
        each block's gate fires; return an iterable of predicted expert
        indices for the NEXT block, or None/empty to fall back to pure
        history for that call. Predictions are UNIONED with the
        historical top-`prefetch_topn` list, never used to replace it ,
        this is a hint that can widen the prefetch set, not a correctness
        dependency (the synchronous pre-hook/gate-hook restore paths
        remain the actual correctness guarantee regardless of what this
        oracle predicts). Default None: behavior is then byte-identical to
        before this parameter existed. Speculative decoding is universally
        used to amortize compute; using its predictions as a memory-tiering
        oracle is the actually-new idea here, not a proven technique , no
        MTP-capable model was available to test this against live in this
        session, so treat it as a documented, tested-at-the-plumbing-level
        extension point rather than a validated performance claim.
    """

    def __init__(self, model: torch.nn.Module, hot_threshold: float = 0.05,
                 cold_bits: "int | str" = 4, prefetch_topn: int = 2,
                 hysteresis_margin: float = 0.25, tier_manager=None,
                 mtp_oracle=None):
        self.model = model
        self.hot_threshold = hot_threshold
        self.cold_bits = cold_bits
        self.mtp_oracle = mtp_oracle
        self.prefetch_topn = prefetch_topn
        self.hysteresis_margin = hysteresis_margin

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
                handle = mod.register_forward_pre_hook(self._make_pre_hook(st, key, mod, entry_name))
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

    def _make_pre_hook(self, st: _BlockState, key: str, mod: torch.nn.Module, entry_name: str):
        def _pre_hook(module, inputs):
            entry = self.tm._entries.get(entry_name)
            if entry is not None and entry.tier != "T1_HBM":
                # Real per-token routing can select a cold expert regardless
                # of its historical frequency (see attach()'s comment on
                # this hook) — _restore_expert_module decompresses it
                # bit-exact before promoting, rather than the old bare
                # self.tm.promote(entry_name), which would have handed the
                # forward pass a still-1-element placeholder parameter.
                self._restore_expert_module(st, key, mod, entry_name)
                st.hot.add(key)
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
            self._prefetch_next_batched(bst, gate_output=logits)
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
            # Restore any still-cold experts BEFORE removing hooks/clearing
            # state, so the model is left with real weights, not 1-element
            # placeholders, after detach. This mirrors the batched path's
            # existing cpu_bufs restore below — required here because
            # _demote_expert_module swaps each cold parameter's .data out
            # to a placeholder; skipping this would silently strand the
            # model with unusable experts instead of merely stale tier
            # bookkeeping (a materially worse failure mode than the
            # pre-existing quantize-in-place behavior this replaced, which
            # never removed the parameter's real storage).
            for key, mod in st.experts:
                if key in st.lossless_bufs:
                    entry_name = f"moe::{st.block_name}::{key}"
                    self._restore_expert_module(st, key, mod, entry_name)
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
            self._prefetch_next(st, gate_output=logits)
        return _hook

    # ── compression / tier rebalance ─────────────────────────────────────────

    def _is_hot(self, freq_norm: float, was_hot: bool, threshold: "float | None" = None) -> bool:
        """Hysteresis-gated hot/cold decision , see hysteresis_margin
        docstring on __init__. Not-yet-classified experts (was_hot=False on
        an expert never seen resident) use the plain threshold, same as a
        demoted expert would on the cold side.

        threshold: overrides self.hot_threshold when given — see
        _entropy_adjusted_threshold, the entropy-aware placement caller."""
        base = self.hot_threshold if threshold is None else threshold
        if was_hot:
            return freq_norm >= base * (1.0 - self.hysteresis_margin)
        return freq_norm >= base * (1.0 + self.hysteresis_margin)

    def _entropy_adjusted_threshold(self, st: _BlockState, key: str) -> float:
        """Entropy-aware placement (spark-parity-survey.md Track 2.2, in
        ~/Dev/kernel_inference): every tiering scheme in gb_moe.py before
        this ranked hot/cold purely by access frequency, which is
        entropy-blind. An expert that compresses well is a CHEAPER
        eviction than an equally-hot peer that doesn't (less to re-fetch/
        decompress on promote), so it can be held to a stricter ("more
        must stay cold") bar , freeing VRAM budget for poorly-compressing,
        expensive-to-refetch experts to stay resident instead.

        st.compress_ratio[key] defaults to 1.0 (assume worst case,
        expensive to evict) for any expert never yet demoted, so a fresh
        model with zero compression history behaves EXACTLY as it did
        before this feature existed , the adjustment only kicks in once an
        expert has gone cold at least once and its real ratio is known.
        Scoped to the non-batched (state_dict-module) path, the only one
        that currently tracks compress_ratio; the batched 3D-param path's
        cold_bits>4 branch is lossless-but-currently-uncompressed (stores
        a raw CPU copy), so it has no ratio signal to adjust with yet.
        """
        ratio = st.compress_ratio.get(key, 1.0)
        return self.hot_threshold * (2.0 - ratio)

    def _emit_expert_placement(self, block_name: str, key: str, action: str,
                               freq_norm: float) -> None:
        """Best-effort dataflux event for a real hot/cold expert-tier
        transition , Rule #1's MoE clause is that the VRAM occupied by
        experts must be the subset ACTUALLY used by inference (real routing
        frequency), not an arbitrary or load-order-determined one; without
        this, that placement decision was invisible to dataflux entirely."""
        try:
            import gb_dataflux
            gb_dataflux.emit({
                "node": "host", "label": "gb_moe", "kind": "placement",
                "runtime": "moe_expert", "block": block_name, "expert": key,
                "action": action, "freq_norm": round(freq_norm, 4),
                "cold_bits": self.cold_bits,
            })
        except Exception:
            pass

    def _demote_expert_module(self, st: _BlockState, key: str,
                              mod: torch.nn.Module, entry_name: str) -> None:
        """Lossless demote: ModelTierManager still owns the actual device
        transfer (self.tm.demote -> module.to("cpu"), unchanged, already
        correct), then each parameter's now-CPU-resident bytes are
        compressed bit-exact and the module's own copy is replaced with a
        1-element placeholder so its nominal footprint tracks what's
        actually being kept resident. See the _lossless_compress_tensor
        module comment for why this replaces gb_quant.quantize_module
        here specifically."""
        self.tm.demote(entry_name)
        buf: dict = {}
        total_raw = 0
        total_stored = 0
        for pname, p in mod.named_parameters():
            is_compressed, payload, shape, dtype = _lossless_compress_tensor(p.data)
            buf[pname] = (is_compressed, payload, shape, dtype)
            total_raw += p.data.numel() * p.data.element_size()
            total_stored += len(payload)
            with torch.no_grad():
                p.data = torch.zeros(1, dtype=p.dtype)
        st.lossless_bufs[key] = buf
        if total_raw > 0:
            # Capped at 1.0: an incompressible tensor stores its raw bytes
            # (is_compressed=False, see _lossless_compress_tensor), so this
            # ratio never exceeds "no space saved", by construction — the
            # entropy-aware placement signal this feeds should never see a
            # value implying compression made things worse.
            st.compress_ratio[key] = min(1.0, total_stored / total_raw)

    def _restore_expert_module(self, st: _BlockState, key: str,
                               mod: torch.nn.Module, entry_name: str) -> None:
        """Inverse of _demote_expert_module — decompresses each parameter
        back to its exact original bytes before handing off to
        self.tm.promote() (which does the actual `.to("cuda")` transfer,
        unchanged). Safe/idempotent to call on an already-hot expert:
        lossless_bufs.pop() returns None and only tm.promote() (itself a
        no-op for an already-T1 entry) runs."""
        buf = st.lossless_bufs.pop(key, None)
        if buf is not None:
            for pname, p in mod.named_parameters():
                stored = buf.get(pname)
                if stored is None:
                    continue
                is_compressed, payload, shape, dtype = stored
                with torch.no_grad():
                    p.data = _lossless_decompress_tensor(is_compressed, payload, shape, dtype)
        self.tm.promote(entry_name)

    def _rebalance(self, st: _BlockState):
        total = st.freq.sum().clamp_min(1e-8)
        norm = st.freq / total
        for i, (key, mod) in enumerate(st.experts):
            was_hot = key in st.hot
            threshold = self._entropy_adjusted_threshold(st, key)
            is_hot = self._is_hot(float(norm[i]), was_hot, threshold)
            entry_name = f"moe::{st.block_name}::{key}"
            if is_hot and not was_hot:
                self._restore_expert_module(st, key, mod, entry_name)
                st.hot.add(key)
                self._emit_expert_placement(st.block_name, key, "promote", float(norm[i]))
            elif not is_hot and was_hot:
                self._demote_expert_module(st, key, mod, entry_name)
                st.hot.discard(key)
                self._emit_expert_placement(st.block_name, key, "demote", float(norm[i]))

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
            was_hot = str(i) in bst.hot
            is_hot = self._is_hot(float(norm[i]), was_hot)
            if is_hot and not was_hot:
                self._restore_expert_slice(bst, i)
                self._emit_expert_placement(bst.block_name, str(i), "promote", float(norm[i]))
            elif not is_hot and was_hot:
                self._demote_expert_slice(bst, i)
                self._emit_expert_placement(bst.block_name, str(i), "demote", float(norm[i]))

    def _prefetch_next_batched(self, bst: _BatchedBlockState, gate_output=None) -> None:
        """Copy the historically-hottest cold expert slices of the NEXT batched
        block back to GPU before that block's GEMM runs.

        gate_output: this block's just-computed gate logits, forwarded to
        self.mtp_oracle if one is configured (see __init__ docstring) —
        None (the default) skips the oracle call, matching pre-existing
        pure-history behavior for any caller that doesn't pass it.

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

        top = set(torch.topk(
            nxt.freq, k=min(self.prefetch_topn, nxt.num_experts)
        ).indices.tolist())
        top |= self._mtp_predicted_indices(nxt.block_name, gate_output, nxt.num_experts)
        for i in top:
            if str(i) in nxt.hot or i not in nxt.cpu_bufs:
                continue
            self._restore_expert_slice(nxt, i)

    # ── statistical (+ optional MTP-predictive) prefetch ────────────────────

    def _mtp_predicted_indices(self, block_name: str, gate_output, num_experts: int) -> set:
        """Best-effort call into self.mtp_oracle (see its __init__
        docstring) — a predictive hint, never a correctness dependency, so
        any failure here (wrong return type, an oracle that raises, no
        oracle configured) degrades silently to "no extra predictions",
        leaving the historical-frequency prefetch as the sole source,
        exactly as if this feature didn't exist."""
        if self.mtp_oracle is None or gate_output is None:
            return set()
        try:
            predicted = self.mtp_oracle(block_name, gate_output)
            if not predicted:
                return set()
            return {int(i) for i in predicted if 0 <= int(i) < num_experts}
        except Exception:
            return set()

    def _prefetch_next(self, st: _BlockState, gate_output=None):
        """Prefetch the historically-hottest experts of the *next* MoE block
        on the transfer stream while this block's gate/expert compute runs.
        Heuristic only (see module docstring) , never blocks, never required
        for correctness (promote() on the critical path is the fallback).

        gate_output: this block's just-computed gate logits, forwarded to
        self.mtp_oracle if one is configured (see __init__ docstring) —
        None (the default) skips the oracle call entirely, so any existing
        caller that doesn't pass it gets pure-history behavior unchanged.
        """
        if not _vram_budget_ok():
            return
        idx = self._block_order.get(st.block_name)
        if idx is None or idx + 1 >= len(self._blocks):
            return
        nxt = self._blocks[idx + 1]
        if nxt.freq.sum() <= 0:
            return
        top = set(torch.topk(nxt.freq, k=min(self.prefetch_topn, nxt.freq.numel())).indices.tolist())
        top |= self._mtp_predicted_indices(nxt.block_name, gate_output, nxt.freq.numel())

        for i in top:
            key, mod = nxt.experts[i]
            entry_name = f"moe::{nxt.block_name}::{key}"
            entry = self.tm._entries.get(entry_name)
            if entry is None or entry.tier == "T1_HBM":
                continue
            # Must go through _restore_expert_module, not a bare
            # mod.to("cuda") + tier flip: if this expert was demoted via
            # _demote_expert_module its parameters are 1-element
            # placeholders on CPU, not real weights — moving those to CUDA
            # and marking the entry T1_HBM directly would leave the model
            # holding placeholder weights for real compute, AND the
            # pre-hook safety net (_make_pre_hook) would see tier==T1_HBM
            # and skip restoring, since it only checks tier, not whether
            # the data is real. _restore_expert_module decompresses first,
            # then calls self.tm.promote() — which already has its own
            # async-transfer handling (gs.on("transfer") + event sync, see
            # ModelTierManager.promote), so this also drops the need for a
            # separate manual transfer-stream block here.
            self._restore_expert_module(nxt, key, mod, entry_name)
            nxt.hot.add(key)
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
