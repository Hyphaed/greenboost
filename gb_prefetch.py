#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_prefetch.py — GreenBoost generic dense-LLM layer-sequential prefetch.

WHY: gb_moe.py predictively prefetches MoE experts (routing is stochastic,
so the next-layer expert set is a frequency-based guess). gb_diffusion_orch.py
prefetches the VAE ahead of the last denoising step. Neither covers the more
common case: a dense transformer's decoder-layer stack, executed in a fixed,
known order every forward pass. Unlike MoE routing, dense-layer order needs
no prediction at all — layer i is always followed by layer i+1 — so this is
a strictly simpler, deterministic instance of the same overlap pattern
(DeepSpeed ZeRO-Infinity's overlap-centric offload: bring tier N+1 onto the
GPU on the transfer stream while tier N computes on the compute stream),
applied to GreenBoost's own explicit T1/T2/T3 tiers via the existing
ModelTierManager and gb_stream_sched primitives — no new tier or kernel
change needed.

Design (mirrors gb_moe.py's validated per-expert pre-hook pattern, scaled
down since there is no misprediction to correct here):

  - `_find_layer_stack()` generically detects the decoder-layer ModuleList
    (covers `model.layers` / `transformer.h` / `gpt_neox.layers` / etc.
    without hardcoding a model family) by picking the largest top-level
    `nn.ModuleList` of parameterized submodules in the model.
  - Each layer is registered with `ModelTierManager` individually (same
    "any nn.Module registers fine" property gb_moe.py already confirmed).
  - A `forward_pre_hook` on layer i does two things, in order:
      1. If layer i was asynchronously prefetched while layer i-1 (or
         earlier) computed, wait for that transfer to land, then mark it
         resident. If it was NOT prefetched (cold start, or
         `keep_resident`/`lookahead` misconfigured), promote it
         synchronously — this is the only "fallback" path, and unlike
         MoE there is no steady-state case where it should ever fire.
      2. Demote the layer `keep_resident` steps behind (sliding window)
         back to T2, then kick off an async prefetch of the layer
         `lookahead` steps ahead on the named "transfer" CUDA stream, so
         the H2D copy overlaps layer i's own compute instead of blocking
         layer i+1's pre-hook.

Limitation (by design, not yet validated against a multi-billion-parameter
model): correctness was validated against a real, GPU-resident synthetic
decoder stack (see tests/test_gb_prefetch.py) with real H2D/D2H transfers
on the "transfer" CUDA stream and real overlap timing; it has not yet been
run against a full-size checkpoint (Llama/Qwen/Mistral) end to end the way
gb_moe.py's Track 3 was. Treat `keep_resident`/`lookahead` defaults as a
starting point for a real model, not a validated value.

Usage:
    import gb_prefetch

    pf = gb_prefetch.LayerPrefetcher(model, keep_resident=2, lookahead=1)
    pf.attach()
    ...run inference...
    print(pf.status())
    pf.detach()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


def _find_layer_stack(model: torch.nn.Module) -> Optional[torch.nn.ModuleList]:
    """Generic decoder-layer-stack detector: the largest top-level
    `nn.ModuleList` in the model whose elements are parameterized modules
    and are not themselves an MoE expert container (gb_moe.py owns those -
    skip any ModuleList directly named `experts` so the two modules never
    fight over the same submodules).

    Returns None if no suitable stack is found (e.g. the model is too
    small, or uses the batched-*Experts MoE convention exclusively with no
    separate dense layer stack).
    """
    best: Optional[torch.nn.ModuleList] = None
    best_len = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList):
            continue
        if name.rsplit(".", 1)[-1] == "experts":
            continue
        if len(mod) < 2:
            continue
        if not any(p.numel() > 0 for p in mod[0].parameters()):
            continue
        if len(mod) > best_len:
            best, best_len = mod, len(mod)
    return best


@dataclass
class _PrefetchState:
    in_flight: Dict[int, bool] = field(default_factory=dict)
    promote_count: int = 0     # layers that arrived already-prefetched
    fallback_count: int = 0    # layers that had to synchronously promote


class LayerPrefetcher:
    """
    Predictive (deterministic) prefetch for a dense transformer's decoder
    layers, overlapping T2(DDR)→T1(VRAM) H2D copies with the previous
    layer's compute.

    Parameters
    ----------
    model : torch.nn.Module
        The model containing a sequential decoder-layer stack.
    tm : ModelTierManager, optional
        Reuses an existing manager (e.g. one already created by the
        caller's orchestrator) instead of constructing a private one.
    keep_resident : int
        Number of layers behind the current one to keep on T1 before
        demoting (sliding window). 2 means the current and previous layer
        stay resident; layers more than 2 behind are demoted to T2.
    lookahead : int
        How many layers ahead to prefetch. 1 means "start moving layer
        i+1 to GPU while layer i computes" - the minimum useful overlap.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tm: Optional["ModelTierManager"] = None,
        keep_resident: int = 2,
        lookahead: int = 1,
    ):
        self.model = model
        self.keep_resident = max(1, keep_resident)
        self.lookahead = max(1, lookahead)
        self._tm = tm
        self._owns_tm = tm is None
        self._layers: Optional[torch.nn.ModuleList] = None
        self._names: List[str] = []
        self._hooks = []
        self._state = _PrefetchState()
        self._attached = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def attach(self):
        """Find the decoder-layer stack, register each layer with
        ModelTierManager (T2 except the first `keep_resident` layers, which
        start on T1 since the first forward pass needs them immediately),
        and install the prefetch pre-hooks."""
        if self._attached:
            return

        self._layers = _find_layer_stack(self.model)
        if self._layers is None:
            print("[gb_prefetch] no decoder-layer stack found - no-op", flush=True)
            return

        if self._tm is None:
            from gb_model_tier import ModelTierManager
            self._tm = ModelTierManager()

        from gb_model_tier import Tier

        n = len(self._layers)
        self._names = [f"gb_prefetch_layer_{i}" for i in range(n)]
        for i, (name, layer) in enumerate(zip(self._names, self._layers)):
            start_tier = Tier.T1 if i < self.keep_resident else Tier.T2
            self._tm.register(name, layer, tier=start_tier)
            # register() only records bookkeeping - actually place the
            # layer's parameters to match the tier we just claimed.
            layer.to("cuda" if start_tier == Tier.T1 else "cpu")

            hook = layer.register_forward_pre_hook(self._make_hook(i))
            self._hooks.append(hook)

        self._attached = True
        print(f"[gb_prefetch] attached: {n} layers, keep_resident="
              f"{self.keep_resident}, lookahead={self.lookahead}", flush=True)

    def detach(self):
        """Remove hooks and demote every layer back to T2 (releases T1
        VRAM). Safe to call even if attach() was a no-op."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        if self._tm is not None and self._names:
            from gb_model_tier import Tier
            for name in self._names:
                e = self._tm._entries.get(name)
                if e is not None and e.tier == Tier.T1:
                    self._tm.demote(name)

        self._attached = False

    def status(self) -> dict:
        return {
            "attached": self._attached,
            "num_layers": len(self._names),
            "keep_resident": self.keep_resident,
            "lookahead": self.lookahead,
            "prefetch_hits": self._state.promote_count,
            "fallback_promotions": self._state.fallback_count,
        }

    # ── hook ─────────────────────────────────────────────────────────────────

    def _make_hook(self, idx: int):
        from gb_model_tier import Tier

        def _hook(module, args):
            import gb_stream_sched as gs

            name = self._names[idx]
            e = self._tm._entries[name]

            if self._state.in_flight.pop(idx, False):
                # Prefetched while an earlier layer computed - just sync the
                # transfer stream before this layer's compute touches it.
                gs.wait_for("transfer", on="gemm")
                e.tier = Tier.T1
                self._state.promote_count += 1
            elif e.tier != Tier.T1:
                # Cold start / misconfigured window - correctness fallback,
                # not the steady-state path (dense-layer order has nothing
                # to mispredict, unlike gb_moe.py's expert routing).
                self._tm.promote(name)
                self._state.fallback_count += 1

            # Slide the resident window: demote the layer that just fell
            # outside keep_resident behind the current one.
            demote_idx = idx - self.keep_resident
            if demote_idx >= 0:
                demote_name = self._names[demote_idx]
                de = self._tm._entries[demote_name]
                if de.tier == Tier.T1:
                    self._tm.demote(demote_name)

            # Kick off the next async prefetch, overlapping with this
            # layer's own forward compute.
            prefetch_idx = idx + self.lookahead
            if prefetch_idx < len(self._names) and prefetch_idx not in self._state.in_flight:
                pf_name = self._names[prefetch_idx]
                pf_e = self._tm._entries[pf_name]
                if pf_e.tier == Tier.T2:
                    with gs.on("transfer"):
                        pf_e.module.to("cuda", non_blocking=True)
                    self._state.in_flight[prefetch_idx] = True

            return args

        return _hook
