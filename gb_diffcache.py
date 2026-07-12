#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_diffcache.py , GreenBoost diffusion activation caching (TeaCache/DeepCache-style).

WHY: Diffusion denoising calls the same DiT/transformer module once per
timestep. Consecutive timesteps are often near-identical inputs (especially
mid-schedule, away from the first/last few steps), so the transformer's
output for this step can sometimes be approximated by reusing the previous
step's output instead of recomputing it , trading a small, bounded quality
cost for skipping the most expensive call in the loop.

This is a *generic* implementation: it wraps the whole denoiser module's
forward call (not individual internal blocks, which would require per-
architecture knowledge of DiT block boundaries) and decides whether to skip
recomputation based on a cheap signal computed from the call's input. The
default signal is the mean-pooled relative L1 distance between this step's
input hidden-states and the previous step's , an architecture-agnostic
proxy. Callers with architecture-specific knowledge (e.g. a FLUX model's
`time_text_embed` modulation vector, which is the signal the published
TeaCache paper uses for higher fidelity) can pass `signal_fn` to override
the proxy with a cheaper/more accurate one.

Usage:
    import gb_diffcache

    with gb_diffcache.diffcache(pipe.transformer, threshold=0.1, max_skip=2):
        for i, t in enumerate(timesteps):
            noise_pred = pipe.transformer(latents, t, ...)

    print(gb_diffcache.status(pipe.transformer))
    # {"calls": 28, "skipped": 9, "skip_rate": 0.32, "cache_key": "..."}

A skipped call returns the previous call's *output* unchanged , this only
makes sense for modules whose API returns the predicted noise/velocity
tensor directly (the common Diffusers UNet/Transformer2DModel contract).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Optional

import torch

_state_lock = threading.Lock()
# module id -> _CacheState. Plain dict (not WeakKeyDictionary) because the
# patched forward closure itself holds a strong ref to the module for the
# duration of the `with` block; state is dropped explicitly on unpatch.
_states: dict[int, "_CacheState"] = {}


class _CacheState:
    __slots__ = ("decider", "signal_fn", "lora_sig", "orig_forward")

    def __init__(self, threshold: float, max_skip: int,
                 signal_fn: Callable, lora_sig: str, orig_forward):
        self.decider = SkipDecider(threshold, max_skip)
        self.signal_fn = signal_fn
        self.lora_sig = lora_sig
        self.orig_forward = orig_forward


class SkipDecider:
    """The core threshold/skip-budget algorithm, decoupled from *how* the
    wrapped call is intercepted. `patch()` below uses this internally to
    monkeypatch a module's `forward`; callers with a different interception
    point (e.g. ComfyUI's `model_function_wrapper`, which is not a module
    forward call at all , see `comfy/samplers.py`'s
    `model_options['model_function_wrapper'](apply_model, args_dict)` call
    site) can drive the same algorithm directly without monkeypatching."""

    __slots__ = ("threshold", "max_skip", "prev_signal", "prev_output",
                 "skip_run", "calls", "skipped")

    def __init__(self, threshold: float = 0.1, max_skip: int = 2):
        self.threshold = threshold
        self.max_skip = max_skip
        self.prev_signal: Optional[torch.Tensor] = None
        self.prev_output = None
        self.skip_run = 0
        self.calls = 0
        self.skipped = 0

    def maybe_skip(self, signal: Optional[torch.Tensor]) -> bool:
        """Return True if the caller should skip recompute and use
        `self.cached_output` instead. Does not mutate skip/call counters
        unless it actually decides to skip."""
        if (signal is None or self.prev_signal is None
                or self.prev_output is None or self.skip_run >= self.max_skip):
            return False
        denom = self.prev_signal.clamp_min(1e-8)
        rel_dist = (signal - self.prev_signal).abs() / denom
        if float(rel_dist) < self.threshold:
            self.skip_run += 1
            self.calls += 1
            self.skipped += 1
            return True
        return False

    @property
    def cached_output(self):
        return self.prev_output

    def record(self, signal: Optional[torch.Tensor], output) -> None:
        """Call after an actual (non-skipped) compute to update the cache."""
        self.prev_signal = signal
        self.prev_output = output
        self.skip_run = 0
        self.calls += 1

    def reset(self) -> None:
        self.prev_output = None
        self.prev_signal = None
        self.skip_run = 0

    def status(self) -> dict:
        return {
            "calls": self.calls,
            "skipped": self.skipped,
            "skip_rate": (self.skipped / self.calls) if self.calls else 0.0,
            "threshold": self.threshold,
            "max_skip": self.max_skip,
        }


def _default_signal_fn(args, kwargs) -> Optional[torch.Tensor]:
    """Mean-pooled |hidden_states| proxy , the first positional/keyword
    tensor argument, detached and reduced to a single scalar per call so
    the comparison below is O(1) regardless of sequence length."""
    candidate = None
    for a in args:
        if torch.is_tensor(a):
            candidate = a
            break
    if candidate is None:
        for v in kwargs.values():
            if torch.is_tensor(v):
                candidate = v
                break
    if candidate is None:
        return None
    return candidate.detach().float().abs().mean()


def _lora_signature(module: torch.nn.Module) -> str:
    """Cheap fingerprint of currently-active LoRA adapters, if any, so a
    LoRA swap invalidates the cache instead of reusing a stale activation.
    Works with PEFT-style `active_adapters()` / `peft_config`; no-op (empty
    signature) for plain modules without LoRA support."""
    try:
        active = module.active_adapters()  # type: ignore[attr-defined]
        return ",".join(sorted(active))
    except Exception:
        pass
    cfg = getattr(module, "peft_config", None)
    if cfg:
        try:
            return ",".join(sorted(cfg.keys()))
        except Exception:
            return str(id(cfg))
    return ""


def patch(module: torch.nn.Module, threshold: float = 0.1, max_skip: int = 2,
           signal_fn: Optional[Callable] = None) -> None:
    """Wrap `module.forward` with the skip-decision logic. Idempotent no-op
    if already patched (re-patching first calls unpatch())."""
    if id(module) in _states:
        unpatch(module)

    sig_fn = signal_fn or _default_signal_fn
    orig_forward = module.forward
    lora_sig = _lora_signature(module)
    state = _CacheState(threshold, max_skip, sig_fn, lora_sig, orig_forward)

    def _patched_forward(*args, **kwargs):
        with _state_lock:
            st = _states.get(id(module))
        if st is None:
            return orig_forward(*args, **kwargs)

        cur_sig = st.signal_fn(args, kwargs)
        cur_lora = _lora_signature(module)
        lora_changed = cur_lora != st.lora_sig
        st.lora_sig = cur_lora

        if not lora_changed and st.decider.maybe_skip(cur_sig):
            return st.decider.cached_output

        out = orig_forward(*args, **kwargs)
        st.decider.record(cur_sig, out)
        return out

    module.forward = _patched_forward
    with _state_lock:
        _states[id(module)] = state


def unpatch(module: torch.nn.Module) -> None:
    """Restore the module's original forward and drop cached state."""
    with _state_lock:
        st = _states.pop(id(module), None)
    if st is not None:
        module.forward = st.orig_forward


def reset(module: torch.nn.Module) -> None:
    """Clear cached output/signal without removing the patch , call at the
    start of a new image/video generation so the first step of a new run
    never reuses the previous run's last step."""
    with _state_lock:
        st = _states.get(id(module))
    if st is not None:
        st.decider.reset()


def status(module: torch.nn.Module) -> dict:
    with _state_lock:
        st = _states.get(id(module))
    if st is None:
        return {"patched": False}
    return {"patched": True, "lora_signature": st.lora_sig, **st.decider.status()}


@contextmanager
def diffcache(module: torch.nn.Module, threshold: float = 0.1,
              max_skip: int = 2, signal_fn: Optional[Callable] = None):
    """Context manager: patch `module` for the duration of the block, then
    unpatch on exit (success or exception)."""
    patch(module, threshold=threshold, max_skip=max_skip, signal_fn=signal_fn)
    try:
        yield module
    finally:
        unpatch(module)
