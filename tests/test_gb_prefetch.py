#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_prefetch , layer-stack detection, and a real-CUDA end-to-end
forward pass through a synthetic decoder stack exercising the actual
T2<->T1 transfers on gb_stream_sched's "transfer" stream.

Mirrors gb_moe.py's validation approach: a structural unit test first
(no CUDA), then a real forward pass with real tensors on the real GPU
(skipped if no CUDA available).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn


# ── _find_layer_stack ───────────────────────────────────────────────────────

class _DecoderLayer(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _ExpertMLP(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)


class _MoEBlock(nn.Module):
    """Has its own `experts` ModuleList - must be ignored by the dense-layer
    detector so gb_prefetch and gb_moe never fight over the same modules."""
    def __init__(self, n=4, dim=8):
        super().__init__()
        self.gate = nn.Linear(dim, n, bias=False)
        self.experts = nn.ModuleList([_ExpertMLP(dim) for _ in range(n)])

    def forward(self, x):
        return x


class _DenseModel(nn.Module):
    def __init__(self, n_layers=6, dim=8):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderLayer(dim) for _ in range(n_layers)])
        self.moe = _MoEBlock(dim=dim)  # decoy - smaller ModuleList, should lose

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_find_layer_stack_picks_largest_parameterized_modulelist():
    from gb_prefetch import _find_layer_stack
    model = _DenseModel(n_layers=6)
    stack = _find_layer_stack(model)
    assert stack is not None
    assert len(stack) == 6
    assert stack is model.layers


def test_find_layer_stack_skips_experts_modulelist():
    from gb_prefetch import _find_layer_stack

    class _OnlyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = _MoEBlock(n=8)

    stack = _find_layer_stack(_OnlyMoE())
    assert stack is None  # the only ModuleList present is named "experts"


def test_find_layer_stack_returns_none_for_tiny_model():
    from gb_prefetch import _find_layer_stack

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_DecoderLayer(4)])  # len 1, too small

    assert _find_layer_stack(_Tiny()) is None


# ── Real CUDA end-to-end ─────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_layer_prefetcher_real_forward_pass():
    from gb_prefetch import LayerPrefetcher
    from gb_model_tier import ModelTierManager, Tier

    torch.manual_seed(0)
    model = _DenseModel(n_layers=6, dim=16)

    tm = ModelTierManager(t3_dir="/tmp/gb_prefetch_test_t3")
    pf = LayerPrefetcher(model, tm=tm, keep_resident=2, lookahead=1)
    pf.attach()

    try:
        x = torch.randn(2, 16, device="cuda")
        out = model(x)

        assert out.device.type == "cuda"
        assert out.shape == (2, 16)
        assert torch.isfinite(out).all()

        st = pf.status()
        assert st["attached"] is True
        assert st["num_layers"] == 6
        # Layers 0/1 start resident (keep_resident=2) and need no promotion.
        # Layers 2-5 should have arrived via async prefetch (steady state),
        # not the synchronous fallback - that's the entire point of this
        # module, so assert it actually happened rather than just "ran".
        assert st["prefetch_hits"] >= 3
        assert st["fallback_promotions"] == 0

        # Sliding window: layer 0 should have been demoted back off T1 by
        # the time the whole stack finished (keep_resident=2 means once
        # layer 3 starts, layer 0 falls more than 2 behind and is demoted).
        e0 = tm._entries[pf._names[0]]
        assert e0.tier == Tier.T2
    finally:
        pf.detach()
        tm.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_layer_prefetcher_detach_demotes_all():
    from gb_prefetch import LayerPrefetcher
    from gb_model_tier import ModelTierManager, Tier

    model = _DenseModel(n_layers=4, dim=8)
    tm = ModelTierManager(t3_dir="/tmp/gb_prefetch_test_t3")
    pf = LayerPrefetcher(model, tm=tm, keep_resident=2, lookahead=1)
    pf.attach()

    x = torch.randn(1, 8, device="cuda")
    model(x)
    pf.detach()

    for name in pf._names:
        assert tm._entries[name].tier == Tier.T2

    tm.close()
