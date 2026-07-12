#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse Stage-2b: the KV-estimate geometry fix and tensor-split v2.

CPU-only. No GGUF, no cluster, no CUDA — estimate_kv_gb is pure arithmetic and
_compute_tensor_split takes plain objects with a .t1_free_mb attribute.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_synapse as gs


class _FakeFeeder:
    def __init__(self, t1_free_mb):
        self.t1_free_mb = t1_free_mb
        self.online = True


# ── estimate_kv_gb: real geometry vs bucket fallback ─────────────────────────

def test_kv_real_geometry_matches_formula():
    # 40 layers, 8 KV heads, head_dim 128, q8_0 (1 B/elem), 8192 ctx.
    # 2 * 40 * 8 * 128 * 1 = 81920 B/tok; * 8192 / 2^30 ≈ 0.625 GiB
    kv = gs.estimate_kv_gb(8192, n_bytes=0, quant="Q4_K_M",
                           n_layers=40, n_kv_heads=8, head_dim=128)
    assert abs(kv - (8192 * 81920) / (1024 ** 3)) < 1e-6


def test_kv_real_geometry_is_gb_scale_not_bucket():
    """The bug: bucket heuristic predicted ~0.06 GB where the real GQA KV needs
    GB-scale at long ctx. Real geometry must be >> the bucket value here."""
    ctx = 262144
    real = gs.estimate_kv_gb(ctx, n_bytes=9 * 10**9, quant="Q4_K_M",
                             n_layers=40, n_kv_heads=8, head_dim=128)
    bucket = gs.estimate_kv_gb(ctx, n_bytes=9 * 10**9, quant="Q4_K_M")  # no geometry
    assert real > 5.0                       # GB-scale, would have caught the OOM
    assert real > bucket * 10               # ~100x class error the fix removes


def test_kv_falls_back_to_bucket_without_geometry():
    """Missing geometry (old manifest) => bucket heuristic, unchanged."""
    kv = gs.estimate_kv_gb(4096, n_bytes=9 * 10**9, quant="Q4_K_M")
    assert kv > 0                            # produces a number, doesn't crash


# ── _compute_tensor_split: v1 default identity, v2 KV-aware + host bias ───────

def test_split_v1_default_is_free_vram(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V2", raising=False)
    monkeypatch.delenv("GB_SYNAPSE_HOST_BIAS", raising=False)
    split = gs._compute_tensor_split(11000, [_FakeFeeder(7000)], kv_total_gb=4.0)
    assert split == "11000,7000"             # exact v1 string, KV ignored


def test_split_v2_subtracts_kv(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_SPLIT_V2", "1")
    monkeypatch.delenv("GB_SYNAPSE_HOST_BIAS", raising=False)
    # 2 GiB KV = 2048 MB spread proportionally over 11000+7000 free.
    split = gs._compute_tensor_split(11000, [_FakeFeeder(7000)], kv_total_gb=2.0)
    parts = [int(x) for x in split.split(",")]
    # host loses 2048*11/18 ≈ 1252 -> ~9748 ; feeder loses ~796 -> ~6204
    assert parts[0] == pytest.approx(9748, abs=5)
    assert parts[1] == pytest.approx(6204, abs=5)


def test_split_host_bias(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_SPLIT_V2", "1")
    monkeypatch.setenv("GB_SYNAPSE_HOST_BIAS", "1.5")
    split = gs._compute_tensor_split(10000, [_FakeFeeder(10000)], kv_total_gb=0.0)
    parts = [int(x) for x in split.split(",")]
    assert parts[0] == pytest.approx(15000, abs=5)   # host boosted 1.5x
    assert parts[1] == pytest.approx(10000, abs=5)


def test_split_bias_identity_is_v1(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V2", raising=False)
    monkeypatch.setenv("GB_SYNAPSE_HOST_BIAS", "1.0")
    split = gs._compute_tensor_split(11000, [_FakeFeeder(7000)], kv_total_gb=3.0)
    assert split == "11000,7000"
