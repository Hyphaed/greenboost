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
    def __init__(self, t1_free_mb, link_mbps_ewma=0.0, t1_total_mb=0):
        self.t1_free_mb = t1_free_mb
        self.online = True
        self.link_mbps_ewma = link_mbps_ewma
        self.t1_total_mb = t1_total_mb


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
#
# Every share below is free VRAM minus a %-derived per-device compute/graph
# workspace reserve (gb_synapse._compute_reserve_gb == gb_topology's shared
# max(0.75 GiB, 8% of that device's own VRAM) formula — see its docstring:
# a flat "raw free VRAM" share used to hand a small feeder more than it could
# actually hold). host_free=11000 -> reserve 880 MB -> free 10120;
# feeder t1_free=7000 (no t1_total_mb, falls back to its own free as "total")
# -> reserve 768 MB -> free 6232. Expectations are computed via the shared
# helper below instead of hardcoded so a future reserve-formula tweak doesn't
# silently desync the tests from the code.

def _reserved_free_mb(mb: int) -> int:
    return int(mb - gs._compute_reserve_gb(mb) * 1024.0)


def test_split_v1_default_is_free_vram(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V2", raising=False)
    monkeypatch.delenv("GB_SYNAPSE_HOST_BIAS", raising=False)
    split = gs._compute_tensor_split(11000, [_FakeFeeder(7000)], kv_total_gb=4.0)
    host = _reserved_free_mb(11000)
    feeder = _reserved_free_mb(7000)
    assert split == f"{host},{feeder}"       # KV ignored in v1; reserve still applied
    assert split == "10120,6232"             # pinned so a reserve-formula change is visible


def test_split_v2_subtracts_kv(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_SPLIT_V2", "1")
    monkeypatch.delenv("GB_SYNAPSE_HOST_BIAS", raising=False)
    # Reserve applies first (10120/6232 free, as above), THEN 2 GiB KV spreads
    # proportionally over that reserved-free total (16352 MB).
    split = gs._compute_tensor_split(11000, [_FakeFeeder(7000)], kv_total_gb=2.0)
    parts = [int(x) for x in split.split(",")]
    assert parts[0] == pytest.approx(8853, abs=5)
    assert parts[1] == pytest.approx(5451, abs=5)


def test_split_host_bias(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_SPLIT_V2", "1")
    monkeypatch.setenv("GB_SYNAPSE_HOST_BIAS", "1.5")
    split = gs._compute_tensor_split(10000, [_FakeFeeder(10000)], kv_total_gb=0.0)
    parts = [int(x) for x in split.split(",")]
    reserved = _reserved_free_mb(10000)      # 9200 — same reserve on both sides
    assert parts[0] == pytest.approx(reserved * 1.5, abs=5)   # host boosted 1.5x
    assert parts[1] == pytest.approx(reserved, abs=5)         # feeder untouched by bias


def test_split_bias_identity_is_v1(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V2", raising=False)
    monkeypatch.setenv("GB_SYNAPSE_HOST_BIAS", "1.0")
    split = gs._compute_tensor_split(11000, [_FakeFeeder(7000)], kv_total_gb=3.0)
    assert split == f"{_reserved_free_mb(11000)},{_reserved_free_mb(7000)}"


def test_split_reserve_scales_with_each_devices_own_total(monkeypatch):
    """The reserve is per-device, derived from THAT device's own total VRAM
    (t1_total_mb when known) — not a shared/host-derived figure. Two feeders
    with identical free VRAM but different total capacity must get different
    absolute reserves, and thus different final shares. This is the exact
    failure mode _compute_reserve_gb's docstring cites: a small feeder handed
    more share than it can actually hold when reserve wasn't per-device."""
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V2", raising=False)
    monkeypatch.delenv("GB_SYNAPSE_HOST_BIAS", raising=False)
    small = _FakeFeeder(4000, t1_total_mb=4000)     # small card: reserve floors at 0.75 GiB
    large = _FakeFeeder(4000, t1_total_mb=40000)    # same free, much bigger card: reserve scales up
    split = gs._compute_tensor_split(30000, [small, large], kv_total_gb=0.0)
    parts = [int(x) for x in split.split(",")]
    assert parts[1] == 4000 - int(gs._compute_reserve_gb(4000) * 1024.0)
    assert parts[2] == 4000 - int(gs._compute_reserve_gb(40000) * 1024.0)
    assert parts[1] > parts[2]               # identical free VRAM, but the bigger card lost more to its reserve


# ── v3 (GB_SYNAPSE_SPLIT_V3): link-quality-scaled feeder shares ─────────────

class _FakeTopo:
    def __init__(self, net_link_mbps):
        self.net_link_mbps = net_link_mbps


def test_split_v3_off_by_default(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V3", raising=False)
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V2", raising=False)
    monkeypatch.delenv("GB_SYNAPSE_HOST_BIAS", raising=False)
    split = gs._compute_tensor_split(10000, [_FakeFeeder(10000, link_mbps_ewma=1.0)])
    parts = [int(x) for x in split.split(",")]
    assert parts[0] == parts[1]   # equal free VRAM, no v3 penalty applied -> stay equal


def test_split_v3_penalizes_slow_feeder_link(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V2", raising=False)
    monkeypatch.delenv("GB_SYNAPSE_HOST_BIAS", raising=False)
    monkeypatch.setattr("gb_topology.get_topology", lambda: _FakeTopo(net_link_mbps=100.0))
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V3", raising=False)
    off_parts = [int(x) for x in gs._compute_tensor_split(
        10000, [_FakeFeeder(10000, link_mbps_ewma=10.0)]).split(",")]
    monkeypatch.setenv("GB_SYNAPSE_SPLIT_V3", "1")
    on_parts = [int(x) for x in gs._compute_tensor_split(
        10000, [_FakeFeeder(10000, link_mbps_ewma=10.0)]).split(",")]
    assert on_parts[1] < off_parts[1]                            # slow link (10/100) shrinks the feeder's share
    assert on_parts[1] == pytest.approx(off_parts[1] * 0.1, rel=0.05)  # scaled by 10/100 = 0.1
    assert on_parts[0] == off_parts[0]                           # host share untouched by v3


def test_split_v3_no_penalty_when_link_unmeasured(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V2", raising=False)
    monkeypatch.delenv("GB_SYNAPSE_HOST_BIAS", raising=False)
    monkeypatch.setattr("gb_topology.get_topology", lambda: _FakeTopo(net_link_mbps=100.0))
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V3", raising=False)
    off_split = gs._compute_tensor_split(10000, [_FakeFeeder(10000, link_mbps_ewma=0.0)])
    monkeypatch.setenv("GB_SYNAPSE_SPLIT_V3", "1")
    on_split = gs._compute_tensor_split(10000, [_FakeFeeder(10000, link_mbps_ewma=0.0)])
    assert on_split == off_split   # link_mbps_ewma==0 (never measured) -> no penalty, byte-identical


def test_split_v3_no_penalty_when_host_link_undetectable(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V2", raising=False)
    monkeypatch.delenv("GB_SYNAPSE_HOST_BIAS", raising=False)
    monkeypatch.setattr("gb_topology.get_topology", lambda: _FakeTopo(net_link_mbps=0))
    monkeypatch.delenv("GB_SYNAPSE_SPLIT_V3", raising=False)
    off_split = gs._compute_tensor_split(10000, [_FakeFeeder(10000, link_mbps_ewma=5.0)])
    monkeypatch.setenv("GB_SYNAPSE_SPLIT_V3", "1")
    on_split = gs._compute_tensor_split(10000, [_FakeFeeder(10000, link_mbps_ewma=5.0)])
    assert on_split == off_split   # no reference bandwidth -> never guess a penalty, byte-identical
