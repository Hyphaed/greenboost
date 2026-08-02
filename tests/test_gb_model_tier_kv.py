#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
End-to-end tests for KV-cache tier-serde compression wired into
ModelTierManager (missing_features.md item (e)): register(kv_bits=...) ->
evict() -> T3 disk -> _load_from_t3() -> reconstructed buffers, plus the
dataflux event fields and the byte-identical-when-unset default.

CPU-only, same fixture pattern as test_gb_model_tier.py: entries registered
directly at Tier.T2 so evict() never calls demote() (which would touch
gb_stream_sched/CUDA).
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn


class _KVHolder(nn.Module):
    """A module whose state_dict is entirely KV-cache-shaped buffers , the
    honest per-register opt-in target this feature is designed for."""
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("k_cache", torch.randn(2, 8, 128, 128, generator=g))
        self.register_buffer("v_cache", torch.randn(2, 8, 128, 128, generator=g))


@pytest.fixture
def tm(tmp_path):
    from gb_model_tier import ModelTierManager

    with patch("gb_model_tier._gb_session"):
        mgr = ModelTierManager(hbm_headroom_mb=512, t3_dir=str(tmp_path / "model_pages"))
    return mgr


# ── default (kv_bits=None) stays byte-identical ─────────────────────────────

def test_register_default_kv_bits_none_evict_unchanged(tm):
    from gb_model_tier import Tier
    mod = _KVHolder()
    orig_k = mod.k_cache.clone()
    tm.register("kv", mod, tier=Tier.T2)   # no kv_bits -> default path

    with patch("gb_dataflux.emit") as mock_emit:
        tm.evict("kv")

    ev = mock_emit.call_args[0][0]
    assert "kv_codec" not in ev
    assert "kv_bits" not in ev
    entry = tm._entries["kv"]
    assert entry.tier == Tier.T3
    # Loading it back must NOT go through gb_tier_kv's manifest path.
    import gb_model_tier
    raw = gb_model_tier._t3_load_raw(entry.t3_path)
    assert "__gb_kv__" not in raw
    assert torch.equal(raw["k_cache"], orig_k)


# ── kv_bits opt-in: end-to-end evict -> disk -> restore ─────────────────────

def test_kv_bits_evict_restore_roundtrip(tm):
    from gb_model_tier import Tier
    mod = _KVHolder()
    orig_k = mod.k_cache.clone()
    orig_v = mod.v_cache.clone()
    tm.register("kv", mod, tier=Tier.T2, kv_bits=4)

    with patch("gb_dataflux.emit") as mock_emit:
        tm.evict("kv")

    entry = tm._entries["kv"]
    assert entry.tier == Tier.T3
    ev = mock_emit.call_args[0][0]
    assert ev["kv_codec"] == "polarquant"
    assert ev["kv_bits"] == 4
    assert ev["kv_tensors"] == 2   # k_cache + v_cache
    assert ev["kv_ratio"] > 1.0    # smaller than raw

    tm._load_from_t3(entry)
    assert entry.tier == Tier.T2
    assert mod.k_cache.shape == orig_k.shape
    assert mod.v_cache.shape == orig_v.shape
    cos_k = torch.nn.functional.cosine_similarity(
        mod.k_cache.reshape(-1, 128), orig_k.reshape(-1, 128), dim=-1)
    assert cos_k.mean() > 0.90


def test_kv_bits_disk_footprint_smaller_than_uncompressed(tm, monkeypatch):
    """Disk footprint with kv_bits=4 must beat the same module WITHOUT KV
    compression, both with zstd forced off so the comparison isolates the
    KV quantization's own contribution."""
    from gb_model_tier import Tier
    import gb_model_tier

    monkeypatch.setenv("GB_T3_COMPRESS", "0")
    gb_model_tier._T3_ZSTD_BACKEND = None

    plain = _KVHolder(seed=1)
    tm.register("plain", plain, tier=Tier.T2)
    tm.evict("plain")
    plain_bytes = __import__("os").path.getsize(tm._entries["plain"].t3_path)

    tm2_dir = tm.t3_dir
    compressed = _KVHolder(seed=1)
    tm.register("compressed", compressed, tier=Tier.T2, kv_bits=4)
    tm.evict("compressed")
    compressed_bytes = __import__("os").path.getsize(tm._entries["compressed"].t3_path)

    assert compressed_bytes < plain_bytes * 0.6
    gb_model_tier._T3_ZSTD_BACKEND = None


# ── kv_keys restricts which tensors get encoded ─────────────────────────────

def test_kv_keys_restricts_encoding(tm):
    from gb_model_tier import Tier
    mod = _KVHolder()
    orig_v = mod.v_cache.clone()
    tm.register("kv", mod, tier=Tier.T2, kv_bits=4, kv_keys=("k_cache",))

    with patch("gb_dataflux.emit") as mock_emit:
        tm.evict("kv")

    ev = mock_emit.call_args[0][0]
    assert ev["kv_tensors"] == 1   # only k_cache

    entry = tm._entries["kv"]
    import gb_model_tier
    raw = gb_model_tier._t3_load_raw(entry.t3_path)
    assert "k_cache" in raw["__gb_kv__"]["entries"]
    assert "v_cache" not in raw["__gb_kv__"]["entries"]
    # v_cache never went through gb_tier_kv at all -> bit-exact on restore.
    tm._load_from_t3(entry)
    assert torch.equal(mod.v_cache, orig_v)


# ── kill switch stays wired at the ModelTierManager level too ──────────────

def test_kv_compress_kill_switch_via_evict(tm, monkeypatch):
    from gb_model_tier import Tier
    monkeypatch.setenv("GB_TIER_KV_COMPRESS", "0")
    mod = _KVHolder()
    tm.register("kv", mod, tier=Tier.T2, kv_bits=4)

    with patch("gb_dataflux.emit") as mock_emit:
        tm.evict("kv")

    ev = mock_emit.call_args[0][0]
    assert ev.get("kv_tensors", 0) == 0
    entry = tm._entries["kv"]
    import gb_model_tier
    raw = gb_model_tier._t3_load_raw(entry.t3_path)
    assert "__gb_kv__" not in raw


# ── legacy checkpoints (written before this feature) still load ────────────

def test_legacy_checkpoint_without_manifest_loads_unchanged(tm):
    from gb_model_tier import Tier
    mod = _KVHolder()
    tm.register("kv", mod, tier=Tier.T2)   # no kv_bits -> plain legacy-shaped save
    tm.evict("kv")
    entry = tm._entries["kv"]
    tm._load_from_t3(entry)
    assert entry.tier == Tier.T2
