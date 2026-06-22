#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for ModelTierManager.auto_evict routing and _lru_in_tier ordering.

demote/evict are patched so no CUDA or filesystem access is needed.
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from gb_telemetry import GpuMetrics, GbPoolInfo


# ── Helpers ──────────────────────────────────────────────────────────────────

def _metrics(fb_used_pct=50.0, t2_pressure=0) -> GpuMetrics:
    m = GpuMetrics()
    m.fb_used_pct = fb_used_pct
    m.fb_total_mb = 12000
    m.fb_used_mb  = int(12000 * fb_used_pct / 100)
    m.fb_free_mb  = 12000 - m.fb_used_mb
    if t2_pressure:
        gb = GbPoolInfo()
        gb.t2_pressure = t2_pressure
        m.gb = gb
    return m


def _make_module(name: str) -> torch.nn.Module:
    """Tiny CPU nn.Linear — represents a model component for tier tracking."""
    m = torch.nn.Linear(4, 4, bias=False)
    return m


@pytest.fixture
def tm(tmp_path):
    """ModelTierManager with T3 in a temp dir; ioctl calls patched out."""
    from gb_model_tier import ModelTierManager, Tier, _gb_session

    with patch("gb_model_tier._gb_session"):        # no /dev/greenboost
        mgr = ModelTierManager(
            hbm_headroom_mb=512,
            t3_dir=str(tmp_path / "model_pages"),
        )
    return mgr


# ── _lru_in_tier ──────────────────────────────────────────────────────────────

def test_lru_in_tier_picks_oldest(tm):
    from gb_model_tier import Tier
    a = _make_module("a")
    b = _make_module("b")
    c = _make_module("c")
    tm.register("a", a, tier=Tier.T1)
    tm.register("b", b, tier=Tier.T1)
    tm.register("c", c, tier=Tier.T1)

    # Manually stagger last_used
    tm._entries["a"].last_used = time.time() - 10   # oldest
    tm._entries["b"].last_used = time.time() - 5
    tm._entries["c"].last_used = time.time()          # most recent

    result = tm._lru_in_tier(Tier.T1)
    assert result == "a"


def test_lru_in_tier_returns_none_when_empty(tm):
    from gb_model_tier import Tier
    assert tm._lru_in_tier(Tier.T1) is None


def test_lru_in_tier_ignores_different_tier(tm):
    from gb_model_tier import Tier
    tm.register("x", _make_module("x"), tier=Tier.T2)
    # No T1 entries → lru_in_tier(T1) is None
    assert tm._lru_in_tier(Tier.T1) is None


# ── auto_evict routing ────────────────────────────────────────────────────────

def test_auto_evict_demotes_lru_t1_when_vram_above_90(tm):
    from gb_model_tier import Tier
    tm.register("model_a", _make_module("a"), tier=Tier.T1)
    tm.register("model_b", _make_module("b"), tier=Tier.T1)
    tm._entries["model_a"].last_used = time.time() - 100  # LRU
    tm._entries["model_b"].last_used = time.time()

    with patch.object(tm, "demote") as mock_demote, \
         patch.object(tm, "evict"):
        tm.auto_evict(_metrics(fb_used_pct=91.0))
        mock_demote.assert_called_once_with("model_a")


def test_auto_evict_skips_demote_when_vram_below_90(tm):
    from gb_model_tier import Tier
    tm.register("model_a", _make_module("a"), tier=Tier.T1)

    with patch.object(tm, "demote") as mock_demote, \
         patch.object(tm, "evict"):
        tm.auto_evict(_metrics(fb_used_pct=85.0))
        mock_demote.assert_not_called()


def test_auto_evict_evicts_lru_t2_when_t2_critical(tm):
    from gb_model_tier import Tier
    tm.register("heavy", _make_module("h"), tier=Tier.T2)

    with patch.object(tm, "demote"), \
         patch.object(tm, "evict") as mock_evict:
        tm.auto_evict(_metrics(fb_used_pct=50.0, t2_pressure=2))
        mock_evict.assert_called_once_with("heavy")


def test_auto_evict_does_nothing_when_all_ok(tm):
    from gb_model_tier import Tier
    tm.register("model_a", _make_module("a"), tier=Tier.T1)

    with patch.object(tm, "demote") as mock_demote, \
         patch.object(tm, "evict") as mock_evict:
        tm.auto_evict(_metrics(fb_used_pct=50.0, t2_pressure=0))
        mock_demote.assert_not_called()
        mock_evict.assert_not_called()


def test_auto_evict_both_conditions_simultaneously(tm):
    """High VRAM (>90%) AND T2 critical → both demote and evict."""
    from gb_model_tier import Tier
    tm.register("t1_model", _make_module("t1"), tier=Tier.T1)
    tm.register("t2_model", _make_module("t2"), tier=Tier.T2)

    with patch.object(tm, "demote") as mock_demote, \
         patch.object(tm, "evict") as mock_evict:
        tm.auto_evict(_metrics(fb_used_pct=95.0, t2_pressure=2))
        mock_demote.assert_called_once_with("t1_model")
        mock_evict.assert_called_once_with("t2_model")


# ── evict() → T3 NVMe path ────────────────────────────────────────────────────

def test_evict_saves_to_t3_and_clears_params(tm, tmp_path):
    """evict() torch.saves state_dict to t3_dir, marks tier T3, zeros params."""
    from gb_model_tier import Tier
    mod = torch.nn.Linear(4, 4, bias=False)
    tm.register("big_model", mod, tier=Tier.T2)

    with patch("gb_model_tier._gb_session"):
        tm.evict("big_model")

    e = tm._entries["big_model"]
    assert e.tier == Tier.T3
    assert e.t3_path is not None
    assert Path(e.t3_path).exists(), "T3 checkpoint file must exist on disk"

    # Module should be on meta device (memory freed, shapes preserved)
    assert next(mod.parameters()).device.type == "meta"


def test_evict_noop_if_already_t3(tm):
    """evict() on an already-T3 entry is a no-op."""
    from gb_model_tier import Tier
    mod = torch.nn.Linear(2, 2, bias=False)
    tm.register("cold", mod, tier=Tier.T3)
    tm._entries["cold"].t3_path = "/nonexistent/path.pt"

    with patch.object(tm, "demote") as mock_demote:
        tm.evict("cold")
        mock_demote.assert_not_called()

    assert tm._entries["cold"].tier == Tier.T3


def test_load_from_t3_restores_state_dict(tm, tmp_path):
    """_load_from_t3 loads a torch-saved state_dict back into the module."""
    from gb_model_tier import Tier, _ModelEntry
    mod = torch.nn.Linear(4, 4, bias=False)
    original_weight = mod.weight.data.clone()

    # Manually save, then move to meta (simulates what evict() now does)
    t3_path = str(tmp_path / "model_pages" / "test.pt")
    Path(t3_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(mod.state_dict(), t3_path)
    mod.to("meta")  # simulate evict() — frees memory, preserves shapes

    entry = _ModelEntry(name="test", module=mod, tier=Tier.T3, t3_path=t3_path)

    tm._load_from_t3(entry)

    assert entry.tier == Tier.T2
    assert mod.weight.shape == original_weight.shape


def test_evict_demotes_t1_first_then_saves(tm, tmp_path):
    """evict() on a T1 module calls demote() first, then saves to T3."""
    from gb_model_tier import Tier
    mod = torch.nn.Linear(2, 2, bias=False)
    tm.register("active", mod, tier=Tier.T1)

    # Patch demote to just flip tier without actual CUDA calls
    def _fake_demote(name):
        tm._entries[name].module.to("cpu")
        tm._entries[name].tier = Tier.T2

    with patch.object(tm, "demote", side_effect=_fake_demote):
        tm.evict("active")

    e = tm._entries["active"]
    assert e.tier == Tier.T3
    assert Path(e.t3_path).exists()


# ── register + promote T2 → T1 (no CUDA) ─────────────────────────────────────

def test_promote_noop_if_already_t1(tm):
    """promote() on a T1 entry just updates last_used (no I/O)."""
    from gb_model_tier import Tier
    mod = torch.nn.Linear(2, 2, bias=False)
    tm.register("hot", mod, tier=Tier.T1)
    before = tm._entries["hot"].last_used

    with patch("gb_model_tier._gb_session"):
        tm.promote("hot")

    assert tm._entries["hot"].tier == Tier.T1


# ── session lifecycle ─────────────────────────────────────────────────────────

def test_session_idle_and_active_do_not_raise(tm):
    """session_idle / session_active work without /dev/greenboost."""
    with patch("gb_model_tier._gb_session"):
        tm.session_idle()
        tm.session_active()


# ── demote edge cases ─────────────────────────────────────────────────────────

def test_demote_noop_if_already_t2(tm):
    """demote() on a T2 entry is a no-op (line 172)."""
    from gb_model_tier import Tier
    mod = torch.nn.Linear(2, 2, bias=False)
    tm.register("cold", mod, tier=Tier.T2)
    before = tm._entries["cold"].last_used

    with patch.object(mod, "to") as mock_to:
        tm.demote("cold")
        mock_to.assert_not_called()

    assert tm._entries["cold"].tier == Tier.T2


def test_demote_non_async_path(tm):
    """demote() with async_transfers=False uses blocking module.to("cpu") (line 180)."""
    from gb_model_tier import Tier
    mod = torch.nn.Linear(2, 2, bias=False)
    tm.async_transfers = False
    tm.register("model", mod, tier=Tier.T1)

    with patch("gb_model_tier._gb_session"):
        tm.demote("model")

    assert tm._entries["model"].tier == Tier.T2
    # Module should be on CPU now
    assert all(p.device.type == "cpu" for p in mod.parameters())


def test_load_from_t3_raises_when_file_missing(tm):
    """_load_from_t3 raises FileNotFoundError if t3_path doesn't exist (line 209)."""
    from gb_model_tier import Tier, _ModelEntry
    import pytest
    mod = torch.nn.Linear(2, 2, bias=False)
    entry = _ModelEntry(name="x", module=mod, tier=Tier.T3, t3_path="/nonexistent.pt")

    with pytest.raises(FileNotFoundError):
        tm._load_from_t3(entry)
