#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_evict_policy , prioritized-LRU eviction, per tier.

No GPU, no daemon, no torch. The policy module is deliberately free of both so
it can be tested in isolation from ModelTierManager.

The load-bearing test here is test_degenerates_to_plain_lru: with every entry
at the default priority, the new controller must select exactly the victim the
old `sorted(last_used)[0]` scan selected. That equivalence is what makes this
safe to land before any caller assigns a priority.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gb_evict_policy import (
    PRIORITY_DEFAULT,
    PRIORITY_MAX,
    PRIORITY_MIN,
    LRUEvictionPolicy,
    PerTierEvictionController,
    PrioritizedLRUEvictionPolicy,
    _SimpleEvictable,
    clamp_priority,
)

T1, T2, T3 = "T1_HBM", "T2_DDR", "T3_NVME"


def mk(name, tier=T1, priority=PRIORITY_DEFAULT, last_used=0.0):
    return _SimpleEvictable(name=name, tier=tier, priority=priority, last_used=last_used)


def old_lru_in_tier(entries, tier):
    """The exact pre-change implementation, kept here as the oracle."""
    candidates = [(e.last_used, e.name) for e in entries if e.tier == tier]
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


# ── the equivalence guarantee ─────────────────────────────────────────────────

def test_degenerates_to_plain_lru():
    """Default priority everywhere => identical victim to the old scan."""
    entries = [
        mk("unet", last_used=30.0),
        mk("vae", last_used=10.0),
        mk("text_encoder", last_used=20.0),
        mk("controlnet", tier=T2, last_used=5.0),
    ]
    ctl = PerTierEvictionController()
    for e in entries:
        ctl.schedule_for_eviction(e)

    for tier in (T1, T2, T3):
        victim = ctl.victim(tier)
        assert (victim.name if victim else None) == old_lru_in_tier(entries, tier)


@pytest.mark.parametrize("seed", range(8))
def test_degenerates_to_plain_lru_randomised(seed):
    import random

    rng = random.Random(seed)
    entries = [
        mk(f"m{i}", tier=rng.choice([T1, T2, T3]), last_used=rng.random() * 100)
        for i in range(12)
    ]
    ctl = PerTierEvictionController()
    for e in entries:
        ctl.schedule_for_eviction(e)
    for tier in (T1, T2, T3):
        victim = ctl.victim(tier)
        assert (victim.name if victim else None) == old_lru_in_tier(entries, tier)


# ── what priority buys ────────────────────────────────────────────────────────

def test_lower_priority_evicted_first_despite_being_newer():
    """The case plain LRU cannot express: a freshly-touched but expendable
    buffer should go before an old but precious one."""
    precious = mk("kv_cache", priority=PRIORITY_MAX, last_used=1.0)   # oldest
    expendable = mk("scratch", priority=PRIORITY_MIN, last_used=999.0)  # newest
    ctl = PerTierEvictionController()
    ctl.schedule_for_eviction(precious)
    ctl.schedule_for_eviction(expendable)

    assert ctl.victim(T1).name == "scratch"
    assert old_lru_in_tier([precious, expendable], T1) == "kv_cache"  # old answer


def test_lru_within_a_priority_band():
    a = mk("a", priority=50, last_used=3.0)
    b = mk("b", priority=50, last_used=1.0)
    c = mk("c", priority=50, last_used=2.0)
    pol = PrioritizedLRUEvictionPolicy()
    for e in (a, b, c):
        pol.push(e)
    assert [pol.pop().name for _ in range(3)] == ["b", "c", "a"]


def test_evict_first_jumps_the_queue():
    old = mk("old", last_used=1.0)
    flagged = mk("flagged", last_used=999.0)
    pol = LRUEvictionPolicy()
    pol.push(old)
    pol.push(flagged, evict_first=True)
    assert pol.pop().name == "flagged"


# ── liveness: last_used is read at pop time, not frozen at push time ──────────

def test_reordering_after_push_is_respected():
    """ModelTierManager mutates entry.last_used directly on promote. A policy
    that froze order at push time would go stale; this asserts it does not."""
    a = mk("a", last_used=1.0)
    b = mk("b", last_used=2.0)
    ctl = PerTierEvictionController()
    ctl.schedule_for_eviction(a)
    ctl.schedule_for_eviction(b)
    assert ctl.victim(T1).name == "a"
    a.last_used = 99.0            # 'a' just got used
    assert ctl.victim(T1).name == "b"


# ── bookkeeping ───────────────────────────────────────────────────────────────

def test_retier_moves_tracking():
    e = mk("unet", tier=T1, last_used=1.0)
    ctl = PerTierEvictionController()
    ctl.schedule_for_eviction(e)
    assert ctl.victim(T1).name == "unet"

    ctl.retier(e, T1, T2)
    e.tier = T2
    assert ctl.victim(T1) is None
    assert ctl.victim(T2).name == "unet"
    assert len(ctl) == 1


def test_forget_is_idempotent_and_tolerates_untracked():
    e = mk("x")
    ctl = PerTierEvictionController()
    ctl.forget(e)                      # never pushed: must not raise
    ctl.schedule_for_eviction(e)
    ctl.forget(e)
    ctl.forget(e)                      # already gone: must not raise
    assert len(ctl) == 0


def test_double_push_does_not_duplicate():
    e = mk("x")
    ctl = PerTierEvictionController()
    ctl.schedule_for_eviction(e)
    ctl.schedule_for_eviction(e)
    assert len(ctl) == 1


def test_counts_per_tier():
    ctl = PerTierEvictionController()
    for e in (mk("a", T1), mk("b", T1), mk("c", T2)):
        ctl.schedule_for_eviction(e)
    assert ctl.counts() == {T1: 2, T2: 1}


def test_empty_tier_returns_none_not_raises():
    ctl = PerTierEvictionController()
    assert ctl.victim(T1) is None
    with pytest.raises(IndexError):
        ctl.evict(T1)


def test_priority_bucket_is_dropped_when_emptied():
    pol = PrioritizedLRUEvictionPolicy()
    e = mk("x", priority=7)
    pol.push(e)
    pol.remove(e)
    assert len(pol) == 0
    assert pol.peek() is None


# ── clamping ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("given,expected", [
    (None, PRIORITY_DEFAULT),
    (-50, PRIORITY_MIN),
    (10_000, PRIORITY_MAX),
    (0, PRIORITY_MIN),
    (100, PRIORITY_MAX),
    (35, 35),
])
def test_clamp_priority(given, expected):
    assert clamp_priority(given) == expected
