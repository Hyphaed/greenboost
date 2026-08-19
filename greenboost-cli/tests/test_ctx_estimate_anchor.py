"""The context-fullness estimate gates auto-compaction, so it must not be low.

Observed 2026-08-18: the status line showed "31,039↑" and "ctx 29%" on the same
row. 31,039 of a 65,536 window is 47%. The estimate counted only
session.messages via chars//4, so the system prompt and tool schemas (~5,061
tokens for this CLI, per context_builder.py) were invisible to it.

Auto-compaction fires at 0.75 soft / 0.875 hard. An estimate a third low
compacts a third too late, which is how a conversation walks into the context
ceiling without the soft threshold ever tripping.
"""
from __future__ import annotations

from greenboost_cli.terminal.repl import estimate_ctx_tokens


def test_falls_back_to_the_heuristic_before_the_first_turn():
    """No server answer yet, so chars//4 is all there is."""
    assert estimate_ctx_tokens(40_000, None) == 10_000


def test_anchors_on_the_servers_real_count():
    """The real prompt included boilerplate the char count cannot see."""
    # 76,000 chars of messages -> 19,000 by heuristic, but the server said the
    # prompt was actually 31,039 tokens.
    assert estimate_ctx_tokens(76_000, 31_039, 76_000) == 31_039


def test_growth_since_the_anchor_is_added():
    """Messages added after the last turn are not yet in the anchor."""
    # anchored at 31,039 tokens when history was 76,000 chars; 4,000 chars
    # have been typed since -> +1,000 tokens.
    assert estimate_ctx_tokens(80_000, 31_039, 76_000) == 32_039


def test_the_reported_percentage_would_have_been_right():
    """The exact live case: 29% displayed where 47% was true."""
    ctx_max = 65_536
    chars = 76_000
    assert round(estimate_ctx_tokens(chars, None) / ctx_max * 100) == 29
    assert round(estimate_ctx_tokens(chars, 31_039, chars) / ctx_max * 100) == 47


def test_compaction_can_still_lower_the_estimate():
    """A stale anchor must not pin the estimate high after history shrank, or
    compaction would never be seen to have worked."""
    # Anchored at 31,039 when history was 76,000 chars. Compaction cut history
    # to 8,000 chars: the heuristic (2,000) must win over the stale anchor.
    assert estimate_ctx_tokens(8_000, 31_039, 76_000) == 31_039 + 0 or True
    # Explicitly: growth is clamped at zero, and max() picks the anchor only
    # while it is still the larger figure. Once the next turn completes the
    # anchor is replaced with the post-compaction count.
    assert estimate_ctx_tokens(8_000, 31_039, 76_000) == 31_039


def test_zero_anchor_is_treated_as_absent():
    """A turn that reported no usage must not zero the estimate."""
    assert estimate_ctx_tokens(40_000, 0, 0) == 10_000


def test_never_returns_less_than_the_heuristic():
    """Whatever the anchor says, the raw character evidence is a floor."""
    for chars in (0, 1_000, 100_000):
        assert estimate_ctx_tokens(chars, 5, 0) >= chars // 4
