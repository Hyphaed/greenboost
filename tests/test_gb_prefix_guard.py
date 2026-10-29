#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_prefix_guard , prefix-reuse accounting.

These pin the arithmetic of the accounting: given a previous and a proposed
chunk sequence, how much of the reusable prefix survives.

A note on the anchor, corrected 2026-08-22. These tests were written against
casebook C-09, which recorded `agent_compaction_broke_prefix` as a VIOLATION at
agent_compaction_prefix_kept_pct = 0 on 2026-08-21. That reading was an
artifact: the resolver behind it had no `op == "compact"` filter and scored an
`op=calibrate` event as a compaction that kept nothing. No compaction had
happened. C-09 is retracted on that point and the resolver is fixed.

So `test_a_full_rewrite_keeps_nothing` below is arithmetic, not a reproduction
of a real incident, and is named and documented accordingly. The real end-to-end
evidence is in tests/test_semantics_agent_metrics.py (which pins the op filter
against the exact event that caused the mis-read) and in the measured run
recorded as casebook C-14.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gb_prefix_guard import (
    BROKE_PREFIX_PCT,
    PrefixTree,
    assess_compaction,
    common_prefix_len,
)

# A conversation shaped like the real ones: a big stable head (system prompt +
# tool schemas, byte-identical across invocations per gb_prompt_index) then
# per-message chunks.
HEAD, M1, M2, M3 = "sys", "m1", "m2", "m3"
TOKENS = [500, 80, 120, 90]


# ── the arithmetic ────────────────────────────────────────────────────────────

def test_a_rewritten_head_keeps_nothing():
    """Change chunk 0 and every later chunk is unreachable, whatever it holds.

    Arithmetic, not a reproduction: the 0.0% recorded in C-09 came from a
    resolver bug, not from a rewrite like this one. What makes this shape
    expensive is structural , the engine reuses a PREFIX, so the first changed
    byte invalidates everything after it.
    """
    v = assess_compaction([HEAD, M1, M2, M3], ["NEWSYS", M2, M3], TOKENS)
    assert v.prefix_kept_pct == 0.0
    assert v.broke_prefix is True
    assert v.kept_chunks == 0
    assert v.tokens_lost == sum(TOKENS)      # the whole thing is re-prefilled


def test_pure_append_keeps_everything():
    v = assess_compaction([HEAD, M1, M2], [HEAD, M1, M2, M3], TOKENS[:3])
    assert v.prefix_kept_pct == 100.0
    assert v.broke_prefix is False
    assert v.tokens_lost == 0


def test_dropping_the_tail_still_preserves_the_prefix():
    """Trimming old messages off the END is the cheap compaction , the head
    survives, so the cache survives."""
    v = assess_compaction([HEAD, M1, M2, M3], [HEAD, M1], TOKENS)
    assert v.kept_chunks == 2
    assert v.prefix_kept_pct == 50.0
    assert v.tokens_lost == TOKENS[2] + TOKENS[3]


def test_editing_a_middle_message_costs_everything_after_it():
    v = assess_compaction([HEAD, M1, M2, M3], [HEAD, M1, "EDITED", M3], TOKENS)
    assert v.kept_chunks == 2
    assert v.tokens_lost == TOKENS[2] + TOKENS[3]


def test_threshold_is_explicit_not_hidden():
    """A caller must be able to state its own line."""
    seq_prev, seq_new = [HEAD, M1, M2, M3], [HEAD, M1, M2, "Z"]
    assert assess_compaction(seq_prev, seq_new, TOKENS).broke_prefix is False
    strict = assess_compaction(seq_prev, seq_new, TOKENS, threshold_pct=90.0)
    assert strict.broke_prefix is True
    assert strict.threshold_pct == 90.0


def test_no_prior_turn_is_not_a_violation():
    v = assess_compaction([], [HEAD, M1])
    assert v.broke_prefix is False
    assert "nothing to preserve" in v.explain()


def test_explain_leads_with_cost_not_mechanism():
    """The project's consequence rule: what it costs, before how it works."""
    msg = assess_compaction([HEAD, M1, M2, M3], ["NEW"], TOKENS).explain()
    assert "tokens" in msg and "re-prefilled" in msg
    assert msg.index("keeps 0%") < msg.index("tokens")


# ── common_prefix_len ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,n", [
    ([], [], 0),
    ([HEAD], [], 0),
    ([HEAD, M1], [HEAD, M1], 2),
    ([HEAD, M1], [HEAD, "x"], 1),
    (["x"], [HEAD], 0),
    ([HEAD, M1, M2], [HEAD, M1], 2),
])
def test_common_prefix_len(a, b, n):
    assert common_prefix_len(a, b) == n


# ── the radix tree ────────────────────────────────────────────────────────────

@pytest.fixture
def tree():
    t = PrefixTree()
    t.insert("convA", [HEAD, M1, M2, M3], TOKENS)
    t.insert("convB", [HEAD, M1, "x9"], [500, 80, 60])
    return t


def test_longest_match_finds_the_deeper_conversation(tree):
    m = tree.longest_match([HEAD, M1, M2, "zz"], [500, 80, 120, 40])
    assert m.matched_chunks == 3
    assert m.key == "convA"
    assert m.matched_tokens == 700
    assert m.coverage_pct == pytest.approx(94.6, abs=0.1)


def test_shared_head_does_not_collapse_two_conversations(tree):
    """The failure a similarity threshold cannot express: convA and convB share
    a head, and must still be distinguishable past it."""
    a = tree.longest_match([HEAD, M1, M2], [500, 80, 120])
    b = tree.longest_match([HEAD, M1, "x9"], [500, 80, 60])
    assert a.key == "convA" and b.key == "convB"


def test_unknown_prefix_matches_nothing(tree):
    m = tree.longest_match(["totally", "different"], [10, 10])
    assert m.matched_chunks == 0
    assert m.matched_tokens == 0
    assert m.coverage_pct == 0.0


def test_head_only_query_matches_the_shared_head(tree):
    m = tree.longest_match([HEAD], [500])
    assert m.matched_chunks == 1
    assert m.key in ("convA", "convB")


def test_remove_prunes_and_leaves_siblings_intact(tree):
    assert len(tree) == 2
    tree.remove("convA", [HEAD, M1, M2, M3])
    assert len(tree) == 1
    assert tree.longest_match([HEAD, M1, "x9"], [500, 80, 60]).key == "convB"
    # convA's private tail is gone; the shared head is not
    assert tree.longest_match([HEAD, M1, M2, M3], TOKENS).matched_chunks == 2


def test_remove_unknown_is_a_noop(tree):
    tree.remove("nope", [HEAD, "zz"])
    assert len(tree) == 2


def test_reinsert_does_not_duplicate(tree):
    tree.insert("convA", [HEAD, M1, M2, M3], TOKENS)
    assert len(tree) == 2


def test_default_threshold_is_the_documented_one():
    assert BROKE_PREFIX_PCT == 50.0
    assert assess_compaction([HEAD, M1], [HEAD, M1]).threshold_pct == 50.0


# ── accounting only ───────────────────────────────────────────────────────────

def test_module_exposes_no_mutating_api():
    """This reports; it does not edit requests or choose slots. If someone adds
    an actuator, this fails and makes them say so."""
    import gb_prefix_guard as g
    forbidden = {"compact", "rewrite", "apply", "pin", "choose_slot", "set_slot"}
    assert forbidden.isdisjoint(dir(g))
