"""A retry must be sized against the server's own numbers, not a guess.

Live 2026-08-20, mid-session: a turn 400'd with
`request (24654 tokens) exceeds the available context size (24576 tokens)`,
the CLI auto-compacted and retried, and the retry 400'd at the same size ,
because compaction leaves the live tail verbatim and the one large tool result
sitting in that tail is exactly what did not fit. The operator got the raw 400.

Two things had to change, and these tests pin both:

  1. the estimator learns the real chars-per-token from `usage.prompt_tokens`,
     instead of assuming the prose figure of 4 against a context that is mostly
     code, JSON and paths;
  2. a last-resort path evicts until a hard token budget is met, and reports
     honestly when it cannot , so the caller can explain instead of firing a
     request that is known to fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from greenboost_cli.workflow import intelligence as I


class _Session:
    def __init__(self, messages):
        self.messages = messages


@pytest.fixture(autouse=True)
def _fresh_calibration():
    I.reset_calibration()
    yield
    I.reset_calibration()


# ── calibration ────────────────────────────────────────────────────────────

def test_estimator_starts_at_the_prose_ratio():
    s = _Session([{"role": "user", "content": "x" * 4000}])
    assert I._estimate_tokens(s) == 1000


def test_a_real_server_count_moves_the_ratio_down():
    """The live case: 4 chars/token predicted less than the server charged."""
    before = I._estimate_tokens(_Session([{"role": "user", "content": "x" * 4000}]))
    # Server said the same bytes were 1333 tokens (3 chars/token).
    I.note_real_prompt_tokens(actual=1333, estimated=before)
    after = I._estimate_tokens(_Session([{"role": "user", "content": "x" * 4000}]))
    assert after > before, "an under-estimate must be corrected upward"


def test_an_absurd_sample_is_ignored_rather_than_believed():
    """A sample that cannot describe the same bytes must not poison the ratio."""
    assert I.note_real_prompt_tokens(actual=1, estimated=100_000) is None
    assert I.calibration()["samples"] == 0


def test_calibration_reports_whether_it_has_ever_seen_the_server():
    assert I.calibration()["samples"] == 0
    assert "default" in I.calibration()["source"]
    I.note_real_prompt_tokens(actual=1100, estimated=1000)
    assert I.calibration()["samples"] == 1
    assert "usage.prompt_tokens" in I.calibration()["source"]


# ── the whole-request estimate ─────────────────────────────────────────────

def test_request_estimate_counts_the_overhead_history_cannot_reach():
    msgs = [{"role": "user", "content": "hi"}]
    system = "S" * 8000
    tools = [{"name": "T", "description": "D" * 4000}]
    hist_only = I._estimate_tokens(_Session(msgs))
    whole = I.estimate_request_tokens(system, tools, msgs)
    assert whole > hist_only + 2000, "system prompt and tool schemas must count"


# ── hard trim ──────────────────────────────────────────────────────────────

def _big_session():
    return _Session([
        {"role": "user", "content": "the original question"},
        {"role": "assistant", "content": "working on it"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "Bash", "input": {}}]},
        {"role": "tool", "name": "Bash", "content": "OUT" * 8000},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "name": "Read", "input": {}}]},
        {"role": "tool", "name": "Read", "content": "FILE" * 8000},
        {"role": "user", "content": "and now the follow-up question"},
    ])


def test_no_work_when_it_already_fits():
    s = _big_session()
    out = I.hard_trim_to_fit(s, budget_tokens=10 ** 9)
    assert out == {"needed": 0, "freed": 0, "met": True, "steps": []}


def test_it_reaches_a_budget_compaction_cannot():
    s = _big_session()
    target = I._estimate_tokens(s) // 4
    out = I.hard_trim_to_fit(s, budget_tokens=target)
    assert out["met"] is True
    assert I._estimate_tokens(s) <= target
    assert out["freed"] > 0


def test_the_live_question_survives_the_trim():
    """Trimming away the turn's own question would make the retry pointless."""
    s = _big_session()
    I.hard_trim_to_fit(s, budget_tokens=50)
    assert any(m.get("role") == "user"
               and "follow-up question" in str(m.get("content", ""))
               for m in s.messages)


def test_an_impossible_budget_reports_failure_instead_of_pretending():
    s = _Session([{"role": "user", "content": "a question that cannot be shrunk"}])
    out = I.hard_trim_to_fit(s, budget_tokens=1)
    assert out["met"] is False


def test_no_orphan_tool_messages_are_left_behind():
    """A tool result without its calling assistant message is a malformed
    request on every backend , a context problem must not become a 400 of a
    different kind."""
    s = _big_session()
    I.hard_trim_to_fit(s, budget_tokens=30)
    roles = [m.get("role") for m in s.messages]
    for i, r in enumerate(roles):
        if r == "tool":
            assert i > 0 and roles[i - 1] in ("assistant", "tool"), roles
