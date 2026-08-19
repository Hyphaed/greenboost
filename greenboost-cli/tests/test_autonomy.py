"""Unattended operation: continue decisions, auto-answers, and the audit trail.

Anchored to the 2026-08-19 stall: a 48-minute session ended with the model
writing "I'll make those script fixes now." and emitting no tool call. Nothing
had failed and nothing was finished.
"""
import json

import pytest

from greenboost_cli.core.autonomy import (
    AutonomyState, choose_answer, render_report, export_report,
    export_journal_json, CONTINUE_PROMPT,
    STALL_DELTA_TOKENS, STALL_MIN_CONTINUES,
)


# ── the decision ─────────────────────────────────────────────────────────────

def test_the_real_stall_continues():
    s = AutonomyState()
    go, why = s.decide_continue("I'll make those script fixes now.")
    assert go is True
    assert "stated it was about to continue" in why


@pytest.mark.parametrize("text", [
    "I'll clean up a few Godot APIs that I know are off.",
    "Let me run the tests.",
    "Next, wire the plunger impulse.",
    "I'm going to add icon.svg.",
    "- I'll fix the collision shape",
])
def test_stated_intent_continues(text):
    assert AutonomyState().decide_continue(text)[0] is True


@pytest.mark.parametrize("text", [
    "All done , the project builds and the tests pass.",
    "Everything is working now.",
    "Task complete.",
])
def test_reported_completion_stops(text):
    go, why = AutonomyState().decide_continue(text)
    assert go is False and "finished" in why


@pytest.mark.parametrize("text", [
    "Which option would you like me to take?",
    "Let me know how you want to handle Android signing.",
    "Should I proceed with the destructive migration?",
])
def test_a_question_to_the_user_always_stops(text):
    go, why = AutonomyState().decide_continue(text)
    assert go is False and "asked the user" in why


def test_a_pending_todo_beats_prose_that_sounds_finished():
    """Objective state outranks the model's own narration."""
    go, why = AutonomyState().decide_continue("All done!", pending_todos=3)
    assert go is True and "3 todo" in why


def test_a_direct_question_beats_even_a_pending_todo():
    go, _ = AutonomyState().decide_continue(
        "Which one would you like?", pending_todos=5)
    assert go is False


def test_no_intent_and_no_todos_stops():
    go, why = AutonomyState().decide_continue("The file has 42 lines.")
    assert go is False and "no stated intent" in why


def test_nonstop_is_on_by_default_and_can_be_turned_off():
    s = AutonomyState()
    assert s.nonstop is True                  # owner's requested default
    s.nonstop = False
    go, why = s.decide_continue("I'll keep going.")
    assert go is False and "off" in why


def test_auto_answer_is_off_by_default():
    """Continuing work the user asked for != choosing on their behalf."""
    assert AutonomyState().auto_answer is False


def test_the_continue_ceiling_stops_a_runaway():
    s = AutonomyState(max_auto_continues=3)
    for _ in range(3):
        assert s.decide_continue("I'll continue.")[0] is True
        s.note_continue("intent")
    go, why = s.decide_continue("I'll continue.")
    assert go is False and "ceiling" in why


def test_real_progress_resets_the_ceiling():
    s = AutonomyState(max_auto_continues=2)
    s.note_continue("intent"); s.note_continue("intent")
    assert s.decide_continue("I'll continue.")[0] is False
    s.note_progress()
    assert s.decide_continue("I'll continue.")[0] is True


def test_continue_prompt_does_not_redirect_the_work():
    assert "Continue with the work you just described" in CONTINUE_PROMPT
    assert "ask" in CONTINUE_PROMPT.lower()


# ── auto-answers ─────────────────────────────────────────────────────────────

def test_recommended_option_wins():
    q = {"options": [{"label": "Rewrite everything"},
                     {"label": "Patch in place (Recommended)"}]}
    idx, why = choose_answer(q)
    assert idx == 1 and "Recommended" in why


def test_first_option_is_the_fallback():
    q = {"options": [{"label": "A"}, {"label": "B"}]}
    idx, why = choose_answer(q)
    assert idx == 0 and "first" in why


def test_a_question_with_no_options_is_not_answered():
    idx, why = choose_answer({"options": []})
    assert idx == -1 and "no options" in why


# ── the audit trail ──────────────────────────────────────────────────────────

def test_report_covers_tools_skills_questions_and_why_it_stopped(tmp_path):
    s = AutonomyState()
    s.record("tool", name="Bash"); s.record("tool", name="Bash")
    s.record("tool", name="Write")
    s.record("skill", name="godot-check")
    s.record("question", question="Which export preset?",
             chosen="portrait", why="marked Recommended",
             options=["portrait", "landscape"])
    s.note_continue("intent")
    s.record("stop", reason="the model reported the work finished")

    out = render_report(s, "pinball overnight")
    assert "pinball overnight" in out
    assert "Bash x2" in out and "Write x1" in out
    assert "godot-check" in out
    assert "Which export preset?" in out and "portrait" in out
    assert "marked Recommended" in out
    assert "the model reported the work finished" in out
    assert "1 automatic continuation" in out

    p = export_report(s, tmp_path / "r" / "report.md", "pinball overnight")
    assert p.exists() and "Bash x2" in p.read_text()

    j = export_journal_json(s, tmp_path / "r" / "journal.json")
    data = json.loads(j.read_text())
    assert {e["kind"] for e in data} == {"tool", "skill", "question",
                                         "continue", "stop"}


def test_report_is_honest_when_nothing_ran():
    out = render_report(AutonomyState())
    assert "- none" in out
    assert "still running" in out


# ── stall detection (Claude Code query/tokenBudget.ts) ───────────────────────

def _turn(s, tokens, text="I'll continue."):
    """One turn of the real chain, in the real order: the model generates,
    the token count lands, then the chain decides whether to go again."""
    s.note_output_tokens(tokens)
    go, why = s.decide_continue(text)
    if go:
        s.note_continue(why)
    return go, why


_BUSY = STALL_DELTA_TOKENS * 2
_IDLE = 5


def test_a_productive_chain_is_never_called_a_stall():
    s = AutonomyState()
    for _ in range(12):
        assert _turn(s, _BUSY)[0] is True


def test_a_chain_that_stops_producing_output_is_stopped():
    s = AutonomyState()
    for _ in range(STALL_MIN_CONTINUES):
        assert _turn(s, _BUSY)[0] is True
    assert _turn(s, _IDLE)[0] is True          # one quiet turn is not a stall
    go, why = _turn(s, _IDLE)                  # two in a row is
    assert go is False and "circles" in why


def test_tool_activity_cannot_reset_the_stall_check():
    """note_progress() clears the runaway ceiling by design , a chain that is
    genuinely working should not hit it. The stall check is the guard a loop
    of cheap tool calls must NOT be able to defeat by looking busy."""
    s = AutonomyState()
    for _ in range(STALL_MIN_CONTINUES):
        assert _turn(s, _BUSY)[0] is True
    s.note_progress(); assert _turn(s, _IDLE)[0] is True
    s.note_progress(); go, why = _turn(s, _IDLE)
    assert go is False and "circles" in why


def test_short_early_turns_are_not_a_stall():
    """Reading one file on turn one is short and completely legitimate."""
    s = AutonomyState()
    for _ in range(STALL_MIN_CONTINUES):
        assert _turn(s, _IDLE)[0] is True
