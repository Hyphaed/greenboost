"""The non-stop chain, its controls, and its audit trail are actually wired."""
import inspect

import pytest

from greenboost_cli.core.autonomy import get_state, reset_state
from greenboost_cli.terminal import repl as R


@pytest.fixture(autouse=True)
def _fresh():
    reset_state()
    yield
    reset_state()


def test_turn_chain_repeats_while_the_model_is_mid_task(monkeypatch):
    replies = ["I'll make those script fixes now.",   # the real 2026-08-19 stall
               "I'll run the tests next.",
               "All done , everything passes."]
    seen = []

    def _fake(user_input, session, settings):
        seen.append(user_input)
        return replies[len(seen) - 1]

    monkeypatch.setattr(R, "process_query", _fake)
    monkeypatch.setattr(R, "_pending_todo_count", lambda: 0)
    R._run_turn_chain("build the pinball game", object(), {})

    assert len(seen) == 3, "should have continued twice, then stopped"
    assert seen[0] == "build the pinball game"
    from greenboost_cli.core.autonomy import CONTINUE_PROMPT
    assert seen[1] == seen[2] == CONTINUE_PROMPT
    assert get_state().counts()["continue"] == 2
    assert "finished" in [e.detail["reason"] for e in get_state().journal
                          if e.kind == "stop"][0]


def test_an_interrupted_turn_never_auto_resumes(monkeypatch):
    monkeypatch.setattr(R, "process_query", lambda *a, **k: None)
    monkeypatch.setattr(R, "_pending_todo_count", lambda: 5)
    R._run_turn_chain("go", object(), {})
    reasons = [e.detail["reason"] for e in get_state().journal if e.kind == "stop"]
    assert reasons == ["interrupted by the user"]


def test_nonstop_off_runs_exactly_one_turn(monkeypatch):
    get_state().nonstop = False
    calls = []
    monkeypatch.setattr(R, "process_query",
                        lambda u, s, st: calls.append(u) or "I'll continue.")
    monkeypatch.setattr(R, "_pending_todo_count", lambda: 3)
    R._run_turn_chain("go", object(), {})
    assert calls == ["go"]


def test_open_todos_keep_it_going_even_when_prose_sounds_final(monkeypatch):
    todos = {"n": 2}
    calls = []

    def _fake(user_input, session, settings):
        calls.append(user_input)
        if len(calls) == 2:
            todos["n"] = 0
        return "All done."

    monkeypatch.setattr(R, "process_query", _fake)
    monkeypatch.setattr(R, "_pending_todo_count", lambda: todos["n"])
    R._run_turn_chain("go", object(), {})
    assert len(calls) == 2       # continued once because a todo was open


def test_pending_todo_count_reads_the_instrument_store(monkeypatch):
    from greenboost_cli.instruments import handlers as H
    monkeypatch.setattr(H, "_session_todos", [
        {"status": "pending"}, {"status": "in_progress"},
        {"status": "completed"},
    ], raising=False)
    assert R._pending_todo_count() == 2


def test_both_toggles_are_bound_to_keys_and_shown_in_the_hint():
    src = inspect.getsource(R)
    assert '@_pt_kb.add("c-n")' in src
    assert '@_pt_kb.add("c-y")' in src
    assert "ctrl+n=nonstop" in src and "ctrl+y=autoanswer" in src


def test_the_hint_reports_live_state_not_a_fixed_string():
    src = inspect.getsource(R)
    assert "'ON' if _st.nonstop else 'off'" in src


def test_slash_commands_registered():
    from greenboost_cli.terminal.commands import COMMAND_TABLE
    for n in ("session-report", "nonstop", "auto-answer"):
        assert n in COMMAND_TABLE


def test_auto_answer_path_records_every_choice():
    """Whatever it decides overnight must be readable in the morning."""
    src = inspect.getsource(R)
    assert "if _st.auto_answer:" in src
    assert '_st.record(' in src and 'question=' in src and 'why=' in src
