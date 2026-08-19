"""Exactly one status row per turn reaches the scrollback.

Reported 2026-08-19: four "Processing" rows stacked inside a single turn,
every one showing 0.6s and a different token count. Cause: the status line
restarts after each tool result, and the next StreamFragment stopped it with
sl.stop(), which commits a final static row plus a newline. One row per tool
call, each timing only the gap between restart and first token.
"""
import re
import inspect

from greenboost_cli.terminal import repl as R


def _repl_source():
    return inspect.getsource(R)


def test_only_the_end_of_turn_stop_commits():
    """Every _stop_sl() inside the event loop must erase, not commit."""
    src = _repl_source()
    # all call sites, with their argument text
    calls = re.findall(r"(?<!def )_stop_sl\(([^)]*)\)", src)
    calls = [c.strip() for c in calls]
    committing = [c for c in calls
                  if "commit=False" not in c and "cancel=True" not in c]
    # exactly one bare _stop_sl() survives: the one after the event loop
    assert committing == [""], (
        f"expected exactly one committing _stop_sl(), found {committing}")


def test_stop_sl_accepts_commit_and_routes_it_to_cancel():
    sig = inspect.signature(R.__dict__.get("_stop_sl", lambda: None))
    src = _repl_source()
    assert "def _stop_sl(cancel: bool = False, commit: bool = True)" in src
    assert "if cancel or not commit:" in src
    assert "sl.cancel()" in src


def test_cancel_erases_without_printing_a_final_row():
    """cancel() is what mid-turn stops rely on; it must not print a row."""
    from greenboost_cli.terminal import statusline as S
    src = inspect.getsource(S.StatusLine.cancel)
    assert "_ERASE_LINE" in src
    assert "final=True" not in src      # no static summary row
    assert '"\\n"' not in src            # and no newline that would commit it


def test_stop_still_commits_a_final_row():
    from greenboost_cli.terminal import statusline as S
    src = inspect.getsource(S.StatusLine.stop)
    assert "final=True" in src
