"""The pause/resume lever must be reachable AND visible.

Owner requirement, 2026-08-18: "at the bottom of greenboost-cli must show the
keyboard shortcut to stop/resume a session". A lever nobody can see is a lever
nobody uses, and the moment you want the GPU back is usually the moment you are
about to launch something else — so it needs a key, not just a slash command.
"""
from __future__ import annotations

import inspect

from greenboost_cli.terminal import repl, statusline
from greenboost_cli.slash_commands import greenboost_cmds


def test_ctrl_p_is_bound_in_the_interactive_app():
    src = inspect.getsource(repl.run_interactive)
    assert '_pt_kb.add("c-p")' in src, "ctrl+p is not bound"
    assert "cmd_resume" in src and "cmd_pause" in src, (
        "ctrl+p must toggle between pause and resume, not one direction only")


def test_idle_toolbar_advertises_the_shortcut():
    src = inspect.getsource(repl.run_interactive)
    assert "ctrl+p=pause/resume" in src, "idle footer does not show the shortcut"


def test_running_turn_hint_advertises_the_shortcut():
    src = inspect.getsource(statusline.StatusLine._show_hint)
    assert "ctrl+p" in src, "the mid-turn footer does not show the shortcut"
    assert "esc to interrupt" in src, "the existing interrupt hint must remain"


def test_hint_line_still_fits_a_narrow_terminal():
    """The hint grew; it must not become the next thing that wraps."""
    from greenboost_cli.terminal.width import display_width
    hint = ("  esc to interrupt  ·  ctrl+p pause/resume  ·  "
            "type to queue next prompt")
    assert display_width(hint) <= 78, (
        f"hint is {display_width(hint)} columns — too wide for an 80-column term")


def test_slash_commands_exist_for_both_directions():
    table: dict = {}
    greenboost_cmds.register(table)
    for name in ("pause", "resume", "paused"):
        assert name in table, f"/{name} is not registered"
        assert callable(table[name])
