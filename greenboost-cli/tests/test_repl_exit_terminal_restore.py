"""Exit must leave a clean, usable terminal — /exit, /quit and Ctrl-D alike.

The cooperative half of shutdown (joining the stdin-reader thread so
prompt_toolkit's own cleanup erases the bottom box) is a bounded wait and has
failed twice in the field, leaving the status bar painted under the returned
shell prompt. restore_terminal() is the unconditional half; these tests pin
its contract rather than the race it compensates for.
"""
from __future__ import annotations

import io

from greenboost_cli.terminal.repl import restore_terminal


class _FakeTTY(io.StringIO):
    def __init__(self, tty: bool = True) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_writes_nothing_when_stdout_is_not_a_tty(monkeypatch) -> None:
    """`gb -p` and the AI Factory capture stdout — escapes there are pollution."""
    out = _FakeTTY(tty=False)
    monkeypatch.setattr("sys.stdout", out)
    restore_terminal()
    assert out.getvalue() == ""


def test_resets_the_terminal_state_prompt_toolkit_owns(monkeypatch) -> None:
    out = _FakeTTY()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.delenv("GB_CLI_NO_CLEAR_ON_EXIT", raising=False)
    restore_terminal()
    written = out.getvalue()

    assert "\033[r" in written          # scroll region — what pins a bottom bar
    assert "\033[?25h" in written       # cursor visible
    assert "\033[?2004l" in written     # bracketed paste off
    assert "\033[?1049l" in written     # alternate screen left
    assert "\033[?1000l" in written     # mouse reporting off


def test_clears_the_screen_like_the_clear_command(monkeypatch) -> None:
    out = _FakeTTY()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.delenv("GB_CLI_NO_CLEAR_ON_EXIT", raising=False)
    restore_terminal()
    written = out.getvalue()

    # The exact trio ncurses' `clear` emits: home, erase screen, erase scrollback.
    assert "\033[H\033[2J\033[3J" in written
    # The clear must come last — resetting SGR after it would repaint nothing,
    # but leaving a scroll region set after it would re-pin the bottom rows.
    assert written.index("\033[r") < written.index("\033[H\033[2J\033[3J")


def test_clear_is_opt_out_but_restore_is_not(monkeypatch) -> None:
    out = _FakeTTY()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setenv("GB_CLI_NO_CLEAR_ON_EXIT", "1")
    restore_terminal()
    written = out.getvalue()

    assert "\033[2J" not in written     # transcript kept
    assert "\033[r" in written          # terminal still made usable
    assert "\033[?25h" in written


def test_crash_path_restores_without_erasing_the_traceback(monkeypatch) -> None:
    """atexit registers clear=False so a stack trace survives the exit."""
    out = _FakeTTY()
    monkeypatch.setattr("sys.stdout", out)
    restore_terminal(clear=False)
    written = out.getvalue()

    assert "\033[2J" not in written
    assert "\033[?25h" in written
