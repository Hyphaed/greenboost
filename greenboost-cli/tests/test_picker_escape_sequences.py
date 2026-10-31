import os
import pty
import sys
import threading
import tty

import greenboost_cli.terminal.wizard_prompt as W


def _feed(monkeypatch, data: bytes) -> str:
    """Run _getch against a real pty that already holds `data`."""
    master, slave = pty.openpty()
    tty.setraw(slave)
    stdin = os.fdopen(os.dup(slave), "rb", buffering=0)
    monkeypatch.setattr(sys, "stdin", stdin)
    threading.Timer(0.2, os.write, (master, data)).start()     # _getch's setraw flushes earlier input
    try:
        return W._getch(timeout=3.0)
    finally:
        stdin.close(); os.close(master); os.close(slave)


def test_cursor_position_reply_is_not_escape(monkeypatch):
    assert _feed(monkeypatch, b"\x1b[24;1R") == W.IGNORED


def test_arrows_still_work_including_application_mode(monkeypatch):
    assert _feed(monkeypatch, b"\x1b[A") == "UP"
    assert _feed(monkeypatch, b"\x1b[B") == "DOWN"
    assert _feed(monkeypatch, b"\x1bOA") == "UP"


def test_a_lone_escape_still_cancels(monkeypatch):
    assert _feed(monkeypatch, b"\x1b") == "ESC"


def test_other_sequences_are_swallowed_whole(monkeypatch):
    assert _feed(monkeypatch, b"\x1b[1;5C") == W.IGNORED        # ctrl+right
    assert _feed(monkeypatch, b"\x1b[15~") == W.IGNORED         # F5
    assert _feed(monkeypatch, b"\x1bx") == W.IGNORED            # alt+x
