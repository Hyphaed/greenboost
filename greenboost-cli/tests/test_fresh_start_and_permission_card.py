"""Fresh start by default, slash-command argv stays interactive, permission text verbatim."""
import json
import os
import pty
import select
import signal
import sys
import time
from pathlib import Path

import pytest

from greenboost_cli.terminal import repl as R
from greenboost_cli.terminal.wizard_prompt import (
    _summarize_call, neutralise_text, verbatim_lines,
)
from greenboost_cli.terminal.width import display_width

ROOT = Path(__file__).resolve().parents[1]

NASTY = [
    'for i in $(seq 1 40); do echo "  $i = $((i))" ; done',
    "echo '${HOME}' `date` \"$(printf '%s' \"a (b) c\")\"",
    "echo [bold red]not markup[/] [/]",
    "printf '\\x1b[2J\\x1b[31mred\\x1b[0m'",
    "echo \x1b[2J\x1b]0;title\x07 done",
    "echo héllo 日本語 ✓ 🙂 — done",
    "line one\nline two\n\tindented",
    "x" * 900 + " tail",
]


def test_owner_command_is_not_eaten():
    cmd = NASTY[0]
    assert _summarize_call("Run: " + cmd) == "Run: " + cmd
    lines, trunc = verbatim_lines("Run: " + cmd, 90, 4)
    assert not trunc and "".join(lines) == "Run: " + cmd


def test_real_call_form_still_summarised():
    assert _summarize_call("Foo(['/tmp/some/file'])") == "Foo · /tmp/some/file"


@pytest.mark.parametrize("cmd", NASTY)
def test_lines_fit_and_are_safe(cmd):
    for width in (20, 60, 100):
        lines, trunc = verbatim_lines("Run: " + cmd, width, 4)
        assert 1 <= len(lines) <= 4
        for ln in lines:
            assert display_width(ln) <= width
            assert all(ord(c) >= 0x20 and ord(c) != 0x7F for c in ln)
        full, t2 = verbatim_lines("Run: " + cmd, width, None)
        assert not t2
        assert "".join(full) == neutralise_text("Run: " + cmd)


def test_ansi_and_newlines_neutralised():
    out = neutralise_text("a\x1b[31mb\nc\td\x07")
    assert "\x1b" not in out and "\n" not in out and "\x07" not in out
    assert "␛" in out and "⏎" in out


def test_long_command_is_truncated_with_ellipsis():
    lines, trunc = verbatim_lines("Run: " + "y" * 900, 50, 4)
    assert trunc and len(lines) == 4 and lines[-1].endswith("…")


def test_previous_session_hint(tmp_path):
    p = tmp_path / "_last.json"
    p.write_text(json.dumps({"messages": [1] * 105}))
    old = time.time() - 3 * 3600
    os.utime(p, (old, old))
    assert R._previous_session_hint(p, time.time()) == \
        "Previous session available: /resume (105 messages, 3h ago)"
    p.write_text(json.dumps({"messages": []}))
    assert R._previous_session_hint(p, time.time()) == ""


def test_main_slash_argv_becomes_command_not_chat(monkeypatch):
    import greenboost_cli.__main__ as M
    from greenboost_cli.slash_commands import backend_cmds
    seen = {}
    monkeypatch.setattr(sys, "argv", ["greenboost-cli", "/llamaserve", "load"])
    monkeypatch.setattr("greenboost_cli.terminal.repl.run_interactive",
                        lambda settings, **kw: seen.update(kw))
    monkeypatch.setattr(backend_cmds, "cmd_llamaserve",
                        lambda *a, **k: pytest.fail("argv command must not auto-start a model"))
    M.main()
    assert seen["initial_command"] == "/llamaserve load"
    assert seen["initial_prompt"] is None and seen["resume"] is None


def test_main_resume_flags(monkeypatch):
    import greenboost_cli.__main__ as M
    for flag, want in (("-c", "last"), ("--resume", "pick")):
        seen = {}
        monkeypatch.setattr(sys, "argv", ["greenboost-cli", flag])
        monkeypatch.setattr("greenboost_cli.terminal.repl.run_interactive",
                            lambda settings, **kw: seen.update(kw))
        M.main()
        assert seen["resume"] == want


def test_main_clears_persisted_accept_all(monkeypatch):
    import greenboost_cli.__main__ as M
    from greenboost_cli.environment import settings as S
    seen = {}
    real = S.load_settings
    monkeypatch.setattr(S, "load_settings",
                        lambda: dict(real(), permission_mode="accept-all"))
    monkeypatch.setattr(sys, "argv", ["greenboost-cli"])
    monkeypatch.setattr("greenboost_cli.terminal.repl.run_interactive",
                        lambda settings, **kw: seen.update(s=settings))
    M.main()
    assert "permission_mode" not in seen["s"]


def test_cli_stays_alive_after_argv_slash_command():
    """Real process in a pty: `greenboost-cli /gb-status` runs it, then keeps the prompt."""
    env = dict(os.environ, PYTHONPATH=str(ROOT), TERM="xterm-256color",
               COLUMNS="110", LINES="30")
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe(sys.executable,
                   [sys.executable, "-m", "greenboost_cli", "/gb-status"], env)
    buf, t0, seen = b"", time.monotonic(), False
    try:
        while time.monotonic() - t0 < 120:
            r, _, _ = select.select([fd], [], [], 0.2)
            if r:
                try:
                    d = os.read(fd, 65536)
                except OSError:
                    break
                if not d:
                    break
                buf += d
                for _ in range(d.count(b"\x1b[6n")):
                    os.write(fd, b"\x1b[1;1R")
            if not seen and b"GreenBoost Status" in buf:
                seen, t_seen = True, time.monotonic()
            if seen and time.monotonic() - t_seen > 6:
                break
        done, _ = os.waitpid(pid, os.WNOHANG)
        assert seen, "slash command did not run"
        assert done == 0, "CLI exited by itself after the argv command"
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except Exception:
            pass
