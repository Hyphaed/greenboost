"""/exit must not leave a late cursor-position reply ("^[[1;1R") on the prompt line.

A terminal that answers ESC[6n after our timeout used to find echo switched back on. The cursor query now leaves echo off
when it timed out, and restore_terminal() drains late replies before it turns echo on again."""
import os
import pty
import subprocess
import sys
import termios

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str) -> str:
    master, slave = pty.openpty()
    try:
        subprocess.run([sys.executable, "-c", code], stdin=slave, stdout=slave, stderr=subprocess.DEVNULL,
                       cwd=HERE, env={**os.environ, "PYTHONPATH": HERE}, timeout=20)
        os.set_blocking(master, False)
        try:
            return os.read(master, 65536).decode(errors="replace")
        except OSError:
            return ""
    finally:
        os.close(master)
        os.close(slave)


def test_cursor_query_timeout_keeps_echo_off():
    out = _run(
        "import sys, termios\n"
        "from greenboost_cli.terminal.dock import query_cursor_row\n"
        "fd = sys.stdin.fileno(); a = termios.tcgetattr(fd); a[3] |= termios.ECHO | termios.ICANON; termios.tcsetattr(fd, termios.TCSANOW, a)\n"
        "r = query_cursor_row(timeout=0.2)\n"
        "print(r, bool(termios.tcgetattr(fd)[3] & termios.ECHO))\n")
    assert out.strip().endswith("None False"), out


def test_restore_terminal_swallows_late_reply_and_turns_echo_on():
    out = _run(
        "import sys, termios\n"
        "from greenboost_cli.terminal.repl import restore_terminal\n"
        "fd = sys.stdin.fileno(); a = termios.tcgetattr(fd); a[3] &= ~termios.ECHO; termios.tcsetattr(fd, termios.TCSANOW, a)\n"
        "restore_terminal(clear=False)\n"
        "print(bool(termios.tcgetattr(fd)[3] & termios.ECHO))\n")
    assert out.strip().endswith("True"), out
