"""Real-PTY smoke test of the bottom dock: the last rows are the dock in every state.

A tiny driver builds the real DockSession (no model, no network), the test types
into it through a pseudo-terminal and reads the screen back through a terminal
emulator. Skipped when pyte is not installed.
"""
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import textwrap
import time

import pytest

pyte = pytest.importorskip("pyte")

_DRIVER = textwrap.dedent("""
    import sys
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.patch_stdout import patch_stdout
    from greenboost_cli.terminal.dock import DockSession, DockState
    st = DockState(name="smoke", server=("fg:#a3e635", "● model running"), nonstop=True, ctx_pct=12.0,
                   tiers=[("T1 1.0/11.9G", ""), ("T2 0.0/12.2G", "")],
                   hints=["esc interrupt", "ctrl+p pause", "type to queue"])
    s = DockSession(history=InMemoryHistory(), completer=WordCompleter(["/help", "/doctor", "/design"], WORD=True),
                    complete_while_typing=True, key_bindings=None, style=None, get_state=lambda: st)
    print("scrollback line 1"); print("scrollback line 2")
    with patch_stdout(raw=True):
        r = s.prompt()
    print("GOT:" + r)
""")


class _Term:
    def __init__(self, cols, rows):
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.ByteStream(self.screen)
        env = dict(os.environ, TERM="xterm-256color", COLUMNS=str(cols), LINES=str(rows))
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execvpe(sys.executable, [sys.executable, "-c", _DRIVER], env)
        self.size(cols, rows)

    def size(self, cols, rows):
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def pump(self, sec):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            r, _, _ = select.select([self.fd], [], [], 0.05)
            if not r:
                continue
            try:
                d = os.read(self.fd, 65536)
            except OSError:
                return
            if not d:
                return
            parts = d.split(b"\x1b[6n")        # answer each cursor request at its place in the stream
            for i, part in enumerate(parts):
                self.stream.feed(part)
                if i < len(parts) - 1:
                    os.write(self.fd, f"\x1b[{self.screen.cursor.y + 1};{self.screen.cursor.x + 1}R".encode())

    def rows(self):
        return [l.rstrip() for l in self.screen.display]

    def close(self):
        try:
            os.kill(self.pid, signal.SIGKILL)
            os.waitpid(self.pid, 0)
        except Exception:
            pass


def _assert_dock(rows):
    n = len(rows)
    assert "model running" in rows[n - 2] and "ctx 12%" in rows[n - 2]      # status 1
    assert "T1 1.0/11.9G" in rows[n - 1]                                    # status 2 = last row
    assert rows[n - 3].startswith("────")                                   # lower rule
    assert rows[n - 5].startswith("────")                                   # upper rule (session name)
    assert rows[n - 4].startswith("❯")                                      # input directly above lower rule


@pytest.mark.parametrize("cols,rows", [(80, 24), (120, 36), (200, 50)])
def test_dock_is_the_last_rows_idle_typing_menu_resize(cols, rows):
    t = _Term(cols, rows)
    try:
        t.pump(2.0)
        _assert_dock(t.rows())
        t.os_write = lambda b: os.write(t.fd, b)
        t.os_write(b"hello")
        t.pump(0.6)
        r = t.rows()
        _assert_dock(r)
        assert r[-4].startswith("❯ hello")
        t.os_write(b"\x15/d")                      # ctrl+u, then "/d" -> completion menu
        t.pump(0.8)
        r = t.rows()
        assert "/doctor" in "\n".join(r) and "/design" in "\n".join(r)
        assert "model running" in r[-2] and "T1" in r[-1]      # dock still the last rows with the menu open
        t.os_write(b"\x15")                        # clear: the menu closes
        t.pump(0.4)
        # shrink: the dock re-anchors on the new last rows
        k = t.screen.cursor.y - 17
        if k > 0:
            y = t.screen.cursor.y
            t.screen.cursor_to_line(t.screen.lines)
            for _ in range(k):
                t.screen.index()
            t.screen.cursor.y = y - k
        t.screen.resize(18, 90)
        t.size(90, 18)
        os.kill(t.pid, signal.SIGWINCH)
        t.pump(1.2)
        _assert_dock(t.rows())
        t.os_write(b"\x15done\r")
        t.pump(0.8)
        assert any(l.startswith("GOT:done") for l in t.rows())
    finally:
        t.close()
