"""The pinned footer shows the working agent and leaves no stranded status rows.

Runs a scripted "turn" (status line + three tool blocks) in a real pty and
replays the bytes through a terminal emulator, so what is asserted is what a
person would see on screen, not what was written to stdout.
"""
import os
import pty
import select
import sys
import textwrap
import time

import pytest

pyte = pytest.importorskip("pyte")

COLS, ROWS = 100, 24

_DRIVER = textwrap.dedent("""
    import sys, time
    from greenboost_cli.terminal.statusline import StatusLine
    from greenboost_cli.terminal import renderer as R
    from greenboost_cli.terminal.footer import get_footer
    get_footer().agent = "qwen3.8-35b-a3b"
    get_footer().tool_label = lambda: R._tool_label
    sl = StatusLine(); sl.start("Thinking"); time.sleep(0.4)
    for i in range(3):
        sl.cancel()
        R.show_instrument_start("Bash", {"command": f"nvidia-smi -q # {i}"}, False)
        time.sleep(0.4)
        R.show_instrument_result("Bash", f"line a{i}\\nline b{i}", False)
        sl.start("Running"); time.sleep(0.3)
    sl.update(in_tokens=1234, out_tokens=56, tok_s=33.0)
    sys.stdout.write("final answer text\\n"); sys.stdout.flush()
    time.sleep(0.3)
    sl.stop()
""")


def _run_turn(snapshot_at: float):
    """Run the driver in a pty; return (screen_mid_turn, screen_end)."""
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)
    env = dict(os.environ, COLUMNS=str(COLS), LINES=str(ROWS), TERM="xterm-256color")
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe(sys.executable, [sys.executable, "-c", _DRIVER], env)
    import fcntl, struct, termios
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    mid, t0 = None, time.monotonic()
    while True:
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            stream.feed(data)
        # Snapshot when the footer shows a tool running, not at a fixed time:
        # a loaded CI box starts the driver late and a clock-based grab misses it.
        if mid is None and time.monotonic() - t0 > snapshot_at and "Running" in screen.display[-2]:
            mid = list(screen.display)
        if time.monotonic() - t0 > 20:
            break
    os.waitpid(pid, 0)
    return mid or list(screen.display), list(screen.display)


def test_footer_shows_agent_and_leaves_no_status_rows():
    mid, end = _run_turn(snapshot_at=0.5)
    bottom = "\n".join(mid[-2:])
    assert "qwen3.8-35b-a3b" in bottom and "Running" in bottom, mid[-3:]
    body = "\n".join(mid[:-2])
    assert "──" not in body, "an inline status row was stranded in the scrollback"
    assert body.count("Running") == 0
    # the bottom two rows are blank again once the turn is over, and exactly
    # one committed summary row remains
    assert not "".join(end[-2:]).strip(), end[-3:]
    assert sum("Done" in row for row in end) == 1, end


def test_dead_child_is_reaped():
    from greenboost_cli.terminal.repl import _reap_zombie_children
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    time.sleep(0.2)
    assert _reap_zombie_children() >= 1
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_fit_hints_drops_from_the_end_and_never_overflows():
    from greenboost_cli.terminal.footer import fit_hints
    items = ["shift+tab mode", "ctrl+p pause", "ctrl+n nonstop on", "esc clear"]
    assert fit_hints(items, 200) == "  ·  ".join(items)
    out = fit_hints(items, 30)
    assert len(out) <= 30 and out.startswith("shift+tab mode")
    assert fit_hints(items, 5) == ""


def test_activity_fragments_empty_when_idle_and_labelled_when_live():
    from greenboost_cli.terminal import statusline as S
    assert S.activity_fragments() == []
    sl = S.StatusLine()
    sl.update(phase="Planning", out_tokens=2500, tok_s=40.0)
    S._live_sl = sl
    try:
        text = "".join(t for _, t in S.activity_fragments())
        assert "Planning" in text and "2.5k tokens" in text and "40 t/s" in text
        assert "Running" in "".join(t for _, t in S.activity_fragments("Bash echo"))
    finally:
        S._live_sl = None
