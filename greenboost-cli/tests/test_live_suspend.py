"""A multi-row block must never be interrupted by a live painter.

Reproduces the mangled permission card from the 2026-08-18 screenshots: the
tool spinner WAS halted before the approval prompt, but the status line runs on
its own thread and repaints every 80 ms. Because the card prints its rows one
at a time, and the cursor legitimately sits at column 0 BETWEEN two of them, a
per-row `at_line_start()` guard is not sufficient — the frame lands squarely in
the middle of the box.
"""
import io
import threading
import time

import pytest

from greenboost_cli.terminal import width as W


@pytest.fixture(autouse=True)
def _fresh_stdout(monkeypatch):
    """Give width.py its own stdout, out of pytest's capture's reach.

    Patching the real `sys.stdout` does not survive here: pytest's capture
    plugin swaps that object around the test, so the writes land in its buffer
    and the assertions see nothing. Replacing the `sys` NAME that width.py
    resolves through is unambiguous and captures exactly the writes under test.
    """
    import types

    buf = io.StringIO()
    monkeypatch.setattr(W, "sys", types.SimpleNamespace(stdout=buf))
    yield buf


def _painter(stop: threading.Event, tag: str) -> None:
    """Stand-in for the status thread, using the real painter API.

    `live=True` is the whole contract: the decision to drop the frame is made
    inside tty_write while it holds the lock. A painter that instead tested
    live_suspended() first and then wrote would have released the lock in
    between, and a block starting in that gap would still be corrupted — which
    is precisely what this test caught the first implementation doing.
    """
    while not stop.is_set():
        W.tty_write(f"\r{tag}", live=True)
        time.sleep(0.002)


def test_a_multi_row_block_is_drawn_intact(_fresh_stdout):
    stop = threading.Event()
    t = threading.Thread(target=_painter, args=(stop, "STATUS"), daemon=True)
    t.start()
    try:
        time.sleep(0.02)                      # let it paint freely first
        with W.suspend_live():
            _fresh_stdout.truncate(0)
            _fresh_stdout.seek(0)
            for row in ("┌────┐\n", "│ a  │\n", "│ b  │\n", "└────┘\n"):
                W.tty_write(row)
                time.sleep(0.02)              # ample room to be interrupted
            drawn = _fresh_stdout.getvalue()
    finally:
        stop.set()
        t.join(timeout=1)

    assert "STATUS" not in drawn, f"a live painter interrupted the block: {drawn!r}"
    assert drawn == "┌────┐\n│ a  │\n│ b  │\n└────┘\n"


def test_painters_resume_after_the_block(_fresh_stdout):
    assert not W.live_suspended()
    with W.suspend_live():
        assert W.live_suspended()
    assert not W.live_suspended(), "suspension leaked past the block"


def test_suspend_nests():
    with W.suspend_live():
        with W.suspend_live():
            assert W.live_suspended()
        assert W.live_suspended(), "inner exit released an outer suspension"
    assert not W.live_suspended()


def test_suspend_releases_on_exception():
    with pytest.raises(RuntimeError):
        with W.suspend_live():
            raise RuntimeError("card draw blew up")
    assert not W.live_suspended(), "an exception left every painter frozen"


def test_rich_output_updates_row_tracking(_fresh_stdout):
    """rich bypassed tty_write entirely, so the tracking never saw its output."""
    from greenboost_cli.terminal.theme import console

    console.print("a line")
    assert W.at_line_start()
    W.tty_write("half a line")
    assert not W.at_line_start()


# ── Host RAM warning in the CLI vitals ────────────────────────────────────────

def _seg_stub(matched, avail=None):
    ev = [{"host_mem_available_gb": avail}] if avail is not None else []
    return {"segment": "host_oom_imminent", "matched": matched, "evidence": ev}


@pytest.mark.parametrize("matched,expect_shown", [
    (True, True),
    (False, False),
    (None, False),      # cannot tell — never dressed up as a warning
])
def test_host_ram_warning_only_when_it_matters(monkeypatch, matched, expect_shown):
    """The tier readouts describe GreenBoost's own pools.

    The memory that ran out on 2026-08-18 was ordinary system RAM the shim
    never sees, so every tier gauge looked healthy while the OOM killer took
    the terminal. This segment appears only when there is something to say —
    including staying silent when the probe could not tell.
    """
    from greenboost_cli.terminal import repl

    class _FakeSem:
        @staticmethod
        def evaluate_segment(_name):
            return _seg_stub(matched, 3.2)

    monkeypatch.setattr(repl, "gb_module", lambda _n: _FakeSem(), raising=False)
    monkeypatch.setattr("greenboost_cli.gb_paths.gb_module",
                        lambda _n: _FakeSem(), raising=False)
    seg = repl._host_mem_warning_seg()
    assert (seg is not None) is expect_shown
    if expect_shown:
        assert "RAM LOW" in seg[0] and "3.2" in seg[0]


def test_host_ram_warning_survives_a_broken_probe(monkeypatch):
    def _boom(_n):
        raise RuntimeError("semantics import failed")

    monkeypatch.setattr("greenboost_cli.gb_paths.gb_module", _boom, raising=False)
    from greenboost_cli.terminal import repl
    assert repl._host_mem_warning_seg() is None


# ── The chokepoints, not just the one card that exposed the bug ───────────────

def test_slash_dispatch_suspends_the_painters(monkeypatch):
    """/gb-status alone makes 50 console.print calls.

    Wrapping ~18 handlers individually would also leave every FUTURE handler
    exposed by default, which is why the suspension lives at the dispatcher.
    """
    from greenboost_cli.terminal import commands as C

    seen = {}

    def _handler(_args, _session, _settings):
        seen["suspended"] = W.live_suspended()

    monkeypatch.setitem(C.COMMAND_TABLE, "unittest-probe", _handler)
    assert C.dispatch_command("/unittest-probe", None, {}) is True
    assert seen["suspended"] is True
    assert not W.live_suspended(), "suspension leaked past the command"


def test_unknown_command_also_releases_the_suspension(monkeypatch):
    from greenboost_cli.terminal import commands as C

    assert C.dispatch_command("/definitely-not-a-command", None, {}) is True
    assert not W.live_suspended()


def test_a_raising_handler_does_not_freeze_the_painters(monkeypatch):
    """A crashing slash command must not leave the status line dead."""
    from greenboost_cli.terminal import commands as C

    def _boom(_a, _s, _c):
        raise RuntimeError("handler exploded")

    monkeypatch.setitem(C.COMMAND_TABLE, "unittest-boom", _boom)
    with pytest.raises(RuntimeError):
        C.dispatch_command("/unittest-boom", None, {})
    assert not W.live_suspended()


def test_drawn_as_block_suspends_and_preserves_identity():
    calls = {}

    @W.drawn_as_block
    def banner(a, b=2):
        """A docstring worth keeping."""
        calls["suspended"] = W.live_suspended()
        return a + b

    assert banner(1) == 3
    assert calls["suspended"] is True
    assert not W.live_suspended()
    assert banner.__name__ == "banner"
    assert "worth keeping" in banner.__doc__
