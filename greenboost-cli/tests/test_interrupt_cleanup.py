"""Ctrl-C during a turn must not crash, and cleanup must survive a second one.

Owner report, 2026-08-18. Two Ctrl-C presses during a long turn produced:

    File ".../greenboost_cli/terminal/repl.py", line 1863, in run_interactive
        still_prefilling = getattr(sl, "_first_out_ts", None) is None
    NameError: name 'sl' is not defined

and then, from the atexit handler that runs `greenboost clear memory-pool`:

    File ".../subprocess.py", line 2153, in _communicate
        ready = selector.select(timeout)
    KeyboardInterrupt

So the first Ctrl-C killed the CLI outright and the second aborted the memory
pool release half-way through it.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import time

from greenboost_cli.terminal import statusline as sl_mod
from greenboost_cli.terminal.statusline import StatusLine, is_prefilling


# ── the NameError ────────────────────────────────────────────────────────────

def test_is_prefilling_is_none_with_no_live_status_line() -> None:
    """No live turn: the answer is "unknown", not a guess in either direction."""
    sl_mod._live_sl = None
    assert is_prefilling() is None


def test_is_prefilling_tracks_the_raw_render_path(monkeypatch) -> None:
    """The raw \\r path is the one a real turn uses, and it used to register
    nothing — which is why repl.py had no live instance to consult."""
    monkeypatch.setattr(sl_mod, "_is_tty", lambda: True)
    monkeypatch.setattr(sl_mod, "_use_pt", lambda: False)
    sl_mod._live_sl = None

    sl = StatusLine()
    sl.start("Processing")
    try:
        assert sl_mod._live_sl is sl, "raw path did not register the live statusline"
        assert is_prefilling() is True, "no output token yet == still prefilling"
        sl._first_out_ts = time.monotonic()
        assert is_prefilling() is False, "first token landed == decoding"
    finally:
        sl.cancel()


def test_live_reference_is_dropped_on_every_exit_route(monkeypatch) -> None:
    """A stale instance would make is_prefilling() answer about a finished turn."""
    monkeypatch.setattr(sl_mod, "_is_tty", lambda: True)
    monkeypatch.setattr(sl_mod, "_use_pt", lambda: False)
    for finish in ("cancel", "stop"):
        sl_mod._live_sl = None
        sl = StatusLine()
        sl.start("Processing")
        getattr(sl, finish)()
        assert sl_mod._live_sl is None, f"{finish}() left a stale live reference"


def test_repl_interrupt_handler_does_not_reference_a_local_sl() -> None:
    """Pins the actual defect: the handler must not read a name from another
    function's scope. Checked structurally so it cannot silently come back."""
    import inspect
    from greenboost_cli.terminal import repl
    src = inspect.getsource(repl.run_interactive)
    assert 'getattr(sl, "_first_out_ts"' not in src, (
        "run_interactive is reaching for `sl` again — it is not in this scope"
    )
    assert "is_prefilling()" in src


# ── the interrupted cleanup ──────────────────────────────────────────────────

def test_pool_release_subprocess_ignores_sigint() -> None:
    """A second Ctrl-C must not abort `greenboost clear memory-pool`.

    Runs a real child that outlives a SIGINT delivered to this process: if the
    handler were not swapped to SIG_IGN, the KeyboardInterrupt would propagate
    out of subprocess.run()'s wait and the call would return before the child
    finished.
    """
    prev = signal.getsignal(signal.SIGINT)
    try:
        prev_ign = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            t0 = time.monotonic()
            p = subprocess.Popen([sys.executable, "-c",
                                  "import time; time.sleep(0.4); print('done')"],
                                 stdout=subprocess.PIPE, text=True)
            signal.raise_signal(signal.SIGINT)   # the second Ctrl-C
            out, _ = p.communicate(timeout=10)
            assert "done" in out, "child did not run to completion"
            assert time.monotonic() - t0 >= 0.4, "wait was cut short"
        finally:
            signal.signal(signal.SIGINT, prev_ign)
    finally:
        signal.signal(signal.SIGINT, prev)


def test_release_memory_pool_guards_its_subprocess() -> None:
    """Structural: the cleanup path must actually install the SIGINT guard."""
    import inspect
    from greenboost_cli.terminal import repl
    src = inspect.getsource(repl.release_memory_pool)
    assert "SIG_IGN" in src, "cleanup subprocess is still interruptible by Ctrl-C"
    assert "KeyboardInterrupt" in src
