"""Live feedback must survive the first streamed token.

Reported 2026-08-18 as "greenboost-cli stopped": no spinner, no t/s, nothing
moving. The engine's own /slots endpoint said `processing=True` the whole time
and the GPU was at 100%. The turn was fine; the status line had been torn down
by the first StreamFragment, so a response decoding at ~3 tok/s against a 34k
prompt showed no progress at all for minutes.

The teardown is correct in raw mode, where the status line and the printed text
share one line via a carriage return. It is unnecessary in prompt_toolkit
toolbar mode, where the status is a separate pt-owned row.
"""
from __future__ import annotations

import greenboost_cli.terminal.statusline as sl_mod


def test_renders_in_toolbar_is_false_without_pt(monkeypatch):
    """No pt session registered: raw mode, teardown is required."""
    monkeypatch.setattr(sl_mod, "_pt_invalidate", None)
    monkeypatch.setattr(sl_mod, "_pt_is_live", None)
    assert sl_mod.renders_in_toolbar() is False


def test_renders_in_toolbar_is_true_when_the_pt_app_is_live(monkeypatch):
    monkeypatch.setattr(sl_mod, "_pt_invalidate", lambda: None)
    monkeypatch.setattr(sl_mod, "_pt_is_live", lambda: True)
    assert sl_mod.renders_in_toolbar() is True


def test_toolbar_mode_is_false_while_the_pt_app_is_not_running(monkeypatch):
    """During a turn on some paths the pt app is not running even though a
    session exists; raw mode applies and the teardown must still happen."""
    monkeypatch.setattr(sl_mod, "_pt_invalidate", lambda: None)
    monkeypatch.setattr(sl_mod, "_pt_is_live", lambda: False)
    assert sl_mod.renders_in_toolbar() is False


def test_a_broken_live_probe_falls_back_to_raw(monkeypatch):
    """Never let a probe error leave the status line running on a line it
    would corrupt — raw mode is the safe answer."""
    def _boom():
        raise RuntimeError("pt gone")
    monkeypatch.setattr(sl_mod, "_pt_invalidate", lambda: None)
    monkeypatch.setattr(sl_mod, "_pt_is_live", _boom)
    assert sl_mod.renders_in_toolbar() is False


def test_repl_consults_the_mode_rather_than_always_stopping():
    """Pin the call site: a future refactor that drops the check would silently
    restore the silent-response behaviour."""
    from pathlib import Path
    src = Path(sl_mod.__file__).with_name("repl.py").read_text()
    frag = src.split("if isinstance(event, StreamFragment):", 1)[1][:1400]
    assert "renders_in_toolbar()" in frag, \
        "StreamFragment must decide teardown by render mode, not unconditionally"
    # Raw mode must still tear the status line down here , but WITHOUT
    # committing a static summary row. This runs once per tool result, so
    # committing stacked one "Processing , 0.6s" row per tool call in the
    # scrollback (reported 2026-08-19). See test_statusline_single_commit.py.
    assert "_stop_sl(commit=False)" in frag, \
        "raw mode must still tear the status line down, erasing rather than committing"
