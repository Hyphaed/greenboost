"""The status line's t/s must be a DECODE rate, not a wall-clock average.

Real measurement, 2026-08-18, from dataflux: a turn with 27,438 prompt tokens
ran 73.5 s wall, of which ttft_ms recorded 68.3 s. It produced 81 output
tokens, so decode itself took ~5 s and ran at ~16 tok/s. The status line showed
"1t/s", because it divided output tokens by the whole turn including prefill.

That number is what made decode look like the bottleneck for months while the
actual cost was prefill growing with conversation depth (5.4 s at 13.8k reused
tokens, 34.2 s at 18.5k, 68.3 s at 27.5k, all at 96-99% prompt-cache hit).
"""
from __future__ import annotations

import time

from greenboost_cli.terminal.statusline import StatusLine


def _render_of(sl: StatusLine) -> str:
    return sl._render()


def test_tps_measures_the_decode_span_not_the_whole_turn(monkeypatch):
    sl = StatusLine()
    sl.start("Thinking")

    # Simulate a long prefill: 10 s pass with no output tokens.
    sl._start = time.monotonic() - 10.0

    # First output token arrives now, then 20 tokens over a short decode span.
    sl.update(in_tokens=27438, out_tokens=1)
    assert sl._first_out_ts is not None, "first output token must be timestamped"
    sl._first_out_ts = time.monotonic() - 1.0     # 1 s of decode
    sl.update(out_tokens=20)

    out = _render_of(sl)
    # 20 tokens / 1 s of decode = 20 t/s. The old formula would have given
    # 20 / ~11 s ≈ 2 t/s.
    assert "20t/s" in out or "19t/s" in out or "21t/s" in out, out
    assert "2t/s" not in out.replace("20t/s", "").replace("21t/s", ""), out


def test_prefill_is_surfaced_rather_than_hidden_in_elapsed():
    sl = StatusLine()
    sl.start("Thinking")
    sl._start = time.monotonic() - 30.0
    sl.update(out_tokens=1)
    sl._first_out_ts = sl._start + 25.0    # 25 s prefill

    out = _render_of(sl)
    assert "ttft" in out, "time-to-first-token must be visible on a long prefill"
    assert "ttft 25s" in out, out


def test_no_tps_and_no_ttft_before_the_first_token():
    """While still prefilling there is no decode rate to report, and reporting
    a fabricated one is what the old code effectively did."""
    sl = StatusLine()
    sl.start("Thinking")
    sl._start = time.monotonic() - 5.0
    sl.update(in_tokens=27438)             # prompt counted, nothing generated yet

    out = _render_of(sl)
    assert "t/s" not in out, out
    assert "ttft" not in out, out


def test_short_prefill_does_not_clutter_the_line():
    sl = StatusLine()
    sl.start("Thinking")
    sl.update(out_tokens=5)                # first token essentially immediately
    out = _render_of(sl)
    assert "ttft" not in out, "sub-second prefill should not be shown"


def test_first_token_timestamp_resets_between_turns():
    sl = StatusLine()
    sl.start("Thinking")
    sl.update(out_tokens=3)
    assert sl._first_out_ts is not None
    sl.stop()
    sl.start("Thinking")
    assert sl._first_out_ts is None, "a new turn must re-measure its own prefill"
