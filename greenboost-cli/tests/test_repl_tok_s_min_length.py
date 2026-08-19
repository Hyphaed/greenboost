"""CLI-side mirror of gb_synapse_api's minimum-sample-length guard.

An agentic session produces many very short final turns (a bare tool call, a
one-word answer). Their tok/s is fixed-overhead noise, and feeding it to
gb_synapse.record_measured_tok_s() poisons the measured history that
`recommend()` prefers over its bandwidth heuristic — so junk samples steer
placement decisions, not just dashboards.

See tests/test_tok_s_min_sample_length.py in the parent repo for the proxy-side
half and the live 2026-08-17 numbers that motivated both.
"""
from __future__ import annotations

from greenboost_cli.terminal import repl


class _Spy:
    def __init__(self):
        self.calls = []

    def record_measured_tok_s(self, model, tok_s, source=""):
        self.calls.append((model, tok_s, source))


def _wire(monkeypatch):
    """Patch the lazy imports inside _record_measured_tok_s."""
    spy = _Spy()
    import greenboost_cli.slash_commands.backend_cmds as bc
    monkeypatch.setattr(bc, "_import_gb_synapse", lambda: spy, raising=False)
    monkeypatch.setattr(bc, "_llamaserve_model_name", lambda s: "test-model", raising=False)
    return spy


def test_threshold_matches_the_proxy_side() -> None:
    # Drift between the two would mean the same turn is a valid sample from one
    # vantage point and junk from the other.
    assert repl._MIN_TOK_S_SAMPLE_TOKENS >= 8


def test_short_turn_is_not_recorded(monkeypatch) -> None:
    spy = _wire(monkeypatch)
    repl._record_measured_tok_s(180.0, {}, output_tokens=3)
    assert spy.calls == []


def test_long_turn_is_recorded(monkeypatch) -> None:
    spy = _wire(monkeypatch)
    repl._record_measured_tok_s(5.0, {}, output_tokens=400)
    assert len(spy.calls) == 1
    assert spy.calls[0][1] == 5.0
    assert spy.calls[0][2] == "cli"


def test_unknown_length_is_skipped(monkeypatch) -> None:
    """Older callers pass no length. A sample that cannot be validated is
    dropped rather than trusted: a wrong measurement is worse than a missing
    one here, because recommend() weights measured history above its own
    heuristic."""
    spy = _wire(monkeypatch)
    repl._record_measured_tok_s(5.0, {})
    assert spy.calls == []


def test_never_raises_into_the_turn_loop(monkeypatch) -> None:
    # It is called from the REPL's event loop; telemetry must never break a turn.
    import greenboost_cli.slash_commands.backend_cmds as bc

    def boom():
        raise RuntimeError("gb_synapse unavailable")

    monkeypatch.setattr(bc, "_import_gb_synapse", boom, raising=False)
    repl._record_measured_tok_s(5.0, {}, output_tokens=400)  # must not raise
