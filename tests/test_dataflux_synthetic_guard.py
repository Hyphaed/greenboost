"""A benchmark's events must not reach the stream an operator reads.

On 2026-08-22 an A/B harness called the real _compress_context — which is the
point of an A/B, it exercises the real path — and wrote ten compaction events.
The governed metric agent_compaction_prefix_kept_pct then reported "28.7% over
10 of 10 compactions" while a live session was running that had performed ZERO
compactions. The metric was correct about its inputs and wrong about the world,
and only a filter on timestamp and pid caught it.

See docs/performance-casebook.md C-20.
"""
import os

import pytest

import gb_dataflux as d
import gb_semantics as S


def test_events_are_not_tagged_by_default(monkeypatch, tmp_path):
    """The tag is opt-in: normal operation must not mark anything synthetic."""
    monkeypatch.delenv("GB_DATAFLUX_SYNTHETIC", raising=False)
    monkeypatch.setattr(d, "_log_path", lambda: tmp_path / "df.jsonl")
    d.emit({"kind": "agent_context_edit", "op": "compact"})
    assert "synthetic" not in (tmp_path / "df.jsonl").read_text()


def test_synthetic_env_tags_the_event(monkeypatch, tmp_path):
    monkeypatch.setenv("GB_DATAFLUX_SYNTHETIC", "1")
    written = []
    monkeypatch.setattr(d, "_log_path", lambda: tmp_path / "df.jsonl")
    d.emit({"kind": "agent_context_edit", "op": "compact"})
    line = (tmp_path / "df.jsonl").read_text().strip()
    assert '"synthetic": true' in line.replace(" ", " ")


def test_resolvers_ignore_synthetic_events(monkeypatch):
    """The whole point: a governed metric must not see benchmark events."""
    real = {"kind": "turn_bench", "ts": 2, "median_prompt_ms": 1300.0}
    fake = {"kind": "turn_bench", "ts": 3, "median_prompt_ms": 99.0, "synthetic": True}
    monkeypatch.setattr(d, "read_events", lambda **kw: [real, fake])
    r = S.resolve("warm_prefill_fixed_ms")
    assert r["value"] == 1300.0, "the synthetic event won on ts and must have been filtered"


def test_a_stream_of_only_synthetic_events_resolves_to_none(monkeypatch):
    """Not to a reassuring number computed from test data."""
    fake = {"kind": "turn_bench", "ts": 1, "median_prompt_ms": 99.0, "synthetic": True}
    monkeypatch.setattr(d, "read_events", lambda **kw: [fake])
    assert S.resolve("warm_prefill_fixed_ms")["value"] is None
