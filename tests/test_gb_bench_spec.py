#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""GB-6: the speculative-decode measurement, tested without a server.

What matters here is the ACCOUNTING, not the engine: that a sweep picks the
best depth rather than the deepest (the 2026-08-05 sweep was non-monotonic ,
2:5.15, 3:5.58, 4:6.50, 6:4.40, 8:5.76 tok/s , so "pick the largest" would
have chosen 8 and lost 0.74 tok/s against 4), and that a missing acceptance
rate is explained rather than reported as zero.
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_bench_spec as B


class _FakeSynapse:
    """Stands in for gb_synapse: records serve/stop, reports what is served."""
    def __init__(self, depth=4):
        self.depth = depth
        self.calls = []

    def ps(self):
        return [{"model": "m", "mtp_draft_n": self.depth}]

    def stop(self, model):
        self.calls.append(("stop", model))

    def serve(self, model, mtp_draft_n=None, **kw):
        self.calls.append(("serve", model, mtp_draft_n))
        self.depth = mtp_draft_n


def _install(monkeypatch, fake, per_depth=None):
    monkeypatch.setitem(sys.modules, "gb_synapse", fake)

    def _once(base_url, model, prompt, max_tokens):
        tok_s = (per_depth or {}).get(fake.depth, 5.0)
        return {"decode_tok_s": tok_s,
                "accept_rate": None if fake.depth == 0 else 0.62}
    monkeypatch.setattr(B, "_measure_once", _once)


def test_a_sweep_picks_the_best_depth_not_the_deepest(monkeypatch):
    """The whole reason this tool exists: deeper is not better."""
    fake = _FakeSynapse()
    # The real 2026-08-05 shape.
    _install(monkeypatch, fake, {0: 4.60, 2: 5.15, 4: 6.50, 6: 4.40, 8: 5.76})
    res = B.sweep("http://x", model="m", depths=(0, 2, 4, 6, 8),
                  repeats=1, confirm=True)
    assert res["best"]["draft_n"] == 4
    assert res["best"]["median_tok_s"] == 6.50
    assert max(r["draft_n"] for r in res["depths"]) == 8   # deeper was measured


def test_a_sweep_refuses_to_restart_the_server_without_confirm(monkeypatch):
    """It stops and restarts the live engine. That must not happen because a
    benchmark was run by accident."""
    fake = _FakeSynapse()
    _install(monkeypatch, fake)
    res = B.sweep("http://x", model="m", depths=(0, 4), confirm=False)
    assert "confirm" in res["error"]
    assert res["would_measure"] == [0, 4]
    assert fake.calls == []                    # nothing was touched


def test_a_sweep_serves_every_depth_it_reports(monkeypatch):
    fake = _FakeSynapse()
    _install(monkeypatch, fake, {0: 4.6, 4: 6.5})
    B.sweep("http://x", model="m", depths=(0, 4), repeats=1, confirm=True)
    served = [c[2] for c in fake.calls if c[0] == "serve"]
    assert served == [0, 4]


def test_depth_zero_never_reports_an_acceptance_rate(monkeypatch):
    """Measured 2026-08-20: at depth 0 the ENGINE returns acceptance = 1.0,
    because nothing was drafted so nothing was rejected. Passing that through
    would read as 'drafting is working perfectly' when drafting is off."""
    fake = _FakeSynapse(depth=0)

    def _once(base_url, model, prompt, max_tokens):
        return {"decode_tok_s": 3.8, "accept_rate": 1.0}   # what the engine says
    monkeypatch.setitem(sys.modules, "gb_synapse", fake)
    monkeypatch.setattr(B, "_measure_once", _once)

    s = B.measure_current("http://x", model="m", repeats=1, emit=False)["summary"]
    assert s["median_accept_rate"] is None
    assert "depth 0" in s["accept_rate_source"]
    assert s["median_tok_s"] == 3.8          # the throughput is still real


def test_a_present_acceptance_rate_names_its_source(monkeypatch):
    fake = _FakeSynapse(depth=4)
    _install(monkeypatch, fake, {4: 6.5})
    s = B.measure_current("http://x", model="m", repeats=1, emit=False)["summary"]
    assert s["median_accept_rate"] == 0.62
    assert s["accept_rate_source"] == "engine timings"
    assert s["draft_n"] == 4


def test_measuring_without_a_server_says_so(monkeypatch):
    class _Empty(_FakeSynapse):
        def ps(self):
            return []
    monkeypatch.setitem(sys.modules, "gb_synapse", _Empty())
    assert "no serve session" in B.measure_current("http://x", emit=False)["error"]


def test_the_prompt_set_spans_more_than_one_shape():
    """A draft head is confident on formulaic text and poor on novel text.
    Measuring one shape reports that shape's acceptance and calls it the
    model's."""
    assert len(B.PROMPTS) >= 3
    assert len({p.split()[0].lower() for p in B.PROMPTS}) >= 2


def test_a_sweep_leaves_the_box_at_the_best_depth_not_the_last(monkeypatch):
    """Measured 2026-08-20: a sweep ending at depth 0 left the server at
    3.80 tok/s against the 6.03 it had before the benchmark ran, silently.
    A tool that measures the machine must not degrade it."""
    fake = _FakeSynapse()
    _install(monkeypatch, fake, {0: 3.80, 2: 4.51, 4: 6.03, 6: 4.61})
    res = B.sweep("http://x", model="m", depths=(0, 2, 4, 6),
                  repeats=1, confirm=True)
    assert res["best"]["draft_n"] == 4
    assert res["restored_depth"] == 4
    assert fake.depth == 4                       # actually re-served, not just reported
    assert [c[2] for c in fake.calls if c[0] == "serve"][-1] == 4


def test_no_restore_when_the_best_depth_was_measured_last(monkeypatch):
    """Nothing to undo means no extra re-serve — a restart costs minutes."""
    fake = _FakeSynapse()
    _install(monkeypatch, fake, {2: 4.51, 4: 6.03})
    res = B.sweep("http://x", model="m", depths=(2, 4), repeats=1, confirm=True)
    assert res["best"]["draft_n"] == 4
    assert res["restored_depth"] is None
    assert [c[2] for c in fake.calls if c[0] == "serve"] == [2, 4]   # no third serve
