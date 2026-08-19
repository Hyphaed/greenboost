#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""GB-4: the KV-quantization quality curve, tested without a server.

The claim this file defends is that quantized KV must be gated on RECALL, not
on a smoke test. Quantized weights fail by collapsing into repetition, which a
smoke gate catches. Quantized KV fails by staying fluent while quietly losing
the ability to retrieve things from far back in the context — a smoke gate
passes that, which is why NIAH is the gate here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_bench_kv as B


class _FakeSynapse:
    def __init__(self, kv_type="f16", kv_gb=2.12):
        self.kv_type, self.kv_gb, self.calls = kv_type, kv_gb, []

    def ps(self):
        return [{"model": "m", "kv_type": self.kv_type,
                 "kv_gb": self.kv_gb, "ctx": 24576}]

    def stop(self, model):
        self.calls.append(("stop", model))

    def serve(self, model, kv_type=None, **kw):
        self.calls.append(("serve", model, kv_type))
        self.kv_type = kv_type
        self.kv_gb = {"f16": 2.12, "q8_0": 1.06, "q4_0": 0.53}.get(kv_type, 2.12)


class _FakeAviary:
    """Recall degrades as KV gets coarser — the failure this gate exists for.

    Returns the REAL shape of `niah_certify`: `score` is a COUNT of needles
    retrieved, not a fraction. An earlier version of this fake invented a
    `recall` float, and the caller was written against the fake rather than the
    function — so the first live run reported "recall=8" for an 8-needle test.
    A fixture that does not match the thing it stands in for tests nothing.
    """
    RECALL = {"f16": 1.0, "q8_0": 0.875, "q4_0": 0.5}

    def __init__(self, fake):
        self._f = fake
        self.calls = []

    def niah_certify(self, model, tokens, needles=8, kv_type="unknown", **kw):
        self.calls.append((model, tokens, needles, kv_type))
        r = self.RECALL.get(self._f.kv_type, 1.0)
        score = int(round(r * needles))
        return {"kind": "niah_cert", "score": score, "needles": needles,
                "status": "ok" if score == needles else "error",
                "prompt_tokens": tokens, "kv_type": kv_type}


def _install(monkeypatch, fake):
    av = _FakeAviary(fake)
    monkeypatch.setitem(sys.modules, "gb_synapse", fake)
    monkeypatch.setitem(sys.modules, "gb_aviary", av)
    return av


def test_the_gate_used_is_niah_not_smoke(monkeypatch):
    """A smoke gate passes quantized KV that has lost long-range retrieval.
    This asserts the recall gate is the one actually invoked."""
    fake = _FakeSynapse()
    av = _install(monkeypatch, fake)
    B.measure_current("m", emit=False)
    assert av.calls, "niah_certify was never called"
    assert av.calls[0][1] == B.DEFAULT_NIAH_TOKENS


def test_the_haystack_is_big_enough_to_certify_anything():
    """A 2k-token haystack passes on every setting and certifies nothing."""
    assert B.DEFAULT_NIAH_TOKENS >= 8192


def test_the_curve_reports_vram_saved_against_recall_lost(monkeypatch):
    fake = _FakeSynapse()
    _install(monkeypatch, fake)
    res = B.sweep("m", kv_types=("f16", "q8_0"), confirm=True)
    q8 = next(r for r in res["kv_types"] if r["kv_type"] == "q8_0")
    assert q8["vram_saved_gb"] == 1.06        # 2.12 -> 1.06
    assert q8["recall_lost"] == 0.125         # 1.0 -> 0.875


def test_recall_is_a_ratio_not_a_needle_count(monkeypatch):
    """The live 2026-08-20 run reported "recall=8" for an 8-needle test,
    because `niah_certify` returns `score` as a COUNT. A count cannot be
    compared across runs with different needle budgets."""
    fake = _FakeSynapse()
    _install(monkeypatch, fake)
    s = B.measure_current("m", needles=8, emit=False)["summary"]
    assert s["found"] == 8
    assert s["recall"] == 1.0
    assert 0.0 <= s["recall"] <= 1.0


def test_kv_gb_is_populated_even_when_ps_omits_it(monkeypatch):
    """A cost curve with a blank cost column answers nothing. ps() drops
    kv_gb right after a restart, which is exactly when a sweep reads it."""
    fake = _FakeSynapse()
    fake.kv_gb = None
    _install(monkeypatch, fake)
    s = B.measure_current("m", emit=False)["summary"]
    assert s["kv_gb"] and s["kv_gb"] > 0


def test_kv_type_is_recorded_from_what_was_served_not_what_was_asked(monkeypatch):
    fake = _FakeSynapse()
    _install(monkeypatch, fake)
    res = B.sweep("m", kv_types=("f16", "q8_0", "q4_0"), confirm=True)
    assert [r["kv_type"] for r in res["kv_types"]] == ["f16", "q8_0", "q4_0"]


def test_a_sweep_refuses_to_restart_the_engine_without_confirm(monkeypatch):
    fake = _FakeSynapse()
    _install(monkeypatch, fake)
    res = B.sweep("m", kv_types=("f16", "q8_0"), confirm=False)
    assert "confirm" in res["error"]
    assert fake.calls == []


def test_a_sweep_restores_f16_rather_than_leaving_a_degraded_config(monkeypatch):
    """Same defect gb_bench_spec.sweep had (found 2026-08-20): a benchmark
    must not leave the box on the last, worst setting it measured."""
    fake = _FakeSynapse()
    _install(monkeypatch, fake)
    res = B.sweep("m", kv_types=("f16", "q4_0"), confirm=True)
    assert res["restored_kv_type"] == "f16"
    assert fake.kv_type == "f16"
    assert [c[2] for c in fake.calls if c[0] == "serve"][-1] == "f16"


def test_no_restore_needed_when_f16_ran_last(monkeypatch):
    fake = _FakeSynapse()
    _install(monkeypatch, fake)
    res = B.sweep("m", kv_types=("q8_0", "f16"), confirm=True)
    assert res["restored_kv_type"] is None
    assert [c[2] for c in fake.calls if c[0] == "serve"] == ["q8_0", "f16"]


def test_measuring_with_nothing_served_says_so(monkeypatch):
    class _Empty(_FakeSynapse):
        def ps(self):
            return []
    monkeypatch.setitem(sys.modules, "gb_synapse", _Empty())
    assert "no serve session" in B.measure_current(emit=False)["error"]


def test_a_saturated_sweep_says_it_cannot_distinguish(monkeypatch):
    """Measured 2026-08-20: f16, q8_0 and q4_0 all scored 10/10 at 20k tokens.
    A run where every setting is perfect measures the haystack, not the
    setting, and must not be readable as 'quantization is free'."""
    fake = _FakeSynapse()
    av = _install(monkeypatch, fake)
    av.RECALL = {"f16": 1.0, "q8_0": 1.0, "q4_0": 1.0}      # all perfect
    res = B.sweep("m", kv_types=("f16", "q8_0", "q4_0"), confirm=True)
    assert "SATURATED" in res["warning"]
    assert "not" in res["warning"].lower()


def test_a_discriminating_sweep_carries_no_warning(monkeypatch):
    fake = _FakeSynapse()
    _install(monkeypatch, fake)                              # q4_0 -> 0.5
    res = B.sweep("m", kv_types=("f16", "q4_0"), confirm=True)
    assert "warning" not in res


def test_too_many_needles_is_refused_before_the_engine_is_touched(monkeypatch):
    """Found 2026-08-20: asking for 24 needles against a 15-city list raised
    'Sample larger than population' from inside gb_aviary — minutes in, after
    the model had already been re-served. A caller mistake must fail fast."""
    fake = _FakeSynapse()
    _install(monkeypatch, fake)
    monkeypatch.setattr(B, "max_needles", lambda: 15)
    res = B.sweep("m", kv_types=("f16", "q4_0"), needles=24, confirm=True)
    assert "exceeds" in res["error"] and "15" in res["error"]
    assert fake.calls == []                    # nothing was restarted


def test_a_needle_count_within_the_cap_runs(monkeypatch):
    fake = _FakeSynapse()
    _install(monkeypatch, fake)
    monkeypatch.setattr(B, "max_needles", lambda: 15)
    res = B.sweep("m", kv_types=("f16",), needles=15, confirm=True)
    assert "error" not in res


def test_a_failure_mid_sweep_still_restores_the_engine(monkeypatch):
    """A benchmark that dies must not leave the box with NOTHING serving —
    that is worse than leaving it mis-tuned. The restore used to sit after the
    loop, so any exception skipped it."""
    fake = _FakeSynapse()
    av = _install(monkeypatch, fake)

    calls = {"n": 0}

    def _boom(model, tokens, needles=8, kv_type="unknown", **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("502 Bad Gateway")
        return {"score": needles, "needles": needles, "status": "ok",
                "prompt_tokens": tokens}
    av.niah_certify = _boom

    res = B.sweep("m", kv_types=("f16", "q4_0"), confirm=True)
    assert "502" in res["failed"]
    assert res["restored_kv_type"] == "f16"
    assert fake.kv_type == "f16"                      # actually re-served
    assert any(r.get("kv_type") == "(interrupted)" for r in res["kv_types"])


def test_a_failing_restore_is_reported_not_swallowed(monkeypatch):
    """If the restore itself fails, the caller must learn that the box is in
    an unknown state rather than reading a clean result."""
    fake = _FakeSynapse()
    av = _install(monkeypatch, fake)
    av.niah_certify = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))

    def _no_serve(model, kv_type=None, **kw):
        raise RuntimeError("engine will not start")
    monkeypatch.setattr(fake, "serve", _no_serve)

    res = B.sweep("m", kv_types=("f16", "q4_0"), confirm=True)
    assert str(res["restored_kv_type"]).startswith("FAILED")
