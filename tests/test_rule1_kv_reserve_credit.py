"""Rule #1 must not count COMMITTED KV reserve as unfilled VRAM.

The shim holds `kv_reserve_effective_mb` from the moment a model loads and
consumes it as the conversation grows, so a fresh session at a large ctx
legitimately shows a gap between "VRAM used" and "VRAM committed".

Measured on a healthy serve 2026-08-18: fill 75.7%, kv_reserve_effective_mb
2009, kv_t1_tracked_mb 218 , about 1.8 GB of the free space was reservation,
not waste, and the tripwire called it a violation. Crying wolf on the one
signal Rule #1 depends on is worse than the occasional real miss, because it
teaches the operator to ignore it.
"""
import pytest

import gb_semantics as S


@pytest.fixture
def stub(monkeypatch):
    """Drive the evaluator from fixed inputs, not the live box."""
    def _apply(fill, overflow_mb, held, used, total_mb=11264):
        def _resolve(name, *a, **k):
            return {
                "vram_fill_pct":         {"value": fill, "metric": name},
                "t2_overflow_active_mb": {"value": overflow_mb, "metric": name},
                "t2_pressure_fraction":  {"value": 0, "metric": name},
                "t3_spill_active_mb":    {"value": 0, "metric": name},
            }.get(name, {"value": None, "metric": name})
        monkeypatch.setattr(S, "resolve", _resolve)
        # shim_stats is a flat key=value FILE , every field is a string.
        monkeypatch.setattr(S, "_shim_stats",
                            lambda *a, **k: ({"kv_reserve_effective_mb": str(held),
                                              "kv_t1_tracked_mb": str(used)}, "stub"))

        class _Snap:
            vram_physical_mb = total_mb
        monkeypatch.setitem(__import__("sys").modules, "gb_monitor",
                            type("M", (), {"snapshot": staticmethod(lambda *a, **k: _Snap())}))
    return _apply


def test_committed_reserve_is_not_a_violation(stub):
    """The exact healthy serve that was wrongly flagged."""
    stub(fill=75.7, overflow_mb=10240, held=2009, used=218)
    matched, ev = S._seg_rule1_underfilled()
    assert matched is False
    note = [e for e in ev if e.get("metric") == "kv_reserve_pending_mb"]
    assert note and note[0]["value"] == pytest.approx(1791, abs=1)
    assert note[0]["vram_fill_pct_effective"] > 85.0


def test_a_real_underfill_still_fires(stub):
    """The case the tripwire exists for: VRAM idle, weights in T2, no reserve."""
    stub(fill=50.3, overflow_mb=16312, held=0, used=0)
    matched, _ = S._seg_rule1_underfilled()
    assert matched is True


def test_a_reserve_already_consumed_earns_no_credit(stub):
    """Filled reserve is already inside vram_fill_pct , counting it twice hides a real miss."""
    stub(fill=60.0, overflow_mb=8000, held=2000, used=2000)
    matched, ev = S._seg_rule1_underfilled()
    assert matched is True
    assert not [e for e in ev if e.get("metric") == "kv_reserve_pending_mb"]


def test_string_fields_are_parsed(stub):
    """shim_stats values arrive as strings; an isinstance check silently ignored them."""
    stub(fill=75.7, overflow_mb=10240, held="2009", used="218")
    matched, ev = S._seg_rule1_underfilled()
    assert matched is False
    assert any(e.get("metric") == "kv_reserve_pending_mb" for e in ev)


def test_unreadable_shim_gives_no_credit(monkeypatch, stub):
    """'Cannot read' must never soften the verdict."""
    stub(fill=50.0, overflow_mb=9000, held=2000, used=0)
    monkeypatch.setattr(S, "_shim_stats", lambda *a, **k: (None, "untrusted"))
    matched, ev = S._seg_rule1_underfilled()
    assert matched is True
    assert not [e for e in ev if e.get("metric") == "kv_reserve_pending_mb"]


def test_no_overflow_is_never_a_violation(stub):
    stub(fill=40.0, overflow_mb=0, held=0, used=0)
    matched, _ = S._seg_rule1_underfilled()
    assert matched is False
