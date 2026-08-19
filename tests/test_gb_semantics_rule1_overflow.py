"""The Rule #1 tripwire must fire on real T2 overflow, not on pool pressure.

`rule1_underfilled` is CLAUDE.md's flagship tripwire: "physical VRAM sits well
below the 85-92% target band while overflow to T2/T3 is active". It used to test
`t2_pressure_fraction` for the overflow half of that condition — but pressure is
how full the T2 POOL is, not whether weights are being served from T2.

On a large pool the two diverge completely. Measured 2026-08-17 while serving a
15.85 GiB model on 11.26 GB of VRAM: the shim reported
`t2_overflow_total_mb=14326` into a 43 GB pool whose pressure read 0.0, with
`vram_fill_pct=69`. The tripwire returned `matched: False` — silent in exactly
the scenario it exists to catch. (The kmod's own `t2_allocated_mb` also read 0
in that sample, which is why the shim counter is the trustworthy source.)
"""
from __future__ import annotations

import pytest

import gb_semantics as S


@pytest.fixture(autouse=True)
def _fresh_definitions():
    S.load(force=True)
    yield
    S.load(force=True)


def _patch(monkeypatch, *, vram, overflow_mb, pressure, t3_mb):
    """Pin the four metrics the segment reads, so the test exercises the
    segment's LOGIC rather than whatever this machine happens to be doing."""
    values = {
        "vram_fill_pct": {"value": vram, "unit": "percent"},
        "t2_overflow_active_mb": {"value": overflow_mb, "unit": "megabytes"},
        "t2_pressure_fraction": {"value": pressure, "unit": "fraction_0_1"},
        "t3_spill_active_mb": {"value": t3_mb, "unit": "megabytes"},
    }
    real = S.resolve

    def fake(name, *a, **kw):
        if name in values:
            return {"metric": name, **values[name]}
        return real(name, *a, **kw)

    monkeypatch.setattr(S, "resolve", fake)


def test_fires_on_t2_overflow_even_when_pool_pressure_is_zero(monkeypatch) -> None:
    """The exact production sample that exposed the bug."""
    _patch(monkeypatch, vram=69.0, overflow_mb=14326.0, pressure=0.0, t3_mb=0)
    matched, evidence = S._seg_rule1_underfilled()
    assert matched is True
    assert any(e.get("metric") == "t2_overflow_active_mb" for e in evidence), \
        "the overflow counter must appear in the evidence chain"


def test_quiet_when_vram_is_in_target_despite_overflow(monkeypatch) -> None:
    # Overflow alone is normal GreenBoost operation — it is only a violation
    # when VRAM is ALSO underfilled.
    _patch(monkeypatch, vram=88.0, overflow_mb=14326.0, pressure=0.0, t3_mb=0)
    matched, _ = S._seg_rule1_underfilled()
    assert matched is False


def test_quiet_when_underfilled_with_no_overflow_anywhere(monkeypatch) -> None:
    # A small model simply not needing the VRAM is not a Rule #1 violation.
    _patch(monkeypatch, vram=40.0, overflow_mb=0.0, pressure=0.0, t3_mb=0)
    matched, _ = S._seg_rule1_underfilled()
    assert matched is False


def test_still_fires_via_pressure_or_t3(monkeypatch) -> None:
    """The original signals stay wired — this widened the condition, it did not
    swap one blind spot for another."""
    _patch(monkeypatch, vram=50.0, overflow_mb=0.0, pressure=0.7, t3_mb=0)
    assert S._seg_rule1_underfilled()[0] is True

    _patch(monkeypatch, vram=50.0, overflow_mb=0.0, pressure=0.0, t3_mb=512)
    assert S._seg_rule1_underfilled()[0] is True


def test_unknown_vram_never_asserts_a_violation(monkeypatch) -> None:
    # Absent data must not be read as a violation; fail quiet, not loud-and-wrong.
    _patch(monkeypatch, vram=None, overflow_mb=14326.0, pressure=0.0, t3_mb=0)
    assert S._seg_rule1_underfilled()[0] is False


def test_unknown_overflow_is_indeterminate_not_healthy(monkeypatch) -> None:
    """The other half of "fail quiet": don't assert HEALTH on absent data either.

    The shim rewrites shim_stats every 250 ms from its CUDA hooks, so the file
    goes stale whenever inference is idle and t2_overflow_active_mb resolves to
    None. Measured 2026-08-17: vram_fill_pct=50.3 (below target) with
    t2_overflow_total_mb=16312 actually sitting in the file the resolver had
    just declined to trust — and the segment returned False, a clean bill of
    health for a live violation. None means "cannot tell"; evaluate_segment
    already carries that value on its exception path.
    """
    _patch(monkeypatch, vram=50.3, overflow_mb=None, pressure=0.0, t3_mb=0)
    matched, evidence = S._seg_rule1_underfilled()
    assert matched is None, "unknown overflow must not read as 'no violation'"
    assert any(e.get("metric") == "t2_overflow_active_mb" for e in evidence)


def test_unknown_overflow_still_fires_when_another_signal_is_positive(monkeypatch) -> None:
    """Indeterminacy only applies when NO signal can answer. A positive
    pressure or T3 reading is still a definitive violation on its own."""
    _patch(monkeypatch, vram=50.0, overflow_mb=None, pressure=0.7, t3_mb=0)
    assert S._seg_rule1_underfilled()[0] is True

    _patch(monkeypatch, vram=50.0, overflow_mb=None, pressure=0.0, t3_mb=512)
    assert S._seg_rule1_underfilled()[0] is True


def test_unknown_overflow_is_irrelevant_when_vram_is_in_target(monkeypatch) -> None:
    """VRAM in the target band is a definitive non-violation regardless of what
    the overflow counter is doing — no need to degrade to indeterminate."""
    _patch(monkeypatch, vram=88.0, overflow_mb=None, pressure=0.0, t3_mb=0)
    assert S._seg_rule1_underfilled()[0] is False


def test_overflow_metric_is_defined_and_traps_the_lookalikes() -> None:
    defs = S.load(force=True)
    metrics = defs["metrics"] if isinstance(defs, dict) and "metrics" in defs else defs
    m = metrics["t2_overflow_active_mb"]
    banned = {n["field"] if isinstance(n, dict) else n for n in (m.never_use or [])}
    # Both are the fields that made this bug possible in the first place.
    assert "t2_pressure" in banned
    assert "t2_allocated_mb" in banned


def test_resolver_reports_nothing_rather_than_stale_overflow(tmp_path, monkeypatch) -> None:
    """A stale shim_stats describes a process that may be long gone. Reporting
    its overflow as current would be worse than reporting nothing."""
    stats = tmp_path / "shim_stats"
    stats.write_text("t2_overflow_total_mb=14326\n")
    import os, time as _t
    old = _t.time() - 3600
    os.utime(stats, (old, old))
    monkeypatch.setattr(S, "Path", lambda p: stats if "shim_stats" in str(p) else __import__("pathlib").Path(p))
    out = S._res_t2_overflow_active_mb(None, 30.0)
    assert out["value"] is None
    assert "stale" in out["raw_source"]
