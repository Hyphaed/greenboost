"""A persisted KV measurement must be the HIGH-WATER mark, never a snapshot.

`GREENBOOST_KV_RESERVE_MB` is sized from this value on every later serve of the
same (model, ctx, kv_type). Under-reserving is not the safe direction: weights
fill the VRAM the KV was meant to get, and the KV then spills to T2 mid
conversation , exactly what the reserve exists to prevent.

Live 2026-08-18: `Qwen3.8-27B-Cold-Fusion-MTP-IQ4_XS::24576::f16` was persisted
as 218 MB, sampled ELEVEN SECONDS into the serve against an empty conversation.
A 24,576-token context genuinely needs ~1.6 GB. The peak was consulted only
when the instantaneous read came back zero, so a small non-zero read was
trusted as if it were the maximum.
"""
import pytest

import gb_synapse as gs


@pytest.fixture
def capture(monkeypatch):
    """Run _persist_kv_measurement against stubbed inputs; return what it saved."""
    saved = {}

    class _Snap:
        loaded = True
        shim_stale = False
        shim = {}

    def _apply(live_mb, peak_mb):
        _Snap.shim = {"kv_t1_tracked_mb": live_mb}
        monkeypatch.setitem(__import__("sys").modules, "gb_monitor",
                            type("M", (), {"snapshot": staticmethod(lambda *a, **k: _Snap())}))
        monkeypatch.setattr(gs, "_peak_kv_used_mb", lambda *a, **k: peak_mb)
        monkeypatch.setattr(gs, "_save_kv_measurement",
                            lambda name, ctx, kv, mb: saved.update(
                                name=name, ctx=ctx, kv=kv, mb=mb))
        entry = type("E", (), {"name": "M"})()
        gs._persist_kv_measurement(entry, {"ctx": 24576, "kv_type": "f16",
                                           "started_ts": 0})
        return saved
    return _apply


def test_the_peak_wins_over_an_early_snapshot(capture):
    """The exact 2026-08-18 defect: 218 MB read 11 s in, real peak far higher."""
    out = capture(live_mb=218, peak_mb=1640)
    assert out["mb"] == 1640, "an early snapshot was persisted over the peak"


def test_a_live_read_higher_than_the_recorder_is_kept(capture):
    """The recorder samples every 5 s and can miss a spike the shim saw."""
    out = capture(live_mb=1800, peak_mb=1200)
    assert out["mb"] == 1800


def test_peak_is_used_when_the_live_read_is_zero(capture):
    """Teardown zeroes the shim counters , the original fallback case."""
    out = capture(live_mb=0, peak_mb=900)
    assert out["mb"] == 900


def test_nothing_is_persisted_when_both_are_empty(capture):
    """Pinning the reserve at zero is worse than staying on the formula."""
    out = capture(live_mb=0, peak_mb=0)
    assert out == {}, "a zero measurement was persisted"


def test_the_signature_is_recorded_exactly(capture):
    out = capture(live_mb=100, peak_mb=1500)
    assert out["ctx"] == 24576 and out["kv"] == "f16" and out["name"] == "M"
