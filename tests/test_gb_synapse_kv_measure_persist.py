"""The KV measurement must actually get written, or Rule #1 stays blocked.

`_persist_kv_measurement()` exists so a second serve of the same
(model, ctx, kv_type) sizes GREENBOOST_KV_RESERVE_MB from a real shim
measurement instead of estimate_kv_gb()'s formula, which this hybrid
Gated-DeltaNet architecture inflates. It was only ever called immediately after
the server reported ready, and it bailed on two conditions that are both TRUE at
exactly the moments it runs:

  * `kv_t1_tracked_mb` is 0 right after load, because the engine allocates KV
    lazily on the first request; and
  * the shim rewrites its counters to 0 during teardown.

So for a model whose KV allocates lazily, nothing was ever persisted and every
future serve kept the formula. Measured live 2026-08-17, first serve of
Qwen3.8-27B-Cold-Fusion at ctx=32768/f16: reserve 2908 MB against a real 258 MB
(11.3x), no entry in kv_measurements.json, physical VRAM stuck at 66.7% while
14326 MB streamed from T2 over PCIe every forward pass.

After the fix (stop-time capture + peak-from-history fallback), the same serve:
reserve 264 MB, VRAM fill 88.0% (`in_target`, `rule1_underfilled` false),
T2 overflow 14326 -> 12140 MB, decode 4.90 -> ~5.48 tok/s.
"""
from __future__ import annotations

import json
import types

import pytest

import gb_synapse


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Redirect the measurement store to a temp file."""
    p = tmp_path / "kv_measurements.json"
    monkeypatch.setattr(gb_synapse, "MODEL_STORE_DIR", tmp_path, raising=False)
    return p


def _snap(loaded=True, stale=False, kv_t1=0.0):
    return types.SimpleNamespace(
        loaded=loaded, shim_stale=stale, shim={"kv_t1_tracked_mb": kv_t1},
    )


def _patch_monitor(monkeypatch, snap):
    import gb_monitor
    monkeypatch.setattr(gb_monitor, "snapshot", lambda **kw: snap)


def test_peak_helper_ignores_the_teardown_zero(monkeypatch) -> None:
    """It must take the high-water mark, not the latest sample , the latest
    sample is precisely the zeroed teardown value we cannot trust."""
    import gb_dataflux
    import time
    now = time.time()
    monkeypatch.setattr(gb_dataflux, "read_events", lambda *a, **k: [
        {"kind": "snapshot", "ts": now - 300, "kv_used_mb": 120},
        {"kind": "snapshot", "ts": now - 200, "kv_used_mb": 258},
        {"kind": "snapshot", "ts": now - 10, "kv_used_mb": 0},
    ])
    assert gb_synapse._peak_kv_used_mb(window_s=3600) == 258.0


def test_peak_helper_respects_its_window(monkeypatch) -> None:
    import gb_dataflux
    import time
    now = time.time()
    monkeypatch.setattr(gb_dataflux, "read_events", lambda *a, **k: [
        {"kind": "snapshot", "ts": now - 99999, "kv_used_mb": 9999},
        {"kind": "snapshot", "ts": now - 10, "kv_used_mb": 258},
    ])
    assert gb_synapse._peak_kv_used_mb(window_s=3600) == 258.0


def test_peak_helper_scopes_to_this_serve_not_the_previous_model(monkeypatch) -> None:
    """`snapshot` events carry no model field, so a 6-hour window spans model
    swaps. Right after switching reference models the window is still full of
    the PREVIOUS model's KV — persisting that under the NEW model's key writes
    a confidently wrong reserve, the same cross-contamination class as the
    tok_s keying fix (missing_features.md item (k))."""
    import gb_dataflux
    import time
    now = time.time()
    serve_started = now - 60
    monkeypatch.setattr(gb_dataflux, "read_events", lambda *a, **k: [
        # Previous reference model, big KV, still inside the 6h window.
        {"kind": "snapshot", "ts": now - 3000, "kv_used_mb": 5628},
        # This serve.
        {"kind": "snapshot", "ts": now - 30, "kv_used_mb": 1322},
    ])
    unscoped = gb_synapse._peak_kv_used_mb(window_s=6 * 3600)
    scoped = gb_synapse._peak_kv_used_mb(window_s=6 * 3600, since_ts=serve_started)
    assert unscoped == 5628.0, "sanity: unscoped really does see the old model"
    assert scoped == 1322.0, "scoped scan must not inherit the previous model's KV"


def test_peak_helper_since_ts_never_widens_the_window(monkeypatch) -> None:
    """since_ts narrows, never widens: an older since_ts must not resurrect
    samples the window_s cutoff already excluded."""
    import gb_dataflux
    import time
    now = time.time()
    monkeypatch.setattr(gb_dataflux, "read_events", lambda *a, **k: [
        {"kind": "snapshot", "ts": now - 99999, "kv_used_mb": 9999},
        {"kind": "snapshot", "ts": now - 10, "kv_used_mb": 258},
    ])
    assert gb_synapse._peak_kv_used_mb(window_s=3600, since_ts=now - 99999) == 258.0


def test_peak_helper_survives_a_broken_log(monkeypatch) -> None:
    import gb_dataflux
    monkeypatch.setattr(gb_dataflux, "read_events",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no log")))
    assert gb_synapse._peak_kv_used_mb() == 0.0


def test_falls_back_to_history_when_the_live_read_is_zero(store, monkeypatch) -> None:
    """The load-time case: KV not allocated yet, so the live shim reads 0."""
    _patch_monitor(monkeypatch, _snap(kv_t1=0.0))
    monkeypatch.setattr(gb_synapse, "_peak_kv_used_mb", lambda **kw: 258.0)
    saved = {}
    monkeypatch.setattr(gb_synapse, "_save_kv_measurement",
                        lambda m, c, k, v: saved.update(model=m, ctx=c, kv=k, mb=v))

    gb_synapse._persist_kv_measurement(types.SimpleNamespace(name="m"),
                                       {"ctx": 32768, "kv_type": "f16"})
    assert saved == {"model": "m", "ctx": 32768, "kv": "f16", "mb": 258.0}


def test_a_stale_shim_no_longer_makes_this_a_no_op(store, monkeypatch) -> None:
    """The teardown case, and the reason stop() could never record anything.

    A stale shim_stats means the LIVE read is untrustworthy, not that the
    recorded history is.
    """
    _patch_monitor(monkeypatch, _snap(stale=True, kv_t1=0.0))
    monkeypatch.setattr(gb_synapse, "_peak_kv_used_mb", lambda **kw: 258.0)
    saved = {}
    monkeypatch.setattr(gb_synapse, "_save_kv_measurement",
                        lambda m, c, k, v: saved.update(mb=v))

    gb_synapse._persist_kv_measurement(types.SimpleNamespace(name="m"),
                                       {"ctx": 32768, "kv_type": "f16"})
    assert saved.get("mb") == 258.0


def test_live_reading_wins_when_it_is_the_larger_evidence(store, monkeypatch) -> None:
    """A live read is kept , but it is now reconciled against the peak, not
    trusted blindly.

    This test used to assert the history "must not" be consulted whenever a
    live reading existed. That is what allowed the 2026-08-18 defect: a 218 MB
    read taken ELEVEN SECONDS into a serve, against an empty conversation, was
    persisted as the measurement for a 24,576-token context that genuinely
    needs ~1.6 GB. The reserve must cover the HIGH-WATER mark, so the larger of
    the two wins , here that is still the live reading.
    """
    _patch_monitor(monkeypatch, _snap(kv_t1=788.0))
    monkeypatch.setattr(gb_synapse, "_peak_kv_used_mb", lambda **kw: 300.0)
    saved = {}
    monkeypatch.setattr(gb_synapse, "_save_kv_measurement",
                        lambda m, c, k, v: saved.update(mb=v))

    gb_synapse._persist_kv_measurement(types.SimpleNamespace(name="m"),
                                       {"ctx": 65536, "kv_type": "q8_0"})
    assert saved.get("mb") == 788.0


def test_an_early_snapshot_never_beats_the_peak(store, monkeypatch) -> None:
    """The defect itself, pinned: small live read, much larger real peak."""
    _patch_monitor(monkeypatch, _snap(kv_t1=218.0))
    monkeypatch.setattr(gb_synapse, "_peak_kv_used_mb", lambda **kw: 1640.0)
    saved = {}
    monkeypatch.setattr(gb_synapse, "_save_kv_measurement",
                        lambda m, c, k, v: saved.update(mb=v))

    gb_synapse._persist_kv_measurement(types.SimpleNamespace(name="m"),
                                       {"ctx": 24576, "kv_type": "f16"})
    assert saved.get("mb") == 1640.0, "an early snapshot was persisted over the peak"


def test_records_nothing_when_there_is_no_evidence_at_all(store, monkeypatch) -> None:
    """No measurement is better than a fabricated one: the caller stays on the
    conservative formula."""
    _patch_monitor(monkeypatch, _snap(kv_t1=0.0))
    monkeypatch.setattr(gb_synapse, "_peak_kv_used_mb", lambda **kw: 0.0)
    monkeypatch.setattr(gb_synapse, "_save_kv_measurement",
                        lambda *a: pytest.fail("must not save without evidence"))

    gb_synapse._persist_kv_measurement(types.SimpleNamespace(name="m"),
                                       {"ctx": 32768, "kv_type": "f16"})


def test_unloaded_kmod_is_still_a_hard_stop(store, monkeypatch) -> None:
    _patch_monitor(monkeypatch, _snap(loaded=False))
    monkeypatch.setattr(gb_synapse, "_save_kv_measurement",
                        lambda *a: pytest.fail("no kmod means no measurement"))
    gb_synapse._persist_kv_measurement(types.SimpleNamespace(name="m"),
                                       {"ctx": 32768, "kv_type": "f16"})
