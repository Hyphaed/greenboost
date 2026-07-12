"""gb_pilot analyze/advise — pure-function tests over synthetic dataflux events.
No CUDA, no real log (analysis functions take event lists directly).
"""
import gb_pilot as gp


def _stage(stage, wall, status="ok", ts=0, vram=None):
    ev = {"kind": "stage_profile", "stage": stage, "duration_s": wall,
          "status": status, "ts": ts, "node": "host", "label": "jobqueue",
          "n_items": 1}
    if vram is not None:
        ev["vram_peak_mb"] = vram
    return ev


def _tok(model, tok_s, ts=0):
    return {"kind": "tok_s_measured", "model": model, "tok_s": tok_s, "ts": ts,
            "node": "host", "label": "gb_synapse", "n_items": 1}


def _snap(ts=0, **kw):
    ev = {"kind": "snapshot", "ts": ts, "node": "host", "label": "system_snapshot",
          "t2_pressure": 0, "t3_used_mb": 0, "fb_used_pct": 50,
          "kv_used_mb": 0, "kv_reserve_mb": 2048, "shim_phase": "STEADY"}
    ev.update(kw)
    return ev


# ── analyze ──────────────────────────────────────────────────────────────────
def test_analyze_empty():
    a = gp.analyze([])
    assert a["stages"] == {} and a["models"] == {}
    assert a["event_count"] == 0


def test_analyze_malformed_rows_tolerated():
    a = gp.analyze([None, "junk", {"kind": "stage_profile"},
                    {"kind": "stage_profile", "stage": "x", "duration_s": "NaN?"},
                    _stage("image:generate", 10.0)])
    assert a["stages"]["image:generate"]["count"] == 1


def test_analyze_flags_stage_regression():
    evs = [_stage("image:generate", w, ts=i) for i, w in
           enumerate([30.0, 31.0, 29.0, 45.0])]     # latest +50% vs median
    a = gp.analyze(evs)
    s = a["stages"]["image:generate"]
    assert s["regressed"] is True
    assert s["latest_s"] == 45.0
    assert s["median_s"] == 30.0


def test_analyze_no_flag_below_min_samples():
    evs = [_stage("video:denoise", 10.0), _stage("video:denoise", 20.0)]
    a = gp.analyze(evs)
    assert a["stages"]["video:denoise"]["regressed"] is False


def test_analyze_flags_tok_s_degradation():
    evs = [_tok("qwen3-vl:30b", t) for t in [20.0, 21.0, 19.0, 10.0]]
    a = gp.analyze(evs)
    m = a["models"]["qwen3-vl:30b"]
    assert m["degraded"] is True
    assert m["latest"] == 10.0


def test_analyze_improving_tok_s_not_flagged():
    evs = [_tok("m", t) for t in [10.0, 12.0, 15.0, 20.0]]
    assert gp.analyze(evs)["models"]["m"]["degraded"] is False


def test_analyze_latest_snapshot_wins():
    evs = [_snap(ts=100, t2_pressure=0), _snap(ts=200, t2_pressure=2)]
    assert gp.analyze(evs)["pressure"]["t2_pressure"] == 2


# ── advise ───────────────────────────────────────────────────────────────────
def test_advise_all_clear_on_clean_data():
    a = gp.analyze([_stage("image:generate", 30.0), _snap(ts=1)])
    adv = gp.advise(a)
    assert len(adv) == 1 and adv[0]["topic"] == "all_clear"


def test_advise_t3_spill_is_critical():
    a = gp.analyze([_snap(ts=1, t3_used_mb=4096)])
    adv = gp.advise(a)
    topics = {x["topic"]: x for x in adv}
    assert topics["t3_spill"]["severity"] == "critical"
    assert "quantize_to_fit" in topics["t3_spill"]["action"]


def test_advise_t2_critical_names_safe_levers():
    a = gp.analyze([_snap(ts=1, t2_pressure=2)])
    adv = gp.advise(a)
    lever = {x["topic"]: x for x in adv}["t2_pressure"]["lever"]
    assert "set_workstation_reserve_mb" in lever
    assert "set_prefetch_throttle" in lever


def test_advise_stage_regression_evidence():
    evs = [_stage("image:generate", w, ts=i) for i, w in
           enumerate([30.0, 31.0, 29.0, 60.0])]
    adv = gp.advise(gp.analyze(evs))
    reg = {x["topic"]: x for x in adv}["stage_regression"]
    assert "image:generate" in reg["evidence"]
    assert "60.0s" in reg["evidence"] or "60.0" in reg["evidence"]


def test_advise_tok_s_drop():
    evs = [_tok("m", t) for t in [20.0, 21.0, 19.0, 5.0]]
    adv = gp.advise(gp.analyze(evs))
    assert any(x["topic"] == "tok_s_drop" for x in adv)


def test_advise_stage_error_surfaces():
    evs = [_stage("mesh:bake", 10.0, status="error", ts=5)]
    adv = gp.advise(gp.analyze(evs))
    assert any(x["topic"] == "stage_error" for x in adv)


# ── panel rendering smoke (no TTY, no real log) ─────────────────────────────
def test_snapshot_renders_without_log(monkeypatch):
    monkeypatch.setattr(gp, "_read", lambda days: [
        _stage("image:generate", 30.0), _tok("m", 15.0), _snap(ts=1)])
    text = gp._cmd_pilot_snapshot(5.0)
    assert "GreenBoost Pilot" in text
    assert "image:generate" in text
    assert "ADVICE" in text
    assert "│" not in text          # UI paradigm: no vertical pipes
