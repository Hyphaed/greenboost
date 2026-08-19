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
    evs = [_tok("qwen3-vl:30b", t) for t in
            [20.0, 21.0, 19.0, 20.0, 21.0, 19.0, 20.0, 10.0]]
    a = gp.analyze(evs)
    m = a["models"]["qwen3-vl:30b"]
    assert m["degraded"] is True
    assert m["latest"] == 10.0


def test_analyze_improving_tok_s_not_flagged():
    evs = [_tok("m", t) for t in [8.0, 9.0, 10.0, 12.0, 14.0, 15.0, 18.0, 20.0]]
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
    item = {x["topic"]: x for x in adv}["t2_pressure"]
    lever = item["lever"]
    assert lever["call"] == "set_prefetch_throttle"
    assert lever["args"] == [True]   # boolean toggle needs no current-state read , auto-appliable
    assert "workstation reserve" in item["action"]  # named in prose, not exec'd (needs current value)


def test_every_advice_lever_names_a_real_gbcontrol_method():
    import gb_control
    tok_events = [_tok("m", 100, ts=i) for i in range(8)] + [_tok("m", 40, ts=8)]
    scenarios = [
        [_snap(ts=1, t3_used_mb=4096)],
        [_snap(ts=1, t2_pressure=2)],
        tok_events,   # drives the tok_s_drop / set_kv_size_threshold_mb lever
    ]
    for events in scenarios:
        a = gp.analyze(events)
        for item in gp.advise(a):
            lever = item.get("lever")
            if lever is None:
                continue
            assert hasattr(gb_control.GbControl, lever["call"]), (
                f"lever names {lever['call']!r}, not a real GbControl method")
            if lever["args"] is not None:
                assert isinstance(lever["args"], list)


def test_advise_stage_regression_evidence():
    evs = [_stage("image:generate", w, ts=i) for i, w in
           enumerate([30.0, 31.0, 29.0, 60.0])]
    adv = gp.advise(gp.analyze(evs))
    reg = {x["topic"]: x for x in adv}["stage_regression"]
    assert "image:generate" in reg["evidence"]
    assert "60.0s" in reg["evidence"] or "60.0" in reg["evidence"]


def test_advise_tok_s_drop():
    evs = [_tok("m", t) for t in [20.0, 21.0, 19.0, 20.0, 21.0, 19.0, 20.0, 5.0]]
    adv = gp.advise(gp.analyze(evs))
    assert any(x["topic"] == "tok_s_drop" for x in adv)


def test_advise_stage_error_surfaces():
    evs = [_stage("mesh:bake", 10.0, status="error", ts=5)]
    adv = gp.advise(gp.analyze(evs))
    assert any(x["topic"] == "stage_error" for x in adv)


# ── DI-13: rebalance advisory (petals should_rebalance gate) ────────────────

def _split(nodes, ts=0, v3=False):
    return {"kind": "tensor_split", "split": ",".join(["1000"] * nodes),
            "nodes": nodes, "v3": v3, "ts": ts, "node": "host", "label": "gb_synapse"}


def test_rebalance_advisory_fires_on_severe_drop_with_multinode_split():
    evs = [_split(2, ts=0)] + [_tok("m", t, ts=i) for i, t in
                               enumerate([100.0] * 8 + [40.0])]
    a = gp.analyze(evs)
    assert a["last_split"]["nodes"] == 2
    adv = gp.advise(a)
    assert any(x["topic"] == "rebalance_advice" for x in adv)


def test_rebalance_advisory_silent_on_single_node():
    evs = [_split(1, ts=0)] + [_tok("m", t, ts=i) for i, t in
                               enumerate([100.0] * 8 + [40.0])]
    adv = gp.advise(gp.analyze(evs))
    assert not any(x["topic"] == "rebalance_advice" for x in adv)


def test_rebalance_advisory_silent_below_threshold():
    # Drop clears the generic tok_s_drop bar (20%) but not the stricter
    # rebalance bar (25%). Baseline is the MEDIAN, not the mean (changed
    # 2026-08-17 — a duration-blind tok/s distribution is too skewed for a mean
    # to be a usable baseline): median=100, drop=(100-78)/100=22%.
    # The old fixture's 74.0 was computed against mean=94.8 (21.9%); under
    # median semantics that same sample is a 26% drop and correctly clears the
    # rebalance bar too, so the value moved to keep this test asserting the
    # in-between band it was written to cover.
    evs = [_split(2, ts=0)] + [_tok("m", t, ts=i) for i, t in
                               enumerate([100.0] * 8 + [78.0])]
    a = gp.analyze(evs)
    adv = gp.advise(a)
    assert any(x["topic"] == "tok_s_drop" for x in adv)
    assert not any(x["topic"] == "rebalance_advice" for x in adv)


def test_rebalance_advisory_silent_without_split_history():
    evs = [_tok("m", t, ts=i) for i, t in enumerate([100.0] * 8 + [40.0])]
    a = gp.analyze(evs)
    assert a["last_split"]["nodes"] == 1   # default when no tensor_split event seen
    adv = gp.advise(a)
    assert not any(x["topic"] == "rebalance_advice" for x in adv)


# ── panel rendering smoke (no TTY, no real log) ─────────────────────────────
def test_snapshot_renders_without_log(monkeypatch):
    monkeypatch.setattr(gp, "_read", lambda days: [
        _stage("image:generate", 30.0), _tok("m", 15.0), _snap(ts=1)])
    text = gp._cmd_pilot_snapshot(5.0)
    assert "GreenBoost Pilot" in text
    assert "image:generate" in text
    assert "ADVICE" in text
    assert "│" not in text          # UI paradigm: no vertical pipes


def test_tok_s_drop_needs_a_real_population():
    """A big drop on a handful of samples must NOT raise an advisory.

    Regression for 2026-08-18: five standing "decode speed degraded" warnings
    were built on n=3, n=5 and n=9 populations, each carrying a real
    set_kv_size_threshold_mb retune lever. A tok/s sample is duration-blind and
    the 24-token sample floor discards much of an agentic session's traffic, so
    what survives is both few and skewed — a latest-vs-median comparison over
    three of them is noise, not a finding.
    """
    few = [_tok("m", t, ts=i) for i, t in enumerate([100.0, 100.0, 20.0])]
    a = gp.analyze(few)
    assert a["models"]["m"]["degraded"] is False
    assert a["models"]["m"]["insufficient_samples"] is True
    assert not any(x["topic"] == "tok_s_drop" for x in gp.advise(a))

    # The same drop, once there is a population behind it, still fires.
    many = [_tok("m", t, ts=i) for i, t in enumerate([100.0] * 8 + [20.0])]
    a = gp.analyze(many)
    assert a["models"]["m"]["degraded"] is True
    assert a["models"]["m"]["insufficient_samples"] is False
    assert any(x["topic"] == "tok_s_drop" for x in gp.advise(a))


def test_stage_regressions_keep_the_smaller_sample_floor():
    """Only decode rate got the bigger floor; stage wall-times are not rates."""
    evs = [_stage("image:generate", w, ts=i) for i, w in
           enumerate([30.0, 31.0, 29.0, 60.0])]
    assert gp.analyze(evs)["stages"]["image:generate"]["regressed"] is True
