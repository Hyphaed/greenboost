"""gb_pilot's throughput advisories must be computed on real, comparable samples.

Found via `dataflux_critic` on 2026-08-17 while inspecting a live greenboost-cli
session. The critic reported:

    [pilot:warn] Decode speed for Qwen3.8-27B-Cold-Fusion-MTP-IQ4_XS degraded.
    (evidence: latest 47.8 tok/s vs avg 649.7 (93% down, n=34))

An average of 649.7 tok/s is physically impossible for a 27B model on this box.
Two independent defects produced it, both already fixed in
`gb_dataflux.summarize()` but never mirrored into `gb_pilot.analyze()`:

1. Samples the WRITER rejected were counted. `record_measured_tok_s()` emits
   over-ceiling samples with `status="error"` for auditability; one impossible
   21065.2 reading dragged a 34-sample mean to 649.7. The `stage_profile` branch
   directly above already checked `status`; the `tok_s_measured` branch did not.
2. Series were keyed on the bare model name, so `[proxy]` and `[cli]` vantage
   points and different quant/ctx/kv_type configurations blended into one
   average matching none of them.

This is not cosmetic: `degraded` gates an advisory that carries a
`set_kv_size_threshold_mb` retune lever, so bad data could drive a real change.
"""
from __future__ import annotations

import gb_pilot


def _tok(model: str, tok_s: float, ts: float, status: str = "ok", **extra) -> dict:
    ev = {"kind": "tok_s_measured", "node": "host", "model": model,
          "tok_s": tok_s, "status": status, "ts": ts}
    ev.update(extra)
    return ev


def _models(events: list[dict]) -> dict:
    return gb_pilot.analyze(events)["models"]


def test_rejected_samples_do_not_enter_the_average() -> None:
    events = [_tok("m", 5.0, 1.0), _tok("m", 5.0, 2.0),
              _tok("m", 21065.2, 3.0, status="error",
                   ceiling_tok_s=2566.0,
                   error="sample exceeds node/model-derived sanity ceiling, dropped")]
    (stats,) = _models(events).values()
    assert stats["samples"] == 2
    assert stats["avg"] == 5.0


def test_a_rejected_sample_cannot_trigger_a_degradation_advisory() -> None:
    """The actual production failure: one impossible reading inflates the mean
    so every subsequent normal turn looks like a collapse."""
    events = [_tok("m", 21065.2, 0.0, status="error")]
    events += [_tok("m", 5.0, float(i)) for i in range(1, 12)]
    (stats,) = _models(events).values()
    assert stats["degraded"] is False, "a normal, steady series must not read as degraded"


def test_proxy_and_cli_vantage_points_are_kept_apart() -> None:
    # Real 2026-08-01 incident: proxy=0.3, cli=2.4, engine truth=2.18,
    # blended=0.6 , an average that matched neither observer.
    events = [_tok("m", 0.3, 1.0, source="proxy"), _tok("m", 2.4, 2.0, source="cli")]
    rows = _models(events)
    assert len(rows) == 2
    assert all(r["samples"] == 1 for r in rows.values())


def test_different_serve_configurations_are_kept_apart() -> None:
    # Same model name at a different quant/ctx is a different throughput
    # population; blending them hides real regressions and invents fake ones.
    events = [
        _tok("m", 5.0, 1.0, source="proxy", quant="IQ4_XS", ctx=32768, kv_type="f16"),
        _tok("m", 40.0, 2.0, source="proxy", quant="IQ2_M", ctx=8192, kv_type="q8_0"),
    ]
    assert len(_models(events)) == 2


def test_normal_series_still_aggregates() -> None:
    events = [_tok("m", 4.0, 1.0, source="cli"), _tok("m", 6.0, 2.0, source="cli")]
    (stats,) = _models(events).values()
    assert stats["samples"] == 2
    assert stats["avg"] == 5.0
    assert stats["latest"] == 6.0


def test_a_real_regression_is_still_reported() -> None:
    """The guard must not blunt the detector it protects."""
    events = [_tok("m", 20.0, float(i), source="cli") for i in range(10)]
    events.append(_tok("m", 1.0, 99.0, source="cli"))
    (stats,) = _models(events).values()
    assert stats["degraded"] is True


def test_a_short_generation_outlier_does_not_manufacture_a_regression() -> None:
    """The third defect in this family: even correctly-keyed, ok-status samples
    are duration-blind, so a couple of tiny replies score hundreds of tok/s and
    drag a MEAN baseline far above anything the model really does.

    Live evidence 2026-08-17, reference workload, n=41 at ctx=32768: min 1.7,
    median 13.9, mean 33.2, max 283.8. Against the mean a perfectly ordinary
    17.8 tok/s sample scored "46% down" and raised a `tok_s_drop` advisory
    carrying a real set_kv_size_threshold_mb retune lever. Against the median it
    is not a regression at all.
    """
    # Steady ~14 tok/s, plus two short-reply outliers, ending on a normal sample.
    series = [14.0, 13.0, 15.0, 14.0, 13.5, 254.9, 14.2, 283.8, 13.8, 17.8]
    events = [_tok("m", v, float(i), source="proxy") for i, v in enumerate(series)]
    (stats,) = _models(events).values()
    assert stats["median"] < stats["avg"], "sanity: this series really is skewed"
    assert stats["degraded"] is False, (
        f"17.8 tok/s against a median of {stats['median']} is normal; "
        f"only the outlier-inflated mean ({stats['avg']}) made it look degraded")


def test_median_baseline_is_what_the_advisory_quotes() -> None:
    """Evidence strings must report the statistic the decision actually used —
    quoting `avg` while deciding on `median` is how a reader loses the thread."""
    events = [_tok("m", 20.0, float(i), source="cli") for i in range(10)]
    events.append(_tok("m", 1.0, 99.0, source="cli"))
    adv = gb_pilot.advise(gb_pilot.analyze(events))
    drops = [x for x in adv if x["topic"] == "tok_s_drop"]
    assert drops, "a real 95% drop must still be reported"
    assert "median" in drops[0]["evidence"]
