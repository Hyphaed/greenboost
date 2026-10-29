"""tools/compare_goldens.py — score a measurement set against pinned goldens.

Ported from Megatron-LM's update-golden-values skill (crosswalk P5-10). The
scoring idea transfers; the GitHub-Actions plumbing around it does not, because
GreenBoost's numbers come off this box rather than off CI artifacts.

The behaviours worth pinning are the ones that decide whether a number is
trustworthy: an improvement must never fail, a mismatched metric set must never
report a pass, and direction must be per-metric because this repo mixes
latencies (lower is better) with cosines and speedups (higher is better).
"""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "compare_goldens", Path(__file__).resolve().parents[1] / "tools" / "compare_goldens.py")
cg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cg)


def _g(metrics, direction=None):
    return {"metrics": metrics, "direction": direction or {}}


def test_identical_sets_have_no_regressions():
    m = {"gemm_ms": {"M512": 0.046}}
    r = cg.compare(_g(m, {"gemm_ms.M512": "lower_is_better"}), _g(m))
    assert r["regressions"] == []
    assert r["avg_rel_diff"] == 0.0


def test_getting_slower_regresses_a_lower_is_better_metric():
    r = cg.compare(_g({"t": {"a": 1.0}}, {"t.a": "lower_is_better"}),
                   _g({"t": {"a": 1.5}}), tolerance=0.10)
    assert [x["metric"] for x in r["regressions"]] == ["t.a"]


def test_getting_faster_never_regresses():
    """A kernel that got 5x faster must not fail its own golden."""
    r = cg.compare(_g({"t": {"a": 1.0}}, {"t.a": "lower_is_better"}),
                   _g({"t": {"a": 0.2}}), tolerance=0.10)
    assert r["regressions"] == []


def test_direction_is_per_metric():
    """Latencies and cosines live in the same file and move opposite ways."""
    golden = _g({"ms": {"x": 1.0}, "cos": {"x": 0.99}},
                {"ms.x": "lower_is_better", "cos.x": "higher_is_better"})
    r = cg.compare(golden, _g({"ms": {"x": 0.5}, "cos": {"x": 0.5}}), tolerance=0.10)
    assert [x["metric"] for x in r["regressions"]] == ["cos.x"]


def test_within_tolerance_is_not_a_regression():
    r = cg.compare(_g({"t": {"a": 1.0}}, {"t.a": "lower_is_better"}),
                   _g({"t": {"a": 1.05}}), tolerance=0.10)
    assert r["regressions"] == []


def test_metric_set_mismatch_is_reported_separately():
    """A missing metric is neither a pass nor a regression — it means the two
    files do not describe the same thing, and must not be scored as agreement."""
    r = cg.compare(_g({"t": {"a": 1.0, "b": 2.0}}), _g({"t": {"a": 1.0, "c": 3.0}}))
    assert r["missing_in_measured"] == ["t.b"]
    assert r["extra_in_measured"] == ["t.c"]


def test_zero_golden_does_not_divide_by_zero():
    r = cg.compare(_g({"t": {"a": 0.0}}), _g({"t": {"a": 0.0}}))
    assert r["rows"][0]["rel_diff"] == 0.0
    r2 = cg.compare(_g({"t": {"a": 0.0}}), _g({"t": {"a": 1.0}}))
    assert r2["rows"][0]["rel_diff"] == float("inf")


def test_booleans_are_not_treated_as_numbers():
    r = cg.compare(_g({"t": {"ok": True, "a": 1.0}}), _g({"t": {"ok": True, "a": 1.0}}))
    assert [x["metric"] for x in r["rows"]] == ["t.a"]


# ── the shipped golden file itself ───────────────────────────────────────────

_GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden" / "nvfp4_gemm.json"


def test_shipped_golden_declares_a_direction_for_every_metric():
    """A metric with no declared direction silently defaults to
    higher_is_better, which would score a latency backwards."""
    import json
    doc = json.loads(_GOLDEN.read_text())
    flat = cg._flat(doc["metrics"])
    undeclared = sorted(set(flat) - set(doc["direction"]))
    assert not undeclared, f"metrics with no declared direction: {undeclared}"


def test_shipped_golden_records_how_it_was_measured():
    """A golden without provenance is a number nobody can re-derive."""
    import json
    prov = json.loads(_GOLDEN.read_text())["_provenance"]
    for field in ("case", "measured", "box", "producer", "method",
                  "tolerance_rationale"):
        assert prov.get(field), f"golden provenance missing {field}"
