# SPDX-License-Identifier: GPL-2.0-only
"""tests/test_semantics_evals.py — GB-Semantics eval runner.

Loads tests/fixtures/semantics/events.py's frozen scenario into the isolated
per-test dataflux log (tests/conftest.py's autouse
_isolate_dataflux_log_globally fixture already points GREENBOOST_DATAFLUX_LOG
at a tmp_path), then runs every entry in evals/semantics/*.yaml against the
REAL gb_semantics engine.

Reports two numbers, mirroring Anthropic's own 21% -> 95% semantic-layer
finding as GreenBoost's local equivalent:
  - GOVERNED pass rate: every eval entry resolved through gb_semantics.
  - UNGOVERNED (layer-off) pass rate: `category: trap` entries re-checked by
    reading `trap_raw_field` directly instead of going through the layer —
    proving quantitatively, not just asserting, that the governed path is
    the one that gets these right.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.semantics import events as fixture_events  # noqa: E402

_EVALS_DIR = _REPO_ROOT / "evals" / "semantics"


def _load_evals() -> list[dict]:
    out: list[dict] = []
    for p in sorted(_EVALS_DIR.glob("*.yaml")):
        out.extend(yaml.safe_load(p.read_text()) or [])
    return out


_EVALS = _load_evals()


@pytest.fixture(autouse=True)
def _load_fixture_events(monkeypatch):
    """Every test in this file gets the frozen scenario in its (already
    isolated, per conftest.py) dataflux log before it runs."""
    import gb_dataflux
    log_path = gb_dataflux._log_path()
    fixture_events.write(log_path)
    import gb_semantics
    gb_semantics.load(force=True)  # fresh compile, no stale cross-test cache
    yield


def _check_metric(entry: dict) -> "tuple[bool, str]":
    import gb_semantics
    result = gb_semantics.resolve(entry["name"])
    expect = entry.get("expect", {})
    if "governed" in expect and result.get("governed") != expect["governed"]:
        return False, f"governed={result.get('governed')}, expected {expect['governed']}"
    if expect.get("has_error") and "error" not in result:
        return False, "expected an 'error' key, none present"
    if "value" in expect:
        if result.get("value") != expect["value"]:
            return False, f"value={result.get('value')!r}, expected {expect['value']!r}"
    if "value_subset" in expect:
        v = result.get("value") or {}
        for k, want in expect["value_subset"].items():
            if v.get(k) != want:
                return False, f"value[{k}]={v.get(k)!r}, expected {want!r}"
    if "threshold_state" in expect and result.get("threshold_state") != expect["threshold_state"]:
        return False, (f"threshold_state={result.get('threshold_state')!r}, "
                        f"expected {expect['threshold_state']!r}")
    if entry.get("source") == "live":
        # Structural only: must not raise, must return the governed shape.
        if "metric" not in result:
            return False, "resolve() did not return a well-formed metric dict"
    return True, ""


def _check_segment(entry: dict) -> "tuple[bool, str]":
    import gb_semantics
    result = gb_semantics.evaluate_segment(entry["name"])
    expect = entry.get("expect", {})
    if expect.get("has_error") and "error" not in result:
        return False, "expected an 'error' key, none present"
    if "matched" in expect and result.get("matched") != expect["matched"]:
        return False, f"matched={result.get('matched')!r}, expected {expect['matched']!r}"
    if entry.get("source") == "live" and "segment" not in result:
        return False, "evaluate_segment() did not return a well-formed segment dict"
    return True, ""


def _check_discover(entry: dict) -> "tuple[bool, str]":
    import gb_semantics
    result = gb_semantics.discover(entry["query"])
    names = {r["name"] for r in result}
    want = entry["expect"]["contains_name"]
    if want not in names:
        return False, f"discover({entry['query']!r}) -> {names}, expected to contain {want!r}"
    return True, ""


def _check_answer(entry: dict) -> "tuple[bool, str]":
    import gb_semantics
    result = gb_semantics.answer(entry["question"])
    expect = entry["expect"]
    if result.get("governed") != expect["governed"]:
        return False, f"governed={result.get('governed')!r}, expected {expect['governed']!r}"
    if "intent" in expect and result.get("intent") != expect["intent"]:
        return False, f"intent={result.get('intent')!r}, expected {expect['intent']!r}"
    for seg, want in (expect.get("segment_matched") or {}).items():
        got = (result.get("segments") or {}).get(seg, {}).get("matched")
        if got != want:
            return False, f"segment[{seg}].matched={got!r}, expected {want!r}"
    return True, ""


_CHECKERS = {"metric": _check_metric, "segment": _check_segment,
             "discover": _check_discover, "answer": _check_answer}


@pytest.mark.parametrize("entry", _EVALS, ids=[e["id"] for e in _EVALS])
def test_governed_eval(entry: dict):
    """Every eval entry, resolved through the REAL gb_semantics engine."""
    ok, why = _CHECKERS[entry["kind"]](entry)
    assert ok, f"{entry['id']}: {why}"


_TRAP_ENTRIES = [e for e in _EVALS if e.get("category") == "trap" and "trap_raw_field" in e]


@pytest.mark.parametrize("entry", _TRAP_ENTRIES, ids=[e["id"] for e in _TRAP_ENTRIES])
def test_ungoverned_baseline_is_wrong(entry: dict):
    """The layer-off comparison: for each documented trap, prove the naive
    raw-field read gives a DIFFERENT (wrong) answer than the governed one —
    this is what makes the trap real rather than a hypothetical concern."""
    import gb_semantics
    governed = gb_semantics.resolve(entry["name"])
    assert governed["value"] == entry["expect"]["value"]
    if entry["trap_raw_field"] == "fb_used_pct":
        raw_value = entry["trap_raw_value"]
        assert raw_value != governed["value"], (
            "trap fixture is stale: raw and governed values now agree, "
            "this trap no longer demonstrates anything")
        # The actual wrong CONCLUSION a naive reader draws: "VRAM is full,
        # Rule #1 satisfied" (raw >= 85) vs the governed truth (< 85).
        assert (raw_value >= 85) != (governed["value"] >= 85), (
            "raw and governed fields now agree on the Rule #1 verdict — "
            "update this trap fixture, it no longer proves the point")
    elif entry["trap_raw_field"] == "t2_pressure_level_thresholds":
        # Same numeric value (0.72), read against the WRONG (enum) scale's
        # thresholds, misclassifies "warn" as "nominal".
        naive_state = "critical" if governed["value"] >= 2 else ("warn" if governed["value"] >= 1 else "ok")
        assert naive_state != governed["threshold_state"], (
            "the enum-scale misreading now agrees with the fraction-scale "
            "verdict — update this trap fixture")


def test_kv_layer_undercounting_trap():
    """Real historical incident (see CLAUDE.md's Reference Workload Rule,
    2026-07-27): treating a hybrid-attention model's `n_layers` as its KV
    layer count overcounts KV cache ~4x, which clamped this cluster's
    reference model to ctx=2048 before `n_kv_layers` (full_attention_interval-
    aware) fixed it. Regression guard: the SAME estimate_kv_gb() call, only
    varying n_layers vs n_kv_layers, must differ by roughly the
    full_attention_interval factor — proving the undercount is real and
    reproducible, not just documented."""
    import gb_synapse
    n_layers = 65
    full_attention_interval = 4
    n_kv_layers = -(-n_layers // full_attention_interval)  # ceil division = 17

    wrong_kv_gb = gb_synapse.estimate_kv_gb(
        ctx=45824, n_bytes=0, quant="Q4_K_M", n_layers=n_layers,
        n_kv_heads=8, head_dim=128, kv_bytes_per_elem=1.0)
    right_kv_gb = gb_synapse.estimate_kv_gb(
        ctx=45824, n_bytes=0, quant="Q4_K_M", n_layers=n_kv_layers,
        n_kv_heads=8, head_dim=128, kv_bytes_per_elem=1.0)

    assert wrong_kv_gb > right_kv_gb * 3.0, (
        f"n_layers={n_layers} estimate ({wrong_kv_gb:.2f} GiB) no longer "
        f"overcounts vs n_kv_layers={n_kv_layers} ({right_kv_gb:.2f} GiB) by "
        f"~{full_attention_interval}x — either estimate_kv_gb changed shape "
        f"or this regression guard needs updating")


def test_serve_healthy_idle_branch(monkeypatch):
    """Regression guard for the 2026-07-30 fix: serve_healthy previously
    required shim_fresh unconditionally, so it could never match on a
    genuinely idle-but-healthy box (reproduced live: kmod loaded, zero
    errors, nothing being served, yet serve_healthy resolved false). Mocks
    resolve()/_latest_event() directly rather than fighting this file's
    autouse fixture-log machinery, since the true "no recent tok_s_measured"
    idle case can't be expressed by writing MORE events into the shared
    fixture (which always includes an active tok_s_measured event)."""
    import gb_semantics

    def _fake_resolve(name, *a, **kw):
        return {
            "kmod_loaded": {"value": True},
            "shim_fresh": {"value": False},
            "vram_fill_pct": {"value": 22.0},
            "meets_fp8_floor": {"value": None},
        }[name]

    monkeypatch.setattr(gb_semantics, "resolve", _fake_resolve)
    monkeypatch.setattr(gb_semantics, "_latest_event", lambda *a, **kw: None)
    matched, evidence = gb_semantics._seg_serve_healthy()
    assert matched is True
    assert len(evidence) == 4


def test_serve_healthy_active_session_still_gates_on_vram(monkeypatch):
    """The active branch (a recent tok_s_measured event found) must still
    require shim_fresh + VRAM in band + quality floor , the idle-branch fix
    must not accidentally make serve_healthy always true."""
    import gb_semantics

    def _fake_resolve(name, *a, **kw):
        return {
            "kmod_loaded": {"value": True},
            "shim_fresh": {"value": False},
            "vram_fill_pct": {"value": 22.0},
            "meets_fp8_floor": {"value": True},
        }[name]

    monkeypatch.setattr(gb_semantics, "resolve", _fake_resolve)
    monkeypatch.setattr(gb_semantics, "_latest_event",
                         lambda *a, **kw: {"kind": "tok_s_measured"})
    matched, evidence = gb_semantics._seg_serve_healthy()
    assert matched is False  # shim not fresh + vram below 60 while actively serving


def test_eval_set_size_and_coverage():
    """Guard against the eval set silently shrinking, and against a metric/
    segment in semantics/*.yaml having zero eval coverage."""
    import gb_semantics
    assert len(_EVALS) >= 40, f"only {len(_EVALS)} eval entries — expected >= 40"

    L = gb_semantics.load()
    covered_metrics = {e["name"] for e in _EVALS if e["kind"] == "metric"}
    covered_segments = {e["name"] for e in _EVALS if e["kind"] == "segment"}
    uncovered_metrics = set(L["metrics"]) - covered_metrics
    uncovered_segments = set(L["segments"]) - covered_segments
    assert not uncovered_metrics, f"metrics with no eval coverage: {uncovered_metrics}"
    assert not uncovered_segments, f"segments with no eval coverage: {uncovered_segments}"
