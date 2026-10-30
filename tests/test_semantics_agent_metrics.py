"""Governed metrics over the agent-evolution dataflux kinds (AE-*).

The plan's cross-cutting requirement is that no AE task lands
observability-blind: every new number an operator would ask about gets a
resolver with its never_use trap named. These pin the part that matters most ,
that each one degrades to "I cannot tell" rather than to a reassuring number
when the event has never been emitted. A clean bill of health inferred from
absent data is the failure mode this layer exists to prevent.
"""
import pytest

import gb_semantics as S


def _events(monkeypatch, mapping):
    monkeypatch.setattr(S, "_agent_events",
                        lambda kind, window_s=None, days=2.0: mapping.get(kind, []))


# ── absence is never good news ───────────────────────────────────────────────

@pytest.mark.parametrize("metric,kind", [
    ("agent_eval_overall", "agent_eval_run"),
    ("agent_hallucinated_tool_pct", "agent_tool_schema_miss"),
    ("agent_compaction_prefix_kept_pct", "agent_context_edit"),
    ("agent_memory_recall_chars", "agent_memory_recall"),
])
def test_no_events_resolves_to_none_naming_the_kind(monkeypatch, metric, kind):
    _events(monkeypatch, {})
    r = S.resolve(metric)
    assert r["value"] is None
    assert kind in r["provenance"]["raw_source"]


@pytest.mark.parametrize("segment", [
    "agent_compaction_broke_prefix",
    "agent_hallucinating_tool_names",
    "agent_eval_below_reference",
])
def test_segments_are_none_not_false_without_data(monkeypatch, segment):
    _events(monkeypatch, {})
    assert S.evaluate_segment(segment)["matched"] is None


# ── the numbers ──────────────────────────────────────────────────────────────

def test_eval_score_carries_the_model_it_came_from(monkeypatch):
    """A score without its model invites reading a model change as a harness win."""
    _events(monkeypatch, {"agent_eval_run": [
        {"overall": 0.41, "model": "old"}, {"overall": 0.63, "model": "cold-fusion"}]})
    r = S.resolve("agent_eval_overall")
    assert r["value"] == 0.63                       # latest, not first
    assert "cold-fusion" in r["provenance"]["raw_source"]


def test_eval_below_the_27b_reference_matches(monkeypatch):
    _events(monkeypatch, {"agent_eval_run": [{"overall": 0.50, "model": "m"}]})
    assert S.evaluate_segment("agent_eval_below_reference")["matched"] is True
    _events(monkeypatch, {"agent_eval_run": [{"overall": 0.63, "model": "m"}]})
    assert S.evaluate_segment("agent_eval_below_reference")["matched"] is False


def test_hallucinated_tool_rate(monkeypatch):
    _events(monkeypatch, {"agent_tool_schema_miss": [
        {"outcome": "resolved"}, {"outcome": "resolved"},
        {"outcome": "unknown_tool"}, {"outcome": "unknown_tool"}]})
    assert S.resolve("agent_hallucinated_tool_pct")["value"] == 50.0
    assert S.evaluate_segment("agent_hallucinating_tool_names")["matched"] is True


def test_a_clean_tool_surface_does_not_match(monkeypatch):
    _events(monkeypatch, {"agent_tool_schema_miss": [{"outcome": "resolved"}] * 40})
    assert S.resolve("agent_hallucinated_tool_pct")["value"] == 0.0
    assert S.evaluate_segment("agent_hallucinating_tool_names")["matched"] is False


def test_prefix_preserved_counts_only_head_kept(monkeypatch):
    """Only head_kept > 0 means the prefix survived.

    This used to accept `extended_prior` as an equivalent signal and expect
    66.7% from the events below. That was wrong in the dangerous direction.
    extended_prior means a prior memory block was absorbed OUT of the middle
    and folded into a freshly generated summary, which places it at a new
    offset , the prefix is rewritten, not preserved.

    The compaction bug found on 2026-08-21 emitted exactly
    {"head_kept": 0, "extended_prior": True} on every restored session, and
    this resolver scored each one as a success while the whole prefix was being
    thrown away. The reported 50% was therefore flattering a worse number.
    """
    _events(monkeypatch, {"agent_context_edit": [
        {"op": "compact", "extended_prior": True, "head_kept": 0},    # REWRITTEN
        {"op": "compact", "extended_prior": False, "head_kept": 12},  # kept
        {"op": "compact", "extended_prior": False, "head_kept": 0},   # REWRITTEN
    ]})
    assert S.resolve("agent_compaction_prefix_kept_pct")["value"] == pytest.approx(33.3)
    assert S.evaluate_segment("agent_compaction_broke_prefix")["matched"] is True


def test_extended_prior_alone_is_not_prefix_preservation(monkeypatch):
    """The exact shape the old resolver mis-scored, isolated."""
    _events(monkeypatch, {"agent_context_edit": [
        {"op": "compact", "extended_prior": True, "head_kept": 0}] * 4})
    assert S.resolve("agent_compaction_prefix_kept_pct")["value"] == 0.0
    assert S.evaluate_segment("agent_compaction_broke_prefix")["matched"] is True


def test_every_compaction_preserving_the_prefix_does_not_match(monkeypatch):
    _events(monkeypatch, {"agent_context_edit": [
        {"op": "compact", "extended_prior": True, "head_kept": 8}] * 5})
    assert S.resolve("agent_compaction_prefix_kept_pct")["value"] == 100.0
    assert S.evaluate_segment("agent_compaction_broke_prefix")["matched"] is False


# ── four ops share one kind, and only one of them is a compaction ────────────
#
# Not a hypothetical. On 2026-08-22 the dataflux log contained exactly ONE
# agent_context_edit event in its entire history: an `op=calibrate` emitted at
# 2026-08-21 23:14:37. The resolver had no op filter, counted it as a
# compaction, found no head_kept, and reported 0.0% , which is what casebook
# C-09 recorded as a destroyed prefix. No compaction had happened.

def test_a_calibration_is_not_a_compaction(monkeypatch):
    """The exact event that produced C-09's headline number."""
    _events(monkeypatch, {"agent_context_edit": [
        {"op": "calibrate", "actual_prompt_tokens": 5499,
         "estimated_prompt_tokens": 5157, "estimate_error_pct": -6.22},
    ]})
    r = S.resolve("agent_compaction_prefix_kept_pct")
    assert r["value"] is None, "a calibration must never score as a compaction"
    assert "no compaction" in r["provenance"]["raw_source"]
    assert r["details"]["compactions"] == 0
    assert r["details"]["context_edits"] == 1
    # And the segment must stay silent rather than cry violation.
    assert S.evaluate_segment("agent_compaction_broke_prefix")["matched"] is None


@pytest.mark.parametrize("op", ["calibrate", "microcompact", "hard_trim"])
def test_no_non_compact_op_counts(monkeypatch, op):
    _events(monkeypatch, {"agent_context_edit": [{"op": op, "head_kept": 0}] * 3})
    assert S.resolve("agent_compaction_prefix_kept_pct")["value"] is None


def test_non_compactions_do_not_dilute_the_denominator(monkeypatch):
    """Two real compactions, both good, buried in unrelated context edits."""
    _events(monkeypatch, {"agent_context_edit": [
        {"op": "calibrate", "estimate_error_pct": -6.22},
        {"op": "compact", "head_kept": 2},
        {"op": "microcompact", "chars_freed": 900},
        {"op": "compact", "head_kept": 2},
        {"op": "hard_trim", "met": False},
    ]})
    # 2/2 on the proxy, not 2/5.
    assert S.resolve("agent_compaction_prefix_kept_pct")["value"] == 100.0
    assert S.evaluate_segment("agent_compaction_broke_prefix")["matched"] is False


# ── measured beats proxy, and says which it used ─────────────────────────────

def test_measured_prefix_pct_is_preferred_over_the_head_kept_proxy(monkeypatch):
    """head_kept says the head survived; prefix_kept_pct says how much did.

    The two disagree on purpose here: head_kept is non-zero on every event, so
    the proxy would report a flattering 100%, while the measured share is 40%.
    """
    _events(monkeypatch, {"agent_context_edit": [
        {"op": "compact", "head_kept": 2, "prefix_kept_pct": 30.0, "tokens_lost": 700},
        {"op": "compact", "head_kept": 2, "prefix_kept_pct": 50.0, "tokens_lost": 500},
    ]})
    r = S.resolve("agent_compaction_prefix_kept_pct")
    assert r["value"] == pytest.approx(40.0)
    assert r["details"]["basis"] == "measured"
    assert r["details"]["tokens_lost"] == 1200
    assert "measured prefix_kept_pct" in r["provenance"]["raw_source"]


def test_the_proxy_path_says_it_is_a_proxy(monkeypatch):
    """raw_source is what a reader uses to tell an estimate from a measurement."""
    _events(monkeypatch, {"agent_context_edit": [{"op": "compact", "head_kept": 4}]})
    r = S.resolve("agent_compaction_prefix_kept_pct")
    assert r["details"]["basis"] == "proxy"
    assert "PROXY" in r["provenance"]["raw_source"]


def test_measured_segment_fires_on_the_guard_threshold_not_on_imperfection(monkeypatch):
    """Every compaction moves something, so `< 100` would fire on the cheap case."""
    from gb_prefix_guard import BROKE_PREFIX_PCT
    _events(monkeypatch, {"agent_context_edit": [
        {"op": "compact", "head_kept": 2, "prefix_kept_pct": BROKE_PREFIX_PCT + 20.0},
    ]})
    assert S.evaluate_segment("agent_compaction_broke_prefix")["matched"] is False
    _events(monkeypatch, {"agent_context_edit": [
        {"op": "compact", "head_kept": 2, "prefix_kept_pct": BROKE_PREFIX_PCT - 20.0},
    ]})
    assert S.evaluate_segment("agent_compaction_broke_prefix")["matched"] is True


def test_memory_recall_chars_reads_the_latest(monkeypatch):
    _events(monkeypatch, {"agent_memory_recall": [{"chars": 100}, {"chars": 4200}]})
    assert S.resolve("agent_memory_recall_chars")["value"] == 4200


def test_a_missing_field_is_reported_as_missing_not_zero(monkeypatch):
    """An event without the field must not read as 0 , that is a real value."""
    _events(monkeypatch, {"agent_eval_run": [{"model": "m"}],
                          "agent_memory_recall": [{"n_items": 3}]})
    for metric in ("agent_eval_overall", "agent_memory_recall_chars"):
        r = S.resolve(metric)
        assert r["value"] is None
        assert "no" in r["provenance"]["raw_source"].lower()


# ── the questions route ──────────────────────────────────────────────────────

@pytest.mark.parametrize("question", [
    "is the model making up tool names?",
    "the agent benchmark score",
    "why is it slow after compaction?",
    "the prompt cache dropped",
])
def test_natural_questions_reach_a_governed_answer(question):
    assert S.answer(question).get("governed") is True
