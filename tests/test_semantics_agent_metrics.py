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


def test_prefix_preserved_counts_either_signal(monkeypatch):
    """extended_prior OR a non-zero head_kept means the prefix survived."""
    _events(monkeypatch, {"agent_context_edit": [
        {"extended_prior": True, "head_kept": 0},
        {"extended_prior": False, "head_kept": 12},
        {"extended_prior": False, "head_kept": 0},      # this one moved it
    ]})
    assert S.resolve("agent_compaction_prefix_kept_pct")["value"] == pytest.approx(66.7)
    assert S.evaluate_segment("agent_compaction_broke_prefix")["matched"] is True


def test_every_compaction_preserving_the_prefix_does_not_match(monkeypatch):
    _events(monkeypatch, {"agent_context_edit": [
        {"extended_prior": True, "head_kept": 8}] * 5})
    assert S.resolve("agent_compaction_prefix_kept_pct")["value"] == 100.0
    assert S.evaluate_segment("agent_compaction_broke_prefix")["matched"] is False


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
