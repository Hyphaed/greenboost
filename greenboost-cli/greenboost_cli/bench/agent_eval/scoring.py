"""Scoring for the agent benchmark. Pure , no model, no network, no MCP.

Five dimensions, each in [0,1], because a single number hides the thing you
need to know. A run can get faster by doing less, or select tools perfectly
and never answer; the dimensions keep those apart.

The overall score is the mean of the dimensions that APPLY to a task. A task
that declares no expected tools is not scored on tool selection , averaging in
a free 1.0 for a dimension a task never exercised inflates every comparison
that follows.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

#: dispatcher.py answers a call to a tool that does not exist with this exact
#: prefix. It is the only reliable in-band signal that the model invented a
#: tool name rather than using one it was shown.
HALLUCINATION_MARKER = "Error: Unknown instrument:"


@dataclass
class TaskScore:
    task_id: str
    completion: float | None = None
    tool_selection: float | None = None
    efficiency: float | None = None
    grounding: float | None = None          # 1 - hallucinated tool-call rate
    overall: float = 0.0
    tool_calls: int = 0
    hallucinated: int = 0
    tokens: int = 0
    duration_s: float = 0.0
    errored: bool = False
    skipped_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunScore:
    scores: list = field(default_factory=list)
    overall: float = 0.0
    completion: float = 0.0
    tool_selection: float = 0.0
    efficiency: float = 0.0
    grounding: float = 0.0
    tasks_run: int = 0
    tasks_skipped: int = 0
    total_tokens: int = 0
    total_duration_s: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scores"] = [s.to_dict() if hasattr(s, "to_dict") else s for s in self.scores]
        return d


def _mean(values) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def score_task(task, trace: dict) -> TaskScore:
    """Score one task against one run's trace.

    `trace` is the shape `run_eval` builds from a SubagentResult:
        {"summary": str, "tool_calls": [{"name","result"}...],
         "tokens": int, "duration_s": float, "error": str}
    """
    s = TaskScore(task_id=task.id)
    calls = trace.get("tool_calls") or []
    names = [c.get("name", "") for c in calls]
    s.tool_calls = len(calls)
    s.tokens = int(trace.get("tokens") or 0)
    s.duration_s = float(trace.get("duration_s") or 0.0)
    s.errored = bool(trace.get("error"))

    # Grounding: a call whose RESULT is the unknown-instrument error is a tool
    # the model invented. Always applicable — even a task expecting no tools.
    s.hallucinated = sum(
        1 for c in calls if str(c.get("result", "")).startswith(HALLUCINATION_MARKER))
    s.grounding = 1.0 - (s.hallucinated / len(calls)) if calls else 1.0

    # Completion: every required substring present in the answer.
    if task.expect_substrings:
        text = (trace.get("summary") or "").lower()
        hits = sum(1 for sub in task.expect_substrings if sub.lower() in text)
        s.completion = hits / len(task.expect_substrings)

    # Tool selection: used one of the expected tools, minus forbidden usage.
    if task.expect_tools or task.forbid_tools:
        used_expected = (
            any(_matches_any(n, task.expect_tools) for n in names)
            if task.expect_tools else True)
        used_forbidden = any(_matches_any(n, task.forbid_tools) for n in names)
        s.tool_selection = (1.0 if used_expected else 0.0) * (0.0 if used_forbidden else 1.0)

    # Efficiency: under the call budget is full marks; over it decays to 0.
    if task.max_tool_calls > 0:
        over = max(0, s.tool_calls - task.max_tool_calls)
        s.efficiency = max(0.0, 1.0 - over / task.max_tool_calls)

    if s.errored:
        # An errored run scores zero on everything it was measured on rather
        # than being dropped — a change that makes runs crash must not look
        # like a change that made them faster.
        for f in ("completion", "tool_selection", "efficiency", "grounding"):
            if getattr(s, f) is not None:
                setattr(s, f, 0.0)

    s.overall = _mean([s.completion, s.tool_selection, s.efficiency, s.grounding])
    return s


def _matches_any(name: str, patterns) -> bool:
    """Exact name, or a bare MCP tool name against its mcp__server__tool form."""
    for p in patterns:
        if name == p:
            return True
        if name.startswith("mcp__") and name.endswith(f"__{p}"):
            return True
    return False


def aggregate(scores) -> RunScore:
    run = RunScore(scores=list(scores))
    scored = [s for s in scores if not s.skipped_reason]
    run.tasks_run = len(scored)
    run.tasks_skipped = len(scores) - len(scored)
    run.total_tokens = sum(s.tokens for s in scored)
    run.total_duration_s = sum(s.duration_s for s in scored)
    run.completion = _mean([s.completion for s in scored])
    run.tool_selection = _mean([s.tool_selection for s in scored])
    run.efficiency = _mean([s.efficiency for s in scored])
    run.grounding = _mean([s.grounding for s in scored])
    run.overall = _mean([s.overall for s in scored])
    return run
