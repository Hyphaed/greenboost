"""Where did the run go wrong? Answer it from the trace, not from memory.

The autonomy journal already records every tool, every skill, every
auto-answered question and every continue/stop decision. Dataflux records the
tool-call audit trail, the schema misses and the context edits. All of it was
being written and none of it was being read back after a bad run.

The method is the one automated trajectory diagnosis converges on: first cut
the failure-irrelevant noise, then localise. A 300-step trace where 290 steps
succeeded is not 300 steps of evidence , it is ~10, plus the step where the
run stopped making progress.

Deliberately rule-based. A diagnosis that needs the model to be working is
useless in exactly the case it is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

#: A tool result starting with this is a failure , the same prefix the
#: orchestrator's consecutive-error guard and the renderer key off.
_ERROR_PREFIX = "Error"

#: Repeating the identical call this many times is a loop, not persistence.
_REPEAT_LOOP = 3


@dataclass
class Finding:
    kind: str
    detail: str
    at_step: int = -1
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _steps(journal) -> list:
    """Journal entries that represent an action, in order."""
    return [e for e in journal if getattr(e, "kind", "") in ("tool", "skill")]


def diagnose(state, events=None) -> list:
    """Findings for one run, most specific first.

    `state` is an AutonomyState; `events` is an optional dataflux event list
    (already filtered to the run's window) , when absent, the journal alone is
    used, so this works with telemetry unavailable.
    """
    findings: list = []
    journal = list(getattr(state, "journal", []) or [])
    steps = _steps(journal)

    # 1. The identical call, over and over. The strongest single signal that a
    #    run stopped making progress while still looking busy.
    seen: dict = {}
    for i, e in enumerate(steps):
        key = f"{e.kind}:{(e.detail or {}).get('name', '')}"
        seen.setdefault(key, []).append(i)
    for key, idxs in seen.items():
        if len(idxs) >= _REPEAT_LOOP:
            runs, cur = [], [idxs[0]]
            for a, b in zip(idxs, idxs[1:]):
                (cur.append(b) if b == a + 1 else (runs.append(cur), cur.clear(), cur.append(b)))
            runs.append(cur)
            longest = max(runs, key=len)
            if len(longest) >= _REPEAT_LOOP:
                findings.append(Finding(
                    kind="repeated_call",
                    detail=(f"{key.split(':', 1)[1] or key} was called "
                            f"{len(longest)} times in a row , the run was "
                            f"repeating itself, not advancing"),
                    at_step=longest[0]))

    # 2. Why it actually ended. The stop reason is recorded; a run that ended
    #    on the stall detector or the ceiling did NOT finish its work.
    stops = [e for e in journal if getattr(e, "kind", "") == "stop"]
    if stops:
        reason = (stops[-1].detail or {}).get("reason", "")
        if "circles" in reason:
            findings.append(Finding(
                kind="stalled",
                detail="the run was stopped by the stall detector , it kept "
                       "taking turns while producing almost no new output"))
        elif "ceiling" in reason:
            findings.append(Finding(
                kind="hit_ceiling",
                detail="the run hit the continue ceiling , it was still "
                       "working when it was cut off, so the task is unfinished"))

    # 3. Questions answered without a human, which is where an unattended run
    #    most often diverges from what the user would have chosen.
    qs = [e for e in journal if getattr(e, "kind", "") == "question"]
    if qs:
        findings.append(Finding(
            kind="auto_answered",
            detail=f"{len(qs)} question(s) were answered without you , if the "
                   f"result is wrong, check these first",
            evidence=[(e.detail or {}).get("why", "") for e in qs][:5]))

    # 4. Telemetry, when available: invented tool names and error clusters.
    for ev in (events or ()):
        if ev.get("kind") == "agent_tool_schema_miss" and ev.get("outcome") == "unknown_instrument":
            findings.append(Finding(
                kind="invented_tool",
                detail=f"called a tool that does not exist: "
                       f"{ev.get('requested', '?')}"))
    errs = [ev for ev in (events or ()) if str(ev.get("status", "")) == "error"]
    if len(errs) >= 3:
        findings.append(Finding(
            kind="error_cluster",
            detail=f"{len(errs)} error events in this run's window",
            evidence=[str(e.get("kind", "")) for e in errs[:5]]))

    if not findings and steps:
        findings.append(Finding(
            kind="no_obvious_failure",
            detail=f"{len(steps)} step(s), no repeated calls, no stall, no "
                   f"invented tools , if the result is still wrong the cause "
                   f"is in what the steps DID, not in how the run was driven"))
    return findings


def render(findings) -> str:
    if not findings:
        return "Nothing to diagnose , no actions were recorded for this run."
    lines = []
    for f in findings:
        where = f" (step {f.at_step})" if f.at_step >= 0 else ""
        lines.append(f"- {f.detail}{where}")
        for e in f.evidence:
            if e:
                lines.append(f"    · {e}")
    return "\n".join(lines)
