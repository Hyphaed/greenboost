"""Agent evaluation: does a change to this harness actually help?

Every claim about harness quality in this repo was unfalsifiable before this
package existed. AE-1 in workflow/tasks-agent-evolution.md is the reason it
does: a benchmark first, so the context and tool-surface work that follows can
be measured instead of asserted.

The scoring half (`scoring.py`) is pure and has no model, no network and no
MCP dependency, so it is testable in the ordinary suite. The runner half
(`runner.py`) drives the real served model through the real turn path.
"""
from greenboost_cli.bench.agent_eval.tasks import EvalTask, DEFAULT_TASKS
from greenboost_cli.bench.agent_eval.scoring import (
    TaskScore, RunScore, score_task, aggregate, HALLUCINATION_MARKER,
)

__all__ = [
    "EvalTask", "DEFAULT_TASKS",
    "TaskScore", "RunScore", "score_task", "aggregate", "HALLUCINATION_MARKER",
]
