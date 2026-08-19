"""Task set for the agent benchmark.

Deliberately small and deliberately about THIS harness. A generic agent
benchmark would measure the model; these tasks measure whether the harness
puts the right tools, context and memory in front of it. Each one is scored
on dimensions it actually exercises , see scoring.py.

`requires` names instruments or MCP servers a task needs. A task whose
requirements are absent is SKIPPED, not failed: a box without the dataflux
MCP server has not regressed, it is differently configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalTask:
    id: str
    prompt: str
    #: At least one of these must be called for full tool-selection marks.
    #: A bare MCP tool name also matches its mcp__server__tool form.
    expect_tools: tuple = ()
    #: Calling any of these zeroes tool selection , the wrong instrument for
    #: the job is worse than a clumsy right one.
    forbid_tools: tuple = ()
    #: All must appear (case-insensitive) in the final answer.
    expect_substrings: tuple = ()
    max_tool_calls: int = 8
    requires: tuple = ()
    note: str = ""


#: The default set. Each task states, in `note`, which AE-* item it is the
#: measurement for , a task nobody can explain the purpose of is a task that
#: will not survive its first disagreement.
DEFAULT_TASKS: tuple = (
    EvalTask(
        id="mcp-preferred-over-bash",
        prompt=("What is GreenBoost's current tier pressure and is the kernel "
                "module loaded? Answer from GreenBoost's own tools."),
        expect_tools=("semantic_answer", "greenboost_status", "system_status",
                      "tiering_status"),
        forbid_tools=("Bash",),
        expect_substrings=("kernel module",),
        max_tool_calls=4,
        requires=("mcp",),
        note=("AE-5. The 14-day dataflux baseline recorded 102/124 tool calls "
              "as Bash and ZERO MCP calls. If the model still reaches for "
              "`lsmod` here, the MCP surface is being paid for and not used."),
    ),
    EvalTask(
        id="parallel-read-only-batch",
        prompt=("Read greenboost-cli/greenboost_cli/instruments/concurrency.py "
                "and greenboost_cli/instruments/policy.py and tell me, in one "
                "sentence each, what they gate."),
        expect_tools=("Read",),
        max_tool_calls=4,
        note=("AE-7. Two independent reads should be emitted in ONE batch so "
              "the concurrency path can overlap them."),
    ),
    EvalTask(
        id="plan-survives-compaction",
        prompt=("Make a 4-step todo list for adding a health check to a web "
                "service, then tell me step 3 verbatim."),
        expect_tools=("TodoWrite",),
        expect_substrings=("step 3",),
        max_tool_calls=3,
        note=("AE-3. Baseline recorded 1 TodoWrite in 124 tool calls , the "
              "todo store is barely used, so pinning it is worth nothing "
              "until the model actually writes to it."),
    ),
    EvalTask(
        id="no-invented-tools",
        prompt=("Search this repository for where the prompt cache hit rate is "
                "recorded, and name the file."),
        expect_substrings=("gb_synapse",),
        max_tool_calls=6,
        note="AE-5. Grounding: any Unknown-instrument result is a hallucinated tool.",
    ),
    EvalTask(
        id="bounded-tool-output",
        prompt=("Summarise the last 3 dataflux prompt_cache events in one "
                "line each."),
        expect_tools=("dataflux_events",),
        max_tool_calls=3,
        requires=("mcp",),
        note=("AE-4. A tool that can return 200 events must not be able to "
              "spend the whole context window."),
    ),
)
