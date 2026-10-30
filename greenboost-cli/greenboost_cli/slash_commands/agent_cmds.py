"""Subagent slash command: /agent <prompt> runs a sub-task in an isolated context."""
from __future__ import annotations

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import (
    console, GRAY, VIOLET, LIME, AMBER,
    emit_warn, emit_err, emit_ok, emit_info,
)
from greenboost_cli.security import _cap, _MAX_TEXT_LEN


def _agent(args: str, session, settings: dict) -> None:
    """Run a subagent and pretty-print its result.

    Usage: /agent <prompt>          – uses current model, 600s timeout
           /agent --model M <p>     – override model
           /agent --timeout 120 <p> – override timeout (seconds)
    """
    raw = args.strip()
    if not raw:
        emit_warn("Usage: /agent [--model M] [--timeout S] <prompt>")
        return

    model = None
    timeout_s = 600.0
    tokens = raw.split()
    rest: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--model" and i + 1 < len(tokens):
            model = tokens[i + 1]; i += 2
        elif tok == "--timeout" and i + 1 < len(tokens):
            try:
                timeout_s = float(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            rest.append(tok); i += 1
    # `/agent reviewer <task>` runs a markdown-defined agent (.greenboost/agents)
    from greenboost_cli.agents import custom as _custom
    _defs = _custom.discover()
    _named = None
    if rest and rest[0].lower() in _defs and len(rest) > 1:
        _named = _defs[rest.pop(0).lower()]
    prompt = _cap(" ".join(rest).strip(), _MAX_TEXT_LEN)
    if not prompt:
        emit_warn("Usage: /agent [--model M] [--timeout S] [<agent-name>] <prompt>   (/agents lists the named ones)")
        return
    _tools = None
    if _named:
        _tools, _unknown = _custom.usable_tools(_named.tools)
        if _unknown:
            emit_warn(f"{_named.name}: ignoring unknown tools {', '.join(_unknown)}")
        prompt = _custom.build_prompt(_named, prompt)

    from greenboost_cli.agents.subagent import run_subagent

    console.print(f"  [{VIOLET}]▶[/] Subagent starting "
                  f"[{GRAY}](model={model or settings.get('model','?')}, "
                  f"timeout={int(timeout_s)}s)[/]")
    result = run_subagent(
        prompt, model=model, timeout_s=timeout_s, settings=settings, tools=_tools,
        label=(_named.name if _named else "subagent"),
    )

    if result.error:
        emit_err(f"Subagent error: {result.error}")
    if result.timed_out:
        emit_warn(f"Subagent timed out after {int(timeout_s)}s — partial output below.")

    if result.summary:
        console.print(f"\n  [{LIME}]Summary[/] [{GRAY}]({len(result.summary)} chars)[/]")
        for line in result.summary.splitlines():
            console.print(f"    {line}")
    else:
        console.print(f"  [{AMBER}](no assistant text returned)[/]")

    if result.tool_calls:
        console.print(f"\n  [{LIME}]Tool calls[/] [{GRAY}]({len(result.tool_calls)})[/]")
        for tc in result.tool_calls:
            console.print(f"    [{VIOLET}]{tc.name}[/]  "
                          f"[{GRAY}]permitted={tc.permitted}[/]")
    console.print(f"\n  [{GRAY}]tokens={result.tokens_used}  "
                  f"duration={result.duration_s:.2f}s[/]\n")


register_command("agent", _agent, "Delegate a sub-task to an isolated subagent  (/agent <prompt>)")


def _agents(args: str, session, settings: dict) -> None:
    """List markdown-defined agents."""
    from rich.markup import escape
    from greenboost_cli.agents import custom
    defs = custom.discover()
    if not defs:
        emit_info("No agents found. Put name.md (front matter: description, tools) in "
                  ".greenboost/agents/")
        return
    for name, a in sorted(defs.items()):
        tools = ", ".join(a.tools) if a.tools else "all tools"
        console.print(f"  [{VIOLET}]{escape(name):<16}[/] [{GRAY}]{escape(a.description or '')}  ({escape(tools)})[/]")
register_command("agents", _agents, "List user-defined agents  (.greenboost/agents)")
