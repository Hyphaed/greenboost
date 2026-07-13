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
    prompt = _cap(" ".join(rest).strip(), _MAX_TEXT_LEN)
    if not prompt:
        emit_warn("Usage: /agent [--model M] [--timeout S] <prompt>")
        return

    from greenboost_cli.agents.subagent import run_subagent

    console.print(f"  [{VIOLET}]▶[/] Subagent starting "
                  f"[{GRAY}](model={model or settings.get('model','?')}, "
                  f"timeout={int(timeout_s)}s)[/]")
    result = run_subagent(
        prompt, model=model, timeout_s=timeout_s, settings=settings,
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
