"""/roles : which local model to use for which job (environment/models.yaml)."""
from __future__ import annotations

from greenboost_cli.environment.models import escalation_chain, load_catalog, resolve_role
from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import AMBER, CYAN, DIM, GRAY, LIME, console


def _roles(args: str, session, settings: dict) -> None:
    cat = load_catalog()
    roles = cat.get("roles") or {}
    if not roles:
        console.print(f"  [{AMBER}]No model catalog found.[/]")
        return
    wanted = args.strip()
    if wanted:
        entry = resolve_role(wanted, cat)
        if not entry:
            console.print(f"  [{AMBER}]Unknown role '{wanted}'. Roles: {', '.join(roles)}[/]")
            return
        names = [wanted]
    else:
        names = list(roles)
    chain = [e["role"] for e in escalation_chain(cat)]
    for name in names:
        e = resolve_role(name, cat)
        tag = f"{e.get('status', '?')}" + ("  (env override)" if e.get("override") else "")
        console.print(f"\n  [{LIME}]{name}[/]  [{CYAN}]{e.get('model', '')}[/]  [{DIM}]{tag}[/]")
        if e.get("summary"):
            console.print(f"    {e['summary']}")
        for key, label in (("use_when", "use when"), ("avoid_when", "avoid when")):
            for line in e.get(key) or []:
                console.print(f"    [{GRAY}]{label}:[/] {line}")
        m = e.get("measured") or {}
        if m:
            console.print(f"    [{GRAY}]measured:[/] {m.get('decode_tok_s', '?')} tok/s  {m.get('config', '')}")
            if m.get("warning"):
                console.print(f"    [{AMBER}]warning:[/] {m['warning']}")
        for line in e.get("unmeasured") or []:
            console.print(f"    [{GRAY}]not measured:[/] {line}")
    if not wanted and chain:
        console.print(f"\n  [{DIM}]escalation: {' -> '.join(chain)} (only on a failed deterministic gate)[/]")


register_command("roles", _roles, "Which local model for which job  (/roles [role])")
