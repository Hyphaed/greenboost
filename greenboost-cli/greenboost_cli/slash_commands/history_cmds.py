"""What the agent changed on disk, and putting it back.

    /changes            files this session wrote or edited
    /revert <path>      restore a file to its pre-agent state
    /revert <path> N    restore it to recorded version N

`/undo` is a different thing and stays as it was: it drops the last
conversation exchange. This is about files.
"""
from __future__ import annotations

import time

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import console, GRAY, DIM, emit_ok, emit_warn


def _cmd_changes(args: str, session, settings: dict) -> bool:
    from greenboost_cli.core.file_history import changed_files
    rows = changed_files()
    console.print()
    if not rows:
        console.print(f"  [{GRAY}]No files changed by the agent this session.[/]\n")
        return True
    for r in rows:
        age = time.time() - r["first_ts"]
        when = f"{age/60:.0f}m ago" if age >= 60 else "just now"
        tag = "created" if r["created"] else f"{r['edits']} edit(s)"
        warn = "" if r["revertable"] else f"  [{GRAY}](too large to revert)[/]"
        console.print(f"  {r['path']}")
        console.print(f"      [{DIM}]{tag} · first {when}[/]{warn}")
    console.print()
    console.print(f"  [{DIM}]/revert <path> restores the pre-agent version[/]\n")
    return True


def _cmd_revert(args: str, session, settings: dict) -> bool:
    parts = (args or "").split()
    if not parts:
        emit_warn("Usage: /revert <path> [version]   , /changes lists them")
        return True
    version = 1
    if len(parts) > 1 and parts[-1].isdigit():
        version = int(parts[-1]); parts = parts[:-1]
    from greenboost_cli.core.file_history import revert
    out = revert(" ".join(parts), version)
    (emit_warn if out.startswith("Error") else emit_ok)(out)
    return True


register_command("changes", _cmd_changes,
                 "Files the agent wrote or edited this session")
register_command("revert", _cmd_revert,
                 "Restore a file to its pre-agent state")
