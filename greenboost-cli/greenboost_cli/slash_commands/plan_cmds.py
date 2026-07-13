"""Plan-mode slash commands: /plan /plan-edit /plan-approve /plan-exit /plan-list."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import emit_ok, emit_warn, emit_info, emit_err, GRAY, VIOLET, LIME
from greenboost_cli.terminal.theme import console
from greenboost_cli.security import _cap, _MAX_QUERY_LEN


def _plan(args: str, session, settings: dict) -> None:
    """Enter plan mode and create a plan file. /plan [optional prompt text]."""
    from greenboost_cli.planning.plan import create_plan, PLANS_DIR

    prompt = _cap(args.strip(), _MAX_QUERY_LEN)
    entry = create_plan(prompt)
    session.plan_mode = True
    session.plan_file = entry.path

    console.print(f"  [{LIME}]●[/] Plan mode ON.")
    console.print(f"  [{GRAY}]Plan file:[/] {entry.path}")
    console.print(f"  [{GRAY}]Use /plan-edit to open the file, /plan-approve to exit.[/]")


def _plan_edit(args: str, session, settings: dict) -> None:
    """Open the active plan file in $EDITOR."""
    if not session.plan_file:
        emit_warn("No active plan. Run /plan first.")
        return
    editor = os.environ.get("EDITOR", "nano")
    try:
        subprocess.call([editor, str(session.plan_file)])
    except OSError as e:
        emit_err(f"Could not launch editor '{editor}': {e}")


def _plan_approve(args: str, session, settings: dict) -> None:
    """Exit plan mode and resume normal execution."""
    if not session.plan_mode:
        emit_info("Not currently in plan mode.")
        return
    plan_file = session.plan_file
    session.plan_mode = False
    # Keep `session.plan_file` set so the model can still reference the
    # approved plan path in subsequent turns if it chooses to.
    emit_ok(f"Plan approved — leaving plan mode (plan saved at {plan_file}).")


def _plan_exit(args: str, session, settings: dict) -> None:
    """Discard plan mode without 'approving'. Plan file stays on disk."""
    if not session.plan_mode:
        emit_info("Not in plan mode.")
        return
    session.plan_mode = False
    emit_ok("Plan mode OFF.")


def _plan_list(args: str, session, settings: dict) -> None:
    """List existing plan files."""
    from greenboost_cli.planning.plan import list_plans, PLANS_DIR

    entries = list_plans()
    if not entries:
        console.print(f"  [{GRAY}]No plans yet at {PLANS_DIR}[/]")
        return
    console.print(f"  [{VIOLET}]Plans[/] [{GRAY}]({PLANS_DIR})[/]")
    for e in entries:
        marker = "●" if (session.plan_file and session.plan_file == e.path) else " "
        console.print(f"  {marker}  [{LIME}]{e.id}[/]  [{GRAY}]{e.created_at}[/]  {e.title}")


register_command("plan",         _plan,         "Enter plan mode  (/plan [optional prompt])")
register_command("plan-edit",    _plan_edit,    "Open active plan file in $EDITOR")
register_command("plan-approve", _plan_approve, "Approve plan and resume normal execution")
register_command("plan-exit",    _plan_exit,    "Leave plan mode without approving")
register_command("plan-list",    _plan_list,    "List all plan files")
