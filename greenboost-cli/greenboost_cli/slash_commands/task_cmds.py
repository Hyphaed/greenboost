"""Task-tracker slash commands: /task-add /task-list /task-update /task-delete."""
from __future__ import annotations

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import (
    console, GRAY, VIOLET, LIME, AMBER,
    emit_ok, emit_warn, emit_err, emit_info,
)
from greenboost_cli.security import _cap, _MAX_GOAL_LEN, _MAX_TEXT_LEN


_STATUS_COLOR = {
    "pending":     GRAY,
    "in_progress": AMBER,
    "completed":   LIME,
    "blocked":     "red",
}
_STATUS_ICON = {
    "pending":     "○",
    "in_progress": "◐",
    "completed":   "●",
    "blocked":     "✗",
}


def _proj(settings: dict) -> str | None:
    return settings.get("active_project")


def _task_add(args: str, session, settings: dict) -> None:
    """/task-add "<subject>" [-- <description>]"""
    from greenboost_cli.tasks.tracker import add_task

    raw = args.strip()
    if not raw:
        emit_warn('Usage: /task-add "<subject>" [-- <description>]')
        return

    if "--" in raw:
        subject, desc = raw.split("--", 1)
        subject = subject.strip().strip('"').strip("'")
        desc = desc.strip()
    else:
        subject = raw.strip('"').strip("'")
        desc = ""

    subject = _cap(subject, _MAX_GOAL_LEN)
    desc = _cap(desc, _MAX_TEXT_LEN)
    if not subject:
        emit_warn("Task subject cannot be empty.")
        return

    task = add_task(_proj(settings), subject, description=desc)
    emit_ok(f"Task {task.id} added: {task.subject}")


def _task_list(args: str, session, settings: dict) -> None:
    from greenboost_cli.tasks.tracker import list_tasks

    tasks = list_tasks(_proj(settings))
    if not tasks:
        console.print(f"  [{GRAY}]No tasks. Add one with /task-add \"...\"[/]")
        return
    console.print(f"  [{VIOLET}]Tasks[/]")
    for t in tasks:
        color = _STATUS_COLOR.get(t.status, GRAY)
        icon = _STATUS_ICON.get(t.status, "?")
        console.print(
            f"  [{color}]{icon}[/]  [bold]{t.id}[/]  "
            f"[{color}]{t.status:<11}[/]  {t.subject}"
        )
        if t.description:
            for line in t.description.splitlines():
                console.print(f"           [{GRAY}]{line}[/]")


def _task_update(args: str, session, settings: dict) -> None:
    """/task-update <id> <status>"""
    from greenboost_cli.tasks.tracker import update_task, VALID_STATUSES

    parts = args.strip().split()
    if len(parts) < 2:
        emit_warn(f"Usage: /task-update <id> <{ '|'.join(VALID_STATUSES) }>")
        return
    tid, status = parts[0], parts[1]
    try:
        result = update_task(_proj(settings), tid, status)
    except ValueError as e:
        emit_err(str(e))
        return
    if result is None:
        emit_warn(f"Task {tid} not found.")
        return
    emit_ok(f"Task {result.id} → {result.status}")


def _task_delete(args: str, session, settings: dict) -> None:
    """/task-delete <id>"""
    from greenboost_cli.tasks.tracker import delete_task

    tid = args.strip().split()[0] if args.strip() else ""
    if not tid:
        emit_warn("Usage: /task-delete <id>")
        return
    ok = delete_task(_proj(settings), tid)
    if ok:
        emit_ok(f"Task {tid} deleted.")
    else:
        emit_warn(f"Task {tid} not found.")


register_command("task-add",    _task_add,    'Add a task  (/task-add "<subject>" [-- <desc>])')
register_command("task-list",   _task_list,   "List tasks for active project")
register_command("task-update", _task_update, "Update task status  (/task-update <id> <status>)")
register_command("task-delete", _task_delete, "Delete a task  (/task-delete <id>)")
