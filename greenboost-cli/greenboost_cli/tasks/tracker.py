"""In-session todo tracker, persisted to ~/.greenboost_cli/tasks/<project>.json.

Modelled loosely on Claude Code's TaskCreate, but pared down: each task has
an id, a one-line subject, an optional longer description, a status, an
active form, and created/updated timestamps. The tracker is intentionally
simple — no parent/child, no dependencies — because the goal is "small
in-session todo list the user can drive from the REPL", not a project
management system.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

from greenboost_cli.environment.settings import GB_HOME
from greenboost_cli.security import _safe_project

TASKS_DIR = GB_HOME / "tasks"

VALID_STATUSES = ("pending", "in_progress", "completed", "blocked")


@dataclass
class Task:
    id: str
    subject: str
    description: str = ""
    status: str = "pending"
    active_form: str = ""        # human-readable verb phrase, e.g. "Refactoring router"
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def project_tasks_path(project: str | None) -> Path:
    """Resolve a project name to its on-disk JSON file (safe-named)."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe_project(project) if project else "default"
    if not name:
        name = "default"
    return TASKS_DIR / f"{name}.json"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def _save(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")


def _new_id(existing: list[dict]) -> str:
    """Short monotonically-increasing id keyed on count + timestamp suffix."""
    suffix = int(time.time()) % 10_000
    return f"t{len(existing) + 1:03d}-{suffix:04d}"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ── Public API ────────────────────────────────────────────────────────────────

def add_task(
    project: str | None,
    subject: str,
    description: str = "",
    active_form: str = "",
) -> Task:
    """Append a new pending task to this project's list."""
    path = project_tasks_path(project)
    tasks = _load(path)
    now = _now()
    task = Task(
        id=_new_id(tasks),
        subject=subject.strip(),
        description=description.strip(),
        status="pending",
        active_form=(active_form or subject).strip(),
        created_at=now,
        updated_at=now,
    )
    tasks.append(task.to_dict())
    _save(path, tasks)
    return task


def list_tasks(project: str | None) -> list[Task]:
    """Return all tasks for project (most-recently-updated last)."""
    path = project_tasks_path(project)
    return [Task(**t) for t in _load(path)]


def update_task(project: str | None, task_id: str, status: str) -> Task | None:
    """Set a task's status. Returns the updated Task or None if not found."""
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status '{status}'. "
                         f"Use one of: {', '.join(VALID_STATUSES)}")
    path = project_tasks_path(project)
    tasks = _load(path)
    for t in tasks:
        if t.get("id") == task_id:
            t["status"] = status
            t["updated_at"] = _now()
            _save(path, tasks)
            return Task(**t)
    return None


def delete_task(project: str | None, task_id: str) -> bool:
    """Remove a task. Returns True on success."""
    path = project_tasks_path(project)
    tasks = _load(path)
    new = [t for t in tasks if t.get("id") != task_id]
    if len(new) == len(tasks):
        return False
    _save(path, new)
    return True
