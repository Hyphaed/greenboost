"""Plan-mode storage.

Plan files live under ~/.greenboost_cli/plans/<short-id>.md. Each plan is a
short markdown document the model fills out before executing changes. The
REPL exposes /plan, /plan-edit, /plan-approve, /plan-exit. The headless CLI
exposes `gb plan-create` and `gb plan-list`.

This module is storage-only; the directive that tells the model "you are in
plan mode" lives in greenboost_cli/workflow/intelligence.py.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from greenboost_cli.environment.settings import GB_HOME

PLANS_DIR = GB_HOME / "plans"


@dataclass
class PlanEntry:
    id: str
    path: Path
    created_at: str
    title: str
    size: int


def short_id(seed: str = "") -> str:
    """8-char id from sha256(timestamp+seed). Stable enough for filenames."""
    raw = f"{time.time_ns()}:{seed}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def plan_path(plan_id: str) -> Path:
    """Resolve a plan id to its on-disk markdown path."""
    return PLANS_DIR / f"{plan_id}.md"


def _default_template(prompt: str, plan_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = prompt.strip().splitlines()[0] if prompt.strip() else f"plan {plan_id}"
    return (
        f"# Plan {plan_id}\n\n"
        f"_Created: {ts}_\n\n"
        f"## Goal\n\n{title}\n\n"
        f"## Context\n\n"
        f"_(Notes, constraints, assumptions go here.)_\n\n"
        f"## Steps\n\n"
        f"1. \n"
        f"2. \n"
        f"3. \n\n"
        f"## Risks / open questions\n\n- \n\n"
        f"## Acceptance criteria\n\n- \n"
    )


def create_plan(prompt: str = "") -> PlanEntry:
    """Create a new plan file under PLANS_DIR and return its entry."""
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    pid = short_id(prompt[:40])
    p = plan_path(pid)
    p.write_text(_default_template(prompt, pid), encoding="utf-8")
    return _entry_for(p)


def read_plan(plan_id: str) -> str:
    """Return the plan markdown for plan_id, or '' if not found."""
    p = plan_path(plan_id)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


def list_plans() -> list[PlanEntry]:
    """Return all plan entries sorted newest first."""
    if not PLANS_DIR.is_dir():
        return []
    entries = [_entry_for(p) for p in PLANS_DIR.glob("*.md") if p.is_file()]
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries


def _entry_for(p: Path) -> PlanEntry:
    try:
        first_line = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        title = first_line.lstrip("# ").strip() or p.stem
    except (OSError, IndexError):
        title = p.stem
    stat = p.stat()
    created = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return PlanEntry(
        id=p.stem,
        path=p,
        created_at=created,
        title=title,
        size=stat.st_size,
    )
