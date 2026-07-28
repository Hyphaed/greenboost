"""Second brain — project memory: prime goals, history, snapshots."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yaml

from greenboost_cli.environment.settings import GB_HOME

GLOBAL_DIR = GB_HOME / "projects"


def project_dir(name: str | None = None) -> Path:
    """Return ~/.greenboost_cli/projects/<name>/, creating if needed."""
    import os
    project_name = name or Path(os.getcwd()).name
    d = GLOBAL_DIR / project_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Prime Goals ───────────────────────────────────────────────────────────────

def _goals_file(pdir: Path) -> Path:
    return pdir / "prime-goals.yaml"


def load_goals(pdir: Path) -> list[dict]:
    f = _goals_file(pdir)
    if not f.exists():
        return []
    with open(f) as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("goals", [])


def add_goal(pdir: Path, text: str, priority: int = 5) -> dict:
    goals = load_goals(pdir)
    new_id = max((g["id"] for g in goals), default=0) + 1
    entry = {
        "id": new_id,
        "priority": priority,
        "text": text,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    goals.append(entry)
    goals.sort(key=lambda g: g["priority"])
    with open(_goals_file(pdir), "w") as fh:
        yaml.dump({"goals": goals}, fh, default_flow_style=False, allow_unicode=True)
    return entry


def remove_goal(pdir: Path, goal_id: int) -> bool:
    goals = load_goals(pdir)
    original_len = len(goals)
    goals = [g for g in goals if g["id"] != goal_id]
    if len(goals) == original_len:
        return False
    with open(_goals_file(pdir), "w") as fh:
        yaml.dump({"goals": goals}, fh, default_flow_style=False, allow_unicode=True)
    return True


def build_system_prompt_goals(pdir: Path) -> str:
    """Return goals block for system prompt injection. Empty string if no goals."""
    goals = load_goals(pdir)
    if not goals:
        return ""
    lines = ["=== PROJECT PRIME GOALS (always consider these constraints) ==="]
    for i, g in enumerate(goals, 1):
        lines.append(f"{i}. [P{g['priority']}] {g['text']}")
    lines.append("=" * 62)
    return "\n".join(lines)


def get_goals_summary() -> str:
    """Return goals block for the current project (for context_builder injection)."""
    try:
        pdir = project_dir()
        block = build_system_prompt_goals(pdir)
        if not block:
            return ""
        return f"\n{block}\n"
    except Exception:
        return ""


def print_goals(pdir: Path) -> None:
    goals = load_goals(pdir)
    if not goals:
        print("  No prime goals set. Add one:")
        print("  /goals add \"Your goal here\" [--priority 1-9]")
        return
    print(f"  Prime goals for: {pdir.name}")
    print()
    for g in goals:
        added = g.get("added_at", "")[:10]
        print(f"  [{g['id']:>2}] P{g['priority']}  {g['text']}  ({added})")


# ── History ───────────────────────────────────────────────────────────────────

def _history_file(pdir: Path) -> Path:
    return pdir / "history.md"


VALID_CATEGORIES = {"note", "decision", "milestone", "blocker", "resolved"}


def append_history(pdir: Path, text: str, category: str = "note") -> None:
    if category not in VALID_CATEGORIES:
        category = "note"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## {ts} [{category}]\n{text}\n"
    with open(_history_file(pdir), "a") as fh:
        fh.write(entry)


def read_recent_history(pdir: Path, n_entries: int = 10) -> str:
    f = _history_file(pdir)
    if not f.exists():
        return "(no history yet)"
    text = f.read_text().strip()
    if not text:
        return "(no history yet)"
    sections = [s.strip() for s in text.split("\n## ") if s.strip()]
    if not sections:
        return "(no history yet)"
    recent = sections[-n_entries:]
    return "\n\n## ".join(recent)


def print_history(pdir: Path, n: int = 10) -> None:
    print(f"  Recent history for: {pdir.name}")
    print()
    content = read_recent_history(pdir, n)
    for line in content.splitlines():
        if line.startswith("## "):
            print(f"  \033[38;2;48;200;255m{line}\033[0m")
        elif line.strip():
            print(f"    {line}")


def search_history(pdir: Path, query: str, n_results: int = 20) -> list[dict]:
    """Case-insensitive substring search over history entries.

    Returns a list of dicts with keys: header, body, raw (most-recent last).
    """
    f = _history_file(pdir)
    if not f.exists():
        return []
    text = f.read_text().strip()
    if not text:
        return []
    sections = [s.strip() for s in text.split("\n## ") if s.strip()]
    q = query.lower()
    matches = []
    for s in sections:
        if q in s.lower():
            lines  = s.splitlines()
            header = lines[0] if lines else ""
            body   = "\n".join(lines[1:]).strip()
            matches.append({"header": header, "body": body, "raw": s})
    return matches[-n_results:]


# ── Snapshot ──────────────────────────────────────────────────────────────────

def _snapshot_file(pdir: Path) -> Path:
    return pdir / "context-snapshot.md"


def write_snapshot(pdir: Path, content: str) -> None:
    _snapshot_file(pdir).write_text(content)


def read_snapshot(pdir: Path) -> str:
    f = _snapshot_file(pdir)
    return f.read_text() if f.exists() else "(no snapshot yet)"


def print_snapshot(pdir: Path) -> None:
    print(f"  Context snapshot for: {pdir.name}")
    print()
    for line in read_snapshot(pdir).splitlines():
        print(f"  {line}")
