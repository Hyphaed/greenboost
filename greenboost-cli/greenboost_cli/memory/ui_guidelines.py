"""Per-project UI design guidelines manager.

State is stored under ~/.greenboost_cli/projects/<project>/guidelines/:
  guidelines/index.yaml  — list of {name, file, source, active, added_at}
  guidelines/<name>.md   — the actual guideline markdown content
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from greenboost_cli.environment.settings import GB_HOME

_PROJECTS_DIR = GB_HOME / "projects"


# ── Internal helpers ──────────────────────────────────────────────────────────


def _resolve_project(project: str | None) -> str:
    return project or Path(os.getcwd()).name


def _project_guidelines_dir(project: str | None = None) -> Path:
    """Return ~/.greenboost_cli/projects/<project>/guidelines/"""
    return _PROJECTS_DIR / _resolve_project(project) / "guidelines"


def _index_path(project: str | None = None) -> Path:
    return _project_guidelines_dir(project) / "index.yaml"


def _load_index(project: str | None = None) -> list[dict]:
    """Load index.yaml, return [] if missing."""
    path = _index_path(project)
    if not path.exists():
        return []
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, list):
        return []
    return data


def _save_index(guidelines: list[dict], project: str | None = None) -> None:
    """Save index.yaml, creating dirs as needed."""
    gdir = _project_guidelines_dir(project)
    gdir.mkdir(parents=True, exist_ok=True)
    with open(_index_path(project), "w") as fh:
        yaml.dump(guidelines, fh, default_flow_style=False, allow_unicode=True)


def _unique_name(base: str, existing_names: set[str]) -> str:
    """Return base if not taken, else base_2, base_3, ..."""
    if base not in existing_names:
        return base
    counter = 2
    while f"{base}_{counter}" in existing_names:
        counter += 1
    return f"{base}_{counter}"


# ── Public API ────────────────────────────────────────────────────────────────


def add_guideline(
    source_path: str | Path,
    name: str | None = None,
    project: str | None = None,
) -> str:
    """Copy source file into guidelines dir, register in index (active=True).

    Returns the final name used (may differ from *name* if a conflict exists).
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    index = _load_index(project)
    existing_names = {entry["name"] for entry in index}

    base_name = name or source_path.stem
    final_name = _unique_name(base_name, existing_names)

    gdir = _project_guidelines_dir(project)
    gdir.mkdir(parents=True, exist_ok=True)
    dest = gdir / f"{final_name}.md"
    shutil.copy2(source_path, dest)

    entry: dict = {
        "name": final_name,
        "file": f"{final_name}.md",
        "source": str(source_path),
        "active": True,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    index.append(entry)
    _save_index(index, project)
    return final_name


def add_guideline_from_content(
    name: str,
    content: str,
    project: str | None = None,
) -> str:
    """Create guideline from text content directly.

    Returns the final name used (may differ from *name* if a conflict exists).
    """
    index = _load_index(project)
    existing_names = {entry["name"] for entry in index}
    final_name = _unique_name(name, existing_names)

    gdir = _project_guidelines_dir(project)
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / f"{final_name}.md").write_text(content)

    entry: dict = {
        "name": final_name,
        "file": f"{final_name}.md",
        "source": None,
        "active": True,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    index.append(entry)
    _save_index(index, project)
    return final_name


def update_guideline(name: str, content: str, project: str | None = None) -> None:
    """Overwrite existing guideline content."""
    index = _load_index(project)
    entry = next((e for e in index if e["name"] == name), None)
    if entry is None:
        raise ValueError(f"Guideline '{name}' not found.")
    gdir = _project_guidelines_dir(project)
    (gdir / entry["file"]).write_text(content)


def remove_guideline(name: str, project: str | None = None) -> None:
    """Delete file + remove from index. Raise ValueError if not found."""
    index = _load_index(project)
    entry = next((e for e in index if e["name"] == name), None)
    if entry is None:
        raise ValueError(f"Guideline '{name}' not found.")

    gdir = _project_guidelines_dir(project)
    md_file = gdir / entry["file"]
    if md_file.exists():
        md_file.unlink()

    index = [e for e in index if e["name"] != name]
    _save_index(index, project)


def set_active(name: str, active: bool, project: str | None = None) -> None:
    """Enable or disable a guideline."""
    index = _load_index(project)
    entry = next((e for e in index if e["name"] == name), None)
    if entry is None:
        raise ValueError(f"Guideline '{name}' not found.")
    entry["active"] = active
    _save_index(index, project)


def get_guideline_content(name: str, project: str | None = None) -> str:
    """Read the .md file content, return '' if missing."""
    index = _load_index(project)
    entry = next((e for e in index if e["name"] == name), None)
    if entry is None:
        return ""
    gdir = _project_guidelines_dir(project)
    md_file = gdir / entry["file"]
    if not md_file.exists():
        return ""
    return md_file.read_text()


def get_active_guidelines(project: str | None = None) -> list[tuple[str, str]]:
    """Return [(name, content)] for all active=True guidelines with non-empty content."""
    results: list[tuple[str, str]] = []
    for entry in _load_index(project):
        if not entry.get("active", True):
            continue
        content = get_guideline_content(entry["name"], project)
        if content:
            results.append((entry["name"], content))
    return results


def list_guidelines(project: str | None = None) -> list[dict]:
    """Return the full index list."""
    return _load_index(project)


def get_guidelines_context(project: str | None = None) -> str:
    """Return a formatted string for injection into a system prompt.

    Format::

        \\n# UI Guidelines\\n\\n## <name>\\n<content>\\n...

    Returns '' if no active guidelines.
    """
    active = get_active_guidelines(project)
    if not active:
        return ""
    parts = ["\n# UI Guidelines\n"]
    for name, content in active:
        parts.append(f"## {name}\n{content}")
    return "\n".join(parts)
