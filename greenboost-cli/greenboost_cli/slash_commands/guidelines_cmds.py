"""UI guidelines slash command: /ui-guidelines."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from greenboost_cli.terminal.commands import register_command


# ── helpers ───────────────────────────────────────────────────────────────────


def _open_in_editor(path: Path) -> None:
    """Open *path* in $VISUAL or $EDITOR, falling back to vi."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    subprocess.call([editor, str(path)])


def _print_table(guidelines: list[dict]) -> None:
    """Print a formatted table with Status, Name, Source, Added columns."""
    if not guidelines:
        print("  (no guidelines)")
        return

    # Column widths (dynamic, with minimums)
    name_w   = max(len("Name"),   max(len(e["name"])                        for e in guidelines))
    source_w = max(len("Source"), max(len(str(e.get("source") or "(created)")) for e in guidelines))
    added_w  = max(len("Added"),  max(len(str(e.get("added_at", ""))[:10])  for e in guidelines))

    # Cap widths so the table stays within ~76 chars (2-space indent + columns)
    source_w = min(source_w, 30)
    name_w   = min(name_w,   24)

    header = (
        f"  {'St':<2}  {'Name':<{name_w}}  {'Source':<{source_w}}  {'Added':<{added_w}}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)
    for e in guidelines:
        status  = "✓" if e.get("active", True) else "○"
        name    = e["name"][:name_w]
        source  = str(e.get("source") or "(created)")
        # Truncate long source paths from the left so the filename is visible
        if len(source) > source_w:
            source = "…" + source[-(source_w - 1):]
        added   = str(e.get("added_at", ""))[:10]
        print(f"  {status:<2}  {name:<{name_w}}  {source:<{source_w}}  {added:<{added_w}}")


# ── main handler ──────────────────────────────────────────────────────────────


def _ui_guidelines(args: str, session, settings: dict) -> None:
    from greenboost_cli.memory.ui_guidelines import (
        list_guidelines,
        add_guideline,
        add_guideline_from_content,
        remove_guideline,
        set_active,
        get_guideline_content,
        _project_guidelines_dir,
    )

    project = settings.get("active_project")
    parts   = args.strip().split(None, 2)
    sub     = parts[0].lower() if parts else "list"

    # ── list ─────────────────────────────────────────────────────────────────
    if sub in ("list", ""):
        guidelines = list_guidelines(project)
        if not guidelines:
            print("  No guidelines yet.  Use /ui-guidelines add <file.md> to add one.")
            return
        _print_table(guidelines)

    # ── add <file.md> [name] ─────────────────────────────────────────────────
    elif sub == "add":
        if len(parts) < 2:
            print("  Usage: /ui-guidelines add <file.md> [name]")
            return
        file_arg   = parts[1]
        name_arg   = parts[2].strip() if len(parts) > 2 else None
        source     = Path(file_arg).expanduser().resolve()
        if not source.exists():
            print(f"  ✗  File not found: {source}")
            return
        try:
            final_name = add_guideline(source, name=name_arg, project=project)
            print(f"  ✓  Guideline '{final_name}' added from {source.name}")
        except Exception as exc:
            print(f"  ✗  {exc}")

    # ── create <name> ─────────────────────────────────────────────────────────
    elif sub == "create":
        if len(parts) < 2:
            print("  Usage: /ui-guidelines create <name>")
            return
        name = parts[1].strip()
        # Write a blank file to a temp location, open editor, then register
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix=f"guideline_{name}_",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _open_in_editor(tmp_path)
            content = tmp_path.read_text()
        finally:
            tmp_path.unlink(missing_ok=True)

        try:
            final_name = add_guideline_from_content(name, content, project=project)
            print(f"  ✓  Guideline '{final_name}' created.")
        except Exception as exc:
            print(f"  ✗  {exc}")

    # ── remove <name> ────────────────────────────────────────────────────────
    elif sub == "remove":
        if len(parts) < 2:
            print("  Usage: /ui-guidelines remove <name>")
            return
        name = parts[1].strip()
        try:
            remove_guideline(name, project=project)
            print(f"  ✓  Guideline '{name}' removed.")
        except ValueError as exc:
            print(f"  ✗  {exc}")

    # ── enable <name> ────────────────────────────────────────────────────────
    elif sub == "enable":
        if len(parts) < 2:
            print("  Usage: /ui-guidelines enable <name>")
            return
        name = parts[1].strip()
        try:
            set_active(name, True, project=project)
            print(f"  ✓  Guideline '{name}' enabled.")
        except ValueError as exc:
            print(f"  ✗  {exc}")

    # ── disable <name> ───────────────────────────────────────────────────────
    elif sub == "disable":
        if len(parts) < 2:
            print("  Usage: /ui-guidelines disable <name>")
            return
        name = parts[1].strip()
        try:
            set_active(name, False, project=project)
            print(f"  ○  Guideline '{name}' disabled.")
        except ValueError as exc:
            print(f"  ✗  {exc}")

    # ── show <name> ──────────────────────────────────────────────────────────
    elif sub == "show":
        if len(parts) < 2:
            print("  Usage: /ui-guidelines show <name>")
            return
        name    = parts[1].strip()
        content = get_guideline_content(name, project=project)
        if not content:
            print(f"  ✗  Guideline '{name}' not found or empty.")
            return
        print(f"\n── {name} ──")
        print(content)

    # ── edit <name> ──────────────────────────────────────────────────────────
    elif sub == "edit":
        if len(parts) < 2:
            print("  Usage: /ui-guidelines edit <name>")
            return
        name = parts[1].strip()
        gdir = _project_guidelines_dir(project)
        # Resolve file path from index
        index = list_guidelines(project)
        entry = next((e for e in index if e["name"] == name), None)
        if entry is None:
            print(f"  ✗  Guideline '{name}' not found.")
            return
        md_file = gdir / entry["file"]
        if not md_file.exists():
            md_file.write_text("")
        _open_in_editor(md_file)
        print(f"  ✓  Guideline '{name}' saved.")

    # ── unknown sub-command ──────────────────────────────────────────────────
    else:
        print("  Usage: /ui-guidelines [list|add|create|remove|enable|disable|show|edit]")


# ── registration ──────────────────────────────────────────────────────────────

register_command(
    "ui-guidelines",
    _ui_guidelines,
    "Manage UI design guidelines  (/ui-guidelines [add|list|remove])",
)
