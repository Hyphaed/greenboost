"""Diff applier slash commands: /apply-diff."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from greenboost_cli.terminal.commands import register_command

# ── Block regexes (same as scripts/apply_diff.py) ────────────────────────────

SEARCH_BLOCK = re.compile(
    r"<<<< SEARCH \[(?P<file>[^\]]+)\]\n"
    r"(?P<search>.*?)"
    r"\n==== REPLACE\n"
    r"(?P<replace>.*?)"
    r"\n>>>>",
    re.DOTALL,
)

REPLACE_FN_BLOCK = re.compile(
    r"<<<< REPLACE_FUNCTION (?P<fn>\w+) \[(?P<file>[^\]]+)\]\n"
    r"(?P<new_body>.*?)"
    r"\n>>>>",
    re.DOTALL,
)

CREATE_BLOCK = re.compile(
    r"<<<< CREATE \[(?P<file>[^\]]+)\]\n"
    r"(?P<content>.*?)"
    r"\n>>>>",
    re.DOTALL,
)


def _apply_search_replace(file_path: Path, search: str, replace: str, dry_run: bool) -> bool:
    if not file_path.exists():
        print(f"  ✗  File not found: {file_path}", file=sys.stderr)
        return False
    content = file_path.read_text()
    count   = content.count(search)
    if count == 0:
        print(f"  ✗  SEARCH text not found in {file_path}", file=sys.stderr)
        return False
    if count > 1:
        print(f"  ✗  SEARCH text found {count} times in {file_path} — must be unique", file=sys.stderr)
        return False
    new_content = content.replace(search, replace, 1)
    if dry_run:
        print(f"  ○  [dry-run] Would modify {file_path}")
        return True
    file_path.write_text(new_content)
    print(f"  ✓  Modified {file_path}")
    return True


def _apply_replace_function(file_path: Path, fn_name: str, new_body: str, dry_run: bool) -> bool:
    if not file_path.exists():
        print(f"  ✗  File not found: {file_path}", file=sys.stderr)
        return False
    content = file_path.read_text()
    lines   = content.splitlines(keepends=True)

    fn_start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"def {fn_name}(") or stripped.startswith(f"async def {fn_name}("):
            fn_start = i
            break

    if fn_start is None:
        print(f"  ✗  Function '{fn_name}' not found in {file_path}", file=sys.stderr)
        return False

    base_indent = len(lines[fn_start]) - len(lines[fn_start].lstrip())
    fn_end = len(lines)
    for i in range(fn_start + 1, len(lines)):
        stripped   = lines[i].strip()
        if not stripped:
            continue
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent <= base_indent and stripped:
            fn_end = i
            break

    indent = " " * base_indent
    indented_body = "\n".join(
        (indent + l if l.strip() else l) for l in new_body.splitlines()
    )

    new_lines   = lines[:fn_start] + [indented_body + "\n"] + lines[fn_end:]
    new_content = "".join(new_lines)

    if dry_run:
        print(f"  ○  [dry-run] Would replace function '{fn_name}' in {file_path}")
        return True
    file_path.write_text(new_content)
    print(f"  ✓  Replaced function '{fn_name}' in {file_path}")
    return True


def _apply_create(file_path: Path, content: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"  ○  [dry-run] Would create {file_path}")
        return True
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    print(f"  ✓  Created {file_path}")
    return True


def process_text(text: str, base_dir: Path, dry_run: bool) -> int:
    """Apply all diff blocks in text. Returns number of failures."""
    failures = 0

    for m in CREATE_BLOCK.finditer(text):
        fp = base_dir / m.group("file")
        if not _apply_create(fp, m.group("content"), dry_run):
            failures += 1

    for m in REPLACE_FN_BLOCK.finditer(text):
        fp = base_dir / m.group("file")
        if not _apply_replace_function(fp, m.group("fn"), m.group("new_body"), dry_run):
            failures += 1

    for m in SEARCH_BLOCK.finditer(text):
        fp = base_dir / m.group("file")
        if not _apply_search_replace(fp, m.group("search"), m.group("replace"), dry_run):
            failures += 1

    return failures


def _apply_diff(args: str, session, settings: dict) -> None:
    """Apply SEARCH/REPLACE diff blocks from a file or the last assistant message."""
    parts   = args.strip().split()
    dry_run = "--dry-run" in parts
    files   = [p for p in parts if not p.startswith("--")]

    if files:
        diff_file = Path(files[0]).expanduser()
        if not diff_file.exists():
            print(f"  ✗  File not found: {diff_file}")
            return
        text = diff_file.read_text()
    elif session and session.messages:
        # Use last assistant message
        last_assistant = next(
            (m["content"] for m in reversed(session.messages) if m["role"] == "assistant"),
            None,
        )
        if not last_assistant:
            print("  ✗  No assistant message in session. Provide a file: /apply-diff <file>")
            return
        if isinstance(last_assistant, list):
            text = " ".join(
                b.get("text", "") for b in last_assistant
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = str(last_assistant)
    else:
        print("  Usage: /apply-diff [<diff-file>] [--dry-run]")
        return

    base_dir = Path.cwd()
    failures = process_text(text, base_dir, dry_run)
    if failures:
        print(f"\n  {failures} diff(s) failed to apply.")


register_command(
    "apply-diff",
    _apply_diff,
    "Apply SEARCH/REPLACE diff blocks  (/apply-diff [file] [--dry-run])",
)
