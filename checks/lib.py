#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/lib.py — shared Finding type + helpers for the checks/ suite.

Every check module exposes `run(repo_root: Path) -> list[Finding]`. Findings
carry remediation text so a finding is actionable without re-reading the
check's source (per the harness-engineering practice of injecting
remediation into agent context, not just flagging a violation)."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Finding:
    check: str                  # which check produced this (e.g. "hardware_literals")
    severity: str                # "blocking" | "advisory"
    file: str                   # repo-relative path
    line: int = 0
    message: str = ""
    remediation: str = ""

    def format(self, llm: bool = False) -> str:
        loc = f"{self.file}:{self.line}" if self.line else self.file
        if llm:
            return f"{self.severity.upper()} {self.check} {loc} — {self.message} | fix: {self.remediation}"
        return (f"[{self.severity}] {self.check}\n"
               f"  {loc}\n"
               f"  {self.message}\n"
               f"  remediation: {self.remediation}")


def load_allowlist(path: Path) -> list[str]:
    """One glob-or-literal pattern per line; '#'-prefixed and blank lines
    ignored. Patterns match against repo-relative file paths (fnmatch) or
    'path:line' (exact) for line-level allowlisting."""
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split("#", 1)[0].strip())  # allow trailing '# reason' comments
    return [p for p in out if p]


def is_allowlisted(rel_path: str, line: int, patterns: list[str]) -> bool:
    exact = f"{rel_path}:{line}"
    for pat in patterns:
        if pat == exact or pat == rel_path:
            return True
        if fnmatch.fnmatch(rel_path, pat):
            return True
    return False


def iter_tracked_files(repo_root: Path, suffixes: "tuple[str, ...]") -> "list[Path]":
    """Files git tracks (respects .gitignore) with any of the given suffixes.
    Falls back to a plain rglob (still skipping .git/) if git isn't available
    — e.g. in a tarball checkout with no .git directory."""
    import subprocess
    try:
        out = subprocess.run(["git", "ls-files"], cwd=repo_root, capture_output=True,
                             text=True, check=True, timeout=10).stdout
        files = [repo_root / p for p in out.splitlines() if p]
        return [f for f in files if f.suffix in suffixes and f.is_file()]
    except Exception:
        return [p for p in repo_root.rglob("*")
                if p.suffix in suffixes and p.is_file() and ".git" not in p.parts]
