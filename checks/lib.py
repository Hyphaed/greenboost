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


def content_anchor(text: str) -> str:
    """Stable 8-hex fingerprint of one source line, whitespace-normalised.

    The anchor for `path:~<sha8>` allowlist entries. Normalising whitespace
    means re-indenting a docstring does not invalidate the entry, while
    changing what it actually says does , which is the distinction that
    matters, since the thing being sanctioned is the CONTENT.
    """
    import hashlib
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:8]


def is_allowlisted(rel_path: str, line: int, patterns: list[str],
                   line_text: "str | None" = None) -> bool:
    """Is this finding sanctioned?

    Three entry forms, in increasing robustness:

      path                 , whole file (also fnmatch globs)
      path:line            , one exact line, BY NUMBER
      path:~<sha8>         , one exact line, BY CONTENT (`content_anchor`)

    Prefer the anchored form. Line-numbered entries drift every time anything
    is inserted above them, and drift here is not a cosmetic problem: the
    entry keeps matching a line NUMBER that now holds unrelated code, so it
    silently sanctions something nobody reviewed while the line it was written
    for goes unguarded. That is not hypothetical , on 2026-08-21 a four-entry
    group in secrets_reviewed.txt was found pointing at ordinary prose, having
    been papered over by ADDING a second group rather than repairing the
    first, and the hardware allowlist drifted four times in one session.

    An anchored entry cannot drift: it matches the content wherever it moved
    to, and stops matching the moment that content changes , which is exactly
    when a human should look again.
    """
    exact = f"{rel_path}:{line}"
    for pat in patterns:
        if pat == exact or pat == rel_path:
            return True
        if line_text is not None and pat.startswith(f"{rel_path}:~"):
            if pat.split(":~", 1)[1].strip() == content_anchor(line_text):
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
