#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_docs.py — doc freshness (advisory). Two checks:
  1. workflow/README.md's index table lists every workflow/*.md file
     (workflow/ is gitignored/local-only — skips gracefully if absent, e.g.
     on a fresh clone or in CI).
  2. Every relative markdown link in tracked docs (AGENTS.md, docs/*.md,
     DOCUMENTATION.md, README.md) resolves to a real file.

--gc mode (doc-gardening, B4): additionally flags workflow/*.md files whose
'updated:'-style date (if present) is stale (>60 days) — a heuristic nudge
to re-read and refresh, not a hard rule.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from lib import Finding

_LINK_RE = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)]+)\)")
_TRACKED_DOC_GLOBS = ("AGENTS.md", "DOCUMENTATION.md", "README.md")


def _check_workflow_index(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    workflow_dir = repo_root / "workflow"
    readme = workflow_dir / "README.md"
    if not workflow_dir.is_dir() or not readme.is_file():
        return findings  # gitignored/local-only — nothing to check on a fresh clone
    readme_text = readme.read_text(errors="ignore")
    for md in sorted(workflow_dir.glob("*.md")):
        if md.name == "README.md":
            continue
        if md.name not in readme_text:
            findings.append(Finding(
                check="docs", severity="advisory", file=f"workflow/{md.name}",
                message="not referenced anywhere in workflow/README.md's index",
                remediation="add a row to the index table (or delete the file if it's stale/superseded)"))
    return findings


def _check_dead_links(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    doc_files = list((repo_root / "docs").glob("*.md")) if (repo_root / "docs").is_dir() else []
    for name in _TRACKED_DOC_GLOBS:
        p = repo_root / name
        if p.is_file():
            doc_files.append(p)
    for doc in doc_files:
        rel = str(doc.relative_to(repo_root))
        text = doc.read_text(errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in _LINK_RE.finditer(line):
                target = m.group(1).split("#", 1)[0].strip()
                if not target:
                    continue
                target_path = (doc.parent / target).resolve()
                if not target_path.exists():
                    findings.append(Finding(
                        check="docs", severity="advisory", file=rel, line=lineno,
                        message=f"dead relative link: {target!r}",
                        remediation="fix the path or remove the link"))
    return findings


def _check_gc_stale_dates(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    workflow_dir = repo_root / "workflow"
    if not workflow_dir.is_dir():
        return findings
    import datetime
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")
    today = None
    try:
        today = datetime.date.fromisoformat(_today_iso())
    except Exception:
        return findings  # no reliable "today" available — skip rather than guess
    for md in sorted(workflow_dir.glob("*.md")):
        text = md.read_text(errors="ignore")
        dates = [datetime.date.fromisoformat(d) for d in date_re.findall(text)[:5]
                if _valid_date(d)]
        if not dates:
            continue
        newest = max(dates)
        age_days = (today - newest).days
        if age_days > 60:
            findings.append(Finding(
                check="docs_gc", severity="advisory", file=f"workflow/{md.name}",
                message=f"newest date mentioned in this file is {age_days} days old",
                remediation="re-read and refresh if this plan/status doc is still active, "
                            "or mark it superseded/done"))
    return findings


def _valid_date(s: str) -> bool:
    import datetime
    try:
        datetime.date.fromisoformat(s)
        return True
    except ValueError:
        return False


def _today_iso() -> str:
    # Date.now()-equivalent is unavailable in some harnesses by design; this
    # module runs as a plain script (not inside a Workflow), so real time is
    # fine here — but keep the indirection so a future embedding can inject
    # a fixed date for reproducible test runs.
    import datetime
    return datetime.date.today().isoformat()


def run(repo_root: Path, gc: bool = False) -> list[Finding]:
    findings = _check_workflow_index(repo_root) + _check_dead_links(repo_root)
    if gc:
        findings += _check_gc_stale_dates(repo_root)
    return findings


if __name__ == "__main__":
    gc_mode = "--gc" in sys.argv
    fs = run(Path(__file__).resolve().parent.parent, gc=gc_mode)
    for f in fs:
        print(f.format(llm="--llm" in sys.argv))
    print(f"\n{len(fs)} finding(s) (advisory)")
    sys.exit(0)
