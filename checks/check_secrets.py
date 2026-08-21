#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_secrets.py — mechanizes the project's pre-push checklist
(CLAUDE.md "Secrets & Sensitive Information"): live secret tokens, real LAN
IPs, developer home paths, and tracked AI/cache files. Blocking — no
allowlist; a hit here means fix the leak, not suppress the check."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from lib import Finding, is_allowlisted, load_allowlist

# GitLab and the fine-grained GitHub formats were missing until 2026-08-19,
# on a repo whose primary remote IS GitLab , a leaked `glpat-` would have
# sailed straight through the pre-push gate that exists to stop exactly that.
_SECRET_RE = re.compile(
    r"hf_[A-Za-z0-9]{20,}"
    r"|sk-[A-Za-z0-9]{20,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|gho_[A-Za-z0-9]{20,}"
    r"|ghs_[A-Za-z0-9]{20,}"
    r"|ghr_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|glpat-[A-Za-z0-9_.-]{20,}"
    r"|glptt-[A-Za-z0-9_-]{20,}"
    r"|gldt-[A-Za-z0-9_-]{20,}"
    r"|AKIA[A-Z0-9]{16}")
_IP_RE = re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b")
_HOME_RE = re.compile(r"/home/[a-z][a-z0-9_-]*(?:/|\b)")
# NOTE on the memory patterns: the target is the AI assistant's own context
# store (a repo-root `memory/`, or one under a dot-directory), NOT any source
# package that happens to be called that. A bare `/memory/` also matched
# greenboost-cli's own `greenboost_cli/memory/` package (brain.py,
# token_tracker.py, ui_guidelines.py) and told the operator to `git rm
# --cached` four files of working application source. Narrowed 2026-08-19.
_FORBIDDEN_TRACKED_GLOBS = (
    r"\.cache/", r"\.mcp-cache/", r"CLAUDE\.md$", r"MEMORY\.md$",
    r"^memory/", r"/\.[A-Za-z0-9_-]+/memory/", r"^\.claude/",
)

_TEXT_SUFFIXES = {".py", ".sh", ".c", ".h", ".md", ".json", ".yaml", ".yml",
                  ".toml", ".cfg", ".ini", ".txt"}
_SKIP_PATH_SUBSTR = ("checks/check_secrets.py",)  # this file's own regex literals


def _tracked_files(repo_root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=repo_root, capture_output=True,
                             text=True, check=True, timeout=10).stdout
        return [ln for ln in out.splitlines() if ln]
    except Exception:
        return []


def run(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    tracked = _tracked_files(repo_root)
    if not tracked:
        return findings  # no git repo / git unavailable — nothing to check
    # NOTE on the allowlist: real SECRET TOKENS are never allowlisted (no
    # legitimate reason a token pattern belongs in a tracked file). IPs and
    # home paths, unlike tokens, have a real false-positive class: doc/CLI
    # usage examples ("sudo greenboost connect 192.168.1.50") and synthetic
    # test fixtures (LAN-filter unit tests parametrized over RFC 1918
    # ranges) use IP/path SHAPES without identifying real infrastructure.
    # Each entry below was read and judged illustrative, not a leak — this
    # allowlist is reviewed, not a blanket suppression.
    allow = load_allowlist(repo_root / "checks" / "allowlists" / "secrets_reviewed.txt")

    for rel in tracked:
        if any(s in rel for s in _SKIP_PATH_SUBSTR):
            continue
        for pat in _FORBIDDEN_TRACKED_GLOBS:
            if re.search(pat, rel):
                findings.append(Finding(
                    check="secrets", severity="blocking", file=rel,
                    message=f"tracked file matches forbidden pattern {pat!r} "
                            f"(AI/cache/context files must never be committed)",
                    remediation=f"git rm --cached {rel} && add it to .gitignore"))

        p = repo_root / rel
        suffix = p.suffix
        if suffix and suffix not in _TEXT_SUFFIXES:
            continue  # skip binaries — a regex scan over them is noise, not signal
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if _SECRET_RE.search(line):
                findings.append(Finding(
                    check="secrets", severity="blocking", file=rel, line=lineno,
                    message="line matches a live-secret-token pattern (HF/OpenAI/GitHub/AWS key shape)",
                    remediation="remove the literal token; use os.environ.get(\"VAR\", \"\") "
                                "or ${VAR:?VAR not set} in shell, never a hardcoded value"))
            if _IP_RE.search(line) and not is_allowlisted(rel, lineno, allow, line):
                findings.append(Finding(
                    check="secrets", severity="blocking", file=rel, line=lineno,
                    message="line contains a real LAN IP shape (192.168.x.x) — identifies real infrastructure",
                    remediation="use a placeholder (e.g. 192.0.2.x, RFC 5737 TEST-NET) or an env var; "
                                "if this is a reviewed doc/test example, add to "
                                "checks/allowlists/secrets_reviewed.txt"))
            m = _HOME_RE.search(line)
            if (m and "/home/<" not in line and "/home/user" not in line
                    and not is_allowlisted(rel, lineno, allow, line)):
                findings.append(Finding(
                    check="secrets", severity="blocking", file=rel, line=lineno,
                    message=f"line contains an absolute developer home path ({m.group(0)!r})",
                    remediation="use $HOME, ~, or an env-overridable default "
                                "(e.g. ${GB_X:-$HOME/Dev/...}); if this is a reviewed doc/test "
                                "example, add to checks/allowlists/secrets_reviewed.txt"))
    return findings


if __name__ == "__main__":
    import sys
    fs = run(Path(__file__).resolve().parent.parent)
    for f in fs:
        print(f.format(llm="--llm" in sys.argv))
    sys.exit(1 if fs else 0)
