"""
/init — Initialize or refresh a CLAUDE.md for the current project.

Scans project structure and uses the model to write a comprehensive
CLAUDE.md covering: overview, tech stack, commands, architecture, conventions.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import emit_ok, emit_info, emit_warn


def _gather_project_context() -> str:
    """Collect project signals: manifests, structure, git history, README."""
    cwd = Path.cwd()
    parts: list[str] = [f"Project root: {cwd}\n"]

    # ── Top-level directory listing ──────────────────────────────────────────
    try:
        entries = sorted(
            e.name for e in cwd.iterdir()
            if not e.name.startswith(".") or e.name in (".github",)
        )
        parts.append("Top-level entries:\n" + "  ".join(entries[:40]) + "\n")
    except Exception:
        pass

    # ── Key manifest files ───────────────────────────────────────────────────
    manifests = [
        "pyproject.toml", "setup.py", "setup.cfg",
        "package.json", "package-lock.json",
        "go.mod", "Cargo.toml", "pom.xml", "build.gradle",
        "requirements.txt", "Pipfile", "poetry.lock",
        "Makefile", "CMakeLists.txt", "Dockerfile",
        ".github/workflows",
    ]
    found_manifests: list[str] = []
    for name in manifests:
        p = cwd / name
        if p.exists():
            found_manifests.append(name)
            # Include small manifests verbatim
            if p.is_file() and p.stat().st_size < 6000:
                try:
                    parts.append(f"\n--- {name} ---\n{p.read_text(encoding='utf-8', errors='replace')}\n")
                except Exception:
                    pass
    if found_manifests:
        parts.append(f"\nDetected manifests: {', '.join(found_manifests)}\n")

    # ── README excerpt ───────────────────────────────────────────────────────
    for readme in ("README.md", "README.rst", "README.txt", "README"):
        rp = cwd / readme
        if rp.exists():
            try:
                parts.append(f"\n--- {readme} (first 2000 chars) ---\n{rp.read_text(encoding='utf-8', errors='replace')[:2000]}\n")
            except Exception:
                pass
            break

    # ── Git context ──────────────────────────────────────────────────────────
    def _git(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=8, cwd=str(cwd))
            return r.stdout.strip()
        except Exception:
            return ""

    branch = _git(["git", "branch", "--show-current"])
    log    = _git(["git", "log", "--oneline", "-20"])
    if branch:
        parts.append(f"\nGit branch: {branch}\nRecent commits:\n{log}\n")

    # ── Source language detection ────────────────────────────────────────────
    try:
        exts: dict[str, int] = {}
        for p in cwd.rglob("*"):
            if p.is_file() and not any(seg.startswith(".") for seg in p.parts[-4:]):
                ext = p.suffix.lower()
                if ext:
                    exts[ext] = exts.get(ext, 0) + 1
        top_exts = sorted(exts.items(), key=lambda x: -x[1])[:10]
        if top_exts:
            parts.append("\nFile types (by count): " + ", ".join(f"{e}:{n}" for e, n in top_exts) + "\n")
    except Exception:
        pass

    return "\n".join(parts)[:8000]


def cmd_init(args: str, session, settings: dict) -> bool:
    """Initialize or refresh a CLAUDE.md for the current project."""
    from greenboost_cli.terminal.repl import process_query

    cwd      = Path.cwd()
    force    = "--force" in args or "-f" in args
    target   = cwd / "CLAUDE.md"

    if target.exists() and not force:
        existing = ""
        try:
            existing = target.read_text(encoding="utf-8")
        except Exception:
            pass
        emit_info(f"CLAUDE.md already exists at {target}. Updating it (use --force to fully rewrite).")
        ctx = _gather_project_context()
        prompt = (
            f"Review and improve the CLAUDE.md for the project at `{cwd}`.\n\n"
            f"CURRENT CLAUDE.md:\n```markdown\n{existing[:4000]}\n```\n\n"
            f"PROJECT CONTEXT:\n{ctx}\n\n"
            "Update the file so it accurately reflects the current codebase. "
            "Keep it concise but complete. Preserve any user-written sections. "
            f"Write the improved version to: `{target}`\n"
            "Use the Write tool (absolute path). Report what you changed."
        )
    else:
        if target.exists():
            emit_warn("Rewriting existing CLAUDE.md (--force).")
        ctx = _gather_project_context()
        prompt = (
            f"Create a CLAUDE.md for the project at `{cwd}`.\n\n"
            f"PROJECT CONTEXT:\n{ctx}\n\n"
            "Write a comprehensive CLAUDE.md covering exactly these sections "
            "(use the exact headings):\n\n"
            "```\n"
            "# <Project Name>\n\n"
            "## What it is\n"
            "<1-3 sentences: purpose and primary use case>\n\n"
            "## Tech Stack\n"
            "<bullet list: languages, frameworks, key libraries>\n\n"
            "## Common Commands\n"
            "<code blocks for: install, build/compile, run dev server, run tests, lint/format>\n\n"
            "## Architecture\n"
            "<key modules/directories and their roles — 5-10 bullets>\n\n"
            "## Key Conventions\n"
            "<coding patterns, naming conventions, important rules — 3-8 bullets>\n\n"
            "## Important Notes\n"
            "<non-obvious things a new contributor must know — 3-5 bullets>\n"
            "```\n\n"
            f"Write this to: `{target}` using the Write tool (absolute path). "
            "Be concise — the file should be 60-150 lines. "
            "Report the path when done."
        )

    process_query(prompt, session, settings)
    return True


register_command("init", cmd_init, "Initialize CLAUDE.md for this project  (/init [--force])")
