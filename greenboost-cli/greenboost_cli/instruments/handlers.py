"""Instrument (tool) implementations."""
from __future__ import annotations

import json
import os
import glob as _glob_mod
import subprocess
import threading
from pathlib import Path

# Persistent working directory across Bash calls within one process lifetime.
# The model can run `cd /some/dir` and subsequent calls stay in that directory.
# REPL usage is single-threaded, so this module-global stays authoritative for
# it (repl.py reads it directly for the prompt). The AI Factory runs multiple
# agents concurrently as real OS threads in the same process (factory.py) — a
# bare global here would let one task's `cd` bleed into another's shell calls.
# _bash_cwd_tls carries the thread-local override; handle_shell prefers it
# when present and only mirrors back to the module-global from the main
# thread, so the REPL's own behavior is unchanged.
_bash_cwd: str = ""
_bash_cwd_tls = threading.local()


def set_task_bash_cwd(path: str) -> None:
    """Set the calling thread's Bash working directory (factory task scoping)."""
    _bash_cwd_tls.value = path

# ── In-session todo list (resets on restart, like Claude Code) ────────────────
_session_todos: list = []


def handle_memory_read(file: str = None) -> str:
    """Read CLAUDE.md memory files (project + global)."""
    targets: list[Path] = []
    if file:
        targets.append(Path(file))
    else:
        targets = [
            Path.cwd() / "CLAUDE.md",
            Path.home() / ".claude" / "CLAUDE.md",
        ]
    parts: list[str] = []
    for p in targets:
        if p.exists():
            try:
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(f"=== {p} ===\n{content}")
            except Exception as ex:
                parts.append(f"Error reading {p}: {ex}")
    return "\n\n".join(parts) if parts else "(no CLAUDE.md memory found)"


def handle_memory_write(key: str, content: str, scope: str = "project") -> str:
    """Write a section to CLAUDE.md memory (project or global scope)."""
    import re
    if scope == "global":
        target = Path.home() / ".claude" / "CLAUDE.md"
    else:
        target = Path.cwd() / "CLAUDE.md"

    target.parent.mkdir(parents=True, exist_ok=True)
    section = f"\n## {key}\n{content.strip()}\n"

    if target.exists():
        existing = target.read_text(encoding="utf-8")
        pattern = re.compile(
            rf"^## {re.escape(key)}\s*\n.*?(?=^## |\Z)",
            re.MULTILINE | re.DOTALL,
        )
        if pattern.search(existing):
            new_content = pattern.sub(section.lstrip("\n"), existing)
        else:
            new_content = existing.rstrip("\n") + "\n" + section
    else:
        new_content = f"# Project Memory\n{section}"

    target.write_text(new_content, encoding="utf-8")
    return f"Memory written → {target}  (section: {key})"


def handle_todo_read() -> str:
    return json.dumps(_session_todos, indent=2) if _session_todos else "[]"


def handle_todo_write(todos: list) -> str:
    _session_todos.clear()
    _session_todos.extend(todos if isinstance(todos, list) else [])
    counts: dict = {}
    for t in _session_todos:
        s = t.get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
    return json.dumps({"updated": len(_session_todos), "counts": counts})


_READ_MAX_CHARS = 30_000   # per-call cap; matches Bash handler to bound context growth

def handle_read(file_path: str, limit: int = None, offset: int = None) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    if p.is_dir():
        return f"Error: {file_path} is a directory"
    try:
        lines = p.read_text(errors="replace").splitlines(keepends=True)
        start = offset or 0
        chunk = lines[start : start + limit] if limit else lines[start:]
        if not chunk:
            return "(empty file)"
        result = "".join(f"{start + i + 1}\t{l}" for i, l in enumerate(chunk))
        if len(result) > _READ_MAX_CHARS:
            result = (
                result[:_READ_MAX_CHARS]
                + f"\n\n[... output truncated at {_READ_MAX_CHARS:,} chars; "
                f"use offset={start + result[:_READ_MAX_CHARS].count(chr(10))} to read more ...]\n"
            )
        return result
    except Exception as e:
        return f"Error: {e}"


def handle_write(file_path: str, content: str) -> str:
    p = Path(file_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        lc = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"Wrote {lc} lines to {file_path}"
    except Exception as e:
        return f"Error: {e}"


def handle_edit(
    file_path: str, old_string: str, new_string: str, replace_all: bool = False
) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    try:
        content = p.read_text()
        count = content.count(old_string)
        if count == 0:
            return "Error: old_string not found in file"
        if count > 1 and not replace_all:
            return (
                f"Error: old_string appears {count} times. "
                "Provide more context to make it unique, or use replace_all=true."
            )
        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        p.write_text(new_content)
        label = f"all {count}" if replace_all else "1"
        return f"Replaced {label} occurrence(s) in {file_path}"
    except Exception as e:
        return f"Error: {e}"


def handle_shell(command: str, timeout: int = 120) -> str:
    global _bash_cwd
    thread_cwd = getattr(_bash_cwd_tls, "value", "")
    effective_cwd = thread_cwd or _bash_cwd
    run_cwd = effective_cwd if (effective_cwd and os.path.isdir(effective_cwd)) else os.getcwd()

    # Append a cwd sentinel so we can track `cd` across calls
    _MARKER = "::CWD::"
    tracked = f"{command}\nprintf '\\n{_MARKER}%s\\n' \"$(pwd)\""

    try:
        r = subprocess.run(
            tracked,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=run_cwd,
            env=os.environ.copy(),
        )
        out = r.stdout

        # Parse and strip the cwd sentinel line
        lines = out.splitlines()
        clean: list[str] = []
        for line in lines:
            if line.startswith(_MARKER):
                new_cwd = line[len(_MARKER):].strip()
                if new_cwd and os.path.isdir(new_cwd):
                    if thread_cwd:
                        _bash_cwd_tls.value = new_cwd
                    else:
                        _bash_cwd = new_cwd
            else:
                clean.append(line)
        out = "\n".join(clean)

        if r.stderr:
            out += ("\n" if out.strip() else "") + "[stderr]\n" + r.stderr
        out = out.strip() or "(no output)"
        # Truncate to prevent context explosion (matches Claude Code's 30K limit)
        _LIMIT = 30_000
        if len(out) > _LIMIT:
            half = _LIMIT // 2
            out = (
                out[:half]
                + f"\n\n[... {len(out) - _LIMIT:,} chars truncated ...]\n\n"
                + out[-half:]
            )
        return out
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def handle_glob(pattern: str, path: str = None) -> str:
    base = Path(path) if path else Path.cwd()
    try:
        matches = sorted(base.glob(pattern))
        if not matches:
            return "No files matched"
        return "\n".join(str(m) for m in matches[:500])
    except Exception as e:
        return f"Error: {e}"


_rg_available: bool | None = None

def ripgrep_available() -> bool:
    global _rg_available
    if _rg_available is None:
        try:
            subprocess.run(["rg", "--version"], capture_output=True, check=True)
            _rg_available = True
        except Exception:
            _rg_available = False
    return _rg_available


def handle_grep(
    pattern: str,
    path: str = None,
    glob: str = None,
    output_mode: str = "files_with_matches",
    case_insensitive: bool = False,
    context: int = 0,
) -> str:
    use_rg = ripgrep_available()
    cmd = ["rg" if use_rg else "grep", "--no-heading"]
    if case_insensitive:
        cmd.append("-i")
    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:
        cmd.append("-n")
        if context:
            cmd += ["-C", str(context)]
    if glob:
        cmd += (["--glob", glob] if use_rg else ["--include", glob])
    cmd.append(pattern)
    cmd.append(path or str(Path.cwd()))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        return out[:20000] if out else "No matches found"
    except Exception as e:
        return f"Error: {e}"


def handle_semble(query: str, repo: str = None, top_k: int = 5,
                  content: str = "code") -> str:
    import json, shutil
    if not shutil.which("semble"):
        return "Error: semble not installed (use Grep for literal search)"
    cmd = ["semble", "search", query, repo or str(Path.cwd()),
           "-k", str(top_k), "--content", content]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        data = json.loads(r.stdout or "{}")
    except Exception as e:
        return f"Error: {e}"
    if "error" in data or not data.get("results"):
        return "No matches found"
    out = []
    for res in data["results"]:
        chunk = res.get("chunk", res)   # CLI wraps in chunk; MCP flattens
        fp    = chunk.get("file_path", "?")
        sl    = chunk.get("start_line", "?")
        el    = chunk.get("end_line", "?")
        out.append(f"{fp}:{sl}-{el}  (score {res.get('score', 0):.2f})")
        if chunk.get("content"):
            out.append(chunk["content"].rstrip())
        out.append("")
    return "\n".join(out)[:20000]


def handle_fetch_url(url: str, prompt: str = None) -> str:
    from greenboost_cli.instruments.scrapling_utils import fetch_url as _scrapling_fetch
    return _scrapling_fetch(url)


def handle_screenshot(url: str, output_path: str, width: int = 1280,
                       height: int = 800, full_page: bool = False) -> str:
    from greenboost_cli.instruments.screenshot_utils import capture_screenshot
    return capture_screenshot(url, output_path, width=width, height=height,
                               full_page=full_page)


def handle_web_query(query: str) -> str:
    from greenboost_cli.instruments.scrapling_utils import search_ddg

    hits = search_ddg(query, max_results=8)
    if not hits:
        return "No results found"

    parts = []
    for h in hits:
        block = [f"**{h['title']}**", h["url"]]
        if h.get("snippet"):
            block.append(h["snippet"])
        parts.append("\n".join(block))
    return "\n\n".join(parts)
