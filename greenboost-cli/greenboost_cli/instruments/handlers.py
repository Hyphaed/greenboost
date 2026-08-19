"""Instrument (tool) implementations."""
from __future__ import annotations

import json
import os
import glob as _glob_mod
import shlex
import subprocess
import sys
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


def _ctx_char_budget(default_chars: int, floor_chars: int = 2000) -> int:
    """Scale a per-tool-result char cap to the LIVE served context window,
    instead of a fixed constant sized for a large window.

    Before this, every cap here (_READ_MAX_CHARS=30_000, Bash's 30_000,
    Grep/Semble's 20_000, Glob's unbounded 500 paths) was enormous relative
    to a small window: 30_000 chars is ~7_500 tokens, which alone is ~98%
    of the 7_680-token window this box was actually serving (confirmed
    live) — one Read call could consume the whole budget. A single tool
    result is capped at ~15% of the live window (4 chars/token, matching
    workflow/intelligence.py's _estimate_tokens), leaving room for the
    system prompt, other results in the same turn, and the response itself.
    `default_chars` stays the ceiling on large windows, so nothing shrinks
    for a box that can actually afford it."""
    try:
        from greenboost_cli.environment.settings import load_settings, gb_synapse_ctx
        ctx = gb_synapse_ctx(load_settings())
        if ctx:
            budget = int(ctx * 0.15 * 4)
            return max(floor_chars, min(default_chars, budget))
    except Exception:
        pass
    return default_chars


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


#: Failure kinds a tool result can carry, and whether retrying the SAME call
#: could plausibly succeed. Adapted from NemoClaw's validation-recovery
#: classifier (src/lib/validation-recovery.ts), which separates "transport"
#: failures it retries from validation failures it reports , see
#: third_party/nemoclaw_patterns/NOTICE.
#:
#: The distinction earns its place in the agent loop: a Bash timeout or a busy
#: device is worth one more attempt, while "file not found" will fail
#: identically forever, and the consecutive-error guard currently counts both
#: the same. Retrying a semantic failure burns a turn , minutes, on this
#: hardware , to learn nothing.
_TRANSIENT_MARKERS = (
    "timed out", "timeout", "temporarily unavailable", "resource busy",
    "connection reset", "connection refused", "try again", "device or resource busy",
    "too many open files", "no buffer space",
)
_SEMANTIC_MARKERS = (
    "not found", "is a directory", "not a directory", "old_string",
    "no such file", "already exists", "not unique", "appears",
    "byte-for-byte", "unknown", "invalid",
)


def classify_tool_failure(result: str) -> "tuple[str, bool]":
    """(kind, retry_worthwhile) for a tool result string.

    kind is "ok", "transient", "semantic" or "denied". Only "transient" is
    worth repeating verbatim; a semantic failure needs a DIFFERENT call, which
    is why every semantic message in this module names the tool to use next.
    """
    if not result:
        return "ok", False
    head = result[:400].lower()
    if not (head.startswith("error") or head.startswith("blocked")
            or "permission" in head.split("\n")[0]):
        return "ok", False
    if "blocked" in head or "denied" in head or "not permitted" in head:
        return "denied", False
    if any(m in head for m in _TRANSIENT_MARKERS):
        return "transient", True
    if any(m in head for m in _SEMANTIC_MARKERS):
        return "semantic", False
    return "semantic", False        # unknown errors are NOT retried blindly


def _nearby_suggestions(p: "Path", limit: int = 3) -> str:
    """Up to `limit` real entries whose name resembles the one that was missed.

    A missing path is far more often a typo or a stale memory than a genuinely
    absent file, and the directory listing is right there. Best-effort , an
    unreadable parent just means no suggestion, never an exception on top of
    the error being reported.
    """
    try:
        parent = p.parent if p.parent.exists() else None
        if parent is None:
            return ""
        import difflib

        names = [e.name for e in parent.iterdir()]
        close = difflib.get_close_matches(p.name, names, n=limit, cutoff=0.6)
        return ", ".join(f"{parent}/{c}" for c in close)
    except Exception:
        return ""


_READ_MAX_CHARS = 30_000   # per-call cap; matches Bash handler to bound context growth

#: What the session has read, and the file's identity when it read it:
#: {absolute path -> (mtime, size)}. Cleared per process, never persisted , it
#: describes THIS session's view, and a new session has not read anything.
_read_state: dict = {}


#: Files tracked for the staleness guard. Unattended-For-Days Must-Rule: one
#: entry per file ever read is unbounded over days of exploring a large tree.
#: Eviction is safe by construction , a forgotten file is simply not
#: staleness-checked, which is the pre-guard behaviour, never a false refusal.
MAX_READ_STATE = 4000


def _note_read(p: "Path") -> None:
    try:
        st = p.stat()
        if len(_read_state) >= MAX_READ_STATE:
            for k in list(_read_state)[:MAX_READ_STATE // 2]:
                _read_state.pop(k, None)
        _read_state[str(p.resolve())] = (st.st_mtime, st.st_size)
    except OSError:
        pass


def _changed_since_read(p: "Path") -> str:
    """"" when editing is safe; otherwise why it is not.

    The dangerous case is not the one the "old_string not found" branch below
    already catches. It is the one where old_string DOES still match, but the
    file changed after the model read it , another process, another window, a
    formatter, a git operation, or an unattended run editing the same file
    twice. The edit then lands on content the model has never seen, and
    succeeds, which is how an overwrite happens quietly.

    An unattended `/nonstop` run makes this materially more likely: it can work
    for hours while the user has the same file open elsewhere.

    A file this session never read is NOT refused here. Existing automation
    (the factory workers, subagent flows) writes files it never read, and the
    system prompt already instructs reading first; turning that into a hard
    error would break working paths to restate advice. What is enforced is the
    narrower, unambiguous case: you read it, and it is not what you read.
    """
    try:
        key = str(p.resolve())
    except OSError:
        return ""
    seen = _read_state.get(key)
    if seen is None:
        return ""
    try:
        st = p.stat()
    except OSError:
        return ""
    if (st.st_mtime, st.st_size) == seen:
        return ""
    return (f"Error: {p} changed since you read it "
            f"({seen[1]:,} bytes then, {st.st_size:,} now) , something else "
            f"has written to it.\n"
            f"  Your edit was NOT applied, because it would have been applied "
            f"to content you have not seen.\n"
            f"  Next: Read(file_path=\"{p}\") again, then redo the edit "
            f"against the current text.")


def handle_read(file_path: str, limit: int = None, offset: int = None) -> str:
    """Read a file.

    Failure messages here are written FOR THE MODEL, not for a log. A tool
    error that only says what went wrong costs a whole turn while the model
    guesses a recovery , and on this hardware a turn is minutes, not
    milliseconds. Live 2026-08-18: `Read $HOME/Dev/ai-forge` returned
    "is a directory", and the model spent the next turn discovering that it
    should have listed it instead. Every message below names the tool to call
    next, so the recovery is one step and not a search.
    """
    p = Path(file_path)
    if not p.exists():
        # A wrong path is usually a near-miss, so say what IS there.
        near = _nearby_suggestions(p)
        hint = f"  Did you mean: {near}" if near else ""
        return (f"Error: file not found: {file_path}\n"
                f"  Next: Glob(pattern=\"{p.name}\") to locate it, or "
                f"Bash(ls {p.parent}) to see that directory.{hint}")
    if p.is_dir():
        return (f"Error: {file_path} is a directory, and Read only reads files.\n"
                f"  Next: Bash(ls -la {file_path}) to list it, or "
                f"Glob(pattern=\"{file_path.rstrip('/')}/**/*.py\") to find files "
                f"inside it.")
    try:
        lines = p.read_text(errors="replace").splitlines(keepends=True)
        start = offset or 0
        chunk = lines[start : start + limit] if limit else lines[start:]
        if not chunk:
            return "(empty file)"
        result = "".join(f"{start + i + 1}\t{l}" for i, l in enumerate(chunk))
        _note_read(p)
        _cap = _ctx_char_budget(_READ_MAX_CHARS)
        if len(result) > _cap:
            result = (
                result[:_cap]
                + f"\n\n[... output truncated at {_cap:,} chars; "
                f"use offset={start + result[:_cap].count(chr(10))} to read more ...]\n"
            )
        return result
    except Exception as e:
        return f"Error: {e}"


def _snapshot_before_write(file_path) -> None:
    """Record the pre-edit content so the change is revertable.

    Best-effort by construction: an audit trail must never be the reason an
    edit fails. A snapshot that could not be taken shows up as
    non-revertable in /changes rather than as a silent gap.
    """
    try:
        from greenboost_cli.core.file_history import snapshot
        snapshot(file_path)
    except Exception:
        pass


def handle_write(file_path: str, content: str) -> str:
    _snapshot_before_write(file_path)
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
        near = _nearby_suggestions(p)
        hint = f"  Did you mean: {near}" if near else ""
        return (f"Error: file not found: {file_path}\n"
                f"  Next: Write(file_path=...) to create it, or "
                f"Glob(pattern=\"{p.name}\") to find where it really is.{hint}")
    stale = _changed_since_read(p)
    if stale:
        return stale
    _snapshot_before_write(p)
    try:
        content = p.read_text()
        count = content.count(old_string)
        if count == 0:
            # "not found" alone sends the model guessing. Whitespace and a
            # stale read are the two causes that actually happen, so separate
            # them: a match that differs only in indentation is a different
            # fix from a string that is simply not in the file.
            # Normalise the WHOLE string, not line by line: old_string is very
            # often multi-line, and a per-line comparison can never match it.
            squashed = " ".join(old_string.split())
            if squashed and squashed in " ".join(content.split()):
                # Show the model the real text, at the real indentation, so the
                # retry is a copy rather than another guess.
                first = (old_string.split("\n", 1)[0] or "").strip()
                anchor = next((ln for ln in content.splitlines()
                               if first and first in ln), "")
                where = f"\n  It starts at: {anchor[:200]!r}" if anchor else ""
                return ("Error: old_string is present but not BYTE-for-byte , it "
                        "matches once whitespace is normalised, so the "
                        "indentation or line breaks differ."
                        f"{where}\n"
                        f"  Next: Read(file_path=\"{file_path}\") and copy the "
                        f"text verbatim, leading spaces included.")
            return (f"Error: old_string not found in {file_path}.\n"
                    f"  Next: Read(file_path=\"{file_path}\") to see the current "
                    f"content , it may have changed since you last read it, or "
                    f"Grep(pattern=..., path=\"{file_path}\") to find the right "
                    f"anchor.")
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


def handle_shell(command: str, timeout: int = 120,
                 run_in_background: bool = False) -> str:
    if run_in_background:
        # Deliberately no safety-tier change: a command is exactly as
        # dangerous backgrounded as it is in the foreground, and it has
        # already passed the same approval gate in dispatch() before reaching
        # here. What changes is only who waits for it.
        from greenboost_cli.instruments.background import start
        # Same working directory the foreground path would have used ,
        # a factory task scopes its Bash cwd per thread, and a
        # backgrounded command must not escape that scoping.
        jid = start(command, cwd=getattr(_bash_cwd_tls, "value", None))
        return (f"[started in the background as {jid}]\n"
                f"Carry on with other work. TaskOutput(task_id=\"{jid}\") "
                f"returns what it has printed since you last looked; "
                f"TaskStop(task_id=\"{jid}\") kills it.")
    return _handle_shell_foreground(command, timeout)


def _handle_shell_foreground(command: str, timeout: int = 120) -> str:
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
            executable="/bin/bash",
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
        # Truncate to prevent context explosion (30K matches Claude Code's
        # default; _ctx_char_budget shrinks it against a small served window)
        _limit = _ctx_char_budget(30_000)
        if len(out) > _limit:
            half = _limit // 2
            out = (
                out[:half]
                + f"\n\n[... {len(out) - _limit:,} chars truncated ...]\n\n"
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
        # 500 absolute paths has no char cap at all — on a small served
        # window that alone can be ~20_000 chars (~5_000 tokens). Cap chars
        # the same way every other tool result is capped, not just count.
        out = "\n".join(str(m) for m in matches[:500])
        _cap = _ctx_char_budget(20_000)
        if len(out) > _cap:
            shown = out[:_cap].count("\n") + 1
            out = out[:_cap] + f"\n\n[... {len(matches) - shown} more matches truncated ...]\n"
        return out
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
        return out[:_ctx_char_budget(20_000)] if out else "No matches found"
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
    return "\n".join(out)[:_ctx_char_budget(20_000)]


def handle_fetch_url(url: str, prompt: str = None) -> str:
    # WebFetch is in _ALWAYS_APPROVED , it never asks. Over a multi-day
    # unattended run that is a lot of unreviewed outbound requests, and the URL
    # can come from a fetched page or a file rather than from the user. Judge
    # the destination before going there.
    try:
        from greenboost_cli.instruments.url_guard import check_url
        refusal = check_url(url)
        if refusal:
            return refusal
    except Exception:
        pass                    # a broken guard must not break fetching
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


# ── Skill: user-authored procedures and scripts ───────────────────────────────

def _skill_dirs_and_entries(settings=None):
    from greenboost_cli.skill.router import (
        discover_all_skill_dirs, discover_skills_multi)
    dirs = discover_all_skill_dirs(settings or {})
    return dirs, discover_skills_multi(dirs)


def _normalise_skill_name(n: str) -> str:
    return "".join(ch for ch in (n or "").lower() if ch.isalnum())


def handle_skill(params: dict) -> str:
    """Run or load a user-authored skill.

    Two kinds of skill, decided by what is in the folder , this is the whole
    point of the instrument:

    * **Script skill.** The folder has a `run.sh` or `run.py`. It is executed
      with the caller's `args`, and its OUTPUT comes back as the tool result.
      This is how a personal script becomes a first-class tool the model can
      call, without writing a new MCP server for it.
    * **Procedure skill.** No runnable entrypoint, so the SKILL.md body is
      returned for the model to follow.

    Why this is worth having on a local box specifically: 238 MCP tool schemas
    cost roughly 7k prompt tokens on EVERY request. A skill costs one line in
    the prompt , its name and description , and its body or its script only
    materialises when it is actually invoked. Prefill is super-linear in prompt
    length here, so moving rarely-used capability behind this door is the
    cheapest capability-per-token the CLI has.
    """
    import subprocess
    from greenboost_cli.skill.router import load_skill_body

    name = (params or {}).get("name") or ""
    args = (params or {}).get("args") or ""
    if not name:
        return handle_skill_list()

    try:
        _dirs, entries = _skill_dirs_and_entries(params.get("_settings"))
    except Exception as e:
        return f"Error: could not read the skills directories ({e})."

    if not entries:
        return ("No skills are installed. A skill is a folder containing "
                "SKILL.md (name + description in YAML frontmatter); add an "
                "executable run.sh or run.py to make it a script skill.")

    want = _normalise_skill_name(name)
    match = next((e for e in entries if _normalise_skill_name(e.name) == want), None)
    if match is None:
        near = ", ".join(sorted(e.name for e in entries)[:12])
        return (f"Error: no skill named '{name}'. Available: {near}"
                f"{' , …' if len(entries) > 12 else ''}")

    skill_dir = Path(match.path).parent
    for entry in ("run.sh", "run.py"):
        script = skill_dir / entry
        if not script.is_file():
            continue
        cmd = (["bash", str(script)] if entry == "run.sh"
               else [sys.executable, str(script)])
        if args:
            cmd += shlex.split(args) if isinstance(args, str) else [str(args)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=600, cwd=str(skill_dir))
        except subprocess.TimeoutExpired:
            return (f"Error: skill '{match.name}' timed out after 600s. Its "
                    f"script is {script}.")
        except Exception as e:
            return f"Error: could not run skill '{match.name}' ({e})."
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return (f"Skill '{match.name}' exited {proc.returncode}.\n"
                    f"{err or out or '(no output)'}")
        return out or err or f"Skill '{match.name}' finished with no output."

    body = load_skill_body(Path(match.path))
    return (f"Skill '{match.name}' , follow this procedure:\n\n{body}")


def handle_skill_list(_params: dict | None = None) -> str:
    """Every installed skill, name and description only."""
    try:
        _dirs, entries = _skill_dirs_and_entries()
    except Exception as e:
        return f"Error: could not read the skills directories ({e})."
    if not entries:
        return "No skills are installed."
    return "\n".join(f"- {e.name}: {e.description}" for e in sorted(
        entries, key=lambda x: x.name))
