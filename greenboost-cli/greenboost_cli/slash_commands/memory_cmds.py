"""Memory-related slash commands: /goals /history /snapshot /project /tokens."""
from __future__ import annotations

from greenboost_cli.terminal.commands import register_command


def _goals(args: str, session, settings: dict) -> None:
    from greenboost_cli.memory.brain import project_dir, load_goals, add_goal, remove_goal, print_goals

    pdir = project_dir(settings.get("active_project"))
    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"

    if sub in ("list", ""):
        print_goals(pdir)

    elif sub == "add":
        if len(parts) < 2:
            print("  Usage: /goals add \"goal text\" [--priority N]")
            return
        rest = parts[1]
        priority = 5
        if "--priority" in rest:
            idx = rest.index("--priority")
            try:
                priority = int(rest[idx:].split()[1])
            except (IndexError, ValueError):
                pass
            rest = rest[:idx].strip().strip('"').strip("'")
        else:
            rest = rest.strip('"').strip("'")
        entry = add_goal(pdir, rest, priority)
        print(f"  ✓  Goal #{entry['id']} added (P{entry['priority']}): {entry['text']}")

    elif sub == "remove":
        if len(parts) < 2:
            print("  Usage: /goals remove <id>")
            return
        try:
            gid = int(parts[1].strip())
        except ValueError:
            print("  Goal ID must be a number.")
            return
        if remove_goal(pdir, gid):
            print(f"  ✓  Goal #{gid} removed.")
        else:
            print(f"  ✗  Goal #{gid} not found.")

    else:
        print("  Usage: /goals [list|add|remove]")


def _history_show(args: str, session, settings: dict) -> None:
    from greenboost_cli.memory.brain import project_dir, print_history

    pdir = project_dir(settings.get("active_project"))
    n = 10
    try:
        n = int(args.strip()) if args.strip() else 10
    except ValueError:
        pass
    print_history(pdir, n)


def _history_add(args: str, session, settings: dict) -> None:
    from greenboost_cli.memory.brain import project_dir, append_history, VALID_CATEGORIES

    pdir = project_dir(settings.get("active_project"))
    if not args.strip():
        print("  Usage: /history-add \"text\" [--category note|decision|milestone|blocker|resolved]")
        return

    rest = args.strip()
    category = "note"
    if "--category" in rest:
        idx = rest.index("--category")
        try:
            category = rest[idx:].split()[1]
        except IndexError:
            pass
        rest = rest[:idx].strip().strip('"').strip("'")
    else:
        rest = rest.strip('"').strip("'")

    if category not in VALID_CATEGORIES:
        category = "note"
    append_history(pdir, rest, category)
    print(f"  ✓  History entry added [{category}]")


def _snapshot(args: str, session, settings: dict) -> None:
    from greenboost_cli.memory.brain import (
        project_dir, print_snapshot, write_snapshot
    )

    pdir = project_dir(settings.get("active_project"))
    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else "show"

    if sub in ("show", ""):
        print_snapshot(pdir)

    elif sub == "set":
        if len(parts) < 2:
            print("  Usage: /snapshot set \"content\"")
            return
        write_snapshot(pdir, parts[1].strip('"').strip("'"))
        print("  ✓  Snapshot updated.")

    else:
        print("  Usage: /snapshot [show|set]")


def _project(args: str, session, settings: dict) -> None:
    """Show or switch the active project."""
    from greenboost_cli.memory.brain import project_dir, GLOBAL_DIR

    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else "show"

    if sub in ("show", ""):
        pdir = project_dir(settings.get("active_project"))
        print(f"  Active project: {pdir.name}")
        print(f"  State dir:      {pdir}")

    elif sub == "switch":
        if len(parts) < 2:
            print("  Usage: /project switch <name>")
            return
        name = parts[1].strip()
        settings["active_project"] = name
        from greenboost_cli.environment.settings import save_settings
        save_settings(settings)
        print(f"  ✓  Active project set to: {name}")

    elif sub == "list":
        if GLOBAL_DIR.exists():
            projects = sorted(d.name for d in GLOBAL_DIR.iterdir() if d.is_dir())
            if projects:
                active = settings.get("active_project")
                for p in projects:
                    marker = "✓" if p == active else " "
                    print(f"  {marker}  {p}")
            else:
                print("  No projects yet.")
        else:
            print("  No projects yet.")

    else:
        print("  Usage: /project [show|switch|list]")


def _tokens(args: str, session, settings: dict) -> None:
    from greenboost_cli.memory.brain import project_dir
    from greenboost_cli.memory.token_tracker import get_totals, _save, _load

    pdir = project_dir(settings.get("active_project"))
    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else "show"

    if sub in ("show", ""):
        t = get_totals(pdir)
        print(f"  Token usage for: {pdir.name}")
        print()
        print(f"  Today    API: {t['today_api']:>12,}  Local: {t['today_local']:,}")
        print(f"  Total    API: {t['total_api']:>12,}  Local: {t['total_local']:,}")

    elif sub == "reset":
        _save(pdir, {"sessions": [], "totals": {"api": 0, "local": 0}})
        print("  ✓  Token usage reset.")

    else:
        print("  Usage: /tokens [show|reset]")


def _history_search(args: str, session, settings: dict) -> None:
    from greenboost_cli.memory.brain import project_dir, search_history

    pdir  = project_dir(settings.get("active_project"))
    query = args.strip()
    if not query:
        print("  Usage: /history-search <query>")
        return

    results = search_history(pdir, query)
    if not results:
        print(f"  No history entries matching '{query}' in {pdir.name}")
        return

    print(f"  {len(results)} match(es) for '{query}' in {pdir.name}:")
    print()
    for r in results:
        header = r.get("header", "")
        body   = r.get("body", "")
        print(f"  \033[38;2;48;200;255m## {header}\033[0m")
        if body:
            for line in body.splitlines()[:4]:
                print(f"    {line}")
        print()


def _save_session(args: str, session, settings: dict) -> None:
    """Write a structured plan_session.md for session recovery on next launch."""
    from pathlib import Path
    from greenboost_cli.environment.settings import GB_HOME
    from greenboost_cli.memory.brain import project_dir

    pdir = project_dir(settings.get("active_project"))
    save_path = GB_HOME / "projects" / pdir.name / "plan_session.md"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else "show"

    if sub == "clear":
        if save_path.exists():
            save_path.unlink()
            print("  ✓  plan_session.md cleared.")
        else:
            print("  No plan_session.md for this project.")
        return

    if sub == "show":
        if save_path.exists():
            print(save_path.read_text(encoding="utf-8"))
        else:
            print(f"  No plan_session.md at {save_path}")
        return

    # "save" or bare — inject a session-save task into the next model turn
    # by writing a sentinel that repl picks up, or just open $EDITOR on the template
    template = (
        f"# Session — {pdir.name}\n\n"
        "## Goal\n\n\n"
        "## Accomplished\n\n\n"
        "## Current State\n\n\n"
        "## Key Decisions\n\n\n"
        "## Blockers\n\n\n"
        "## Next Steps\n\n\n"
        "## Key Context\n\n"
    )

    import subprocess
    import os
    import tempfile
    # Write template to temp file, open editor, then save
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tf:
        if save_path.exists():
            tf.write(save_path.read_text(encoding="utf-8"))
        else:
            tf.write(template)
        tmp = tf.name

    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))
    try:
        subprocess.call([editor, tmp])
        content = Path(tmp).read_text(encoding="utf-8").strip()
        if content:
            save_path.write_text(content + "\n", encoding="utf-8")
            print(f"  ✓  Session saved → {save_path}")
        else:
            print("  Empty file — not saved.")
    finally:
        Path(tmp).unlink(missing_ok=True)


def _resume(args: str, session, settings: dict) -> None:
    from greenboost_cli.environment.settings import SESSIONS_PATH
    from greenboost_cli.terminal.theme import emit_err

    if args.strip():
        # Named session
        fname = args.strip()
        if not fname.endswith(".json"):
            fname += ".json"
        path = SESSIONS_PATH / fname
        if not path.exists():
            emit_err(f"Session not found: {path}")
            return
    else:
        # Most recent session
        sessions = sorted(SESSIONS_PATH.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sessions:
            emit_err("No saved sessions found. Use /save first.")
            return
        path = sessions[0]

    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    session.messages            = data.get("messages", [])
    session.turn_count          = data.get("turn_count", 0)
    session.total_input_tokens  = data.get("total_input_tokens", 0)
    session.total_output_tokens = data.get("total_output_tokens", 0)
    print(f"  ✓  Resumed ← {path.stem}  ({len(session.messages)} messages)")


register_command("goals",           _goals,          "Manage project goals  (/goals add|remove|list)")
register_command("history-show",    _history_show,   "Show project history log  (/history-show [N])")
register_command("history-add",     _history_add,    "Add project history entry  (/history-add \"text\")")
register_command("history-search",  _history_search, "Search history log  (/history-search <query>)")
register_command("snapshot",        _snapshot,       "Context snapshot  (/snapshot show|set)")
register_command("project",         _project,        "Active project  (/project show|switch|list)")
register_command("tokens",          _tokens,         "Token usage stats  (/tokens show|reset)")
def _add(args: str, session, settings: dict) -> None:
    """Inject a file's content (or glob) into the conversation as context.

    /add path/to/file.py           — inject one file
    /add src/**/*.ts               — inject all matching files (capped at 20)
    /add path/to/file.py:50-120    — inject lines 50-120 only
    """
    from pathlib import Path
    import glob as _glob

    if not args.strip():
        print("  Usage: /add <path|glob> [:<start>-<end>]")
        return

    # Parse optional line range suffix
    raw = args.strip()
    line_range: tuple[int, int] | None = None
    if ":" in raw and not raw.startswith("http"):
        raw, _, range_str = raw.rpartition(":")
        try:
            if "-" in range_str:
                s, e = range_str.split("-", 1)
                line_range = (int(s), int(e))
            else:
                line_range = (int(range_str), int(range_str))
        except ValueError:
            raw = args.strip()   # restore: colon wasn't a range

    # Glob expansion
    cwd = Path.cwd()
    matches = sorted(cwd.glob(raw)) if "*" in raw or "?" in raw else [cwd / raw if not Path(raw).is_absolute() else Path(raw)]
    matches = [m for m in matches if m.is_file()][:20]

    if not matches:
        print(f"  No files found matching: {raw}")
        return

    injected: list[str] = []
    for path in matches:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            if line_range:
                s, e = line_range
                chunk = lines[s - 1:e]
                content = "".join(f"{s + i}\t{l}" for i, l in enumerate(chunk))
                label = f"{path} (lines {s}-{min(e, len(lines))})"
            else:
                content = "".join(f"{i+1}\t{l}" for i, l in enumerate(lines))
                label = str(path)
            lang = path.suffix.lstrip(".")
            injected.append(f"### {label}\n```{lang}\n{content[:20000]}\n```")
        except Exception as ex:
            injected.append(f"### {path}\n(Error reading: {ex})")

    block = "\n\n".join(injected)
    # Inject as a user message so the model sees it in context
    session.messages.append({"role": "user", "content": f"[Context added via /add]\n\n{block}"})
    session.messages.append({"role": "assistant", "content": f"I've received {len(matches)} file(s) as context: {', '.join(m.name for m in matches)}."})
    print(f"  ✓  {len(matches)} file(s) injected into context")
    for m in matches:
        print(f"     {m}")


def _memory(args: str, session, settings: dict) -> None:
    """View or edit CLAUDE.md memory files.

    /memory           — show both project and global CLAUDE.md
    /memory project   — show ./CLAUDE.md only
    /memory global    — show ~/.claude/CLAUDE.md only
    /memory edit      — open project CLAUDE.md in $EDITOR
    /memory edit global — open global CLAUDE.md in $EDITOR
    /memory write <section> -- <text>  — write/update a section non-interactively
    """
    from pathlib import Path
    from greenboost_cli.terminal.theme import emit_ok, emit_err, console, TEAL, GRAY, DIM, LIME, SEPARATOR

    parts = args.strip().split(None, 1)
    sub   = parts[0].lower() if parts else ""

    project_md = Path.cwd() / "CLAUDE.md"
    global_md  = Path.home() / ".claude" / "CLAUDE.md"

    def _show_file(p: Path, label: str) -> None:
        if not p.exists():
            console.print(f"  [{DIM}]{label}: (none)[/]")
            return
        try:
            text = p.read_text(encoding="utf-8").strip()
            console.print(f"  [{TEAL}]── {label}: {p}[/]")
            console.print(SEPARATOR)
            console.print(f"[{GRAY}]{text}[/]")
            console.print()
        except Exception as ex:
            emit_err(f"Cannot read {p}: {ex}")

    if sub in ("", "show"):
        _show_file(project_md, "Project")
        _show_file(global_md,  "Global")

    elif sub == "project":
        _show_file(project_md, "Project")

    elif sub == "global":
        _show_file(global_md, "Global")

    elif sub == "edit":
        import subprocess, os, tempfile
        rest  = parts[1].lower().strip() if len(parts) > 1 else ""
        target = global_md if rest == "global" else project_md
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("# Project Memory\n", encoding="utf-8")
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))
        subprocess.call([editor, str(target)])
        emit_ok(f"Memory saved → {target}")

    elif sub == "write":
        # /memory write <Section Name> -- <content>
        if len(parts) < 2 or " -- " not in parts[1]:
            print("  Usage: /memory write <Section Name> -- <content>")
            return
        rest = parts[1]
        sec, _, content = rest.partition(" -- ")
        sec     = sec.strip()
        content = content.strip()
        if not sec or not content:
            print("  Usage: /memory write <Section Name> -- <content>")
            return
        from greenboost_cli.instruments.handlers import handle_memory_write
        result = handle_memory_write(sec, content, "project")
        emit_ok(result)

    else:
        print("  Usage: /memory [show|project|global|edit|write <section> -- <text>]")


register_command("save-session",    _save_session,   "Save session state for recovery  (/save-session [show|clear])")
register_command("resume",          _resume,         "Load last (or named) saved session  (/resume [name])")
register_command("add",             _add,            "Inject file content into context  (/add <path|glob> [:<lines>])")
register_command("memory",          _memory,         "View/edit CLAUDE.md memory  (/memory [show|project|global|edit|write])")
