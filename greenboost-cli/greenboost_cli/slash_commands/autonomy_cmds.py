"""Review and export what the agent did while you were not watching.

`/session-report` is the morning-after view of an unattended run: which tools
ran, which skills, every question that was answered without a human and the
reason each choice was made, and why the session finally stopped.

    /nonstop            show or toggle non-stop mode        (also ctrl+n)
    /auto-answer        show or toggle auto-answering       (also ctrl+y)
    /session-report     print the report
    /session-report <path>   also write it to a file (.md, plus .json journal)
"""
from __future__ import annotations

from pathlib import Path

from greenboost_cli.core.autonomy import (
    get_state, render_report, export_report, export_journal_json,
)
from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import console, GRAY, DIM, emit_ok, emit_info


def _cmd_session_report(args: str, session, settings: dict) -> bool:
    st = get_state()
    title = getattr(session, "title", "") or ""
    console.print()
    for line in render_report(st, title).split("\n"):
        console.print(f"  {line}" if line else "")
    console.print()

    dest = (args or "").strip()
    if dest:
        p = Path(dest).expanduser()
        if p.suffix.lower() != ".md":
            p = p.with_suffix(".md")
        export_report(st, p, title)
        j = export_journal_json(st, p.with_suffix(".json"))
        emit_ok(f"Report written to {p}")
        emit_info(f"Full journal (machine-readable): {j}")
    else:
        console.print(f"  [{DIM}]/session-report <path>  to export it as "
                      f"markdown + a JSON journal[/]")
    return True


def _toggle(flag: str, args: str, on_msg: str, off_msg: str) -> bool:
    st = get_state()
    arg = (args or "").strip().lower()
    if arg in ("on", "off"):
        setattr(st, flag, arg == "on")
    elif arg:
        emit_info(f"Usage: on | off  (currently "
                  f"{'on' if getattr(st, flag) else 'off'})")
        return True
    else:
        setattr(st, flag, not getattr(st, flag))
    emit_ok(on_msg if getattr(st, flag) else off_msg)
    return True


def _cmd_nonstop(args: str, session, settings: dict) -> bool:
    st = get_state()
    if not (args or "").strip() and not st.nonstop:
        st.consecutive_continues = 0
    return _toggle(
        "nonstop", args,
        "Non-stop: ON  , keeps working while todos are open or the model says "
        "it is mid-task",
        "Non-stop: OFF , the prompt returns after each turn")


def _cmd_auto_answer(args: str, session, settings: dict) -> bool:
    return _toggle(
        "auto_answer", args,
        "Auto-answer: ON  , picks the Recommended option; every choice is "
        "recorded (/session-report)",
        "Auto-answer: OFF , questions wait for you")


def _cmd_diagnose(args: str, session, settings: dict) -> bool:
    """Why did that run go wrong? Read it out of the trace."""
    from greenboost_cli.core.trajectory import diagnose, render
    st = get_state()
    events = []
    try:
        import gb_dataflux
        since = max(0.05, (Path and 0) or 0.05)
        events = [e for e in gb_dataflux.read_events(since_hours=6.0)
                  if e.get("kind", "").startswith("agent_")
                  or str(e.get("status", "")) == "error"]
    except Exception:
        pass                      # journal-only diagnosis still works
    console.print()
    for line in render(diagnose(st, events)).split("\n"):
        console.print(f"  {line}" if line else "")
    console.print()
    return True


register_command("diagnose", _cmd_diagnose,
                 "Why the last run went wrong, from its own trace")
register_command("session-report", _cmd_session_report,
                 "What ran unattended: tools, skills, auto-answered questions")
register_command("nonstop", _cmd_nonstop,
                 "Keep working without handing the prompt back (ctrl+n)")
register_command("auto-answer", _cmd_auto_answer,
                 "Let the model answer its own questions (ctrl+y)")
