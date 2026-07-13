"""Autonomous-coding mode: /autonomous-coding [on|off|status].

When enabled (permission_mode = "autonomous"):
  - Read/Write/Edit are auto-approved without a permission prompt.
  - Bash auto-approves a wide set of coding commands: test runners, build tools,
    package managers, linters, and git write ops (excluding push).
  - Chain operators (;, &&, ||, |) and destructive commands (rm -rf, git push,
    DROP TABLE …) remain hard-blocked regardless.
  - The model receives an autonomous-coding directive instructing it to work
    methodically, run tests, commit incrementally, and log progress to PROGRESS.md.

Usage:
  /autonomous-coding              show current status
  /autonomous-coding on           enable (saves to settings)
  /autonomous-coding on "goal"    enable with an objective for the model to pursue
  /autonomous-coding off          disable, restore permission_mode to auto
"""
from __future__ import annotations

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import (
    console, AMBER, LIME, GRAY, VIOLET, RED, DIM, TEAL,
    emit_warn, emit_ok, emit_info,
)
from greenboost_cli.environment.settings import save_settings


_SAFETY_SUMMARY = """\
  Auto-approved in autonomous mode:
    • Read / Write / Edit (all files)
    • Bash: pytest · npm test · cargo test · go test · make …
    • Bash: pip install · npm install · poetry run …
    • Bash: git add · git commit · git fetch · git pull …
    • Bash: black · ruff · eslint · prettier · mypy …
    • Bash: python script.py · node script.js …

  Always blocked (even in autonomous mode):
    • git push  ·  rm -rf  ·  sudo rm  ·  DROP TABLE …
    • Any command with chain operators (; && || | ` $( > \\n)\
"""

_WARNING = (
    "  ⚠  The model will write files and run commands WITHOUT asking first.\n"
    "  Enable only when you trust the active model and the current project."
)


def _autonomous(args: str, session, settings: dict) -> None:
    """Toggle autonomous-coding mode.

    Usage:
      /autonomous-coding
      /autonomous-coding on [goal text]
      /autonomous-coding off
    """
    raw = args.strip()
    parts = raw.split(None, 1)
    subcmd = parts[0].lower() if parts else "status"
    goal   = parts[1].strip('"\'') if len(parts) > 1 else ""

    current = settings.get("permission_mode", "auto")
    is_on   = current == "autonomous"

    if subcmd in ("on", "enable", "start"):
        settings["permission_mode"] = "autonomous"
        if goal:
            settings["autonomous_goal"] = goal
        elif "autonomous_goal" in settings:
            del settings["autonomous_goal"]
        save_settings(settings)

        console.print()
        console.print(
            f"  [{AMBER}]◈  Autonomous Coding[/]  [{AMBER}]ENABLED[/]"
            f"  [{DIM}]{'─' * 40}[/]"
        )
        console.print(f"\n  [{RED}]{_WARNING}[/]")
        console.print(f"\n  [{GRAY}]{_SAFETY_SUMMARY}[/]")
        if goal:
            console.print(f"\n  [{VIOLET}]Objective:[/] [{GRAY}]{goal}[/]")
        console.print(
            f"\n  [{DIM}]/autonomous-coding off   to return to manual approval[/]"
        )
        console.print()

    elif subcmd in ("off", "disable", "stop"):
        prev_mode = settings.get("permission_mode", "auto")
        settings["permission_mode"] = "auto"
        settings.pop("autonomous_goal", None)
        save_settings(settings)
        console.print()
        console.print(
            f"  [{LIME}]◈  Autonomous Coding[/]  [{GRAY}]OFF[/]"
            f"  [{DIM}]─ permission_mode restored to auto[/]"
        )
        console.print()

    else:  # status
        console.print()
        if is_on:
            g = settings.get("autonomous_goal", "")
            console.print(
                f"  [{AMBER}]◈  Autonomous Coding:[/]  [{AMBER}]ENABLED[/]"
            )
            if g:
                console.print(f"  [{VIOLET}]Objective:[/] [{GRAY}]{g}[/]")
            console.print(f"  [{GRAY}]{_SAFETY_SUMMARY}[/]")
            console.print(
                f"\n  [{DIM}]/autonomous-coding off   to disable[/]"
            )
        else:
            console.print(
                f"  [{LIME}]◈  Autonomous Coding:[/]  [{GRAY}]off"
                f"  (permission_mode={current})[/]"
            )
            console.print(
                f"  [{DIM}]/autonomous-coding on           enable"
                f"  ·  /autonomous-coding on \"<goal>\"   enable with objective[/]"
            )
        console.print()


register_command(
    "autonomous-coding",
    _autonomous,
    "Headless autonomous mode  (/autonomous-coding [on|off|status])",
)
