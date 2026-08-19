"""Project memory that gets recalled when it applies , and the rule ratchet.

    /remember <text>          codify a correction as a standing rule
    /remember --scope <path> <text>    a fact about one subsystem
    /memories                 list what is stored, and what it costs
    /forget <name>            delete one memory

A rule recorded here is recalled on EVERY turn. That is the whole point: a
correction the user makes once should not have to be made again. Everything
else is recalled only when the turn's words or the files being touched say it
applies.
"""
from __future__ import annotations

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import console, GRAY, DIM, emit_ok, emit_info, emit_warn


def _cmd_remember(args: str, session, settings: dict) -> bool:
    text = (args or "").strip()
    if not text:
        emit_warn("Nothing to remember , /remember <what the agent should do differently>")
        return True
    from greenboost_cli.memory.store import write_memory, record_rule
    scope = ""
    if text.startswith("--scope"):
        parts = text.split(None, 2)
        if len(parts) < 3:
            emit_warn("Usage: /remember --scope <path> <text>")
            return True
        scope, text = parts[1], parts[2]
    from greenboost_cli.memory.store import derivable_reason
    why = derivable_reason(text)
    if why:
        # Not a refusal , it still gets saved. But say what is likely wasted,
        # because on a local model every recalled character competes with the
        # conversation for the window.
        emit_warn(f"That looks like {why}.")
        emit_info("Saved anyway. The part worth keeping is usually what was "
                  "SURPRISING about it , consider /forget and re-recording that.")
    if scope:
        p = write_memory(text[:60], text[:120], text, mtype="project", scope=scope)
        emit_ok(f"Remembered for {scope} , recalled when that subsystem is touched")
    else:
        p = record_rule(text)
        emit_ok("Recorded as a standing rule , recalled on every turn from now on")
    emit_info(f"{p}")
    return True


def _cmd_memories(args: str, session, settings: dict) -> bool:
    from greenboost_cli.memory.store import scan, memory_dir, recall, render_block
    mems = scan()
    console.print()
    if not mems:
        console.print(f"  [{GRAY}]No memories yet. /remember <text> records one.[/]")
        console.print(f"  [{DIM}]{memory_dir()}[/]\n")
        return True
    for m in mems:
        tag = {"rule": "rule ", "project": "proj ", "user": "user ",
               "reference": "ref  "}.get(m.type, "ref  ")
        scope = f"  [{DIM}](scope: {m.scope})[/]" if m.scope else ""
        console.print(f"  [{GRAY}]{tag}[/] {m.name}{scope}")
        if m.description:
            console.print(f"        [{DIM}]{m.description}[/]")
    always = [m for m in mems if m.type == "rule"]
    cost = len(render_block(always))
    console.print()
    console.print(f"  [{DIM}]{len(mems)} stored, {len(always)} recalled on every "
                  f"turn (~{cost:,} chars of context)[/]")
    console.print(f"  [{DIM}]{memory_dir()}[/]\n")
    return True


def _cmd_forget(args: str, session, settings: dict) -> bool:
    name = (args or "").strip()
    if not name:
        emit_warn("Usage: /forget <memory name>")
        return True
    from greenboost_cli.memory.store import scan
    for m in scan():
        if m.name == name or (m.path and m.path.stem == name):
            m.path.unlink()
            emit_ok(f"Forgot '{m.name}'")
            return True
    emit_warn(f"No memory named '{name}' , /memories lists them")
    return True


register_command("remember", _cmd_remember,
                 "Codify a correction as a standing rule (recalled every turn)")
register_command("memories", _cmd_memories,
                 "What this project has learned, and what it costs in context")
register_command("forget", _cmd_forget, "Delete one memory")
