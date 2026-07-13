"""Skill slash commands: /skill-list, /skill-show <name>, /skill-set-dir <path>.

These wrap the headless skill router for in-REPL use, and let the user point
the auto-loader at a skills directory by setting `settings["skills_dir"]`.
"""
from __future__ import annotations

from pathlib import Path

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import (
    console, GRAY, VIOLET, LIME, AMBER,
    emit_ok, emit_warn, emit_err, emit_info,
)
from greenboost_cli.security import _validate_path


def _resolve_dir(settings: dict) -> Path | None:
    raw = settings.get("skills_dir")
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


def _skill_list(args: str, session, settings: dict) -> None:
    """List all auto-discovered skills across all configured directories."""
    from greenboost_cli.skill.router import discover_all_skill_dirs, discover_skills_multi
    dirs = discover_all_skill_dirs(settings)
    if not dirs:
        emit_warn("No skill directories found. Run /skill-set-dir <path> to add one.")
        return

    entries = discover_skills_multi(dirs)
    if not entries:
        console.print(f"  [{GRAY}]No skills found across {len(dirs)} directories.[/]")
        return

    console.print(f"\n  [{VIOLET}]◈[/]  [{VIOLET}]Skills[/]  [{GRAY}]({len(entries)} total across {len(dirs)} dirs)[/]")
    for d in dirs:
        console.print(f"  [{GRAY}]  {d}[/]")
    console.print()
    for e in entries:
        trig = f"  [{GRAY}]triggers: {', '.join(e.triggers[:3])}[/]" if e.triggers else ""
        console.print(f"  [{LIME}]●[/] [bold]{e.name}[/]{trig}")
        console.print(f"      [{GRAY}]{e.description[:140]}[/]")


def _skill_show(args: str, session, settings: dict) -> None:
    """Print the SKILL.md body for a named skill."""
    name = args.strip()
    if not name:
        emit_warn("Usage: /skill-show <name>")
        return
    from greenboost_cli.skill.router import discover_all_skill_dirs, discover_skills_multi, load_skill_body
    dirs    = discover_all_skill_dirs(settings)
    entries = discover_skills_multi(dirs)
    for e in entries:
        if e.name == name:
            body = load_skill_body(Path(e.path), max_chars=8000)
            console.print(f"  [{VIOLET}]{e.name}[/]  [{GRAY}]({e.path})[/]")
            console.print(f"  [{GRAY}]{e.description}[/]\n")
            for line in body.splitlines():
                console.print(f"  {line}")
            return
    emit_warn(f"Skill '{name}' not found.")


def _skill_set_dir(args: str, session, settings: dict) -> None:
    """Set settings['skills_dir'] to enable auto-loading."""
    raw = args.strip()
    if not raw:
        emit_warn("Usage: /skill-set-dir <path>")
        return
    try:
        resolved = _validate_path(raw)
    except ValueError as e:
        emit_err(str(e))
        return
    p = Path(resolved)
    if not p.is_dir():
        emit_err(f"Not a directory: {p}")
        return
    settings["skills_dir"] = str(p)
    try:
        from greenboost_cli.environment.settings import save_settings
        save_settings(settings)
    except Exception as e:
        emit_warn(f"skills_dir updated in memory but could not persist: {e}")
    emit_ok(f"skills_dir → {p}")


def _find_skills(args: str, session, settings: dict) -> None:
    """Search for skills locally and optionally in the skills.sh ecosystem."""
    import subprocess
    query = args.strip()

    from greenboost_cli.skill.router import (
        discover_all_skill_dirs, discover_skills_multi, route_to_json,
    )
    dirs    = discover_all_skill_dirs(settings)
    entries = discover_skills_multi(dirs)
    n_total = len(entries)

    console.print(f"\n  [{VIOLET}]◈[/]  [{VIOLET}]find-skills[/]  [{GRAY}]{n_total} local skills across {len(dirs)} dirs[/]")

    if query:
        # Local semantic search
        hits: list = []
        for d in dirs:
            try:
                res = route_to_json(query, d, top_k=5, min_score=0.15)
                hits.extend(res.get("skills", []))
            except Exception:
                pass
        # Dedupe by name, sort by score
        seen: set[str] = set()
        uniq: list = []
        for h in sorted(hits, key=lambda x: x.get("score", 0), reverse=True):
            if h["name"] not in seen:
                uniq.append(h)
                seen.add(h["name"])
        uniq = uniq[:8]

        if uniq:
            console.print(f"\n  [{GRAY}]Local matches for «{query}»:[/]")
            for h in uniq:
                score_s = f"[{DIM}]{h['score']:.2f}[/]"
                console.print(
                    f"  [{LIME}]●[/] [bold]{h['name']}[/]  {score_s}"
                )
                console.print(f"      [{GRAY}]{h['description'][:120]}[/]")
                console.print(f"      [{DIM}]{h['path']}[/]")
        else:
            console.print(f"  [{GRAY}]No local matches for «{query}».[/]")

        # skills.sh ecosystem search
        console.print(f"\n  [{DIM}]Searching skills.sh ecosystem…[/]")
        try:
            r = subprocess.run(
                ["npx", "skills", "find", query],
                capture_output=True, text=True, timeout=15,
            )
            if r.stdout.strip():
                console.print(f"\n  [{GRAY}]skills.sh results:[/]")
                for line in r.stdout.strip().splitlines()[:20]:
                    console.print(f"  [{GRAY}]{line}[/]")
                console.print(f"\n  [{DIM}]Install: npx skills add <owner/repo@skill> -g -y[/]")
            else:
                console.print(f"  [{GRAY}]No skills.sh results (npx skills find {query})[/]")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            console.print(f"  [{DIM}]npx not available — browse skills.sh manually[/]")
        except Exception:
            pass
    else:
        # No query: list all skills grouped by source dir
        for d in dirs:
            from greenboost_cli.skill.router import discover_skills_in_dir
            dir_entries = discover_skills_in_dir(d)
            if not dir_entries:
                continue
            console.print(f"\n  [{GRAY}]{d}  ({len(dir_entries)} skills)[/]")
            for e in dir_entries[:10]:
                console.print(f"  [{LIME}]●[/] [bold]{e.name}[/]  [{DIM}]{e.description[:80]}[/]")
            if len(dir_entries) > 10:
                console.print(f"  [{DIM}]  … +{len(dir_entries) - 10} more[/]")

        console.print(f"\n  [{DIM}]Usage: /find-skills <query>  to search locally and on skills.sh[/]")
        console.print(f"  [{DIM}]Install: npx skills add <owner/repo@skill> -g -y[/]")


register_command("skill-list",    _skill_list,    "List discovered skills")
register_command("skill-show",    _skill_show,    "Show a SKILL.md body  (/skill-show <name>)")
register_command("skill-set-dir", _skill_set_dir, "Set skills directory  (/skill-set-dir <path>)")
register_command("find-skills",   _find_skills,   "Search for skills  (/find-skills [query])")
