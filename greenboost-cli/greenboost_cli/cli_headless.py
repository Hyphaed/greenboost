"""Headless gb subcommands for programmatic callers (e.g. optimal-claude).

These are invoked when `gb` is called with one of the headless subcommand
names as the first positional argument:

  gb rag-search "<q>"   [--top-k N] [--min-score X] [--project NAME] [--json]
  gb rag-index <folder> [--project NAME] [--json]
  gb rag update         [--all] [--project NAME] [--force] [--json]
  gb rag-status         [--project NAME] [--json]
  gb rag-clear          [--json]
  gb compress           [--target-chars N] (reads stdin)
  gb skill-route "<q>"  --skills-dir PATH [--top-k N] [--min-score X] [--json]
  gb tokens [show|reset] [--project NAME] [--json]

Each handler:
  - Parses its own argv (everything after the subcommand name).
  - Returns an int exit code (0 = success, non-zero = error).
  - Writes structured output to stdout. --json produces machine-readable JSON;
    omitting --json produces a terse human-readable summary.
  - Errors go to stderr.

All inputs flow through greenboost_cli.security so the same path-traversal /
length-cap hardening that protects the MCP server applies here too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from greenboost_cli.security import (
    _validate_path, _cap, _safe_project,
    _MAX_QUERY_LEN, _MAX_TEXT_LEN, _MAX_PATH_LEN,
)

HEADLESS_SUBCOMMANDS = {
    "rag-search", "rag-index", "rag-status", "rag-clear",
    "rag-feed-text", "rag-register-web", "rag-update-web", "rag-update", "rag-web-sources",
    "contextualize",
    "clear-memory-pool",
    "crag-add", "crag-search", "crag-status", "crag-clear",
    "convert",
    "compress", "skill-route", "tokens",
    # AI Factory plane
    "factory-submit", "factory-status", "factory-list",
    "factory-pause", "factory-resume", "factory-agents",
    "factory-hot-swap", "factory-sleep",
    # Skill management, plan-mode, subagent, task tracker
    "skill-list", "skill-show",
    "plan-create", "plan-list",
    "agent",
    "task-add", "task-list", "task-update", "task-delete",
    # Multi-device cluster sharding
    "cluster-status", "cluster-register", "cluster-list",
    "cluster-remove", "cluster-test", "cluster-serve",
    "cluster-load", "cluster-generate", "cluster-bootstrap-peer", "cluster-info",
}


def _emit_json(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def _emit_err(msg: str) -> None:
    sys.stderr.write(f"gb: {msg}\n")


# ── rag-search ────────────────────────────────────────────────────────────────

def cmd_rag_search(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb rag-search", add_help=True)
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-score", type=float, default=0.1)
    p.add_argument("--project", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    query = _cap(args.query, _MAX_QUERY_LEN)
    project = _safe_project(args.project) if args.project else None
    top_k = max(1, min(args.top_k, 20))
    min_score = max(0.0, min(args.min_score, 1.0))

    from greenboost_cli.rag.engine import search, format_for_claude, results_to_json
    results = search(query, project=project, top_k=top_k, min_score=min_score)

    if args.json:
        _emit_json(results_to_json(results, query))
    else:
        print(format_for_claude(results, query))
    return 0


# ── rag-index ─────────────────────────────────────────────────────────────────

def cmd_rag_index(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb rag-index", add_help=True)
    p.add_argument("folder")
    p.add_argument("--project", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    try:
        folder = _validate_path(args.folder)
    except ValueError as e:
        _emit_err(str(e))
        return 2
    project = _safe_project(args.project) if args.project else None

    from greenboost_cli.rag.engine import index_folder
    r = index_folder(Path(folder), project=project)

    if args.json:
        _emit_json(r)
    else:
        print(
            f"Indexed {r['indexed']} chunks from '{r['project']}' "
            f"({r['skipped']} files skipped)."
        )
    return 0


# ── rag-status ────────────────────────────────────────────────────────────────

def cmd_rag_status(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb rag-status", add_help=True)
    p.add_argument("--project", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.rag.engine import (
        _load_store, _load_folders, RAG_DIR, EMBEDDINGS_FILE,
    )
    embeddings, metadata = _load_store()
    folders = _load_folders()
    if args.project:
        proj = _safe_project(args.project)
        folders = [f for f in folders if f.get("project") == proj]

    n_chunks = len(metadata) if metadata else 0
    n_files = len({m["file"] for m in metadata}) if metadata else 0
    db_mb = EMBEDDINGS_FILE.stat().st_size / 1_048_576 if EMBEDDINGS_FILE.exists() else 0.0

    if args.json:
        _emit_json({
            "rag_dir": str(RAG_DIR),
            "chunks": n_chunks,
            "files": n_files,
            "size_mb": round(db_mb, 2),
            "folders": folders,
            "model": "jinaai/jina-embeddings-v2-base-code",
        })
    else:
        from greenboost_cli.rag.engine import print_status
        print_status()
    return 0


# ── rag-clear ─────────────────────────────────────────────────────────────────

def cmd_rag_clear(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb rag-clear", add_help=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.rag.engine import EMBEDDINGS_FILE, METADATA_FILE, FOLDERS_FILE
    removed = []
    for f in (EMBEDDINGS_FILE, METADATA_FILE, FOLDERS_FILE):
        if f.exists():
            f.unlink()
            removed.append(str(f))
    if args.json:
        _emit_json({"cleared": True, "removed": removed})
    else:
        print(f"RAG index cleared ({len(removed)} files).")
    return 0


# ── rag-feed-text ─────────────────────────────────────────────────────────────

def cmd_rag_feed_text(argv: list[str]) -> int:
    """Index arbitrary text (read from stdin) into the RAG store."""
    p = argparse.ArgumentParser(prog="gb rag-feed-text", add_help=True)
    p.add_argument("--source",  required=True, help="Source label (e.g. URL or filename)")
    p.add_argument("--project", default=None,  help="Project scope")
    p.add_argument("--json",    action="store_true")
    args = p.parse_args(argv)

    text = sys.stdin.read()
    if not text.strip():
        if args.json:
            _emit_json({"indexed": 0, "error": "empty input"})
        else:
            _emit_err("rag-feed-text: no text on stdin")
        return 1

    from greenboost_cli.rag.engine import index_text
    result = index_text(text, source_name=args.source, project=args.project or "documents")
    if args.json:
        _emit_json(result)
    else:
        print(f"Indexed {result.get('indexed', 0)} chunks from '{args.source}'")
    return 0


# ── rag-register-web ──────────────────────────────────────────────────────────

def cmd_rag_register_web(argv: list[str]) -> int:
    """Register a URL in the web sources registry for future /rag-update runs."""
    p = argparse.ArgumentParser(prog="gb rag-register-web", add_help=True)
    p.add_argument("url")
    p.add_argument("--project", default="web")
    p.add_argument("--json",    action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.rag.engine import register_web_source
    register_web_source(args.url, args.project)
    if args.json:
        _emit_json({"registered": True, "url": args.url, "project": args.project})
    else:
        print(f"Registered: {args.url}  [{args.project}]")
    return 0


# ── rag-update-web ────────────────────────────────────────────────────────────

def cmd_rag_update_web(argv: list[str]) -> int:
    """Re-fetch and re-index all registered web sources."""
    p = argparse.ArgumentParser(prog="gb rag-update-web", add_help=True)
    p.add_argument("--project", default=None, help="Filter by project")
    p.add_argument("--json",    action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.rag.engine import update_web_sources
    verbose = not args.json
    result = update_web_sources(project_filter=args.project, verbose=verbose)
    if args.json:
        _emit_json(result)
    else:
        print(f"\nDone — {result['updated']} updated · "
              f"{result['failed']} failed · {result['skipped']} skipped")
    return 0


# ── rag-update ────────────────────────────────────────────────────────────────

def cmd_rag_update(argv: list[str]) -> int:
    """Incrementally re-index a registered RAG folder.

    Default scope: the project whose registered folder contains the cwd.
    """
    p = argparse.ArgumentParser(prog="gb rag-update", add_help=True)
    p.add_argument("--all", action="store_true",
                   help="Update every registered folder + web sources")
    p.add_argument("--project", default=None, help="Update folders of this project")
    p.add_argument("--force", action="store_true",
                   help="Full rebuild instead of incremental")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.rag.engine import (
        update_all, update_folder, resolve_folder_entry, _load_folders,
    )
    verbose = not args.json

    if args.all:
        result = update_all(force=args.force, verbose=verbose)
        if args.json:
            _emit_json(result)
        else:
            print(f"\nDone — {result['reindexed_files']} files re-indexed · "
                  f"{result['removed_files']} removed "
                  f"(+{result['chunks_added']}/-{result['chunks_removed']} chunks)")
        return 0

    if args.project:
        proj = _safe_project(args.project)
        targets = [e for e in _load_folders()
                   if e.get("project") == proj and Path(e.get("folder", "")).is_dir()]
        if not targets:
            _emit_err(f"no registered folder for project '{proj}'")
            return 1
        results = [update_folder(Path(e["folder"]), project=proj,
                                 force=args.force, verbose=verbose) for e in targets]
        if args.json:
            _emit_json({"folders": results})
        return 0

    # Default: resolve cwd → registered folder.
    entry = resolve_folder_entry()
    if entry is None:
        _emit_err("cwd is not inside a registered RAG folder — "
                  "run 'gb rag-index .' first or use --all")
        return 2
    result = update_folder(Path(entry["folder"]), project=entry.get("project"),
                           force=args.force, verbose=verbose)
    if args.json:
        _emit_json(result)
    return 0


# ── rag-web-sources ───────────────────────────────────────────────────────────

def cmd_rag_web_sources(argv: list[str]) -> int:
    """List registered web sources."""
    p = argparse.ArgumentParser(prog="gb rag-web-sources", add_help=True)
    p.add_argument("--project", default=None)
    p.add_argument("--json",    action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.rag.engine import _load_web_sources
    sources = _load_web_sources()
    if args.project:
        sources = [s for s in sources if s.get("project") == args.project]

    if args.json:
        _emit_json({"sources": sources, "count": len(sources)})
    else:
        if not sources:
            print("No web sources registered.")
        for s in sources:
            print(f"  {s['url']:<70}  [{s.get('project','?')}]  {s.get('last_indexed','')[:10]}")
    return 0


# ── convert ───────────────────────────────────────────────────────────────────

def cmd_convert(argv: list[str]) -> int:
    """Convert a file or URL to Markdown (using markitdown), optionally feed RAG."""
    p = argparse.ArgumentParser(prog="gb convert", add_help=True)
    p.add_argument("source", help="File path or URL")
    p.add_argument("--project",  default=None)
    p.add_argument("--no-rag",   action="store_true", help="Skip RAG feeding")
    p.add_argument("--output",   default=None, help="Save markdown to this path")
    p.add_argument("--json",     action="store_true")
    args = p.parse_args(argv)

    try:
        from greenboost_cli.converters.markitdown_adapter import convert, convert_and_save
    except ImportError as e:
        _emit_err(f"markitdown not installed: {e}")
        return 1

    feed_rag = not args.no_rag
    try:
        if args.output:
            from pathlib import Path as _Path
            saved = convert_and_save(
                args.source,
                output=_Path(args.output),
                feed_rag=feed_rag,
                project=args.project,
            )
            if args.json:
                _emit_json({"saved": str(saved), "feed_rag": feed_rag})
            else:
                print(f"Saved: {saved}")
        else:
            md = convert(args.source, feed_rag=feed_rag, project=args.project)
            if args.json:
                _emit_json({"markdown": md, "chars": len(md)})
            else:
                sys.stdout.write(md)
                if not md.endswith("\n"):
                    sys.stdout.write("\n")
    except Exception as e:
        _emit_err(str(e))
        return 1
    return 0


# ── compress ──────────────────────────────────────────────────────────────────

def cmd_compress(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb compress", add_help=True)
    p.add_argument("--target-chars", type=int, default=6000,
                   help="Target output length in characters (default 6000).")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON wrapper instead of raw text.")
    args = p.parse_args(argv)

    raw = sys.stdin.read()
    raw = _cap(raw, _MAX_TEXT_LEN)

    from greenboost_cli.workflow.intelligence import compress_text
    out = compress_text(raw, target_chars=args.target_chars)

    if args.json:
        _emit_json({
            "original_chars": len(raw),
            "compressed_chars": len(out),
            "text": out,
        })
    else:
        sys.stdout.write(out)
    return 0


# ── skill-route ───────────────────────────────────────────────────────────────

def cmd_skill_route(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb skill-route", add_help=True)
    p.add_argument("query")
    p.add_argument("--skills-dir", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-score", type=float, default=0.20)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    try:
        skills_dir_str = _validate_path(args.skills_dir)
    except ValueError as e:
        _emit_err(str(e))
        return 2

    query = _cap(args.query, _MAX_QUERY_LEN)
    skills_dir = Path(skills_dir_str)
    top_k = max(1, min(args.top_k, 20))
    min_score = max(0.0, min(args.min_score, 1.0))

    from greenboost_cli.skill.router import route_to_json, route
    if args.json:
        _emit_json(route_to_json(query, skills_dir, top_k=top_k, min_score=min_score))
    else:
        hits = route(query, skills_dir, top_k=top_k, min_score=min_score)
        if not hits:
            print("No skills matched.")
            return 0
        for h in hits:
            print(f"  {h.score:>5.2f}  {h.name:<40}  ({h.reason})")
            print(f"         {h.description[:120]}")
    return 0


# ── tokens ────────────────────────────────────────────────────────────────────

def cmd_tokens(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb tokens", add_help=True)
    p.add_argument("action", nargs="?", default="show", choices=["show", "reset"])
    p.add_argument("--project", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    project = _safe_project(args.project) if args.project else None

    from greenboost_cli.memory.brain import project_dir
    from greenboost_cli.memory.token_tracker import get_totals, _save, _load
    pdir = project_dir(project)

    if args.action == "reset":
        _save(pdir, {"sessions": [], "totals": {"api": 0, "local": 0}})
        if args.json:
            _emit_json({"reset": True, "project": pdir.name})
        else:
            print(f"Token usage reset for project '{pdir.name}'.")
        return 0

    totals = get_totals(pdir)
    if args.json:
        _emit_json({
            "project": pdir.name,
            **totals,
        })
    else:
        print(f"  Project: {pdir.name}")
        print(f"  Today  : API {totals['today_api']:>8}  local {totals['today_local']:>8}")
        print(f"  Total  : API {totals['total_api']:>8}  local {totals['total_local']:>8}")
    return 0


# ── skill-list / skill-show ───────────────────────────────────────────────────

def _default_skills_dir() -> Path | None:
    """Resolve the configured skills directory from gb settings (or None)."""
    from greenboost_cli.environment.settings import load_settings
    cfg = load_settings()
    raw = cfg.get("skills_dir")
    if not raw:
        return None
    return Path(raw).expanduser()


def cmd_skill_list(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb skill-list", add_help=True)
    p.add_argument("--skills-dir", default=None,
                   help="Override skills directory (default: settings['skills_dir']).")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if args.skills_dir:
        try:
            sd = Path(_validate_path(args.skills_dir))
        except ValueError as e:
            _emit_err(str(e)); return 2
    else:
        sd = _default_skills_dir()

    if sd is None or not sd.is_dir():
        if args.json:
            _emit_json({"skills_dir": str(sd) if sd else "", "count": 0, "skills": []})
            return 0
        _emit_err("no skills_dir configured "
                  "(pass --skills-dir or set settings['skills_dir'])")
        return 2

    from greenboost_cli.skill.router import discover_skills
    entries = discover_skills(sd)
    payload = {
        "skills_dir": str(sd),
        "count": len(entries),
        "skills": [
            {"name": e.name, "description": e.description,
             "path": e.path, "triggers": e.triggers}
            for e in entries
        ],
    }
    if args.json:
        _emit_json(payload)
    else:
        if not entries:
            print(f"No skills found under {sd}.")
            return 0
        for e in entries:
            print(f"  {e.name}")
            print(f"    {e.description[:120]}")
            if e.triggers:
                print(f"    triggers: {', '.join(e.triggers[:6])}")
    return 0


def cmd_skill_show(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb skill-show", add_help=True)
    p.add_argument("name")
    p.add_argument("--skills-dir", default=None)
    p.add_argument("--max-chars", type=int, default=8000)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if args.skills_dir:
        try:
            sd = Path(_validate_path(args.skills_dir))
        except ValueError as e:
            _emit_err(str(e)); return 2
    else:
        sd = _default_skills_dir()
    if sd is None or not sd.is_dir():
        _emit_err("no skills_dir configured")
        return 2

    from greenboost_cli.skill.router import discover_skills, load_skill_body
    for e in discover_skills(sd):
        if e.name == args.name:
            body = load_skill_body(
                Path(e.path),
                max_chars=max(200, min(args.max_chars, 50_000)),
            )
            if args.json:
                _emit_json({
                    "name": e.name,
                    "description": e.description,
                    "path": e.path,
                    "triggers": e.triggers,
                    "body": body,
                })
            else:
                print(f"# {e.name}\n")
                print(e.description); print()
                print(body)
            return 0
    _emit_err(f"skill '{args.name}' not found")
    return 1


# ── plan-create / plan-list ───────────────────────────────────────────────────

def cmd_plan_create(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb plan-create", add_help=True)
    p.add_argument("--prompt", default="",
                   help="Optional prompt text seeded into the plan body.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    prompt = _cap(args.prompt, _MAX_TEXT_LEN)
    from greenboost_cli.planning.plan import create_plan
    entry = create_plan(prompt)
    if args.json:
        _emit_json({
            "id": entry.id,
            "path": str(entry.path),
            "title": entry.title,
            "created_at": entry.created_at,
        })
    else:
        print(str(entry.path))
    return 0


def cmd_plan_list(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb plan-list", add_help=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.planning.plan import list_plans
    entries = list_plans()
    if args.json:
        _emit_json([
            {"id": e.id, "path": str(e.path), "title": e.title,
             "created_at": e.created_at, "size": e.size}
            for e in entries
        ])
    else:
        if not entries:
            print("No plans yet.")
            return 0
        for e in entries:
            print(f"  {e.id}  {e.created_at}  {e.title}")
    return 0


# ── agent (subagent runner) ───────────────────────────────────────────────────

def cmd_agent(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb agent", add_help=True)
    p.add_argument("prompt")
    p.add_argument("--model", default=None)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    prompt = _cap(args.prompt, _MAX_TEXT_LEN)
    from greenboost_cli.agents.subagent import run_subagent
    result = run_subagent(
        prompt, model=args.model, timeout_s=max(1.0, args.timeout),
    )
    if args.json:
        _emit_json(result.to_dict())
    else:
        if result.error:
            sys.stderr.write(f"error: {result.error}\n")
        if result.timed_out:
            sys.stderr.write(
                f"(timed out after {args.timeout:.0f}s — partial output below)\n"
            )
        print(result.summary)
        if result.tool_calls:
            sys.stderr.write(
                f"\n[{len(result.tool_calls)} tool calls, "
                f"{result.tokens_used} tokens, "
                f"{result.duration_s:.2f}s]\n"
            )
    return 0 if not result.error else 1


# ── task-add / task-list / task-update / task-delete ──────────────────────────

def _task_project_arg(args) -> str | None:
    if getattr(args, "project", None):
        return _safe_project(args.project)
    return None


def cmd_task_add(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb task-add", add_help=True)
    p.add_argument("subject")
    p.add_argument("--description", default="")
    p.add_argument("--active-form", default="")
    p.add_argument("--project", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.tasks.tracker import add_task
    task = add_task(
        _task_project_arg(args),
        _cap(args.subject, _MAX_TEXT_LEN),
        description=_cap(args.description, _MAX_TEXT_LEN),
        active_form=_cap(args.active_form, _MAX_TEXT_LEN),
    )
    if args.json:
        _emit_json(task.to_dict())
    else:
        print(f"{task.id}  {task.status}  {task.subject}")
    return 0


def cmd_task_list(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb task-list", add_help=True)
    p.add_argument("--project", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.tasks.tracker import list_tasks
    tasks = list_tasks(_task_project_arg(args))
    if args.json:
        _emit_json([t.to_dict() for t in tasks])
    else:
        if not tasks:
            print("No tasks.")
            return 0
        for t in tasks:
            print(f"  {t.id}  {t.status:<11}  {t.subject}")
    return 0


def cmd_task_update(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb task-update", add_help=True)
    p.add_argument("id")
    p.add_argument("status")
    p.add_argument("--project", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.tasks.tracker import update_task
    try:
        result = update_task(_task_project_arg(args), args.id, args.status)
    except ValueError as e:
        _emit_err(str(e))
        return 2
    if result is None:
        _emit_err(f"task '{args.id}' not found")
        return 1
    if args.json:
        _emit_json(result.to_dict())
    else:
        print(f"{result.id}  →  {result.status}")
    return 0


def cmd_task_delete(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb task-delete", add_help=True)
    p.add_argument("id")
    p.add_argument("--project", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    from greenboost_cli.tasks.tracker import delete_task
    ok = delete_task(_task_project_arg(args), args.id)
    if args.json:
        _emit_json({"deleted": ok, "id": args.id})
    else:
        print("deleted" if ok else "not found")
    return 0 if ok else 1


# ── Contextual RAG (crag-*) ───────────────────────────────────────────────────

def cmd_crag_add(argv: list[str]) -> int:
    """Handler for `gb crag-add <path> [--project NAME] [--json]`."""
    p = argparse.ArgumentParser(prog="gb crag-add", add_help=True)
    p.add_argument("path", help="File or folder to ingest.")
    p.add_argument("--project", default="default")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    path_str = _validate_path(args.path)
    project  = _safe_project(args.project)
    from pathlib import Path as _Path
    from greenboost_cli.rag.contextual_rag import ingest_document, ingest_folder  # noqa: PLC0415
    target = _Path(path_str).expanduser().resolve()
    if target.is_dir():
        result = ingest_folder(str(target), project=project)
        if args.json:
            result["project"] = project
            _emit_json(result)
        return 0 if result["errors"] == 0 or result["indexed"] > 0 else 1
    else:
        ok = ingest_document(str(target), project=project)
        if args.json:
            _emit_json({"ok": ok, "path": path_str, "project": project})
        return 0 if ok else 1


def cmd_crag_search(argv: list[str]) -> int:
    """Handler for `gb crag-search <query> [--project NAME] [--top-k N] [--json]`."""
    p = argparse.ArgumentParser(prog="gb crag-search", add_help=True)
    p.add_argument("query")
    p.add_argument("--project", default="default")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    query   = _cap(args.query, _MAX_QUERY_LEN)
    project = _safe_project(args.project)
    top_k   = max(1, min(args.top_k, 20))
    from greenboost_cli.rag.contextual_rag import search, format_hits  # noqa: PLC0415
    hits = search(query, project=project, top_k=top_k)
    if args.json:
        _emit_json([
            {"source": h.source, "heading_path": h.heading_path,
             "score": h.score, "snippet": h.snippet[:2000]}
            for h in hits
        ])
    else:
        print(format_hits(hits, query))
    return 0 if hits else 1


def cmd_crag_status(argv: list[str]) -> int:
    """Handler for `gb crag-status [--project NAME] [--json]`."""
    p = argparse.ArgumentParser(prog="gb crag-status", add_help=True)
    p.add_argument("--project", default="default")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    project = _safe_project(args.project)
    import re as _re, json as _json
    from pathlib import Path as _Path
    d = _Path.home() / ".greenboost_cli" / "contextual_rag" / _re.sub(r"[^\w.-]", "_", project)
    meta_f = d / "meta.json"
    if not meta_f.exists():
        if args.json:
            _emit_json({"project": project, "error": "no index found"})
        else:
            _emit_err(f"no contextual RAG index for '{project}'")
        return 1
    meta = _json.loads(meta_f.read_text())
    if args.json:
        _emit_json({"project": project, **meta})
    else:
        from greenboost_cli.rag.contextual_rag import status  # noqa: PLC0415
        status(project)
    return 0


def cmd_crag_clear(argv: list[str]) -> int:
    """Handler for `gb crag-clear [--project NAME] [--json]`."""
    p = argparse.ArgumentParser(prog="gb crag-clear", add_help=True)
    p.add_argument("--project", default="default")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    project = _safe_project(args.project)
    from greenboost_cli.rag.contextual_rag import clear  # noqa: PLC0415
    clear(project)
    if args.json:
        _emit_json({"ok": True, "project": project})
    return 0


# ── Dispatch ──────────────────────────────────────────────────────────────────

from greenboost_cli.workflow.factory_cli import FACTORY_SUBCOMMANDS
from greenboost_cli.cluster.cli import (
    cmd_cluster_status, cmd_cluster_register, cmd_cluster_list,
    cmd_cluster_remove, cmd_cluster_test, cmd_cluster_serve,
    cmd_cluster_load, cmd_cluster_generate, cmd_cluster_bootstrap_peer,
    cmd_cluster_info,
)

from greenboost_cli.rag.contextualize import cmd_contextualize
from greenboost_cli.rag.memory_pool import cmd_clear_memory_pool

_DISPATCH = {
    "rag-search":       cmd_rag_search,
    "rag-index":        cmd_rag_index,
    "rag-status":       cmd_rag_status,
    "rag-clear":        cmd_rag_clear,
    "rag-feed-text":    cmd_rag_feed_text,
    "rag-register-web": cmd_rag_register_web,
    "rag-update-web":   cmd_rag_update_web,
    "rag-update":       cmd_rag_update,
    "rag-web-sources":  cmd_rag_web_sources,
    "contextualize":    cmd_contextualize,
    "clear-memory-pool": cmd_clear_memory_pool,
    "crag-add":         cmd_crag_add,
    "crag-search":      cmd_crag_search,
    "crag-status":      cmd_crag_status,
    "crag-clear":       cmd_crag_clear,
    "convert":          cmd_convert,
    "compress":     cmd_compress,
    "skill-route":  cmd_skill_route,
    "tokens":       cmd_tokens,
    # Skill management
    "skill-list":   cmd_skill_list,
    "skill-show":   cmd_skill_show,
    # Plan mode
    "plan-create":  cmd_plan_create,
    "plan-list":    cmd_plan_list,
    # Subagent
    "agent":        cmd_agent,
    # Task tracker
    "task-add":     cmd_task_add,
    "task-list":    cmd_task_list,
    "task-update":  cmd_task_update,
    "task-delete":  cmd_task_delete,
    # AI Factory plane — registered from factory_cli.FACTORY_SUBCOMMANDS so
    # the headless dispatcher and the factory module stay in sync.
    **FACTORY_SUBCOMMANDS,
    # Multi-device cluster sharding
    "cluster-status":   cmd_cluster_status,
    "cluster-register": cmd_cluster_register,
    "cluster-list":     cmd_cluster_list,
    "cluster-remove":   cmd_cluster_remove,
    "cluster-test":     cmd_cluster_test,
    "cluster-serve":    cmd_cluster_serve,
    "cluster-load":            cmd_cluster_load,
    "cluster-generate":        cmd_cluster_generate,
    "cluster-bootstrap-peer":  cmd_cluster_bootstrap_peer,
    "cluster-info":            cmd_cluster_info,
}


def dispatch(name: str, argv: list[str]) -> int:
    """Call the headless handler for `name` (one of HEADLESS_SUBCOMMANDS)."""
    handler = _DISPATCH.get(name)
    if handler is None:
        _emit_err(f"unknown headless subcommand: {name}")
        return 64
    try:
        return handler(argv)
    except SystemExit as e:
        # argparse calls sys.exit on --help; honour it
        return int(e.code) if e.code is not None else 0
    except Exception as e:
        _emit_err(f"{name}: {type(e).__name__}: {e}")
        return 1
