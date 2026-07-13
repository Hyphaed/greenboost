"""RAG slash commands: /rag-add /rag-search /rag-status /rag-clear /rag-inject."""
from __future__ import annotations

import sys
from pathlib import Path

from greenboost_cli.terminal.commands import register_command

# Pending inject: RAG results prepended to next user turn
_pending_inject: str | None = None


def get_pending_inject() -> str | None:
    """Return pending RAG context to inject (consumed on first call)."""
    global _pending_inject
    ctx = _pending_inject
    _pending_inject = None
    return ctx


def _rag_add(args: str, session, settings: dict) -> None:
    from greenboost_cli.rag.engine import index_folder

    parts = args.strip().split()
    if not parts:
        print("  Usage: /rag-add <folder> [--project name]")
        return

    folder = Path(parts[0]).expanduser().resolve()
    if not folder.exists():
        print(f"  ✗  Folder not found: {folder}")
        return

    project = None
    if "--project" in parts:
        idx = parts.index("--project")
        if idx + 1 < len(parts):
            project = parts[idx + 1]

    print(f"  Indexing {folder} …")
    result = index_folder(folder, project)
    print(f"  ✓  Indexed {result['indexed']} chunks from '{result['project']}'")
    if result.get("skipped"):
        print(f"     {result['skipped']} files skipped (read errors)")


def _rag_search(args: str, session, settings: dict) -> None:
    from greenboost_cli.rag.engine import search, format_for_claude

    query = args.strip().strip('"').strip("'")
    if not query:
        print("  Usage: /rag-search \"<query>\" [--project name] [--top-k N]")
        return

    parts = query.split()
    project = None
    top_k   = settings.get("rag_top_k", 5)

    clean_query = []
    i = 0
    while i < len(parts):
        if parts[i] == "--project" and i + 1 < len(parts):
            project = parts[i + 1]; i += 2
        elif parts[i] == "--top-k" and i + 1 < len(parts):
            try:
                top_k = int(parts[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            clean_query.append(parts[i]); i += 1

    query = " ".join(clean_query) or args.strip()
    min_score = settings.get("rag_min_score", 0.1)
    results = search(query, project=project, top_k=top_k, min_score=min_score)
    print(format_for_claude(results, query))


def _rag_status(args: str, session, settings: dict) -> None:
    from greenboost_cli.rag.engine import print_status
    print_status()


def _rag_clear(args: str, session, settings: dict) -> None:
    from greenboost_cli.rag.engine import EMBEDDINGS_FILE, METADATA_FILE, FOLDERS_FILE

    for f in (EMBEDDINGS_FILE, METADATA_FILE, FOLDERS_FILE):
        if f.exists():
            f.unlink()
    print("  ✓  RAG index cleared.")


def _rag_inject(args: str, session, settings: dict) -> None:
    """Pre-load RAG results — they'll be prepended to your next message."""
    global _pending_inject
    from greenboost_cli.rag.engine import search, format_for_claude

    query = args.strip().strip('"').strip("'")
    if not query:
        print("  Usage: /rag-inject \"<query>\"")
        return

    min_score = settings.get("rag_min_score", 0.1)
    top_k     = min(settings.get("rag_top_k", 5), 3)  # cap at 3 for injection
    results   = search(query, top_k=top_k, min_score=min_score)
    if not results:
        print(f"  No RAG results for: {query}")
        return

    _pending_inject = format_for_claude(results, query)
    print(f"  ✓  {len(results)} RAG chunk(s) queued — will be injected into your next message.")


def _rag_update(args: str, session, settings: dict) -> None:
    """Incrementally re-index RAG folders (default: cwd's project) or web sources.

    Flags: --all (all folders + web), --project NAME, --force (full rebuild),
    --web (only refresh registered web sources, the legacy behaviour).
    """
    from greenboost_cli.rag.engine import (
        update_all, update_folder, update_web_sources, resolve_folder_entry,
        _load_folders,
    )

    parts   = args.strip().split()
    do_all  = "--all" in parts
    force   = "--force" in parts
    web_only = "--web" in parts
    project = None
    if "--project" in parts:
        idx = parts.index("--project")
        if idx + 1 < len(parts):
            project = parts[idx + 1]

    if web_only:
        result = update_web_sources(project_filter=project, verbose=True)
        print()
        print(f"  Done — {result['updated']} updated · {result['failed']} failed · {result['skipped']} skipped")
        return

    if do_all:
        result = update_all(force=force, verbose=True)
        print()
        print(f"  Done — {result['reindexed_files']} files re-indexed · "
              f"{result['removed_files']} removed "
              f"(+{result['chunks_added']}/-{result['chunks_removed']} chunks)")
        return

    if project:
        targets = [e for e in _load_folders()
                   if e.get("project") == project and Path(e.get("folder", "")).is_dir()]
        if not targets:
            print(f"  ✗  No registered folder for project '{project}'")
            return
        for e in targets:
            update_folder(Path(e["folder"]), project=project, force=force, verbose=True)
        return

    entry = resolve_folder_entry()
    if entry is None:
        print("  cwd is not a registered RAG folder — use /rag-add . or /rag-update --all")
        return
    update_folder(Path(entry["folder"]), project=entry.get("project"), force=force, verbose=True)


register_command("rag-add",    _rag_add,    "Index a folder into RAG  (/rag-add <folder>)")
register_command("rag-search", _rag_search, "Search RAG index  (/rag-search \"<query>\")")
register_command("rag-status", _rag_status, "Show RAG index stats")
register_command("rag-clear",  _rag_clear,  "Clear RAG index")
register_command("rag-inject", _rag_inject, "Pre-load RAG context  (/rag-inject \"<query>\")")
register_command("rag-update", _rag_update, "Incrementally re-index RAG  (/rag-update [--all|--project N|--web] [--force])")
