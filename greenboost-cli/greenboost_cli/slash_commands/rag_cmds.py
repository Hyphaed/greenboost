"""RAG slash commands: /rag-add /rag-search /rag-status /rag-clear /rag-inject."""
from __future__ import annotations

import sys
from pathlib import Path

import time

from greenboost_cli.terminal.commands import register_command

# Pending inject: RAG results prepended to next user turn
_pending_inject: str | None = None


def get_pending_inject() -> str | None:
    """Return pending RAG context to inject (consumed on first call)."""
    global _pending_inject
    ctx = _pending_inject
    _pending_inject = None
    return ctx


def _fmt_eta(seconds: "float | None") -> str:
    if seconds is None:
        return "estimating…"
    s = max(0, int(round(seconds)))
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else (f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s")


def _progress_printer(label: str = "embedding"):
    """A progress callback for the engine: the first call says how much work and on what,
    then one line every ~8 s with the count, percentage and a time-left estimate. The commands
    used to print nothing until they finished, which looked like a hang (a 10,000-chunk
    update is minutes)."""
    import time as _t
    state = {"last": 0.0, "t0": _t.time()}

    def cb(done: int, total: int, eta: "float | None", device: str) -> None:
        now = _t.time()
        if done == 0:
            where = "the GPU" if device == "cuda" else "the CPU (slower)"
            print(f"  {label}: {total:,} chunks on {where}…", flush=True)
            state["last"] = now
            return
        if now - state["last"] >= 8 or done >= total:
            state["last"] = now
            pct = 100 * done / total
            tail = "done" if done >= total else f"about {_fmt_eta(eta)} left"
            print(f"  {label} {done:,}/{total:,}  {pct:4.0f}%  ·  {tail}", flush=True)
    return cb


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

    print(f"  Indexing {folder} …  (scanning and chunking first; embedding progress follows)", flush=True)
    result = index_folder(folder, project, progress=_progress_printer("embedding"))
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


def _rag_reembed(args: str, session, settings: dict) -> None:
    """Rebuild the index's vectors with the current embedder (GPU, resumable)."""
    from greenboost_cli.rag.engine import reembed_store, EMBED_MODEL
    last = [0.0]

    def _progress(done: int, total: int, eta_s: float) -> None:
        if time.time() - last[0] > 20 or done >= total:
            last[0] = time.time()
            print(f"  re-embedding {done:,}/{total:,} chunks  ~{eta_s / 60:.0f} min left")

    print(f"  Re-embedding the index with {EMBED_MODEL} on the GPU. The current index keeps "
          f"working until the swap; Ctrl+C is safe and the next run resumes.")
    try:
        r = reembed_store(progress=_progress)
    except RuntimeError as e:
        print(f"  ✗  {e}")
        return
    if r["status"] == "current":
        print(f"  ✓  Already built with {r['embed_model']}; nothing to do.")
    elif r["status"] == "done":
        print(f"  ✓  {r['chunks']:,} chunks re-embedded ({r['dims']}-dim) in {r['seconds'] // 60} min.")
    else:
        print("  The index is empty; nothing to re-embed.")


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

    prog = _progress_printer("embedding")
    if do_all:
        print("  Updating every registered folder: scanning for changed files first, embedding "
              "progress follows. A folder with nothing new takes a second or two.", flush=True)
        result = update_all(force=force, verbose=True, progress=prog)
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
            print(f"  [{project}] scanning {e['folder']} …", flush=True)
            update_folder(Path(e["folder"]), project=project, force=force, verbose=True, progress=prog)
        return

    entry = resolve_folder_entry()
    if entry is None:
        print("  cwd is not a registered RAG folder — use /rag-add . or /rag-update --all")
        return
    print(f"  [{entry.get('project')}] scanning {entry['folder']} for changed files …", flush=True)
    update_folder(Path(entry["folder"]), project=entry.get("project"), force=force, verbose=True, progress=prog)


register_command("rag-add",    _rag_add,    "Index a folder into RAG  (/rag-add <folder>)")
register_command("rag-search", _rag_search, "Search RAG index  (/rag-search \"<query>\")")
register_command("rag-status", _rag_status, "Show RAG index stats")
register_command("rag-clear",  _rag_clear,  "Clear RAG index")
register_command("rag-reembed", _rag_reembed, "Rebuild RAG vectors with the current embedder on the GPU (resumable)")
register_command("rag-inject", _rag_inject, "Pre-load RAG context  (/rag-inject \"<query>\")")
register_command("rag-update", _rag_update, "Incrementally re-index RAG  (/rag-update [--all|--project N|--web] [--force])")
