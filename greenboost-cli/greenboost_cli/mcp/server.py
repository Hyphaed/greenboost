"""GreenBoost CLI — MCP Server.

Exposes local capabilities to Claude Code and other MCP-compatible AI clients:
  • convert_to_markdown  — convert any file/URL via markitdown + auto-RAG feed
  • rag_search           — semantic + BM25 hybrid search over local index
  • rag_index_folder     — index a folder into RAG
  • rag_index_text       — index arbitrary text into RAG
  • rag_status           — RAG index statistics
  • get_goals / add_goal — per-project goal management
  • get_history          — recent project history
  • system_status        — GreenBoost T1/T2/T3 tier stats
  • factory_submit       — submit a task to the AI factory
  • factory_status       — factory snapshot (agents, queue, GPU)

Security hardening (MCP April-2025 advisory):
  • HTTP mode ALWAYS binds to 127.0.0.1 — no external access possible
  • All path arguments validated against HOME and project roots (path traversal prevention)
  • All text inputs length-capped to prevent prompt injection via oversized context
  • Rate limiter on HTTP mode (30 req/min per tool)
  • Tool names are static — no dynamic registration from user input (no lookalike tools)
  • Tool return values never echo raw user input unsanitised into structured fields

Usage (stdio — for Claude Code):
  python -m greenboost_cli.mcp.server

Usage (HTTP — for remote clients, localhost only):
  python -m greenboost_cli.mcp.server --http [--port 7822]

Claude Code .claude/settings.json:
  {
    "mcpServers": {
      "greenboost": {
        "command": "python",
        "args": ["-m", "greenboost_cli.mcp.server"]
      }
    }
  }
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("greenboost-cli")

# ── Security helpers ──────────────────────────────────────────────────────────

# Validators and length caps live in greenboost_cli.security so the headless
# CLI in cli_headless.py applies the same guards.
from greenboost_cli.security import (
    _validate_path, _cap, _safe_project,
    _MAX_TEXT_LEN, _MAX_QUERY_LEN, _MAX_SOURCE_LEN,
    _MAX_GOAL_LEN, _MAX_PROJECT_LEN, _MAX_PATH_LEN,
)

# Per-tool rate limiting is HTTP-specific so it stays here.
_rate_buckets: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT    = 30
_RATE_WINDOW   = 60.0


def _rate_check(tool: str) -> None:
    """Raise ValueError if tool call exceeds rate limit (HTTP mode guard)."""
    now = time.monotonic()
    bucket = _rate_buckets[tool]
    _rate_buckets[tool] = [t for t in bucket if now - t < _RATE_WINDOW]
    if len(_rate_buckets[tool]) >= _RATE_LIMIT:
        raise ValueError(
            f"Rate limit exceeded for '{tool}' ({_RATE_LIMIT} req/{_RATE_WINDOW:.0f}s). "
            "Slow down or use stdio mode."
        )
    _rate_buckets[tool].append(now)


# ── Document conversion ───────────────────────────────────────────────────────

@mcp.tool()
def convert_to_markdown(source: str, feed_rag: bool = True, project: str = "") -> str:
    """Convert a local file or URL to Markdown.

    Supports PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, EPUB, ZIP,
    images, audio files, Jupyter notebooks, YouTube videos, Wikipedia pages,
    and other URLs.

    Args:
        source:   Local file path or URL to convert.
        feed_rag: Auto-index the result into the local RAG store (default True).
        project:  RAG project label (defaults to filename stem).

    Returns:
        Markdown-formatted content of the document.
    """
    _rate_check("convert_to_markdown")
    source  = _validate_path(source, allow_url=True)
    project = _safe_project(project)
    from greenboost_cli.converters.markitdown_adapter import convert
    return convert(source, feed_rag=feed_rag, project=project or None)


# ── RAG ───────────────────────────────────────────────────────────────────────

@mcp.tool()
def rag_search(
    query: str,
    project: str = "",
    top_k: int = 5,
    min_score: float = 0.1,
) -> str:
    """Search the local GreenBoost RAG knowledge base.

    Uses semantic similarity (jina-embeddings-v2-base-code) + BM25 hybrid
    rescoring. Returns ranked code/doc snippets with file paths.

    Args:
        query:     Natural language or code search query.
        project:   Filter results by project name (empty = all projects).
        top_k:     Max results to return (1–20).
        min_score: Minimum hybrid score threshold (0.0–1.0).

    Returns:
        Formatted markdown with results, file:line references, and scores.
    """
    _rate_check("rag_search")
    query   = _cap(query, _MAX_QUERY_LEN)
    project = _safe_project(project)
    top_k   = max(1, min(top_k, 20))
    min_score = max(0.0, min(min_score, 1.0))
    from greenboost_cli.rag.engine import search, format_for_claude
    results = search(query, project=project or None, top_k=top_k, min_score=min_score)
    return format_for_claude(results, query)


@mcp.tool()
def rag_index_folder(folder: str, project: str = "") -> str:
    """Index a local folder into the GreenBoost RAG knowledge base.

    Recursively indexes all supported code/document files. Re-indexing
    a folder replaces previous entries for that path.

    Args:
        folder:  Absolute or relative path to index.
        project: Project label (defaults to folder name).

    Returns:
        Summary: chunks indexed, files skipped, project name.
    """
    _rate_check("rag_index_folder")
    folder  = _validate_path(folder)
    project = _safe_project(project)
    from pathlib import Path
    from greenboost_cli.rag.engine import index_folder
    r = index_folder(Path(folder), project=project or None)
    return (
        f"Indexed {r['indexed']} chunks from project '{r['project']}' "
        f"({r['skipped']} files skipped)."
    )


@mcp.tool()
def rag_index_text(text: str, source_name: str, project: str = "") -> str:
    """Index arbitrary text (markdown, code, notes) into the RAG knowledge base.

    Uses markdown-aware chunking for prose, code-aware chunking for source files.
    Re-indexing the same source_name replaces previous entries.

    Args:
        text:        Text content to index.
        source_name: Identifier (e.g. 'meeting-notes.md', 'api-docs.txt').
        project:     Project label (defaults to 'documents').

    Returns:
        Summary: chunks indexed.
    """
    _rate_check("rag_index_text")
    text        = _cap(text, _MAX_TEXT_LEN)
    source_name = _cap(source_name, _MAX_PATH_LEN)
    project     = _safe_project(project)
    from greenboost_cli.rag.engine import index_text
    r = index_text(text, source_name=source_name, project=project or None)
    return f"Indexed {r['indexed']} chunks from '{source_name}' (project: {r['project']})."


@mcp.tool()
def rag_status() -> str:
    """Get current GreenBoost RAG index statistics.

    Returns:
        Index path, chunk count, file count, size in MB, and indexed folders.
    """
    from greenboost_cli.rag.engine import (
        _load_store, _load_folders, RAG_DIR, EMBEDDINGS_FILE,
    )
    embeddings, metadata = _load_store()
    folders  = _load_folders()
    n_chunks = len(metadata) if metadata else 0
    n_files  = len({m["file"] for m in metadata}) if metadata else 0
    db_mb    = EMBEDDINGS_FILE.stat().st_size / 1_048_576 if EMBEDDINGS_FILE.exists() else 0.0

    lines = [
        f"RAG Index: {RAG_DIR}",
        f"Chunks: {n_chunks} | Files: {n_files} | Size: {db_mb:.1f} MB",
    ]
    if folders:
        lines.append("\nIndexed sources:")
        for f in folders:
            lines.append(
                f"  [{f['project']}] {f['folder']}"
                f" — {f['chunk_count']} chunks ({f['last_indexed'][:10]})"
            )
    else:
        lines.append("No sources indexed yet.")
    return "\n".join(lines)


# ── Goals & history ───────────────────────────────────────────────────────────

@mcp.tool()
def get_goals(project: str = "") -> str:
    """Get goals for a GreenBoost project.

    Args:
        project: Project name (empty = active project from settings).

    Returns:
        Formatted list of goals with IDs and priorities.
    """
    _rate_check("get_goals")
    project = _safe_project(project)
    from greenboost_cli.memory.brain import project_dir, load_goals
    pdir  = project_dir(project or None)
    goals = load_goals(pdir)
    if not goals:
        return f"No goals for project '{project or 'default'}'."
    lines = [f"Goals — {project or 'default'}:"]
    for g in goals:
        lines.append(f"  [{g['id']}] P{g['priority']} · {g['text']}")
    return "\n".join(lines)


@mcp.tool()
def add_goal(text: str, project: str = "", priority: int = 5) -> str:
    """Add a goal to a GreenBoost project.

    Args:
        text:     Goal description.
        project:  Project name (empty = active project).
        priority: Priority 1–10 (default 5, lower = higher priority).

    Returns:
        Confirmation with the new goal's ID.
    """
    _rate_check("add_goal")
    text     = _cap(text, _MAX_GOAL_LEN)
    project  = _safe_project(project)
    priority = max(1, min(priority, 10))
    from greenboost_cli.memory.brain import project_dir, add_goal as _add
    pdir = project_dir(project or None)
    g    = _add(pdir, text, priority=priority)
    return f"Goal [{g['id']}] added (P{g['priority']}): {g['text']}"


@mcp.tool()
def get_history(project: str = "", n: int = 10) -> str:
    """Get recent history entries for a GreenBoost project.

    Args:
        project: Project name (empty = active project).
        n:       Number of entries to return (default 10).

    Returns:
        Recent history entries with timestamps and categories.
    """
    _rate_check("get_history")
    project = _safe_project(project)
    n       = max(1, min(n, 100))
    from greenboost_cli.memory.brain import project_dir, read_recent_history
    pdir = project_dir(project or None)
    return read_recent_history(pdir, n_entries=n) or f"No history for '{project or 'default'}'."


# ── System status ──────────────────────────────────────────────────────────────

@mcp.tool()
def system_status() -> str:
    """Get GreenBoost GPU memory tier status and system info.

    Returns:
        T1/T2/T3 tier stats, VRAM usage, and active GreenBoost configuration.
    """
    try:
        from greenboost_cli.greenboost.monitor import get_tier_stats, get_banner_line
        stats  = get_tier_stats()
        banner = get_banner_line()
        lines  = [f"GreenBoost — {banner}", ""]
        for tier, data in stats.items():
            lines.append(f"  {tier}: {data}")
        return "\n".join(lines)
    except Exception as e:
        return f"System status unavailable: {e}"


def _gb_repo_module(name: str):
    """Import a module from the greenboost repo checkout (gb_monitor, gb_pilot,
    gb_dataflux…) — via the shared gb_paths resolver."""
    from greenboost_cli.gb_paths import gb_module
    return gb_module(name)


@mcp.tool()
def greenboost_capabilities() -> dict:
    """What the installed/running GreenBoost shim supports, from the canonical
    capability manifest chain (runtime /run/greenboost/capabilities.json written
    by the shim at init → install manifest → binary sniff). Check this before
    assuming a shim feature (gds, kv_compress, expert_pool, cluster fabric,
    gb-quant cudart rebind) is available."""
    try:
        return _gb_repo_module("gb_monitor").capabilities()
    except Exception as e:
        return {"error": f"gb_monitor unavailable: {e}", "features": {}}


@mcp.tool()
def greenboost_pilot(days: float = 5.0) -> dict:
    """The pilot's instrument panel over GreenBoost's dataflux flight recorder:
    per-stage wall-time trends (ai-forge stage_profile events), measured tok/s
    per model, latest VRAM/T2/T3/KV pressure, and evidence-backed advice that
    names the exact GbControl lever or config change (re-quant, GB_TQ_ATTN) to
    consider. Read-only — advice is never auto-applied. Use it to decide how to
    redirect orchestration from real measurements instead of guessing."""
    try:
        gb_dataflux = _gb_repo_module("gb_dataflux")
        gb_pilot = _gb_repo_module("gb_pilot")
        analysis = gb_pilot.analyze(gb_dataflux.read_events(since_hours=days * 24))
        analysis["advice"] = gb_pilot.advise(analysis)
        return analysis
    except Exception as e:
        return {"error": f"pilot unavailable: {e}"}


# ── AI Factory ───────────────────────────────────────────────────────────────

@mcp.tool()
def factory_submit(
    prompt: str,
    agent: str = "",
    priority: int = 10,
) -> str:
    """Submit a task to the GreenBoost AI factory.

    The factory runs background agents that execute tasks autonomously using
    GreenBoost VRAM-aware scheduling.

    Args:
        prompt:   Natural language task description.
        agent:    Agent name to assign (empty = auto-select best agent).
        priority: Task priority 1–20 (1=highest, default 10).

    Returns:
        Task ID confirmation.
    """
    _rate_check("factory_submit")
    prompt   = _cap(prompt, _MAX_TEXT_LEN)
    agent    = _cap(agent, _MAX_PROJECT_LEN)
    priority = max(1, min(priority, 20))
    from greenboost_cli.workflow.factory import get_factory
    factory = get_factory()
    task_id = factory.submit(prompt=prompt, agent_name=agent, priority=priority)
    return f"Task submitted — ID: {task_id}  priority: {priority}"


@mcp.tool()
def factory_status() -> str:
    """Get current AI factory status: agents, queue depth, GPU usage.

    Returns:
        JSON-formatted factory snapshot.
    """
    _rate_check("factory_status")
    import json
    from greenboost_cli.workflow.factory import get_factory
    snap = get_factory().snapshot()
    lines = [
        f"Factory active : {snap['active']}",
        f"Queue depth    : {snap['queue_depth']}",
        f"GPU usage      : {snap['gpu_ratio']:.1f}%",
        "",
        "Agents:",
    ]
    for name, info in snap["agents"].items():
        status = "PAUSED" if info["paused"] else info["current_task"]
        lines.append(
            f"  {name:<16} {status:<40} "
            f"VRAM {info['vram_used_mb']}MB/{info['vram_used_mb']+info['vram_free_mb']}MB"
        )
    if snap["recent"]:
        lines += ["", "Recent completions:"]
        for r in snap["recent"][:5]:
            ok = "✓" if r.get("success") else "✗"
            lines.append(f"  {ok} {r.get('agent_name','?'):<12} {(r.get('prompt') or '')[:50]}")
    return "\n".join(lines)


# ── Contextual RAG ────────────────────────────────────────────────────────────

@mcp.tool()
def contextual_rag_search(
    query: str,
    project: str = "default",
    top_k: int = 5,
) -> list[dict]:
    """Search documents using Contextual Retrieval (hybrid BM25+vector + reranking).

    Uses the Anthropic Contextual Retrieval recipe: each chunk was enriched
    with a situating context before indexing, then searched with hybrid
    BM25+vector RRF fusion and cross-encoder reranking. Retrieves ~67% fewer
    wrong answers compared to standard RAG.

    Prefer this over rag_search for structured documents (PDFs, reports,
    specs, papers) where table and heading fidelity matters.

    Args:
        query:   The search query (semantic or keyword).
        project: Index project to search (default: "default").
        top_k:   Number of results (1–20, default 5).

    Returns:
        List of {source, heading_path, score, snippet} dicts, best-first.
    """
    _rate_check("contextual_rag_search")
    query   = _cap(query, _MAX_QUERY_LEN)
    project = _safe_project(project)
    top_k   = max(1, min(top_k, 20))
    from greenboost_cli.rag.contextual_rag import search  # noqa: PLC0415
    hits = search(query, project=project, top_k=top_k)
    return [
        {
            "source":       h.source,
            "heading_path": h.heading_path,
            "score":        h.score,
            "snippet":      h.snippet[:2000],
        }
        for h in hits
    ]


@mcp.tool()
def contextual_rag_add(path: str, project: str = "default") -> dict:
    """Ingest a document or folder into the Contextual RAG index.

    For each chunk: a cheap LLM (Haiku) writes 50-100 tokens of situating
    context prepended BEFORE embedding and BM25 indexing. This is the
    Anthropic Contextual Retrieval recipe.

    Supported: .md, .txt, .rst, .pdf, .docx, .html, .tex

    Args:
        path:    Absolute or home-relative path to file or directory.
        project: Index project name (default: "default").

    Returns:
        {"ok": true/false, "path": str, "project": str}  for a single file
        {"indexed": N, "errors": N, "project": str}       for a directory
    """
    _rate_check("contextual_rag_add")
    path    = _validate_path(path)
    project = _safe_project(project)
    from pathlib import Path as _Path
    from greenboost_cli.rag.contextual_rag import ingest_document, ingest_folder  # noqa: PLC0415
    target = _Path(path).expanduser().resolve()
    if target.is_dir():
        result = ingest_folder(str(target), project=project)
        result["project"] = project
        return result
    ok = ingest_document(str(target), project=project)
    return {"ok": ok, "path": path, "project": project}


@mcp.tool()
def contextual_rag_status(project: str = "default") -> dict:
    """Return statistics for a Contextual RAG index.

    Args:
        project: Index project name (default: "default").

    Returns:
        {"project", "chunk_count", "doc_count", "model", "indexed_at"}
        or {"project", "error": "no index found"}
    """
    _rate_check("contextual_rag_status")
    project = _safe_project(project)
    import re as _re, json as _json, datetime  # noqa: E401, PLC0415
    from pathlib import Path as _Path
    d = _Path.home() / ".greenboost_cli" / "contextual_rag" / _re.sub(r"[^\w.-]", "_", project)
    meta_f = d / "meta.json"
    if not meta_f.exists():
        return {"project": project, "error": "no index found"}
    meta = _json.loads(meta_f.read_text())
    ts = meta.get("time")
    indexed_at = datetime.datetime.fromtimestamp(ts).isoformat() if ts else "?"
    return {
        "project":     project,
        "chunk_count": meta.get("chunk_count", "?"),
        "doc_count":   meta.get("doc_count",   "?"),
        "model":       meta.get("model",        "?"),
        "indexed_at":  indexed_at,
    }


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("rag://index/status")
def rag_status_resource() -> str:
    """Current RAG index status (chunk count, files, size, indexed folders)."""
    return rag_status()


@mcp.resource("goals://{project}")
def goals_resource(project: str) -> str:
    """Goals for a specific project."""
    return get_goals(project)


@mcp.resource("history://{project}")
def history_resource(project: str) -> str:
    """Recent history for a specific project (last 20 entries)."""
    return get_history(project, n=20)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    use_http = "--http" in sys.argv
    if use_http:
        port = 7822
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 1])
                except ValueError:
                    pass
        print(
            f"GreenBoost MCP server (HTTP) starting on http://127.0.0.1:{port}/mcp",
            file=sys.stderr,
        )
        mcp.run(transport="streamable-http", host="127.0.0.1", port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
