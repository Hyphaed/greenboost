"""Assembles the system context injected into every AI request."""
from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path
from datetime import datetime

_GB_FLAG_DIR  = Path("/etc/greenboost")
_GB_TQ_FLAG   = _GB_FLAG_DIR / "turboquant.enabled"
_GB_CLUSTER   = _GB_FLAG_DIR / "cluster.conf"

_BASE_PROMPT = """\
You are GreenBoost CLI, a unified AI coding assistant running in the terminal.
You help engineers with software tasks: writing code, debugging, refactoring, explaining systems, and shell automation.

# Capabilities
- **Read**: Read file contents with line numbers (use limit/offset for large files; always read before editing)
- **Write**: Create or overwrite files (creates parent dirs automatically; prefer Edit for existing files)
- **Edit**: Replace exact text in a file (old_string must match exactly including whitespace; add surrounding context if needed)
- **Bash**: Execute shell commands — `cd` persists across calls within a session; default timeout 120 s
- **Glob**: Find files by glob pattern (e.g. `**/*.py`, `src/**/*.ts`)
- **Grep**: Search file contents with regex (uses ripgrep when available; supports context lines)
- **WebFetch**: Fetch a URL and return its text content
- **WebSearch**: Search the web via DuckDuckGo (returns titles + URLs + snippets)
- **AskUserQuestion**: Ask clarifying questions via a stepped wizard (options + free-text). Use this **instead of** writing a question as prose — the user picks from options and the turn continues automatically with their answers.
- **TodoWrite**: Write the session task list (replaces entire list). Use to track multi-step work: create tasks at start, set one to `in_progress`, mark `completed` when done. Always maintain at most one `in_progress` task.
- **TodoRead**: Read the current session task list. Call before long tasks to check what's pending.
- **MemoryWrite**: Write a persistent note to CLAUDE.md (project or global). Use to save project conventions, architectural decisions, and key facts discovered during a session so they persist across sessions.
- **MemoryRead**: Read CLAUDE.md memory files. Call at the start of a session or before a task that might benefit from remembered context.

# Coding Architect Workflow
Persona: senior software architect who also writes code. Primary objective: design the correct solution.
Secondary: implement it efficiently. Never optimise for speed at the expense of architecture.

Implementation workflow:
  Understand → Retrieve → Plan → [Confidence ≥85%?] → Implement → Review

Confidence gate: if <85% confident in the approach, retrieve more context or ask ONE clarifying question.
Do not invent architecture. Stop reasoning at >95% confidence — do not continue for completeness.

Token budget: Planning 10% · Retrieval 20% · Implementation 60% · Review 10%
Never spend more on planning than on implementation.

Retrieval order — never skip levels:
  1. knowledge-rag  →  architecture, design decisions, ADRs, constraints
  2. semble         →  symbol lookup, implementations, code search
  3. reranker       →  cut N chunks to top 3 before reading
  4. scrapling      →  external URLs/docs (targeted extraction, never full pages)
  5. Read snippet   →  last resort; state what is missing first

Intent routing:
  "why / design / constraint / ADR"   →  knowledge-rag first
  "where / how coded / symbol"        →  semble first
  "external URL / web content"        →  scrapling (css_selector target)
  both design + code                  →  knowledge-rag + semble in parallel
  exact literal string                →  Grep/Glob only

Progressive retrieval: fetch 3 → reason → confidence ok? → done.
If not, state what is missing, then retrieve 3 more. Stop as soon as sufficient.

PROHIBITED: directory scanning for exploration · speculative file opens · Grep before semantic search ·
reading entire files when snippets exist · fetching full webpages · loading >5 chunks per round ·
assuming conventions without retrieving project docs first · writing code before reaching 85% confidence.

# Code Modification Protocol
When modifying existing files, prefer SEARCH/REPLACE blocks over full rewrites:

  <<<< SEARCH [relative/path/to/file.ext]
  exact original text (must appear exactly once — add context lines if needed)
  ==== REPLACE
  new replacement text
  >>>>

Rules: SEARCH text must be unique in the file · paths relative to project root ·
multiple blocks applied transactionally · never output full file when a diff block suffices ·
new files use: CREATE [path] with full content.

# Tool-calling discipline
- Call **one tool at a time**. Wait for the result before deciding the next step.
- If a tool returns an error or `Denied`, read the message carefully, diagnose the root cause,
  and retry with corrected parameters. **Never retry with identical arguments.**
- Do **not** claim a file was written, edited, or a command was run unless you personally
  issued the tool call and saw its result.
- After Write or Edit, trust the tool result — it raises on failure. Do not re-Read solely to confirm.
- If approval is denied, explain what you were trying to do and ask the user how to proceed.
- When you need a clarifying decision or missing detail — including at the end of a response —
  you MUST call **AskUserQuestion**. NEVER write a question as plain text. This is a hard rule.

  WRONG: "Do you want me to incorporate those before you approve?"
  CORRECT: Call AskUserQuestion with options like [Yes, incorporate] [No, keep as-is].

  Even if you've already written a plan or suggestions, end with AskUserQuestion — not prose.

# Security & Quality
- Never introduce security vulnerabilities: XSS, SQL injection, command injection, path traversal, OWASP Top 10.
- Never hardcode secrets, API keys, or tokens — use environment variables or config files.
- Validate input at system boundaries only (user input, external APIs). Trust internal code.
- No error handling for scenarios that can't happen; no defensive layers for internal invariants.

# Git Protocol
- Never run `git push` or any variant — the developer pushes.
- Never add `Co-Authored-By:` or any AI/Claude attribution in commit messages.
- Create NEW commits; never amend published ones.
- Stage specific files (`git add path/to/file`) rather than `git add -A` to avoid staging secrets.
- Never use `--no-verify` or skip hooks unless the user explicitly asks.

# Operating Principles
- Lead with the answer. Direct, concise — no preamble, no apologies, no hedging.
- Drop filler: avoid "just", "really", "basically", "certainly", "sure, happy to".
  Pattern: [thing] [action] [reason]. [next step].
- Fewest edits: one well-designed change over five incremental ones. One request → one diff.
- No gold-plating: fulfil the request as stated. No unrequested helpers, docs, or refactors.
- Always use absolute paths for file operations.
- Read relevant lines (with offset/limit) before making edits — never guess at content.
- Match the surrounding code's style: comment density, naming, indentation.
- Ask a single, specific clarifying question when the task is genuinely ambiguous.
- Before marking a task complete, verify the change works (run tests, check output).
{project_notes}{goals_context}{rag_summary}{ui_guidelines}{plan_session}
# Current Session
- Date: {date}
- Directory: {cwd}
- Platform: {platform}
{git_context}{gb_context}"""


def _gather_git_context() -> str:
    """Return git branch/status/log summary if inside a git repository."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-5"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        parts = [f"- Git branch: {branch}"]
        if status:
            lines = status.split("\n")[:10]
            parts.append("- Git status:\n" + "\n".join(f"  {l}" for l in lines))
        if log:
            parts.append("- Recent commits:\n" + "\n".join(f"  {l}" for l in log.split("\n")))
        return "\n".join(parts) + "\n"
    except Exception:
        return ""


_MAX_NOTES_CHARS = 4000  # per CLAUDE.md file; full file available via MemoryRead tool


def _cap_notes(text: str, path: Path) -> str:
    """Centre-truncate a CLAUDE.md to _MAX_NOTES_CHARS with a retrieval hint."""
    if len(text) <= _MAX_NOTES_CHARS:
        return text
    half = _MAX_NOTES_CHARS // 2
    return (
        text[:half]
        + f"\n\n…(truncated {len(text) - _MAX_NOTES_CHARS} chars"
        f" — use MemoryRead('{path}') for the full file)…\n\n"
        + text[-half:]
    )


def _gather_project_notes() -> str:
    """Load CLAUDE.md from working directory ancestors (limited to current project tree) and ~/.claude/CLAUDE.md."""
    parts: list[str] = []

    global_md = Path.home() / ".claude" / "CLAUDE.md"
    if global_md.exists():
        try:
            parts.append(f"[Global notes]\n{_cap_notes(global_md.read_text(), global_md)}")
        except Exception:
            pass

    # Walk ancestors only within the current CWD's git root — never cross out of
    # the project tree.  Without this guard, sibling projects (e.g. under a common
    # parent like ~/Dev/greenboost_all/) can have their CLAUDE.md picked up,
    # which confuses the model into thinking it's in that sibling project.
    project_root = Path.cwd()
    for _ in range(10):
        candidate = project_root / "CLAUDE.md"
        if candidate.exists():
            try:
                parts.append(
                    f"[Project notes: {candidate}]\n"
                    f"{_cap_notes(candidate.read_text(), candidate)}"
                )
            except Exception:
                pass
            break
        # If we hit a .git root, stop — never cross into a sibling tree
        git_dir = project_root / ".git"
        if git_dir.exists() or git_dir.is_symlink():
            break
        parent = project_root.parent
        if parent == project_root:
            break
        project_root = parent

    if not parts:
        return ""
    return "\n# Notes\n" + "\n\n".join(parts) + "\n"


def _gather_goals_context() -> str:
    """Return goals/brain summary if the memory module is available."""
    try:
        from greenboost_cli.memory.brain import get_goals_summary
        return get_goals_summary()
    except ImportError:
        return ""
    except Exception:
        return ""


def _gather_rag_context() -> str:
    """Return RAG context summary scoped to the current folder if indexed."""
    try:
        from greenboost_cli.rag.engine import get_context_summary, resolve_folder_entry
        entry = resolve_folder_entry()
        folder = entry["folder"] if entry else None
        return get_context_summary(folder=folder)
    except ImportError:
        return ""
    except Exception:
        return ""


_MAX_GUIDELINES_CHARS = 2000


def _gather_ui_guidelines_context() -> str:
    """Return active UI guidelines formatted for injection into the system prompt."""
    try:
        from greenboost_cli.memory.ui_guidelines import get_guidelines_context
        text = get_guidelines_context()
        if len(text) > _MAX_GUIDELINES_CHARS:
            text = text[:_MAX_GUIDELINES_CHARS] + "\n…(truncated — use /ui-guidelines to view all)\n"
        return text
    except ImportError:
        return ""
    except Exception:
        return ""


def _greenboost_context() -> str:
    """Return a GreenBoost status block for the system prompt."""
    # Prefer the canonical block from gb_monitor.context_summary() (greenboost
    # repo) — the same banner + OOM/T3-spill/T2-pressure warnings this function
    # historically built, now owned in one place so every consumer emits it
    # identically. Fall back to the local monitor path when the checkout is
    # absent, then to the flag-file fallback further down.
    try:
        from greenboost_cli.gb_paths import gb_module
        gb_monitor = gb_module("gb_monitor")
        _summary = gb_monitor.context_summary()
        if _summary:
            return _summary
    except Exception:
        pass

    try:
        from greenboost_cli.greenboost.monitor import get_tier_stats, get_banner_line, get_monitor
        stats = get_tier_stats()
        if not stats:
            return ""
        line = get_banner_line(stats)

        # Critical warnings that should influence model behaviour
        warnings: list[str] = []
        if stats.get("oom_active"):
            warnings.append("WARNING: GreenBoost OOM recovery is ACTIVE — memory critically low.")
        t3_used = stats.get("t3_swap_used_mb", 0)
        if t3_used > 0:
            t3_gb = round(t3_used / 1024, 1)
            warnings.append(
                f"WARNING: T3 NVMe spillover active ({t3_gb} GB on disk swap) — "
                "inference is ~100× slower than normal. Avoid large context expansions."
            )
        t2_pressure = stats.get("t2_pressure", 0)
        if t2_pressure == 2:
            warnings.append("WARNING: T2 DDR pressure is CRITICAL — limit large tool outputs.")
        elif t2_pressure == 1:
            warnings.append("NOTICE: T2 DDR pressure is elevated (warn level).")

        # Shim active path (useful context for debugging inference issues)
        try:
            s = get_monitor().status
            if not s.shim_stale and s.shim_active_path:
                line += f"- GreenBoost shim path: {s.shim_active_path}  phase: {s.shim_phase}\n"
        except Exception:
            pass

        if warnings:
            line += "".join(f"- {w}\n" for w in warnings)
        return line

    except ImportError:
        pass
    except Exception:
        pass

    # Flag-based fallback when monitor module unavailable
    if not _GB_FLAG_DIR.exists():
        return ""
    parts = ["GreenBoost active — T1/T2/T3 memory tiers enabled."]
    if _GB_TQ_FLAG.exists() or os.environ.get("GREENBOOST_TURBOQUANT") == "1":
        parts.append("TurboQuant KV compression: ON.")
    try:
        feeders = sum(1 for ln in _GB_CLUSTER.read_text().splitlines() if ln.strip())
        if feeders:
            parts.append(f"Cluster: {feeders} remote feeder(s) — extended pool available.")
    except Exception:
        pass
    parts.append("Prefer large context windows; memory pool extends beyond local VRAM.")
    return "\n- " + " ".join(parts) + "\n"


def _gather_mcp_context() -> str:
    """Describe MCP servers and tools from .mcp.json in the current project tree."""
    try:
        from greenboost_cli.mcp.client import discover_mcp_json, MCPRegistry
        mcp_path = discover_mcp_json()
        if not mcp_path:
            return ""
        servers = MCPRegistry.load_servers_config(mcp_path)
        if not servers:
            return ""
        lines = [f"\n# MCP Servers (from {mcp_path.name})"]
        for name, cfg in servers.items():
            transport = cfg.get("type", "stdio")
            desc = cfg.get("url", " ".join(cfg.get("args", [])[:1]))
            lines.append(f"- **{name}** ({transport}): {desc}")
        lines.append(
            "\nThese MCP tools are available alongside the built-in tools (Read, Write, Bash, etc.)."
            " Call them by name when generating game assets, 3D models, video, audio, or images."
            " Use `forge gen/doctor/game/list` for batch manifest workflows."
        )
        if any(n.startswith("greenboost") for n in servers):
            lines.append(
                "\nGB-Semantics is the MANDATORY DEFAULT PATH for any question about "
                "GreenBoost's own state (VRAM fill, tier pressure, tok/s, quality floor, "
                "cluster health, etc.): call `semantic_resolve`/`semantic_answer`/"
                "`semantic_segments` (greenboost-orchestrator) FIRST. Reading a raw "
                "dataflux/telemetry field directly is the fallback ONLY, and the answer "
                "must say so — several raw fields in this codebase look plausible but are "
                "wrong (see semantics/metrics.yaml's `never_use` entries)."
            )
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def _gather_plan_session() -> str:
    """Inject plan_session.md if present (session recovery from optimal-claude or gb)."""
    from greenboost_cli.environment.settings import GB_HOME
    project_name = Path.cwd().name
    candidates = [
        GB_HOME / "projects" / project_name / "plan_session.md",
        Path.home() / ".claude_workflow" / "projects" / project_name / "plan_session.md",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            if len(content) > 3000:
                content = content[:3000] + "\n...(truncated)"
            return f"\n\n# Previous Session\n{content}\n"
        except Exception:
            continue
    return ""


def _gather_ollama_system_prompt(model: str) -> str:
    """Extract the SYSTEM block from an Ollama Modelfile. Models with a tuned
    Modelfile often ship real behavioral guidance there — e.g.
    rafw007/qwen36-a3b-claude-coder's "/nothink" + honest-tool-use, no-loop,
    no-hallucination instructions — that gb-synapse's llama-server otherwise
    never sees, since it loads the raw GGUF directly and has no concept of
    Ollama's Modelfile format. Best-effort: not an Ollama model, ollama not
    running/installed, or no SYSTEM block all silently return ""."""
    if not model:
        return ""
    try:
        result = subprocess.run(
            ["ollama", "show", model, "--modelfile"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return ""
        text = result.stdout
        m = re.search(r'^SYSTEM\s+"""([\s\S]*?)"""', text, re.MULTILINE)
        if not m:
            m = re.search(r'^SYSTEM\s+"([\s\S]*?)^"\s*$', text, re.MULTILINE)
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


def assemble_system_context(model: str = "") -> str:
    """Build the full system prompt for the current session."""
    base = _BASE_PROMPT.format(
        date=datetime.now().strftime("%Y-%m-%d %A"),
        cwd=str(Path.cwd()),
        platform=platform.system(),
        git_context=_gather_git_context(),
        project_notes=_gather_project_notes(),
        goals_context=_gather_goals_context(),
        rag_summary=_gather_rag_context(),
        ui_guidelines=_gather_ui_guidelines_context(),
        gb_context=_greenboost_context() + _gather_mcp_context(),
        plan_session=_gather_plan_session(),
    )
    ollama_system = _gather_ollama_system_prompt(model)
    if ollama_system:
        base += (
            "\n\n# Model-Specific Guidance\n\n"
            "This model ships its own tuned system prompt from its Ollama Modelfile. "
            "Follow it alongside the instructions above:\n\n" + ollama_system
        )
    return base
