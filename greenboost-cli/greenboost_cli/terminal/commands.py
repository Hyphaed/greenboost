"""
Slash command handlers and dispatcher.

Each handler has signature:  fn(args: str, session, settings) -> bool
Returns True to indicate the line was handled.
"""
from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path

from greenboost_cli.terminal.theme import (
    emit_ok, emit_err, emit_warn, emit_info, emit_section, VIOLET, GRAY, LIME, AMBER, CYAN, TEAL, DIM
)
from greenboost_cli.terminal.theme import console


# ── Basic commands ─────────────────────────────────────────────────────────

def cmd_help(_args: str, _session, _settings) -> bool:
    console.print(f"""
[bold {VIOLET}]GreenBoost CLI — Commands[/]
[{GRAY}]────────────────────────────────────────────────────[/]

[bold {LIME}]Session[/]
  [{GRAY}]/status[/]                    Session info: model, tokens, mode, cwd
  [{GRAY}]/clear[/]                     Clear conversation history
  [{GRAY}]/history[/]                   Print conversation messages
  [{GRAY}]/context[/]                   Show context window usage
  [{GRAY}]/cost[/]                      Show token usage this session
  [{GRAY}]/save [file][/]               Save session to file
  [{GRAY}]/load [file][/]               Load session from file
  [{GRAY}]/resume [file][/]             Load last (or named) saved session
  [{GRAY}]/sessions [load|search <q>][/] List sessions; load one or search content
  [{GRAY}]/name [name][/]               Show or set session name (like Claude Code)
  [{GRAY}]/fork [name][/]               Snapshot session at this point (rename + reset timer)
  [{GRAY}]/retry[/]                     Re-run the last user message
  [{GRAY}]/undo[/]                      Remove the last user+assistant exchange
  [{GRAY}]/compact[/]                   Compress old history to save context tokens
  [{GRAY}]/note <text>[/]               Inject a note into context (no model call)
  [{GRAY}]/add <path|glob> [:<lines>][/] Inject file content into context (no model call)
  [{GRAY}]/image <path> [caption][/]    Attach an image for the next message (vision)
  [{GRAY}]/init [--force][/]            Generate CLAUDE.md for this project
  [{GRAY}]/permissions [mode][/]        Show or set permission mode  (auto|accept-all|manual)

[bold {LIME}]Model & Backend[/]
  [{GRAY}]/model [list|name][/]         Show current, arrow-key pick, or switch directly
  [{GRAY}]/fetch-model huggingface[/]   Ask for a HF token + repo, pull it via gb-synapse
  [{GRAY}]/fetch-model ollama[/]        List/pull Ollama models, indexed into gb-synapse
  [{GRAY}]/download-models[/]           Browse-and-pull variant (HuggingFace GGUF / Ollama)

[bold {LIME}]GreenBoost[/]
  [{GRAY}]/gb-quant <model> [bits][/]             Quantize LLM with gb-quant (FP8/INT4/TQ3, in-process test)
  [{GRAY}]/turboquant [on|off|status][/]         Toggle TurboQuant KV compression
  [{GRAY}]/llamaserve [start|stop|status|logs|restart][/]  gb-synapse llama-server lifecycle
  [{GRAY}]/llamacache [status|save|restore|erase][/]  Disk-persisted prompt cache
  [{GRAY}]/gb-status[/]                          Show T1/T2/T3/T4 tier statistics
  [{GRAY}]/gb-tiers[/]                           Detailed memory tier breakdown
  [{GRAY}]/gb-vitals [on|off][/]                 Full debug vitals dump (all 9 data sources)

[bold {LIME}]Memory & Brain[/]
  [{GRAY}]/goals [add|remove|list] [text][/]  Manage project goals
  [{GRAY}]/history-show [n][/]               Show last N history entries (default 10)
  [{GRAY}]/history-add <text>[/]             Add a history entry
  [{GRAY}]/history-search <query>[/]         Search history log by keyword
  [{GRAY}]/snapshot [note][/]               Save a project snapshot
  [{GRAY}]/save-session [show|clear][/]     Save session state for recovery on next launch
  [{GRAY}]/project [name][/]               Show or switch active project
  [{GRAY}]/tokens[/]                        Show API/local token usage stats
  [{GRAY}]/resume [file][/]                 Load last (or named) saved session
  [{GRAY}]/todo[/]                          Show current session task list (TodoWrite/Read)
  [{GRAY}]/memory [show|project|global|edit|write][/]  View/edit CLAUDE.md memory

[bold {LIME}]Web[/]
  [{GRAY}]/websearch <query>[/]         Search web via DuckDuckGo (Scrapling)

[bold {LIME}]RAG Search[/]
  [{GRAY}]/rag-add <path>[/]            Index a file or directory
  [{GRAY}]/rag-search <query>[/]        Search indexed code (top 5)
  [{GRAY}]/rag-status[/]                Show RAG index statistics
  [{GRAY}]/rag-clear[/]                 Clear the RAG index
  [{GRAY}]/rag-inject <query>[/]        Inject RAG results into next prompt

[bold {LIME}]Design & Diffusion[/]
  [{GRAY}]/design "<description>"[/]    Run full design pipeline
  [{GRAY}]/design-gen <type> [flags][/] Generate UI asset with FLUX/SD
  [{GRAY}]/design-intel "<query>"[/]    Query design intelligence DB
  [{GRAY}]/design-models[/]             List available diffusion models

[bold {LIME}]Documents & Conversion[/]
  [{GRAY}]/convert <file|url> [flags][/] Convert any format to Markdown + index RAG
  [{GRAY}]/pdf2md <file> [flags][/]      Convert PDF/DOCX/PPTX to Markdown
  [{GRAY}]/apply-diff [file][/]          Apply SEARCH/REPLACE diff blocks

[bold {LIME}]AI Factory[/]
  [{GRAY}]/factory start [workers][/]   Start AI factory (default 2 workers)
  [{GRAY}]/factory stop[/]              Stop factory
  [{GRAY}]/factory status[/]            Queue depth, GPU %, agent states
  [{GRAY}]/factory submit <task>[/]     Submit a task to the factory
  [{GRAY}]/factory agents [add|remove|list][/]  Manage agents
  [{GRAY}]/factory history[/]           Recent completed tasks

[bold {LIME}]Dashboard & MCP[/]
  [{GRAY}]/dashboard [port][/]          Open web dashboard  (default :7821)
  [{GRAY}]/mcp config|status|start|stop|logs|sync-accounts[/]  MCP server management

[bold {LIME}]Diagnostics[/]
  [{GRAY}]/doctor [--fix][/]            Full health check: CUDA, RAG, gb-synapse, deps

[bold {LIME}]Settings[/]
  [{GRAY}]/config [key=val][/]          Show config or set a value
  [{GRAY}]/permissions [mode][/]        Set mode (auto | accept-all | manual)
  [{GRAY}]/verbose[/]                   Toggle verbose mode (full tool I/O)
  [{GRAY}]/quiet[/]                     Toggle quiet mode (tally + ✻ footer only)
  [{GRAY}]/cwd [path][/]               Show or change working directory
  [{GRAY}]/setup[/]                     Run the interactive setup wizard

[bold {LIME}]Git Workflow[/]
  [{GRAY}]/commit [note][/]             Stage all and create a conventional commit
  [{GRAY}]/git-review [aspects][/]      Review current diff (quality/bugs/tests)
  [{GRAY}]/git-clean[/]                 Delete branches marked [gone] on remote
  [{GRAY}]/git-pr [instructions][/]     Create a pull request via gh CLI

[bold {LIME}]GreenBoost Memory[/]
  [{GRAY}]/clear-memory[/]              Run `sudo greenboost clear memory-pool` (T1+T2)
  [{GRAY}]/gb-pool-cap [auto|<gb>][/]   Show or set T2 RAM pool cap

[bold {LIME}]Prompt Queue[/]
  [{GRAY}]/queue[/]                     List queued prompts (typed during thinking)
  [{GRAY}]/queue del N[/]               Delete queued prompt #N
  [{GRAY}]/queue edit N text[/]         Replace queued prompt #N with new text
  [{GRAY}]/queue clear[/]               Discard all queued prompts

[bold {LIME}]Plan Mode[/]
  [{GRAY}]/plan [prompt][/]             Enter plan mode (explore-only, no edits)
  [{GRAY}]/plan-edit[/]                 Edit the current plan file
  [{GRAY}]/plan-approve[/]              Approve plan and exit plan mode
  [{GRAY}]/plan-exit[/]                 Exit plan mode without approving
  [{GRAY}]/plan-list[/]                 List all saved plans

[bold {LIME}]Tasks[/]
  [{GRAY}]/task-add <title>[/]          Create a task for the current session
  [{GRAY}]/task-list[/]                 List tasks (pending / in-progress / done)
  [{GRAY}]/task-update <id> <status>[/] Update task status
  [{GRAY}]/task-delete <id>[/]          Delete a task

[bold {LIME}]Skills & Agents[/]
  [{GRAY}]/skill-list[/]                List available skills
  [{GRAY}]/skill-show <name>[/]         Show skill details / instructions
  [{GRAY}]/skill-set-dir <path>[/]      Set custom skills directory
  [{GRAY}]/agent <task>[/]              Spawn a sub-agent for a task
  [{GRAY}]/autonomous-coding [on|off][/]  Headless mode — code while you sleep
  [{GRAY}]/ui-guidelines [query][/]     Search UI/UX design guidelines (BM25)

[bold {LIME}]Session Control[/]
  [{GRAY}]/exit  /quit[/]               Exit GreenBoost CLI
""")
    return True


def cmd_status(_args: str, session, settings: dict) -> bool:
    """Show Claude Code-style session info panel."""
    import os
    import time as _time
    from greenboost_cli.inference.router import resolve_backend
    from greenboost_cli.instruments.handlers import _bash_cwd
    from greenboost_cli.terminal.theme import TEAL, DIM, BOX_H

    model   = settings.get("model", "?")
    backend = resolve_backend(model)
    msgs    = len(session.messages)
    turns   = msgs // 2
    cwd     = _bash_cwd or os.getcwd()

    # Context estimate
    from greenboost_cli.environment.settings import gb_synapse_ctx
    ctx_chars  = sum(len(str(m.get("content", ""))) for m in session.messages)
    ctx_tokens = ctx_chars // 4
    max_tokens = int(settings.get("context_window", 0)) or gb_synapse_ctx(settings)
    ctx_pct    = min(100, int(ctx_tokens / max(1, max_tokens) * 100))

    # Context bar (30 chars wide)
    bar_width = 30
    filled    = int(bar_width * ctx_pct / 100)
    empty     = bar_width - filled
    if ctx_pct >= 95:
        bar_color = "red"
    elif ctx_pct >= 85:
        bar_color = AMBER
    else:
        bar_color = TEAL
    ctx_bar = f"[{bar_color}]{'█' * filled}[/][{DIM}]{'░' * empty}[/]"

    # Elapsed time
    elapsed_str = ""
    start_t = getattr(session, "_start_time", None)
    if start_t is not None:
        secs = int(_time.time() - start_t)
        mins, s = divmod(secs, 60)
        hrs,  m = divmod(mins, 60)
        if hrs:
            elapsed_str = f"{hrs}h {m:02d}m {s:02d}s"
        elif mins:
            elapsed_str = f"{mins}m {s:02d}s"
        else:
            elapsed_str = f"{s}s"

    # Session name
    sname = getattr(session, "name", None) or "(unnamed)"

    def _kv(key: str, val: str, val_markup: str = "") -> None:
        key_col = f"[{GRAY}]{key:<12}[/]"
        if val_markup:
            console.print(f"  {key_col}  {val_markup}")
        else:
            console.print(f"  {key_col}  [{GRAY}]{val}[/]")

    console.print()
    _kv("name",     sname,   f"[{CYAN}]{sname}[/]")
    _kv("model",    "",      f"[{LIME}]{model}[/]  [{DIM}]·[/]  [{GRAY}]{backend}[/]")
    _kv("messages", "",      f"[{LIME}]{msgs}[/]  [{DIM}]·[/]  [{GRAY}]{turns} turns[/]")

    ctx_right = f"{ctx_tokens:,} / {max_tokens:,} tokens  ({ctx_pct}%)"
    console.print(
        f"  [{GRAY}]{'context':<12}[/]  {ctx_bar}  [{bar_color}]{ctx_right}[/]"
    )

    if elapsed_str:
        _kv("elapsed",  elapsed_str, f"[{VIOLET}]{elapsed_str}[/]")
    _kv("cwd",      cwd,     f"[{CYAN}]{cwd}[/]")
    console.print()
    return True


def cmd_clear(_args: str, session, _settings) -> bool:
    import time as _t
    session.messages.clear()
    session.turn_count = 0
    session._start_time = _t.time()
    session._pending_note = ""
    try:
        import greenboost_cli.instruments.handlers as _h
        _h._bash_cwd = ""
    except Exception:
        pass
    emit_ok("Conversation cleared.")
    return True


def cmd_model(args: str, _session, settings) -> bool:
    """/model              — show current model
    /model list         — arrow-key picker over the gb-synapse manifest
    /model <name>       — switch directly
    /model <name> --remove — delete a model gb-synapse pulled (HF-sourced only;
                              Ollama-managed blobs point you at 'ollama rm')
    """
    arg = args.strip()

    if not arg:
        model = settings["model"]
        emit_info(f"Current model:  [{VIOLET}]{model or '(none — /fetch-model or /model <name>)'}[/]")
        console.print()
        emit_info("Usage: /model list       Pick a model interactively (↑/↓, Enter)")
        emit_info("       /model <name>     Switch directly")
        emit_info("       /model <name> --remove   Delete a pulled model")
        return True

    if arg.lower() == "list":
        return _cmd_model_list(_session, settings)

    parts = arg.split()
    if "--remove" in parts:
        name = " ".join(p for p in parts if p != "--remove").strip()
        return _cmd_model_remove(name, settings)

    _switch_model(arg, _session, settings)
    return True


def _switch_model(m: str, _session, settings: dict) -> None:
    """Persist the model choice and (re)start gb-synapse serving it — gb-synapse
    is the only backend, so picking a model must make it live immediately
    rather than leaving llama-server stopped until the next full CLI restart."""
    previous = settings.get("model")
    settings["model"] = m
    emit_ok(f"Model → [{VIOLET}]{m}[/]")
    from greenboost_cli.environment.settings import save_settings
    save_settings(settings)
    _warn_if_model_exceeds_vram(m)

    from greenboost_cli.slash_commands.backend_cmds import cmd_llamaserve, _llamacpp_running_pid
    action = "restart" if _llamacpp_running_pid(settings) else "start"
    cmd_llamaserve(action, _session, settings)

    # A resolution/gate failure (unknown model, capability-refused quant,
    # etc.) never gets as far as a running engine — cmd_llamaserve prints its
    # own error and returns, but until now settings["model"] was already
    # durably saved above, leaving the CLI pointed at a model that can never
    # start until the user notices and switches again. Real incident,
    # 2026-07-28: this had no rollback at all. Only reverts when the engine
    # itself never came up (not when the engine loaded but the proxy alone
    # died — that's a real running model with a front-door problem, not a
    # bad model choice, and belongs to the caller to fix, not to revert).
    if not _llamacpp_running_pid(settings):
        settings["model"] = previous
        save_settings(settings)
        if previous:
            emit_warn(f"Reverted model back to [{VIOLET}]{previous}[/] "
                       f"(the switch above failed to start).")


def _cmd_model_remove(name: str, settings: dict) -> bool:
    if not name:
        emit_err("Usage: /model <name> --remove")
        return True
    try:
        from greenboost_cli.slash_commands.backend_cmds import _import_gb_synapse
        _import_gb_synapse().rm(name)
    except KeyError:
        emit_err(f"No such model: {name}")
        return True
    except ValueError as e:
        emit_err(str(e))
        return True
    except Exception as e:
        emit_err(f"Could not remove {name}: {e}")
        return True

    emit_ok(f"Removed [{VIOLET}]{name}[/]")
    if settings.get("model") == name:
        settings["model"] = ""
        from greenboost_cli.environment.settings import save_settings
        save_settings(settings)
        emit_info("Current model cleared — pick another with /model list")
    return True


def _cmd_model_list(_session, settings: dict) -> bool:
    """Arrow-key picker over every model gb-synapse knows about (HuggingFace-
    pulled or Ollama-indexed) — all served the same way, through gb-synapse's
    llama-server."""
    try:
        from greenboost_cli.slash_commands.backend_cmds import _import_gb_synapse
        entries = _import_gb_synapse().list_models()
    except Exception as e:
        emit_err(f"Could not load gb-synapse manifest: {e}")
        return True
    if not entries:
        emit_info("No models yet — run /fetch-model huggingface  or  /fetch-model ollama")
        return True

    current = settings.get("model", "")

    def _is_current(name: str) -> bool:
        return name == current or current.endswith(f"/{name}")

    options = [
        {
            "label": e.name,
            "description": f"{e.source} · {e.quant} · {e.n_bytes / (1024 ** 3):.1f} GiB"
                            + ("  (current)" if _is_current(e.name) else ""),
        }
        for e in entries
    ]

    from greenboost_cli.terminal.wizard_prompt import run_picker
    try:
        from greenboost_cli.terminal.repl import _suspend_pt_for_wizard, _resume_pt_after_wizard
    except Exception:
        _suspend_pt_for_wizard = _resume_pt_after_wizard = None

    if _suspend_pt_for_wizard:
        _suspend_pt_for_wizard()
    try:
        chosen = run_picker(options, title="Select a model")
    finally:
        if _resume_pt_after_wizard:
            _resume_pt_after_wizard()

    if not chosen:
        emit_info("No model selected.")
        return True

    m = entries[chosen[0] - 1].name
    _switch_model(m, _session, settings)
    return True


def _warn_if_model_exceeds_vram(model: str) -> None:
    """Warn when a model's on-disk size meaningfully exceeds physical VRAM,
    since GreenBoost has to spill the excess to PCIe-speed DDR (correct but
    much slower than GDDR7) -- see workflow/known-issues.md. Best-effort:
    silently skip on any probe failure, this is informational only."""
    try:
        from greenboost_cli.greenboost.monitor import get_monitor
        from greenboost_cli.slash_commands.backend_cmds import _import_gb_synapse

        s = get_monitor().refresh()
        vram_mb = s.vram_physical_mb
        if not vram_mb:
            return

        bare_name = model.split("/", 1)[1] if model.startswith("gb-synapse/") else model
        entries = _import_gb_synapse().list_models()
        match = next((e for e in entries if e.name == bare_name), None)
        if not match or not match.n_bytes:
            return
        model_mb = match.n_bytes / (1024 * 1024)

        if model_mb <= vram_mb * 0.85:
            return

        smaller = sorted(
            (e for e in entries if e.n_bytes and e.n_bytes / (1024 * 1024) <= vram_mb * 0.85
             and e.name != bare_name),
            key=lambda e: -e.n_bytes,
        )
        emit_warn(
            f"{bare_name} is {model_mb/1024:.1f} GB on disk vs {vram_mb/1024:.1f} GB physical VRAM — "
            f"the excess will spill to PCIe-speed DDR (slower, but correct; see "
            f"workflow/known-issues.md). For full-speed VRAM-resident inference, try a smaller model."
        )
        if smaller:
            best = smaller[0]
            emit_info(f"  Already pulled and fits: {best.name} ({best.n_bytes / 1024 / 1024 / 1024:.1f} GB)")
    except Exception:
        pass


def cmd_config(args: str, _session, settings) -> bool:
    from greenboost_cli.environment.settings import save_settings
    if not args:
        display = {k: v for k, v in settings.items() if k != "api_key"}
        console.print_json(json.dumps(display, indent=2))
    elif "=" in args:
        key, _, val = args.partition("=")
        key, val = key.strip(), val.strip()
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.isdigit():
            val = int(val)
        settings[key] = val
        save_settings(settings)
        emit_ok(f"Set  {key} = {val}")
    else:
        k = args.strip()
        v = settings.get(k, "(not set)")
        emit_info(f"[{VIOLET}]{k}[/] [{GRAY}]=[/] {v}")
    return True


def cmd_save(args: str, session, _settings) -> bool:
    from greenboost_cli.environment.settings import SESSIONS_PATH
    fname = args.strip() or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path  = Path(fname) if "/" in fname else SESSIONS_PATH / fname
    data  = {
        "messages": [
            m if not isinstance(m.get("content"), list) else
            {**m, "content": [
                b if isinstance(b, dict) else b.model_dump()
                for b in m["content"]
            ]}
            for m in session.messages
        ],
        "turn_count":          session.turn_count,
        "total_input_tokens":  session.total_input_tokens,
        "total_output_tokens": session.total_output_tokens,
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    emit_ok(f"Session saved → {path}")
    return True


def cmd_load(args: str, session, _settings) -> bool:
    from greenboost_cli.environment.settings import SESSIONS_PATH
    if not args.strip():
        sessions = sorted(SESSIONS_PATH.glob("*.json"))
        if not sessions:
            emit_info("No saved sessions found.")
        else:
            emit_info("Saved sessions:")
            for s in sessions:
                console.print(f"  [{GRAY}]{s.name}[/]")
        return True
    fname = args.strip()
    path  = Path(fname) if "/" in fname else SESSIONS_PATH / fname
    if not path.exists():
        emit_err(f"File not found: {path}")
        return True
    data = json.loads(path.read_text())
    session.messages             = data.get("messages", [])
    session.turn_count           = data.get("turn_count", 0)
    session.total_input_tokens   = data.get("total_input_tokens", 0)
    session.total_output_tokens  = data.get("total_output_tokens", 0)
    emit_ok(f"Session loaded ← {path}  ({len(session.messages)} messages)")
    return True


def cmd_history(_args: str, session, _settings) -> bool:
    if not session.messages:
        emit_info("(empty conversation)")
        return True
    for i, m in enumerate(session.messages):
        role = m["role"]
        color = VIOLET if role == "user" else LIME
        content = m["content"]
        if isinstance(content, str):
            console.print(f"[{color}][{i}] {role.upper()}:[/] [{GRAY}]{content[:200]}[/]")
        elif isinstance(content, list):
            for block in content:
                btype = block.get("type", "") if isinstance(block, dict) else getattr(block, "type", "")
                if btype == "text":
                    text = block.get("text", "") if isinstance(block, dict) else block.text
                    console.print(f"[{color}][{i}] {role.upper()}:[/] [{GRAY}]{text[:200]}[/]")
                elif btype == "tool_use":
                    name = block.get("name", "") if isinstance(block, dict) else block.name
                    console.print(f"[{color}][{i}] {role.upper()}:[/] [{GRAY}][tool: {name}][/]")
                elif btype == "tool_result":
                    cval = block.get("content", "") if isinstance(block, dict) else block.content
                    console.print(f"[{color}][{i}] {role.upper()}:[/] [{GRAY}][result: {str(cval)[:100]}][/]")
    return True


def cmd_context(_args: str, session, settings) -> bool:
    # Token estimation: chars / 2.8 (code-heavy) + 4 tokens per message framing
    user_chars = sum(len(str(m.get("content", ""))) for m in session.messages if m.get("role") == "user")
    asst_chars = sum(len(str(m.get("content", ""))) for m in session.messages if m.get("role") == "assistant")
    tool_chars = sum(len(str(m.get("content", ""))) for m in session.messages if m.get("role") == "tool")
    user_msgs  = sum(1 for m in session.messages if m.get("role") == "user")
    asst_msgs  = sum(1 for m in session.messages if m.get("role") == "assistant")
    tool_msgs  = sum(1 for m in session.messages if m.get("role") == "tool")
    msg_chars  = user_chars + asst_chars + tool_chars
    est_tokens = int(msg_chars / 2.8) + len(session.messages) * 4
    max_tokens = settings.get("max_tokens", 100_000)
    pct        = min(100, int(est_tokens / max_tokens * 100)) if max_tokens else 0

    # Color ladder
    if pct >= 85:
        bar_color = AMBER
    elif pct >= 60:
        bar_color = VIOLET
    else:
        bar_color = LIME

    def _toks(chars: int) -> str:
        t = int(chars / 2.8)
        return f"{t // 1000}k" if t >= 1000 else str(t)

    # ── 20×10 cell grid (200 cells total) ─────────────────────────────────────
    GRID_W, GRID_H = 20, 10
    TOTAL_CELLS    = GRID_W * GRID_H  # 200

    filled = min(TOTAL_CELLS, int(TOTAL_CELLS * pct / 100))

    # Proportional color allocation per role
    role_cells: list[tuple[str, int]] = []
    if msg_chars > 0:
        role_cells = [
            (CYAN,  max(0, int(TOTAL_CELLS * user_chars / msg_chars))),
            (LIME,  max(0, int(TOTAL_CELLS * asst_chars / msg_chars))),
            (GRAY,  max(0, int(TOTAL_CELLS * tool_chars / msg_chars))),
        ]
    # Determine per-cell color
    cell_colors: list[str | None] = []
    boundary = 0
    color_idx = 0
    for ci in range(TOTAL_CELLS):
        if ci >= filled:
            cell_colors.append(None)
        else:
            while color_idx < len(role_cells) and ci >= boundary + role_cells[color_idx][1]:
                boundary += role_cells[color_idx][1]
                color_idx += 1
            if color_idx < len(role_cells):
                cell_colors.append(role_cells[color_idx][0])
            else:
                cell_colors.append(VIOLET)

    console.print(f"\n  [{VIOLET}]◈  Context window[/]  [{bar_color}]{pct}%[/]"
                  f"  [{GRAY}]~{est_tokens:,} / {max_tokens:,} tokens[/]")
    console.print()
    for row in range(GRID_H):
        parts = []
        for col in range(GRID_W):
            color = cell_colors[row * GRID_W + col]
            parts.append(f"[{color}]█[/]" if color else f"[{DIM}]░[/]")
        console.print("  " + "".join(parts))

    # Legend
    console.print()
    legend = []
    if user_msgs:
        legend.append(f"[{CYAN}]█[/] [{GRAY}]user {user_msgs}msg ~{_toks(user_chars)}[/]")
    if asst_msgs:
        legend.append(f"[{LIME}]█[/] [{GRAY}]asst {asst_msgs}msg ~{_toks(asst_chars)}[/]")
    if tool_msgs:
        legend.append(f"[{GRAY}]█[/] [{DIM}]tool {tool_msgs}msg ~{_toks(tool_chars)}[/]")
    if legend:
        console.print("  " + "  ".join(legend))

    console.print()
    console.print(f"  [{GRAY}]model      [{DIM}]{settings['model']}[/]")
    verbose_s = f"[{LIME}]ON[/]"  if settings.get("verbose") else f"[{DIM}]off[/]"
    quiet_s   = f"[{LIME}]ON[/]"  if settings.get("quiet")   else f"[{DIM}]off[/]"
    console.print(f"  [{GRAY}]verbose    {verbose_s}   quiet  {quiet_s}[/]")
    if pct >= 85:
        console.print(f"\n  [{AMBER}]⚠  Context is {pct}% full — /compact to compress[/]")
    console.print()
    return True


def cmd_cost(_args: str, session, settings) -> bool:
    """Show token usage for this session. gb-synapse is local — no $ cost."""
    emit_info(f"Input tokens:   [{VIOLET}]{session.total_input_tokens:,}[/]")
    emit_info(f"Output tokens:  [{VIOLET}]{session.total_output_tokens:,}[/]")
    return True


def cmd_verbose(_args: str, _session, settings) -> bool:
    settings["verbose"] = not settings.get("verbose", False)
    if settings["verbose"]:
        settings["quiet"] = False   # verbose and quiet are mutually exclusive
    state_str = "ON" if settings["verbose"] else "OFF"
    emit_ok(f"Verbose mode: {state_str}")
    return True


def cmd_quiet(_args: str, _session, settings) -> bool:
    settings["quiet"] = not settings.get("quiet", False)
    if settings["quiet"]:
        settings["verbose"] = False  # quiet takes precedence over verbose
    state_str = "ON" if settings["quiet"] else "OFF"
    emit_ok(f"Quiet mode: {state_str}  ({'tool tally + ✻ footer only' if settings['quiet'] else 'full tool cards'})")
    return True


def cmd_permissions(args: str, _session, settings) -> bool:
    from greenboost_cli.environment.settings import save_settings
    modes = ["auto", "accept-all", "manual"]
    if not args.strip():
        current = settings.get("permission_mode", "auto")
        emit_info(f"Permission mode:   [{VIOLET}]{current}[/]")
        emit_info(f"Available modes:   [{GRAY}]{', '.join(modes)}[/]")
    else:
        m = args.strip()
        if m not in modes:
            emit_err(f"Unknown mode: {m}.  Choose: {', '.join(modes)}")
        else:
            settings["permission_mode"] = m
            save_settings(settings)
            emit_ok(f"Permission mode → {m}")
    return True


def cmd_cwd(args: str, _session, _settings) -> bool:
    if not args.strip():
        emit_info(f"Working directory:  [{VIOLET}]{os.getcwd()}[/]")
    else:
        p = args.strip()
        try:
            os.chdir(p)
            emit_ok(f"Changed directory → {os.getcwd()}")
        except Exception as e:
            emit_err(str(e))
    return True


def cmd_queue(args: str, session, _settings) -> bool:
    """Manage the prompt queue: list | del N | edit N text | clear."""
    pq = getattr(session, "prompt_queue", None)
    if pq is None:
        emit_info("No prompt queue available (start the REPL first).")
        return True

    parts = args.strip().split(None, 2)
    sub   = parts[0].lower() if parts else "list"

    if sub in ("", "list"):
        items = pq.snapshot()
        if not items:
            emit_info("Queue is empty.")
        else:
            console.print(f"\n  [{VIOLET}]◈  Prompt queue  ({len(items)} item(s))[/]")
            for i, item in enumerate(items, 1):
                console.print(f"  [{GRAY}]  [{i}][/] [{VIOLET}]{item.text[:80]}[/]")
            console.print(
                f"\n  [{GRAY}]  /queue del N  ·  /queue edit N text  ·  /queue clear[/]"
            )

    elif sub == "del" and len(parts) >= 2:
        try:
            idx = int(parts[1])
            if pq.delete(idx):
                emit_ok(f"Deleted queued prompt #{idx}")
            else:
                emit_err(f"No prompt at position {idx} (queue has {len(pq)} item(s))")
        except ValueError:
            emit_err("Usage: /queue del <number>")

    elif sub == "edit" and len(parts) >= 3:
        try:
            idx      = int(parts[1])
            new_text = parts[2]
            if pq.edit(idx, new_text):
                emit_ok(f"Edited queued prompt #{idx}")
                console.print(f"  [{GRAY}]  → {new_text[:80]}[/]")
            else:
                emit_err(f"No prompt at position {idx} (queue has {len(pq)} item(s))")
        except ValueError:
            emit_err("Usage: /queue edit <number> <new text>")

    elif sub == "clear":
        count = pq.clear()
        emit_ok(f"Cleared {count} queued prompt(s)")

    else:
        emit_info("Usage: /queue [list | del N | edit N text | clear]")

    return True


def cmd_retry(args: str, session, settings) -> bool:
    """Re-run the last user message after removing it and everything after it."""
    from greenboost_cli.terminal.repl import process_query
    msgs = session.messages
    last_user_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        emit_err("No previous message to retry.")
        return True
    last_user_content = msgs[last_user_idx].get("content", "")
    if not isinstance(last_user_content, str):
        emit_err("Last user message is not plain text; cannot retry.")
        return True
    del session.messages[last_user_idx:]
    emit_info(f"Retrying: {last_user_content[:80]}")
    process_query(last_user_content, session, settings)
    return True


def cmd_undo(_args: str, session, _settings) -> bool:
    """Remove the last user + assistant exchange from conversation history."""
    msgs = session.messages
    if not msgs:
        emit_err("Nothing to undo.")
        return True
    removed = 0
    while msgs and msgs[-1].get("role") != "user":
        msgs.pop()
        removed += 1
    if msgs and msgs[-1].get("role") == "user":
        msgs.pop()
        removed += 1
    emit_ok(
        f"Undid last exchange ({removed} message(s) removed). "
        f"{len(msgs)} message(s) remain."
    )
    return True


def cmd_compact(_args: str, session, settings) -> bool:
    """Compress old conversation history to save context tokens."""
    if len(session.messages) < 10:
        emit_info(f"Nothing to compact ({len(session.messages)} messages — need at least 10).")
        return True
    try:
        from greenboost_cli.workflow.intelligence import _compress_context, _estimate_tokens
        before_msgs = len(session.messages)
        before_tok  = _estimate_tokens(session)
        _compress_context(session, settings, force=True)
        after_msgs = len(session.messages)
        after_tok  = _estimate_tokens(session)
        saved = before_tok - after_tok
        emit_ok(
            f"Compacted: {before_msgs} → {after_msgs} messages  "
            f"·  ~{saved:,} tokens freed  "
            f"·  ~{after_tok:,} tokens remain"
        )
    except Exception as e:
        emit_err(f"Compression failed: {e}")
    return True


def cmd_sessions(args: str, session, settings) -> bool:
    """List saved sessions, search, or load one.  Usage: /sessions [load|search <args>]"""
    from greenboost_cli.environment.settings import SESSIONS_PATH
    from datetime import datetime as _dt

    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else ""

    if sub == "load" and len(parts) >= 2:
        return cmd_load(parts[1], session, settings)

    if sub == "search" and len(parts) >= 2:
        query = parts[1].lower()
        found = []
        for p in SESSIONS_PATH.glob("*.json"):
            try:
                if query in p.read_text(encoding="utf-8", errors="ignore").lower():
                    found.append(p)
            except Exception:
                pass
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not found:
            emit_info(f"No sessions match '{query}'.")
            return True
        console.print(
            f"\n  [{VIOLET}]◈  Sessions matching '{query}'[/]  [{GRAY}]({len(found)} found)[/]"
        )
        for p in found[:15]:
            mtime = _dt.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            console.print(f"  [{GRAY}]{mtime}[/]  [{VIOLET}]{p.stem}[/]")
        console.print(f"\n  [{GRAY}]/sessions load <name> to open[/]")
        return True

    sessions = sorted(
        SESSIONS_PATH.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not sessions:
        emit_info("No saved sessions. Use /save [name] to save the current session.")
        return True

    console.print(
        f"\n  [{VIOLET}]◈  Saved sessions[/]  [{GRAY}]({len(sessions)} files)[/]"
    )
    for p in sessions[:20]:
        size_kb = p.stat().st_size / 1024
        mtime = _dt.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        n_msgs_str = ""
        try:
            data = json.loads(p.read_bytes())
            msgs = data.get("messages", [])
            if msgs:
                n_msgs_str = f"  [{DIM}]{len(msgs)} msg[/]"
        except Exception:
            pass
        console.print(
            f"  [{GRAY}]{mtime}[/]  [{VIOLET}]{p.stem:<30}[/]"
            f"  [{GRAY}]{size_kb:.1f} kb[/]{n_msgs_str}"
        )
    console.print(
        f"\n  [{GRAY}]/sessions load <name>  ·  /sessions search <query>  ·  /save [name][/]"
    )
    return True


def cmd_websearch(args: str, session, settings) -> bool:
    """Search the web via DuckDuckGo (Scrapling) and show formatted results."""
    query = args.strip()
    if not query:
        emit_err("Usage: /websearch <query>")
        return True
    try:
        from greenboost_cli.instruments.handlers import handle_web_query
        results = handle_web_query(query)
        console.print(f"\n  [{VIOLET}]◈  Web search:[/] [{GRAY}]{query}[/]\n")
        console.print(results)
        console.print()
    except Exception as e:
        emit_err(str(e))
    return True


def cmd_note(args: str, session, _settings) -> bool:
    """Inject a plain-text note into the conversation as a user message.

    Useful for adding context mid-session without triggering a model response.
    Usage: /note <text>   — adds a [Note] message; model will see it next turn.
    """
    text = args.strip()
    if not text:
        emit_err("Usage: /note <text>")
        return True
    # Store as a pending prefix so it's prepended to the NEXT user message rather
    # than adding a standalone user turn (which breaks conversation structure) or
    # a fake assistant reply (which pollutes /compact summaries and token counts).
    sep = "\n" if session._pending_note else ""
    session._pending_note += sep + f"[Note]: {text}"
    emit_ok(f"Note queued — will be prepended to your next message: {text[:80]}")
    return True


def cmd_image(args: str, session, _settings) -> bool:
    """Attach an image file to the next message (vision models only).

    Usage: /image <path> [caption]
    The image is base64-encoded and added as a pending attachment.
    gb-synapse is OpenAI-compatible, so this works with any vision-capable
    GGUF it serves (Qwen-VL and similar).
    """
    parts = args.strip().split(None, 1)
    if not parts:
        emit_err("Usage: /image <path> [caption]")
        return True
    img_path = Path(parts[0])
    caption  = parts[1] if len(parts) > 1 else ""
    if not img_path.exists():
        emit_err(f"File not found: {img_path}")
        return True
    try:
        import base64, mimetypes
        mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
        data = base64.standard_b64encode(img_path.read_bytes()).decode()
        content_block = {
            "type":       "image_url",
            "image_url":  {"url": f"data:{mime};base64,{data}"},
        }
        session.pending_attachments.append(content_block)
        emit_ok(
            f"Image attached: {img_path.name}  ({mime})  "
            f"— it will be included in your next message."
        )
        if caption:
            emit_info(f"Caption: {caption}")
    except Exception as e:
        emit_err(f"Failed to attach image: {e}")
    return True


def cmd_name(args: str, session, _settings) -> bool:
    """Set or show the session name (like Claude Code's session label)."""
    name = args.strip()
    if not name:
        current = getattr(session, "name", None)
        if current:
            emit_info(f"Session name: {current}")
        else:
            emit_info("No session name set. Usage: /name <name>")
        return True
    session.name = name
    emit_ok(f"Session name: {name}")
    return True


def cmd_fork(args: str, session, settings) -> bool:
    """Snapshot current session under a new name (checkpoint, not a true branch).

    Usage: /fork [name]
    Renames the session and resets the start timer. All messages are preserved.
    To actually diverge: /clear after /fork to start a new thread.
    """
    import time
    fork_name = args.strip() or f"snap-{len(session.messages)//2}msgs"
    session.name = fork_name
    session._start_time = time.time()
    emit_ok(f"Snapshot → {fork_name}  ({len(session.messages)} messages preserved)")
    emit_info("All history kept. Use /clear to diverge from this point.")
    return True


def cmd_todo(_args: str, _session, _settings) -> bool:
    """Print the session task list written by TodoWrite."""
    from greenboost_cli.instruments.handlers import _session_todos
    if not _session_todos:
        emit_info("No tasks in this session. The model uses TodoWrite to track work.")
        return True
    status_icons = {"pending": "○", "in_progress": "◐", "completed": "✓"}
    pri_colors   = {"high": AMBER, "medium": CYAN, "low": DIM}
    console.print(f"\n  [{VIOLET}]◈  Session Tasks[/]  [{DIM}]({len(_session_todos)})[/]")
    for t in _session_todos:
        st   = t.get("status", "pending")
        pri  = t.get("priority", "medium")
        icon = status_icons.get(st, "○")
        pcol = pri_colors.get(pri, DIM)
        col  = DIM if st == "completed" else GRAY
        console.print(
            f"  [{LIME if st=='completed' else AMBER if st=='in_progress' else DIM}]{icon}[/]"
            f"  [{pcol}]{pri[:3]}[/]  [{col}]{t.get('content','')[:80]}[/]"
        )
    console.print()
    return True


def cmd_exit(_args: str, _session, _settings) -> bool:
    from greenboost_cli.terminal.repl import request_shutdown
    emit_ok("Goodbye.")
    request_shutdown()
    sys.exit(0)


def cmd_setup(_args: str, _session, settings) -> bool:
    from greenboost_cli.wizard.setup import run_wizard
    from greenboost_cli.environment.settings import save_settings
    run_wizard(settings, force=True)
    save_settings(settings)
    return True


# ── Command table + dispatcher ─────────────────────────────────────────────

COMMAND_TABLE: dict = {
    "help":            cmd_help,
    "status":          cmd_status,
    "setup":           cmd_setup,
    "clear":           cmd_clear,
    "model":           cmd_model,
    "config":          cmd_config,
    "save":            cmd_save,
    "load":            cmd_load,
    "history":         cmd_history,
    "context":         cmd_context,
    "cost":            cmd_cost,
    "verbose":         cmd_verbose,
    "quiet":           cmd_quiet,
    "permissions":     cmd_permissions,
    "cwd":             cmd_cwd,
    "exit":            cmd_exit,
    "quit":            cmd_exit,
    "queue":           cmd_queue,
    "retry":           cmd_retry,
    "undo":            cmd_undo,
    "compact":         cmd_compact,
    "sessions":        cmd_sessions,
    "websearch":       cmd_websearch,
    "note":            cmd_note,
    "image":           cmd_image,
    "name":            cmd_name,
    "fork":            cmd_fork,
    "todo":            cmd_todo,
    # populated by backend_cmds, download_cmds, and greenboost_cmds at import time
}

# Register GreenBoost commands
from greenboost_cli.slash_commands import greenboost_cmds as _gb_cmds  # noqa: E402
_gb_cmds.register(COMMAND_TABLE)


def register_command(name: str, handler, description: str = "") -> None:
    """Register a slash command at runtime. Used by slash_commands/* modules."""
    COMMAND_TABLE[name] = handler


def dispatch_command(line: str, session, settings) -> bool:
    """Parse /command [args] and call the appropriate handler.

    Returns True if the line was a slash command (handled or unknown).
    Returns False if the line is not a slash command.
    """
    if not line.startswith("/"):
        return False
    parts = line[1:].split(None, 1)
    if not parts:
        return False
    cmd  = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    handler = COMMAND_TABLE.get(cmd)
    if handler:
        handler(args, session, settings)
    else:
        emit_err(f"Unknown command: /{cmd}  (type /help for commands)")
    return True
