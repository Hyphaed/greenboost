"""
Interactive Read-Eval-Print Loop for GreenBoost CLI.

Architecture
------------
  stdin thread   — always-running daemon; reads one line at a time via
                   readline (preserving history / tab-completion) and
                   puts ("line", text) | ("eof", None) into _stdin_q.

  model worker   — background daemon thread per query; calls _run_once()
                   then drains the PromptQueue before clearing _is_processing.

  main thread    — event loop: routes stdin events to the model worker or
                   to the PromptQueue when the worker is busy.
"""
from __future__ import annotations

import sys
import os
import queue
import readline
import atexit
import shutil
import threading
import traceback
from pathlib import Path

from greenboost_cli.terminal.theme import (
    console, VIOLET, GRAY, AMBER, LIME, TEAL, DIM, LAVENDER, CORAL,
    BOX_H, BOX_TL, BOX_TR, BOX_BL, BOX_BR, BOX_V,
    emit_ok, emit_warn, emit_err,
    CTX_WARN_PCT, CTX_AMBER_PCT,
)

# ── GB memory stats — background refresh for toolbar right corner ─────────────
# Updated every 5 s by _gb_stats_updater (daemon thread). The toolbar reads
# this without blocking.
#
# _gb_stats_segs: list of (plain_text, pt_style_str) pairs, one per tier.
#   T1 = ("T1 x/yG", "fg:<teal>")
#   T2 = ("T2 x/yG", "fg:<lavender>")  — only when GreenBoost loaded + T2 pool active
#   T3 = ("T3 x/yG", "fg:<coral>")     — only when T3 pool active
# A tier's style dims (theme.DIM) when the underlying snapshot is stale,
# instead of silently freezing the last good render forever.
_gb_stats_segs: list[tuple[str, str]] = []
_gb_stats_lock: threading.Lock        = threading.Lock()

# T3 pool total (MB) rarely changes at runtime — the dataflux snapshot event
# doesn't carry it (only t3_used_mb), so cache it once from gb_monitor
# instead of re-probing every 5s tick.
_gb_t3_total_mb_cache: float = 0.0

_GB_STATS_STALE_S = 30.0   # matches gb_monitor's own shim-stats staleness gate


def _read_gb_snapshot() -> "tuple[dict | None, bool]":
    """(snapshot_dict, is_stale). snapshot_dict is None if nothing usable was
    found anywhere.

    Per the owner's standing dataflux rule, live GreenBoost state should
    come from the SAME dataflux SnapshotRecorder the greenboost-dataflux/
    orchestrator MCP servers already write every 5s
    (gb_dataflux.py:SnapshotRecorder), not re-derived less reliably.
    Previously this toolbar used its own broken ioctl struct (T2 showed
    *allocated*, not *available* — genuinely 0 but uninformative — and T3
    was hardcoded to 0 by a regex matching a string the kernel module never
    prints; both fixed in greenboost/monitor.py, but dataflux is still the
    better live source since it already aggregates local + feeder state).

    Tail-reads the last ~64KB of the log and scans backwards for the most
    recent "snapshot" event — no subprocess, no torch (gb_dataflux itself is
    stdlib-only for this path). Falls back to gb_monitor.tier_stats()
    (still torch-free — do NOT use gb_tiering here, it imports torch, ~1.3s)
    only when the log is stale or absent, normalized into the same key
    shape used by the primary path below.
    """
    import time as _t
    import json as _json
    try:
        from greenboost_cli.gb_paths import gb_module
        gb_dataflux = gb_module("gb_dataflux")
        log_path = gb_dataflux._log_path()
        if log_path.exists():
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 65536))
                chunk = f.read().decode("utf-8", errors="ignore")
            for line in reversed(chunk.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = _json.loads(line)
                except ValueError:
                    continue
                if ev.get("kind") == "snapshot":
                    age = _t.time() - float(ev.get("ts", 0))
                    return ev, age > _GB_STATS_STALE_S
    except Exception:
        pass
    # Fallback: log stale/absent (e.g. neither MCP server is running, so
    # nothing is writing snapshots) — gb_monitor.tier_stats() directly.
    try:
        from greenboost_cli.gb_paths import gb_module
        gb_monitor = gb_module("gb_monitor")
        ts = gb_monitor.tier_stats()
        if ts:
            return {
                "fb_used_mb":       ts.get("t1_used_mb"),
                "fb_total_mb":      ts.get("t1_vram_mb"),
                "t2_allocated_mb":  ts.get("t2_allocated_mb"),
                "t2_available_mb":  ts.get("t2_available_mb"),
                "t3_used_mb":       ts.get("t3_swap_used_mb"),
                "t3_total_mb":      ts.get("t3_swap_total_mb"),
            }, False
    except Exception:
        pass
    return None, True


def _gb_stats_updater() -> None:
    """Daemon: poll GreenBoost tier occupancy (dataflux-first) every 5 s."""
    import time as _t
    global _gb_stats_segs, _gb_t3_total_mb_cache
    while True:
        try:
            segs: list[tuple[str, str]] = []
            snap, stale = _read_gb_snapshot()
            if snap:
                dim = f"fg:{DIM}"

                t1_used  = snap.get("fb_phys_used_mb")  or snap.get("fb_used_mb")
                t1_total = snap.get("fb_phys_total_mb") or snap.get("fb_total_mb")
                if t1_used is not None and t1_total:
                    used_g, total_g = round(t1_used / 1024, 1), round(t1_total / 1024, 1)
                    segs.append((f"T1 {used_g}/{total_g}G", dim if stale else f"fg:{TEAL}"))

                t2_alloc = snap.get("t2_allocated_mb")
                t2_avail = snap.get("t2_available_mb")
                if t2_alloc is not None and t2_avail is not None:
                    used_g  = round(t2_alloc / 1024, 1)
                    total_g = round((t2_alloc + t2_avail) / 1024, 1)
                    segs.append((f"T2 {used_g}/{total_g}G", dim if stale else f"fg:{LAVENDER}"))

                t3_used = snap.get("t3_used_mb")
                if t3_used is not None:
                    t3_total = snap.get("t3_total_mb")
                    if t3_total:
                        _gb_t3_total_mb_cache = t3_total
                    elif not _gb_t3_total_mb_cache:
                        try:
                            from greenboost_cli.gb_paths import gb_module
                            _gb_t3_total_mb_cache = gb_module("gb_monitor").snapshot(
                                probe_gpu=False).t3_total_mb
                        except Exception:
                            pass
                    if _gb_t3_total_mb_cache:
                        used_g  = round(t3_used / 1024, 1)
                        total_g = round(_gb_t3_total_mb_cache / 1024, 1)
                        segs.append((f"T3 {used_g}/{total_g}G", dim if stale else f"fg:{CORAL}"))

            if segs:
                with _gb_stats_lock:
                    _gb_stats_segs = segs
        except Exception:
            pass
        _t.sleep(5)


# ── prompt_toolkit (optional, graceful fallback to readline) ──────────────────
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.application import run_in_terminal
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.styles import Style
    from prompt_toolkit.patch_stdout import patch_stdout as _pt_patch_stdout
    _PT_AVAILABLE = True
except ImportError:
    _PT_AVAILABLE = False
    _pt_patch_stdout = None
from greenboost_cli.terminal.renderer import (
    emit_text_fragment, emit_reasoning, finalize_response,
    show_instrument_start, show_instrument_result,
    open_response_block, close_response_block,
    prompt_approval, show_session_banner, halt_tool_spinner,
)
from greenboost_cli.terminal.theme import ANSI_GRAY, ANSI_VIOLET, ANSI_AMBER, ANSI_RESET, ANSI_TEAL
from greenboost_cli.terminal.prompt_queue import PromptQueue
from greenboost_cli.core.session import ConversationSession
from greenboost_cli.core.orchestrator import (
    execute_turn,
    InstrumentInvoked, InstrumentResult, ApprovalNeeded, QuestionsAsked,
    TurnComplete, LoopGuardTriggered,
)
from greenboost_cli.inference.router import StreamFragment, ReasoningFragment
from greenboost_cli.environment.settings import GB_HOME, SESSIONS_PATH

# ── prompt_toolkit / wizard hand-off ────────────────────────────────────────────
# The persistent bottom-box _pt_session.prompt() call (running on the stdin-reader
# thread, waiting for the *next* line) stays alive throughout a model turn so the
# box remains visible and queueable. But wizards (approval picker, AskUserQuestion)
# read the raw terminal directly (termios) on the model-worker thread — if both
# read the same stdin fd at once, prompt_toolkit's own terminal queries (e.g.
# cursor-position reports) get misread by the wizard as a stray Escape key,
# auto-cancelling the picker, and the two renderers tear the screen fighting over
# it. set_pt_suspend_hook() lets run_interactive register a function that forces
# the live prompt() call to exit (releasing the fd) before any wizard runs.
_pt_suspend_hook = None   # type: ignore  # callable() -> None, set by run_interactive

# Set by request_shutdown() (e.g. /exit, /quit) so the stdin-reader daemon
# stops looping back into prompt() instead of re-rendering the bottom box
# after the live app has already been torn down.
_shutdown_evt = threading.Event()


def set_pt_suspend_hook(hook) -> None:
    global _pt_suspend_hook
    _pt_suspend_hook = hook


def request_shutdown() -> None:
    """Signal the stdin reader to stop and force the live prompt_toolkit app
    to exit so it restores the terminal before the process exits."""
    _shutdown_evt.set()
    _suspend_pt_for_wizard()   # app.exit(result=None) -> prompt() returns None


def _suspend_pt_for_wizard() -> None:
    """Force the live prompt_toolkit input prompt to release stdin before a
    wizard (approval picker / AskUserQuestion) takes over the raw terminal.

    Also marks the statusline as "wizard active" so its background repaint
    loop stops calling app.invalidate() — without this, a still-ticking
    invalidate can race the wizard's raw \\033[nA redraw and produce a
    doubled/garbled option list. Callers must pair this with
    _resume_pt_after_wizard() once the wizard returns."""
    from greenboost_cli.terminal.statusline import set_wizard_active
    set_wizard_active(True)
    if _pt_suspend_hook is not None:
        _pt_suspend_hook()


def _resume_pt_after_wizard() -> None:
    """Companion to _suspend_pt_for_wizard() — call once the wizard returns."""
    from greenboost_cli.terminal.statusline import set_wizard_active
    set_wizard_active(False)


# ── Readline setup ─────────────────────────────────────────────────────────────

_CMD_META: dict[str, list[str]] = {
    "gb-quant":       ["--list", "qwen3.6:latest", "qwen3", "llama3.1", "phi4", "fp8", "int8", "int4", "tq3"],
    "fetch-model":    ["huggingface", "ollama"],
    "download-models": ["huggingface", "ollama"],
    "model":          ["list"],
    "factory":        ["start", "stop", "status", "submit", "agents", "history"],
    "plan":           [],
    "plan-edit":      [],
    "plan-approve":   [],
    "plan-exit":      [],
    "gb-vitals":      ["--export"],
    "mcp":            ["list", "enable", "disable"],
    "sessions":       ["load", "search"],
    "apply-diff":     ["--dry-run"],
    "rag-add":        ["--recursive", "--ext"],
    "rag-search":     [],
    "rag-status":     [],
    "rag-clear":      ["--confirm"],
    "rag-inject":     [],
    "design":         ["--output"],
    "design-gen":     ["--model", "--style"],
    "commit":         [],
    "git-review":     [],
    "git-pr":         [],
    "pdf2md":         [],
    "convert":        [],
    "agent":          [],
    "skill-show":     [],
    "skill-set-dir":  [],
    "task-add":       [],
    "task-update":    [],
    "task-delete":    [],
    "turboquant":     ["--bits"],
    "llamaserve":     ["start", "stop", "status", "logs", "restart"],
    "llamacache":     ["status", "save", "restore", "erase"],
    "dashboard":      [],
    "verbose":        [],
    "quiet":          [],
    "init":           ["--force"],
    "resume":         [],
    "permissions":    ["auto", "accept-all", "manual"],
    "todo":           [],
    "add":            [],
    "compact":        [],
    "retry":          [],
    "undo":           [],
    "save":           [],
    "save-session":   ["show", "clear"],
    "snapshot":       ["show", "set"],
    "goals":          ["add", "remove", "list"],
    "history-show":   [],
    "history-add":    ["--category"],
    "history-search": [],
    "project":        ["show", "switch", "list"],
    "tokens":         ["show", "reset"],
    "skill-list":     [],
}


class _GBCompleter(Completer):
    """Two-level slash-command completer for prompt_toolkit."""

    def __init__(self, cmd_meta: dict, command_table_fn) -> None:
        self._cmd_meta = cmd_meta
        self._command_table_fn = command_table_fn

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        rest = text[1:]
        if " " not in rest:
            # Level 1: complete command name
            word = rest
            for cmd in self._command_table_fn():
                if cmd.startswith(word):
                    yield Completion(
                        cmd, start_position=-len(word), display=f"/{cmd}"
                    )
        else:
            # Level 2: complete sub-args for the command
            parts = rest.split(" ", 1)
            cmd = parts[0]
            sub = parts[1] if len(parts) > 1 else ""
            for opt in self._cmd_meta.get(cmd, []):
                if opt.startswith(sub):
                    yield Completion(opt, start_position=-len(sub))


def configure_input_history(history_file: Path) -> None:
    """Set up readline history and two-level tab-completion for slash commands."""
    from greenboost_cli.terminal.commands import COMMAND_TABLE

    try:
        readline.read_history_file(str(history_file))
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, str(history_file))

    def _completer(text: str, state: int):
        try:
            import readline as _rl
            line = _rl.get_line_buffer()
            parts = line.split(None, 1)
            cmd = parts[0].lstrip("/") if parts else ""
            arg_partial = parts[1] if len(parts) > 1 else ""

            # ── Level 2: completing arguments after a known slash command ────
            if len(parts) >= 1 and (len(parts) > 1 or (line.endswith(" ") and parts)):
                # /model → complete model names from registry
                if cmd == "model":
                    from greenboost_cli.inference.registry import BACKEND_REGISTRY
                    completions = []
                    for backend, data in BACKEND_REGISTRY.items():
                        for m in data.get("models", []):
                            full = f"{backend}/{m}" if "/" not in m else m
                            if full.startswith(arg_partial) or m.startswith(arg_partial):
                                completions.append(full)
                    matches = [c for c in completions if c.startswith(arg_partial)]
                    return matches[state] if state < len(matches) else None

                # Generic subcommand/arg completion from _CMD_META
                if cmd in _CMD_META and _CMD_META[cmd]:
                    subs = _CMD_META[cmd]
                    matches = [s for s in subs if s.startswith(arg_partial)]
                    return matches[state] if state < len(matches) else None

            # ── Level 1: completing a slash command name ──────────────────────
            slash_completions = [f"/{c}" for c in COMMAND_TABLE]
            matches = [c for c in slash_completions if c.startswith(text)]
            return matches[state] if state < len(matches) else None

        except Exception:
            return None

    readline.set_completer(_completer)
    readline.parse_and_bind("tab: complete")


# ── GreenBoost tier label ──────────────────────────────────────────────────────

def _get_gb_tier_label() -> str:
    try:
        from greenboost_cli.greenboost.monitor import get_statusline_label
        return get_statusline_label()
    except Exception:
        return ""


# ── Core turn processor ────────────────────────────────────────────────────────

def process_query(user_input: str, session: ConversationSession, settings: dict) -> None:
    """Send one user message through the agent loop and render all events."""
    from greenboost_cli.terminal.statusline import StatusLine
    verbose = settings.get("verbose", False)
    from greenboost_cli.environment.context_builder import assemble_system_context
    system_context = assemble_system_context(settings.get("model", ""))

    # Capture what the user actually typed BEFORE pre_process_query mutates
    # user_input (RAG injection, /note prepend, attachment merge, ...) — the
    # echo band below must show the literal prompt, never internal context
    # that got silently prepended onto it.
    _echo_text        = user_input if isinstance(user_input, str) else ""
    _echo_attachments = len(session.pending_attachments)

    try:
        from greenboost_cli.workflow.intelligence import pre_process_query
        user_input, system_context = pre_process_query(
            user_input, session, settings, system_context
        )
    except ImportError:
        pass

    # UserPromptSubmit hooks — can block the turn
    try:
        from greenboost_cli.instruments.hooks import run_user_prompt_hooks
        _raw = user_input if isinstance(user_input, str) else str(user_input)
        _ok, _reason = run_user_prompt_hooks(_raw)
        if not _ok:
            from greenboost_cli.terminal.theme import emit_warn
            emit_warn(f"Turn blocked by hook: {_reason}")
            return
    except Exception:
        pass

    # Determine effective context window
    # (context_window is separate from max_tokens which is the output token limit)
    def _effective_context_window() -> int:
        explicit = int(settings.get("context_window", 0))
        if explicit > 0:
            return explicit
        from greenboost_cli.environment.settings import gb_synapse_ctx
        return gb_synapse_ctx(settings)

    _ctx_max = _effective_context_window()

    def _compute_ctx() -> tuple[int, float]:
        chars = sum(len(str(m.get("content", ""))) for m in session.messages)
        est   = chars // 4
        return est, est / max(1, _ctx_max)

    _ctx_est, _ctx_pct = _compute_ctx()

    # Auto-compact before sending when context approaches the limit.
    # Thresholds (fraction of context_window):
    #   soft (0.75 = 48K of 64K): compact oldest history → structured memory
    #   hard (0.875 = 56K of 64K): aggressive forced compact
    _soft_pct = float(settings.get("auto_compact_pct",      0.75))
    _hard_pct = float(settings.get("auto_compact_hard_pct", 0.875))
    if _ctx_pct >= _soft_pct:
        try:
            from greenboost_cli.workflow.intelligence import _compress_context
            n_before = len(session.messages)
            _compress_context(session, settings, force=(_ctx_pct >= _hard_pct))
            _ctx_est, _ctx_pct = _compute_ctx()
            n_after = len(session.messages)
            try:
                from greenboost_cli.terminal.renderer import show_compact_progress
                session_elapsed = getattr(session, "_start_time", None)
                import time as _t
                elapsed_s = int(_t.time() - session_elapsed) if session_elapsed else 0
                show_compact_progress(n_before, n_after, _ctx_est, elapsed_s)
            except Exception:
                emit_ok(
                    f"Auto-compacted — context now ~{int(_ctx_pct * 100)}%"
                    f" ({_ctx_est:,} / {_ctx_max:,} tokens)"
                )
        except Exception:
            pass

    if _ctx_pct >= CTX_WARN_PCT:
        emit_warn(
            f"Context {int(_ctx_pct * 100)}% full "
            f"(~{_ctx_est:,} / {_ctx_max:,} tokens) — /compact to free space"
        )

    # Prepend any pending /note text to the message (avoids fake conversation turns)
    pending_note = session._pending_note
    if pending_note:
        if isinstance(user_input, str):
            user_input = f"{pending_note}\n\n{user_input}"
        session._pending_note = ""

    # Attach any pending images (from /image command) to this message
    pending = session.pending_attachments
    if pending:
        text_part = user_input if isinstance(user_input, str) else ""
        if isinstance(user_input, list):
            user_input_content = list(user_input) + pending
        else:
            user_input_content = [{"type": "text", "text": text_part}] + pending
        session.pending_attachments = []
        user_input = user_input_content  # type: ignore[assignment]

    from greenboost_cli.terminal.renderer import echo_user_message
    echo_user_message(_echo_text, _echo_attachments)

    # Snapshot message count so we can roll back on cancel
    _msg_snapshot = len(session.messages)
    _cancel = settings.get("_cancel_event")

    # Show any skills auto-loaded for this turn
    _loaded_skills = settings.pop("_loaded_skills", []) or []
    if _loaded_skills:
        from greenboost_cli.terminal.theme import emit_info
        emit_info(f"↬ Skills: {', '.join(_loaded_skills)}")

    open_response_block(
        model=settings.get("model", ""),
        quiet=settings.get("quiet", False),
        loaded_skills=_loaded_skills,
    )

    sl = StatusLine()
    sl.start("Thinking")
    sl.update(ctx_pct=_ctx_pct)
    gb_label = _get_gb_tier_label()
    if gb_label:
        sl.update(gb_tier=gb_label)
    _sl_running = True

    thinking_started  = False
    reasoning_started = False   # tracks if we've entered a <think> block
    last_in_tok = last_out_tok = 0
    last_tok_s = 0.0
    _interrupted = False

    def _stop_sl(cancel: bool = False) -> None:
        nonlocal _sl_running
        if _sl_running:
            if cancel:
                sl.cancel()
            else:
                sl.stop()
            _sl_running = False

    def _restart_sl(phase: str = "Thinking") -> None:
        nonlocal _sl_running
        if not _sl_running:
            sl.start(phase)
            _sl_running = True

    try:
        for event in execute_turn(user_input, session, settings, system_context):
            # Check for cancel signal from Ctrl-C (set by the stdin thread)
            if _cancel and _cancel.is_set():
                _interrupted = True
                _stop_sl(cancel=True)
                break

            if isinstance(event, StreamFragment):
                if reasoning_started and _sl_running:
                    # Transition from Reasoning → Responding
                    sl.update(phase="Responding")
                elif _sl_running:
                    # Reset phase after auto_compact (shows "Compacting" briefly)
                    sl.update(phase="Thinking")
                _stop_sl()
                emit_text_fragment(event.text)

            elif isinstance(event, ReasoningFragment):
                if not reasoning_started:
                    reasoning_started = True
                    sl.update(phase="Reasoning")   # update spinner label
                if not thinking_started:
                    _stop_sl()
                    from greenboost_cli.terminal.theme import DIM, BOX_H, BOX_ML
                    console.print(f"\n  [{DIM}]{BOX_ML}{BOX_H} thinking {'─' * 46}[/]")
                    thinking_started = True
                emit_reasoning(event.text, verbose)

            elif isinstance(event, InstrumentInvoked):
                _stop_sl(cancel=True)
                finalize_response()
                show_instrument_start(event.name, event.inputs, verbose)

            elif isinstance(event, ApprovalNeeded):
                _stop_sl()
                finalize_response()
                halt_tool_spinner()
                _suspend_pt_for_wizard()
                try:
                    event.granted = prompt_approval(event.description, settings)
                finally:
                    _resume_pt_after_wizard()

            elif isinstance(event, QuestionsAsked):
                _stop_sl()
                finalize_response()
                halt_tool_spinner()
                _suspend_pt_for_wizard()
                from greenboost_cli.terminal.wizard_prompt import run_question_wizard
                try:
                    event.answers = run_question_wizard(event.questions)
                finally:
                    _resume_pt_after_wizard()
                _restart_sl("Processing")

            elif isinstance(event, InstrumentResult):
                show_instrument_result(event.name, event.result, verbose)
                _restart_sl("Processing")

            elif isinstance(event, TurnComplete):
                last_in_tok  = event.input_tokens
                last_out_tok = event.output_tokens
                last_tok_s   = event.tok_s
                sl.update(in_tokens=last_in_tok, out_tokens=last_out_tok)
                _record_turn_tokens(last_in_tok + last_out_tok, settings,
                                    tok_s=event.tok_s if event.is_final else 0.0)
                if event.is_final and event.tok_s > 0:
                    _record_measured_tok_s(event.tok_s, settings)

            elif isinstance(event, LoopGuardTriggered):
                if event.reason == "auto_compact":
                    # Non-fatal: context was full; orchestrator compacted history
                    # and is retrying. Keep the spinner running — update its phase
                    # so the user sees "Compacting" briefly, then "Thinking" resumes
                    # when the first token of the retry arrives.
                    sl.update(phase="Compacting")
                else:
                    _stop_sl(cancel=True)
                    finalize_response()
                    console.print(
                        f"\n  [{AMBER}]⚠  Loop guard ({event.reason}):[/]"
                        f"  [{GRAY}]{event.message}[/]\n"
                    )
    finally:
        _stop_sl()

    if _interrupted:
        # Roll session back to pre-query state — keeps context clean for next turn
        session.messages = session.messages[:_msg_snapshot]
        console.print(f"\n  [{AMBER}]◈[/]  [{GRAY}](cancelled — session state restored)[/]")
        return

    close_response_block(verbose, last_in_tok, last_out_tok, tok_s=last_tok_s)

    # Post-output prose question interceptor: if the model asked a question as
    # plain text instead of using AskUserQuestion, catch it and show a wizard.
    if not _interrupted:
        _intercept_prose_question(session, settings)


def _extract_last_question(text: str) -> str | None:
    """Return the last sentence ending with '?' if it looks like a direct question."""
    import re
    stripped = text.strip()
    if not stripped.endswith("?"):
        return None
    # Split on sentence-ending punctuation, grab last question
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    for part in reversed(parts):
        q = part.strip()
        if q.endswith("?") and len(q) > 10:
            return q
    return None


def _intercept_prose_question(session, settings: dict) -> None:
    """Detect trailing prose question and show inline wizard for quick reply."""
    if not session.messages:
        return
    last = session.messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return

    content = last.get("content", "")
    text = ""
    if isinstance(content, list):
        text = " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    elif isinstance(content, str):
        text = content

    question = _extract_last_question(text)
    if not question:
        return

    # Show a minimal wizard for the prose question
    try:
        from greenboost_cli.terminal.wizard_prompt import run_question_wizard
        from greenboost_cli.terminal.theme import console, AMBER, GRAY, DIM
        console.print(
            f"\n  [{AMBER}]⚑[/]  [{GRAY}]Model asked a question — showing wizard[/]"
        )
        _suspend_pt_for_wizard()
        try:
            answers = run_question_wizard([{
                "question": question,
                "header": "Respond",
                "multiSelect": False,
                "options": [
                    {"label": "Yes", "description": "Proceed as suggested"},
                    {"label": "No",  "description": "Skip / keep current approach"},
                ],
            }])
        finally:
            _resume_pt_after_wizard()
        if answers:
            picked = answers[0].get("answers", [])
            if picked:
                reply_text = picked[0] if picked[0] not in ("Yes", "No") else picked[0]
                # Feed the answer as the next user turn immediately
                process_query(reply_text, session, settings)
    except Exception:
        pass


def _maybe_feed_rag(user_input: str | list, session, settings: dict) -> None:
    """Extract last assistant reply and feed Q&A pair into RAG in a daemon thread."""
    import threading

    if not session.messages:
        return
    last = session.messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return

    content = last.get("content", "")
    if isinstance(content, list):
        reply = " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    elif isinstance(content, str):
        reply = content
    else:
        return

    user_text = (
        user_input if isinstance(user_input, str)
        else " ".join(
            p.get("text", "") for p in user_input
            if isinstance(p, dict) and p.get("type") == "text"
        )
    )

    project = settings.get("active_project")

    def _feed() -> None:
        try:
            from greenboost_cli.rag.engine import feed_qa_turn
            feed_qa_turn(user_text, reply, project)
        except Exception:
            pass

    threading.Thread(target=_feed, daemon=True, name="gb-rag-feed").start()


def _record_turn_tokens(total_tokens: int, settings: dict, tok_s: float = 0.0) -> None:
    # gb-synapse is always local — every turn is a local_tokens record now.
    try:
        from greenboost_cli.memory.brain import project_dir
        from greenboost_cli.memory.token_tracker import record
        pdir = project_dir(settings.get("active_project"))
        record(pdir, api_tokens=0, local_tokens=total_tokens, tok_s=tok_s)
    except Exception:
        pass


def _record_measured_tok_s(tok_s: float, settings: dict) -> None:
    """Feed this turn's real decode speed back into gb-synapse so
    recommend()'s fit/throughput estimates use measured data instead of the
    bandwidth heuristic once enough turns have run for this model — see
    gb_synapse.record_measured_tok_s()."""
    try:
        from greenboost_cli.slash_commands.backend_cmds import _import_gb_synapse, _llamaserve_model_name
        model = _llamaserve_model_name(settings)
        if model:
            _import_gb_synapse().record_measured_tok_s(model, tok_s)
    except Exception:
        pass


# ── REPL ───────────────────────────────────────────────────────────────────────

def run_interactive(settings: dict, initial_prompt: str = None) -> None:
    """
    Launch the interactive REPL.

    If initial_prompt is provided, run it once and exit (non-interactive mode).

    Threading model (interactive mode only)
    ----------------------------------------
    • _stdin_thread  reads from stdin via readline; puts events into _stdin_q.
                     It waits for _is_processing to be clear before showing the
                     prompt so the prompt doesn't interleave with model output.
    • _model_worker  runs _run_once() in a loop, draining the PromptQueue before
                     it clears _is_processing, then puts ("model_done", None) into
                     _stdin_q so the main loop can tick.
    • main thread    dequeues events from _stdin_q and routes them.
    """
    # ── Lazy slash-command imports ─────────────────────────────────────────
    import greenboost_cli.slash_commands.backend_cmds   # noqa: F401
    import greenboost_cli.slash_commands.download_cmds  # noqa: F401
    import greenboost_cli.slash_commands.greenboost_cmds  # noqa: F401

    for _mod in (
        "memory_cmds", "rag_cmds", "pdf_cmds", "diff_cmds",
        "design_cmds", "dashboard_cmds",
        "guidelines_cmds", "convert_cmds", "mcp_cmds",
        "plan_cmds", "agent_cmds", "task_cmds", "skill_cmds",
        "git_cmds", "quant_cmds", "doctor_cmds", "autonomous_cmds",
        "forge_cmds", "init_cmds",
    ):
        try:
            __import__(f"greenboost_cli.slash_commands.{_mod}")
        except ImportError:
            pass

    try:
        from greenboost_cli.slash_commands.factory_cmds import register as _reg_factory
        _reg_factory(settings)
    except ImportError:
        pass

    from greenboost_cli.terminal.commands import dispatch_command, COMMAND_TABLE
    from greenboost_cli.environment.settings import HISTORY_PATH

    # Use readline only as a fallback when prompt_toolkit is unavailable
    if not _PT_AVAILABLE:
        configure_input_history(HISTORY_PATH)
    session = ConversationSession()
    import time as _time
    session._start_time = _time.time()

    # ── Prompt queue (shared between threads) ──────────────────────────────
    prompt_queue = PromptQueue()
    session.prompt_queue = prompt_queue   # accessible from /queue command

    # ── Auto-restore last session ──────────────────────────────────────────
    _last_session_path = SESSIONS_PATH / "_last.json"
    if not initial_prompt and _last_session_path.exists():
        try:
            import json as _json
            _data = _json.loads(_last_session_path.read_text())
            _msgs = _data.get("messages", [])
            if _msgs:
                session.messages            = _msgs
                session.turn_count          = _data.get("turn_count", 0)
                session.total_input_tokens  = _data.get("total_input_tokens", 0)
                session.total_output_tokens = _data.get("total_output_tokens", 0)
                session._start_time         = _time.time()   # fresh start time for restored session
                emit_ok(
                    f"Restored last session  [{len(_msgs)} messages  "
                    f"·  {session.total_input_tokens:,}↑ {session.total_output_tokens:,}↓]"
                    f"  — /clear to start fresh"
                )
        except Exception as _e:
            emit_warn(f"Could not restore last session: {_e}")

    def _autosave_session() -> None:
        try:
            if not session.messages:
                return
            import json as _json
            _last_session_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "messages":            session.messages,
                "turn_count":          session.turn_count,
                "total_input_tokens":  session.total_input_tokens,
                "total_output_tokens": session.total_output_tokens,
            }
            _last_session_path.write_text(_json.dumps(data, default=str))
        except Exception:
            pass

    atexit.register(_autosave_session)

    def _reclaim_residue_on_exit() -> None:
        """Best-effort: reclaim any orphaned GreenBoost GPU residue this
        session's own subprocess launches may have left behind (e.g. a
        llama-server child whose parent CLI exited without its own cleanup
        running — see the studio-orphaned-subprocess-on-restart incident
        class). scope="residue" only — never touches another genuinely
        in-progress GreenBoost job on this box (gb_reclaim.py's own
        classification is what makes that distinction; see task #5/#6)."""
        try:
            from greenboost_cli.gb_paths import gb_module
            gb_module("gb_reclaim").run_reclaim(scope="residue")
        except Exception:
            pass

    atexit.register(_reclaim_residue_on_exit)

    # ── GreenBoost pool auto-tune ──────────────────────────────────────────
    try:
        from greenboost_cli.greenboost.monitor import get_monitor
        m = get_monitor()
        if m.status.loaded:
            ok, cap_mb = m.apply_dynamic_pool_cap(safety_reserve_gb=9, target_pct=0.80)
            if ok and settings.get("verbose"):
                emit_ok(f"GreenBoost T2 pool cap auto-set to {round(cap_mb / 1024, 1)} GB")
    except Exception:
        pass

    # ── TurboQuant auto-enable ─────────────────────────────────────────────
    # Ensure TurboQuant KV compression is on whenever /dev/greenboost is present.
    # Idempotent: greenboost turboquant on is a no-op when already enabled.
    # Requires NOPASSWD sudo for `greenboost turboquant on`; degrades gracefully.
    if settings.get("gb_auto_turboquant", True) and Path("/dev/greenboost").exists():
        import subprocess as _sp_tq
        _tq_flag = Path("/etc/greenboost/turboquant.enabled")
        if not _tq_flag.exists():
            try:
                result = _sp_tq.run(
                    ["sudo", "-n", "greenboost", "turboquant", "on"],
                    capture_output=True, text=True, timeout=8,
                )
                if result.returncode == 0:
                    from greenboost_cli.terminal.theme import emit_ok as _emit_ok_tq
                    _emit_ok_tq("GreenBoost TurboQuant KV compression enabled")
                else:
                    from greenboost_cli.terminal.theme import emit_info as _emit_info_tq
                    _emit_info_tq(
                        "TurboQuant: add NOPASSWD sudo for 'greenboost turboquant on' to auto-enable"
                    )
            except Exception:
                pass

    # ── Non-interactive fast path ──────────────────────────────────────────
    if initial_prompt:
        _run_once_safe(initial_prompt, session, settings)
        return

    # ── MCP auto-connect from .mcp.json (background, non-blocking) ───────────
    def _mcp_autoconnect() -> None:
        try:
            from greenboost_cli.mcp.client import discover_mcp_json, MCPRegistry
            mcp_path = discover_mcp_json()
            if not mcp_path:
                return
            registry = MCPRegistry.from_mcp_json(mcp_path)
            if not registry.server_names():
                return
            # connect_all() BEFORE publishing the registry to the session — a
            # turn starting in the old window between these two lines could
            # see an empty/half-populated tool_schemas and _tool_to_server.
            results = registry.connect_all()
            session.mcp_registry = registry
            ok = sum(1 for v in results.values() if v)
            if ok:
                n_tools = len(registry.tool_schemas)
                servers_ok = [n for n, v in results.items() if v]
                from greenboost_cli.terminal.theme import emit_muted
                emit_muted(
                    f"MCP: {', '.join(servers_ok)} connected"
                    f" · {n_tools} tools available"
                )
        except Exception:
            pass

    threading.Thread(target=_mcp_autoconnect, daemon=True,
                     name="gb-mcp-autoconnect").start()

    threading.Thread(target=_gb_stats_updater, daemon=True,
                     name="gb-mem-stats").start()

    # ── RAG auto-update (background, incremental, never blocks startup) ──────
    if settings.get("rag_auto_update_on_start", True):
        def _rag_autoupdate() -> None:
            import os as _os
            # Suppress tqdm/HuggingFace progress bars from the embedding model load
            # so they don't print over the session banner at startup.
            _os.environ.setdefault("TQDM_DISABLE", "1")
            _os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            try:
                from greenboost_cli.rag.engine import resolve_folder_entry, update_folder
                entry = resolve_folder_entry()
                if not entry:
                    return
                # mtime scan runs first inside update_folder; the embedding model
                # is loaded only when something actually changed.
                r = update_folder(Path(entry["folder"]), project=entry.get("project"))
                if r.get("reindexed_files") or r.get("removed_files"):
                    from greenboost_cli.terminal.theme import emit_muted
                    emit_muted(
                        f"RAG refreshed [{entry.get('project', '?')}]: "
                        f"{r['reindexed_files']} changed · {r['removed_files']} removed"
                    )
            except Exception:
                pass  # never crash or block startup
            finally:
                # Don't leave TQDM_DISABLE set — it would affect user-invoked RAG searches
                _os.environ.pop("TQDM_DISABLE", None)
        threading.Thread(target=_rag_autoupdate, daemon=True,
                         name="gb-rag-autoupdate").start()

    show_session_banner(settings, session)

    # ── Threading primitives ───────────────────────────────────────────────
    _stdin_q: queue.Queue     = queue.Queue()
    _is_processing            = threading.Event()   # set while model runs
    _model_idle               = threading.Event()   # set when model is idle (inverse of _is_processing)
    _model_idle.set()                               # start in idle state
    _cancel_event             = threading.Event()   # set by Ctrl-C to stop current turn

    # ── prompt_toolkit session (bottom-anchored input box) ─────────────────
    _pt_session = None
    _pt_kb      = None
    _pt_style   = None
    _pt_compl   = None

    if _PT_AVAILABLE:
        # Style: override the default dark bottom-toolbar background so the
        # toolbar is transparent (no fill) over the wallpaper.
        _pt_style = Style.from_dict({
            "bottom-toolbar":      "noreverse",
            "bottom-toolbar.text": "noreverse",
        })

        # Key bindings
        _pt_kb = KeyBindings()

        # Ctrl+J → newline (Enter still submits). Shift+Enter is not used because
        # terminals report it identically to plain Enter (ControlM) at the TTY level.
        @_pt_kb.add("c-j")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        # Escape → dismiss completion popup, else clear the input line.
        @_pt_kb.add("escape", eager=True)
        def _esc_clear(event):
            buf = event.current_buffer
            if buf.complete_state:
                buf.cancel_completion()
            else:
                buf.reset()

        # Ctrl+O → expand the last truncated tool result. The truncation hint
        # itself ("… +N lines (ctrl+o to expand)") has always said this, but
        # no binding ever existed to trigger it — confirmed nothing bound
        # "c-o" anywhere in this file before this. run_in_terminal is needed
        # (not a plain console.print) because a pt app is live here; printing
        # directly would corrupt the toolbar/prompt redraw.
        @_pt_kb.add("c-o")
        def _expand_result(event):
            def _do_print():
                from greenboost_cli.terminal import renderer as _renderer_mod
                _renderer_mod.expand_last_result()
            run_in_terminal(_do_print)

        # Shift+Tab → cycle default → plan → auto → default
        @_pt_kb.add("s-tab")
        def _cycle_mode(event):
            plan   = getattr(session, "plan_mode", False)
            is_auto = settings.get("permission_mode") == "autonomous"
            if not plan and not is_auto:
                # default → plan
                session.plan_mode = True
                settings.pop("permission_mode", None)
            elif plan:
                # plan → auto
                session.plan_mode = False
                settings["permission_mode"] = "autonomous"
            else:
                # auto → default
                settings.pop("permission_mode", None)
            event.app.invalidate()

        _pt_compl = _GBCompleter(_CMD_META, lambda: COMMAND_TABLE)

        _pt_session = PromptSession(
            history=FileHistory(str(HISTORY_PATH)),
            completer=_pt_compl,
            complete_while_typing=False,
            key_bindings=_pt_kb,
            style=_pt_style,
            enable_history_search=True,
        )

        # The persistent bottom box now owns the screen for the whole session —
        # tell the renderer/statusline to stop writing raw \r-animation to
        # stdout (it gets garbled by patch_stdout) and drive the live status
        # through the toolbar's repaint instead.
        from greenboost_cli.terminal import renderer as _renderer_mod
        from greenboost_cli.terminal import statusline as _statusline_mod
        _renderer_mod.set_pt_active(True)

        def _pt_invalidate() -> None:
            try:
                app = _pt_session.app
                if app is not None:
                    app.invalidate()
            except Exception:
                pass

        _statusline_mod.set_pt_mode(_pt_invalidate)

        def _pt_is_live() -> bool:
            """True only when the PromptSession app is actively running (user is at
            the idle input prompt). Returns False during every model turn because
            _stdin_reader blocks on _model_idle.wait() while _model_worker runs.
            This lets statusline/renderer use raw \\r animation during turns without
            interference from patch_stdout (which is only installed inside prompt())."""
            try:
                app = _pt_session.app
                return app is not None and app.is_running
            except Exception:
                return False

        _statusline_mod.set_pt_live_probe(_pt_is_live)
        _renderer_mod.set_pt_live_probe(_pt_is_live)

        def _pt_suspend() -> None:
            """Force-exit the live prompt() call and wait for it to actually
            stop, so the wizard gets exclusive ownership of the raw terminal."""
            try:
                app = _pt_session.app
            except Exception:
                return
            if app is None or not app.is_running:
                return
            try:
                app.exit(result=None)
            except Exception:
                return
            import time as _t_pt
            t0 = _t_pt.monotonic()
            # Poll tightly so the wizard never starts drawing while pt still
            # owns the raw terminal — 2s ceiling guards against a stuck app.
            while app.is_running and _t_pt.monotonic() - t0 < 2.0:
                _t_pt.sleep(0.005)

        set_pt_suspend_hook(_pt_suspend)

    def _get_pt_prompt() -> "FormattedText":
        """Dynamic formatted prompt — no top rule (single dim rule lives in the
        toolbar below), no side borders."""
        frags = []
        if getattr(session, "plan_mode", False):
            frags.append((f"bold fg:{VIOLET}", "[PLAN] "))
        if settings.get("permission_mode") == "autonomous":
            frags.append((f"fg:{AMBER}", "[AUTO] "))
        q_len = len(prompt_queue)
        if q_len:
            frags.append((f"fg:{AMBER}", f"[{q_len}q] "))
        frags.append((f"fg:{VIOLET}", "› "))
        return FormattedText(frags)

    def _get_pt_toolbar() -> "FormattedText":
        """Dynamic bottom toolbar — single dim full-width rule, live status + mode hint."""
        from greenboost_cli.terminal.statusline import toolbar_status_fragments
        w = max(20, shutil.get_terminal_size((80, 24)).columns)
        bottom_border = BOX_H * w

        q_len = len(prompt_queue)
        q_hint = f"  ·  {q_len}q" if q_len else ""

        # Context window % estimate
        ctx_str = ""
        try:
            from greenboost_cli.workflow.intelligence import _estimate_tokens
            from greenboost_cli.environment.settings import gb_synapse_ctx
            ctx_est = _estimate_tokens(session)
            ctx_max = int(settings.get("context_window", 0)) or gb_synapse_ctx(settings)
            ctx_pct = ctx_est / ctx_max * 100 if ctx_max else 0
            ctx_str = f"  ·  ctx {ctx_pct:.0f}%"
        except Exception:
            pass

        shift_hint = "shift+tab · ctrl+j=newline · esc=clear"

        # Build line 2 (mode hints) and track plain-text width for right-alignment
        line2_frags: list[tuple[str, str]] = []
        status_frags = toolbar_status_fragments()
        if status_frags:
            line2_frags.extend(status_frags)
            line2_frags.append((f"fg:{DIM}", "  "))

        if getattr(session, "plan_mode", False):
            line2_frags.append((f"bold fg:{VIOLET}", "⏸ plan mode on (shift+tab to cycle)"))
            line2_frags.append((f"fg:{DIM}", f"  ·  esc to interrupt{q_hint}{ctx_str}"))
        elif settings.get("permission_mode") == "autonomous":
            line2_frags.append((f"fg:{AMBER}", "auto-accept on"))
            line2_frags.append((f"fg:{DIM}", f"  ·  {shift_hint}{q_hint}{ctx_str}"))
        else:
            line2_frags.append((f"fg:{DIM}", f"  {shift_hint}{q_hint}{ctx_str}"))

        # Right-align T1+T2+T3 tier usage stats
        with _gb_stats_lock:
            gb_segs = list(_gb_stats_segs)
        if gb_segs:
            # Build plain-text form (for width accounting)
            sep = " · "
            gb_plain = sep.join(txt for txt, _ in gb_segs)
            left_len = sum(len(t) for _, t in line2_frags)
            pad = max(2, w - left_len - len(gb_plain))
            line2_frags.append(("", " " * pad))
            for i, (txt, style) in enumerate(gb_segs):
                if i > 0:
                    line2_frags.append((f"fg:{DIM}", sep))
                line2_frags.append((style, txt))

        frags = [(f"fg:{DIM}", bottom_border + "\n")] + line2_frags
        return FormattedText(frags)

    def _build_prompt() -> str:
        """Readline fallback prompt (used when prompt_toolkit is unavailable)."""
        from greenboost_cli.instruments.handlers import _bash_cwd
        _effective_cwd = _bash_cwd if _bash_cwd else os.getcwd()
        cwd_short      = Path(_effective_cwd).name
        plan_tag = ""
        if getattr(session, "plan_mode", False):
            plan_tag = f"\001{ANSI_VIOLET}\002[PLAN]\001{ANSI_RESET}\002 "
        auto_tag = ""
        if settings.get("permission_mode") == "autonomous":
            auto_tag = f"\001{ANSI_AMBER}\002[AUTO]\001{ANSI_RESET}\002 "
        q_tag = ""
        q_len = len(prompt_queue)
        if q_len:
            q_tag = f"\001{ANSI_AMBER}\002[{q_len}q]\001{ANSI_RESET}\002 "
        name_tag = ""
        sname = getattr(session, "name", None)
        if sname:
            name_tag = f"\001{ANSI_TEAL}\002{sname}\001{ANSI_RESET}\002 "
        return (
            f"\n\001{ANSI_GRAY}\002[{cwd_short}]\001{ANSI_RESET}\002 "
            f"{name_tag}{plan_tag}{auto_tag}{q_tag}"
            f"\001{ANSI_VIOLET}\002❯\001{ANSI_RESET}\002 "
        )

    def _stdin_reader() -> None:
        """Daemon thread: feed lines into _stdin_q.

        Supports multi-line input: end a line with \\ to continue on the next.
        With prompt_toolkit: bottom-anchored box via PromptSession.prompt().
        Fallback: readline input() when prompt_toolkit is unavailable.
        """
        _CONT_PROMPT_RL = f"\001{ANSI_GRAY}\002  › \001{ANSI_RESET}\002"
        _CONT_PROMPT_PT = (
            FormattedText([(f"fg:{GRAY}", "  › ")]) if _PT_AVAILABLE else None
        )

        while True:
            # Block until the model is idle (_model_idle is SET = idle).
            # _model_idle is the inverse of _is_processing: set at startup,
            # cleared when a turn begins, re-set when the turn completes.
            # Event.wait() releases the GIL while blocking — zero CPU spin.
            _model_idle.wait()
            if _shutdown_evt.is_set():
                return
            try:
                parts: list[str] = []
                first = True
                aborted = False
                while True:
                    if _pt_session is not None:
                        # raw=True: pass model/tool output straight through as real
                        # ANSI bytes (write_raw) instead of escaping it — with
                        # raw=False every \x1b the renderer writes shows up as
                        # literal "?[38;2;...m" text on screen.
                        with _pt_patch_stdout(raw=True):
                            if first:
                                raw = _pt_session.prompt(
                                    _get_pt_prompt,
                                    bottom_toolbar=_get_pt_toolbar,
                                )
                            else:
                                raw = _pt_session.prompt(_CONT_PROMPT_PT)
                    else:
                        raw = input(_build_prompt() if first else _CONT_PROMPT_RL)
                    if _shutdown_evt.is_set():
                        return
                    # raw is None when a wizard force-exited the live prompt()
                    # call to take exclusive ownership of the terminal — there
                    # is no line to submit; just loop back and wait for idle.
                    if raw is None:
                        aborted = True
                        break
                    first = False
                    if raw.endswith("\\"):
                        parts.append(raw[:-1])   # strip the backslash
                    else:
                        parts.append(raw)
                        break
                if aborted:
                    continue
                full = "\n".join(parts).strip()
                _stdin_q.put(("line", full))
            except EOFError:
                _stdin_q.put(("eof", None))
                return
            except KeyboardInterrupt:
                print()
                _stdin_q.put(("interrupt", None))

    def _model_worker(first_input: str) -> None:
        """
        Background thread: run first_input, then drain the prompt queue.
        Clears _is_processing only when the queue is empty.
        """
        user_input = first_input
        # Auto-name session from first non-command message (like Claude Code)
        if not getattr(session, "name", None) and not user_input.startswith("/"):
            words = user_input.replace("\n", " ").split()[:6]
            auto_name = " ".join(words)
            if len(auto_name) < len(user_input.replace("\n", " ")):
                auto_name += "…"
            session.name = auto_name[:40]
        while True:
            _model_idle.clear()
            _is_processing.set()
            _cancel_event.clear()
            settings["_cancel_event"] = _cancel_event
            try:
                _run_once_safe(user_input, session, settings)
            except Exception as _escaped_err:
                # _run_once_safe handles/logs its own errors; reaching here means
                # something broke inside its own error handling — never swallow silently.
                console.print(
                    f"\n  [{AMBER}]◈[/]  [{GRAY}]Turn failed unexpectedly: "
                    f"{type(_escaped_err).__name__}: {str(_escaped_err)[:160]}[/]"
                )

            # Auto-feed Q&A into RAG (background, non-blocking)
            if not _cancel_event.is_set() and settings.get("auto_rag", True):
                _maybe_feed_rag(user_input, session, settings)

            # If cancelled, discard queued prompts and exit immediately
            if _cancel_event.is_set():
                _cancel_event.clear()
                _is_processing.clear()
                _model_idle.set()
                _stdin_q.put(("model_done", None))
                return

            next_item = prompt_queue.dequeue()
            if next_item:
                q_remaining = len(prompt_queue)
                suffix = f"  [{q_remaining} more queued]" if q_remaining else ""
                console.print(
                    f"\n  [{AMBER}]◈[/]  [{GRAY}]queued [{next_item.id}]{suffix}:[/] "
                    f"[{VIOLET}]{next_item.text[:80]}[/]"
                )
                user_input = next_item.text
                # Loop to process it
            else:
                # Queue empty — signal the main loop then exit this thread.
                _is_processing.clear()
                _model_idle.set()
                _stdin_q.put(("model_done", None))
                return

    # ── Start stdin reader daemon ──────────────────────────────────────────
    _stdin_thread = threading.Thread(
        target=_stdin_reader, daemon=True, name="gb-stdin"
    )
    _stdin_thread.start()

    # ── Main event loop ────────────────────────────────────────────────────
    while True:
        try:
            ev, data = _stdin_q.get(timeout=0.5)
        except queue.Empty:
            continue

        if ev == "eof":
            print()
            emit_ok("Goodbye.")
            try:
                from greenboost_cli.instruments.hooks import run_stop_hooks
                turns = getattr(session, "turn_count", 0)
                run_stop_hooks(f"Session ended after {turns} turn(s).")
            except Exception:
                pass
            sys.exit(0)

        elif ev == "interrupt":
            if _is_processing.is_set():
                if _cancel_event.is_set():
                    # Second Ctrl-C — user is impatient, warn they may need to wait
                    console.print(
                        f"\n  [{AMBER}]◈[/]  [{GRAY}]Cancel already pending — "
                        f"waiting for next token from the model…[/]"
                    )
                else:
                    _cancel_event.set()
                    console.print(
                        f"\n  [{AMBER}]◈[/]  [{GRAY}]Cancelling…  "
                        f"(stops at next token — Ctrl-C again to see status)[/]"
                    )
            continue

        elif ev == "model_done":
            # Model finished and queue is empty — stdin_reader will show prompt.
            continue

        elif ev == "line":
            user_input = data
            if not user_input:
                continue

            # Slash commands run immediately regardless of model state.
            if dispatch_command(user_input, session, settings):
                continue

            if _is_processing.is_set():
                item = prompt_queue.enqueue(user_input)
                console.print(
                    f"\n  [{AMBER}]◈[/]  [{GRAY}]Queued [{item.id}]:[/] "
                    f"[{VIOLET}]{user_input[:80]}[/]"
                )
                console.print(
                    f"  [{GRAY}]  /queue to list  ·  "
                    f"/queue del {item.id} to remove[/]"
                )
            else:
                t = threading.Thread(
                    target=_model_worker,
                    args=(user_input,),
                    daemon=True,
                    name="gb-model",
                )
                t.start()


# ── Helper: run once with error recovery ────────────────────────────────────────

def _run_once_safe(user_input: str, session: ConversationSession, settings: dict) -> None:
    """Run process_query with KeyboardInterrupt handling. gb-synapse's own
    adapters (inference/adapters.py) already produce actionable RuntimeError
    messages (start/OOM/not-found remediation via /llamaserve, greenboost
    pull, greenboost recommend) — no backend-switch recovery needed here
    since there's only one backend."""
    try:
        process_query(user_input, session, settings)
    except KeyboardInterrupt:
        console.print(f"\n  [{AMBER}]  (interrupted)[/]")
    except Exception as _err:
        err_msg = str(_err)
        if isinstance(_err, RuntimeError):
            emit_err(err_msg)
        else:
            # Unexpected non-RuntimeError — log full traceback so it can be diagnosed
            try:
                _log_dir = GB_HOME / "logs"
                _log_dir.mkdir(parents=True, exist_ok=True)
                _log_file = _log_dir / "repl-errors.log"
                with _log_file.open("a") as _lf:
                    _lf.write(traceback.format_exc())
                emit_warn(
                    f"Turn failed: {type(_err).__name__}: {err_msg[:120]}"
                    f" — see {_log_file}"
                )
            except Exception:
                emit_err(f"Turn failed: {type(_err).__name__}: {err_msg[:120]}")
