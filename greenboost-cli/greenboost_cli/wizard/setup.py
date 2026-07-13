"""
Interactive setup wizard for GreenBoost CLI.

Picks a starting model from gb-synapse's manifest — HuggingFace-pulled or
Ollama-indexed GGUFs, the only inference backend. Modifies settings
in-place; the caller must call save_settings(settings) afterwards.

Entry point:
    from greenboost_cli.wizard.setup import run_wizard
    run_wizard(settings, force=False)
"""
from __future__ import annotations

import sys

from greenboost_cli.terminal.theme import (
    console, emit_ok, emit_err, emit_warn, emit_info, emit_step,
    VIOLET, GRAY, LIME,
    SEPARATOR,
    ANSI_AMBER, ANSI_RESET,
)

# Path to the GreenBoost source tree (gb_synapse.py lives here) — same
# convention as slash_commands/backend_cmds.py and slash_commands/quant_cmds.py.
from greenboost_cli.gb_paths import gb_py_root, gb_root_hint

_GB_SRC = gb_py_root()


def _import_gb_synapse():
    if str(_GB_SRC) not in sys.path:
        sys.path.insert(0, str(_GB_SRC))
    import gb_synapse
    return gb_synapse


# ── Input helpers ──────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        val = input(f"\001{ANSI_AMBER}\002  ❯ {prompt}{hint}: \001{ANSI_RESET}\002").strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print()
        return default


def _ask_int(prompt: str, choices: range, default: int) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            n = int(raw)
            if n in choices:
                return n
        except ValueError:
            pass
        emit_warn(f"Please enter a number between {choices.start} and {choices.stop - 1}.")


# ── Wizard step ────────────────────────────────────────────────────────────

def _step_model_selection(gb_synapse, settings: dict) -> str:
    """Choose a model from the gb-synapse manifest, or pull/index a new one."""
    emit_step(1, 1, "Choose a model")
    console.print()

    models = gb_synapse.list_models()
    if not models:
        emit_info("No models yet — checking for already-downloaded Ollama models…")
        try:
            found = gb_synapse.index_ollama_models()
        except Exception:
            found = []
        if found:
            emit_ok(f"Indexed {len(found)} Ollama model(s).")
            models = gb_synapse.list_models()
        else:
            emit_warn("None found.")

    current = settings.get("model", "")

    if models:
        console.print(f"  [{GRAY}]Available models (gb-synapse manifest):[/]")
        console.print()
        shown = models[:20]
        for i, m in enumerate(shown, 1):
            active = current in (m.name, f"gb-synapse/{m.name}")
            marker = f"[{LIME}]◈ [/]" if active else "  "
            console.print(f"    {marker}[{VIOLET}][{i}][/] [{GRAY}]{m.name}  "
                          f"({m.source}, {m.quant}, {m.n_bytes / (1024 ** 3):.1f} GiB)[/]")
        pull_idx = len(shown) + 1
        console.print(f"    [{VIOLET}][{pull_idx}][/] [{GRAY}]Pull a new model from HuggingFace[/]")
        console.print()
        choice = _ask_int(f"Choose (1-{pull_idx})", range(1, pull_idx + 1), 1)
        if choice <= len(shown):
            return shown[choice - 1].name
    else:
        emit_info("No models available yet.")

    repo = _ask("HuggingFace repo to pull (org/repo[:quant])")
    if not repo:
        return current
    if not gb_synapse.hf_token():
        emit_warn("No HuggingFace token set.")
        token = _ask("HuggingFace token (Enter to skip — only works for public repos)")
        if token:
            gb_synapse.login(token)
    emit_info(f"Pulling {repo} …")
    try:
        entry = gb_synapse.pull(repo)
        emit_ok(f"Pulled {entry.name}  ({entry.n_bytes / (1024 ** 3):.2f} GiB, {entry.quant})")
        return entry.name
    except Exception as e:
        emit_err(f"Pull failed: {e}")
        return current


# ── Public entry point ─────────────────────────────────────────────────────

def run_wizard(settings: dict, force: bool = False) -> None:
    """
    Run the interactive setup wizard.

    Modifies settings in-place (model). The caller must call
    save_settings(settings) afterwards.
    """
    console.print()
    console.print("[bold white]GreenBoost CLI — Setup Wizard[/]")
    console.print(SEPARATOR)
    console.print(f"[{GRAY}]Local-first AI inference for your terminal, via gb-synapse[/]")
    console.print()

    try:
        gb_synapse = _import_gb_synapse()
    except ImportError as e:
        emit_err(f"Cannot import gb_synapse from {_GB_SRC}: {e}")
        emit_info(f"Fix: {gb_root_hint()}.")
        return

    if not gb_synapse.engine_installed():
        emit_warn("gb-synapse engine not built yet.")
        console.print(f"  [{GRAY}]Run: sudo greenboost synapse build-engine[/]")
        console.print()

    model = _step_model_selection(gb_synapse, settings)
    settings["model"] = model

    console.print()
    console.print(f"[bold {LIME}]✓ Setup complete[/]")
    console.print(SEPARATOR)
    console.print(f"  [{GRAY}]Backend   [{VIOLET}]gb-synapse[/]")
    console.print(f"  [{GRAY}]Model     [{VIOLET}]{model or '(none — set with /model)'}[/]")
    console.print()
    emit_info("Run:  greenboost-cli  or  gb")
    console.print()
