"""
/download-models slash command — fetch models for gb-synapse.

Two sources, both ending up in gb-synapse's manifest (`/model` or
`greenboost synapse list` shows the result):
  * HuggingFace — browse trending tool-calling models, then pull a GGUF via
    gb_synapse.pull() (fails loudly if the chosen repo has no GGUF release).
  * Ollama — list already-pulled Ollama models (indexed into the manifest),
    or `ollama pull <name>` a new one and index it immediately.

Registered into commands.COMMAND_TABLE at module import.
"""
from __future__ import annotations

import os
import subprocess
import sys

from greenboost_cli.terminal.theme import (
    console, emit_ok, emit_err, emit_warn, emit_info,
    VIOLET, GRAY, LIME, AMBER,
)

# Path to the GreenBoost source tree (gb_synapse.py lives here) — same
# convention as backend_cmds.py / quant_cmds.py.
from greenboost_cli.gb_paths import gb_py_root, gb_root_hint

_GB_SRC = gb_py_root()


def _import_gb_synapse():
    if str(_GB_SRC) not in sys.path:
        sys.path.insert(0, str(_GB_SRC))
    import gb_synapse
    return gb_synapse


def _hf_token(settings: dict) -> str:
    return settings.get("hf_token", "") or os.environ.get("HF_TOKEN", "")


def _fetch_hf_tool_calling_models(query: str = "", limit: int = 25, hf_token: str = "") -> list:
    try:
        import httpx
        params: dict = {
            "pipeline_tag": "text-generation",
            "tags":         "function-calling",
            "sort":         "trending",
            "direction":    "-1",
            "limit":        str(limit),
            "full":         "false",
        }
        if query:
            params["search"] = query
        hdrs: dict = {"User-Agent": "GreenBoostCLI/1.0"}
        if hf_token:
            hdrs["Authorization"] = f"Bearer {hf_token}"
        r = httpx.get("https://huggingface.co/api/models", params=params, headers=hdrs, timeout=15)
        r.raise_for_status()
        return [
            {
                "name":      m.get("modelId", ""),
                "downloads": m.get("downloads", 0),
                "likes":     m.get("likes", 0),
                "updated":   (m.get("lastModified") or "")[:10],
            }
            for m in r.json()
            if m.get("modelId")
        ]
    except Exception as e:
        emit_warn(f"HuggingFace API error: {e}")
        return []


def cmd_download_models(args: str, _session, settings) -> bool:
    try:
        gb_synapse = _import_gb_synapse()
    except ImportError as e:
        emit_err(f"Cannot import gb_synapse from {_GB_SRC}: {e}")
        emit_info(f"Fix: {gb_root_hint()}.")
        return True

    from greenboost_cli.environment.settings import save_settings

    parts      = args.strip().split(None, 1)
    source_arg = parts[0].lower() if parts else ""
    query_arg  = parts[1].strip() if len(parts) > 1 else ""

    if source_arg in ("ollama", "ol"):
        source = "ollama"
    elif source_arg in ("hf", "huggingface", "hugging-face"):
        source = "hf"
    else:
        console.print()
        console.print(f"[bold {VIOLET}]Download Models[/]")
        console.print(f"[{GRAY}]──────────────────────────────────────────────────[/]")
        console.print()
        console.print(f"  [{GRAY}][1][/] [{VIOLET}]Ollama[/]        [{GRAY}]— already-pulled models, or pull a new one[/]")
        console.print(f"  [{GRAY}][2][/] [{VIOLET}]HuggingFace[/]   [{GRAY}]— browse + pull a GGUF via gb-synapse[/]")
        console.print()
        try:
            choice = input(f"\033[{AMBER}m  ❯ Choose [1/2]: \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return True
        source = "ollama" if choice.strip() == "1" else "hf"

    # ── Ollama path — already-pulled models, or `ollama pull` a new one ────
    if source == "ollama":
        emit_info("Indexing already-pulled Ollama models …")
        try:
            gb_synapse.index_ollama_models()
        except Exception as e:
            emit_warn(f"Could not index Ollama models: {e}")
        existing = [m for m in gb_synapse.list_models() if m.source == "ollama"]

        if existing:
            console.print()
            console.print(f"[bold {GRAY}]  {'#':<4}  {'MODEL':<35}  {'QUANT':<10}  SIZE[/]")
            console.print(f"[{GRAY}]  {'─' * 64}[/]")
            for i, m in enumerate(existing[:20], 1):
                console.print(f"  [{GRAY}]{i:<4}  {m.name[:34]:<35}  {m.quant:<10}  "
                              f"{m.n_bytes / (1024 ** 3):.1f} GiB[/]")
            console.print()

        query = query_arg
        if not query:
            try:
                query = input(f"\033[{AMBER}m  ❯ Model number to use, or a new name to pull: \033[0m").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return True
        if not query:
            return True

        try:
            idx = int(query)
            if 1 <= idx <= len(existing[:20]):
                emit_ok(f"Use:  /model {existing[idx - 1].name}")
                return True
        except ValueError:
            pass

        emit_info(f"Pulling [{VIOLET}]{query}[/] via ollama …")
        result = subprocess.run(["ollama", "pull", query])
        if result.returncode != 0:
            emit_err(f"Failed to pull {query}")
            return True
        emit_ok(f"Downloaded: {query}")
        try:
            gb_synapse.index_ollama_models()
            emit_info(f"Use:  /model {query}")
        except Exception as e:
            emit_warn(f"Pulled, but indexing into gb-synapse failed: {e}")

    # ── HuggingFace path — browse, then pull a GGUF via gb-synapse ─────────
    else:
        token = _hf_token(settings)
        if not token:
            token = gb_synapse.hf_token() or ""
        if not token:
            console.print()
            emit_info("No HuggingFace token found.")
            emit_info("Set via: /config hf_token=hf_...")
            emit_info("Or env:  export HF_TOKEN=hf_...")
            try:
                new_token = input(f"\033[{AMBER}m  ❯ Enter HF token now (or Enter to skip): \033[0m").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                new_token = ""
            if new_token:
                settings["hf_token"] = new_token
                save_settings(settings)
                token = new_token

        query = query_arg
        if not query:
            try:
                query = input(f"\033[{AMBER}m  ❯ Search query (or Enter for trending): \033[0m").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return True

        emit_info("Fetching tool-calling models from HuggingFace …")
        models = _fetch_hf_tool_calling_models(query=query, limit=25, hf_token=token)

        if not models:
            emit_warn("No results or could not reach HuggingFace.")
            emit_info("You can also pull a specific repo directly: greenboost pull <org/repo>[:quant]")
            return True

        console.print()
        console.print(f"[bold {GRAY}]  {'#':<4}  {'MODEL (HuggingFace ID)':<52}  {'DL/mo':>7}  {'❤':>5}  UPDATED[/]")
        console.print(f"[{GRAY}]  {'─' * 78}[/]")
        for i, m in enumerate(models[:20], 1):
            dl   = m["downloads"]
            dl_s = f"{dl/1_000_000:.1f}M" if dl >= 1_000_000 else (f"{dl/1000:.0f}K" if dl >= 1000 else str(dl))
            console.print(f"  [{GRAY}]{i:<4}  {m['name'][:51]:<52}  {dl_s:>7}  {m['likes']:>5}  {m['updated']}[/]")
        console.print()
        emit_info("Picking a repo pulls its GGUF release via gb-synapse — fails if it has none "
                   "(look for a community GGUF reupload, e.g. bartowski/<model>-GGUF).")
        console.print()

        try:
            pick = input(f"\033[{AMBER}m  ❯ Model number to pull, or a repo ID (org/repo[:quant]): \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return True
        if not pick:
            return True

        try:
            idx = int(pick)
            repo = models[idx - 1]["name"] if 1 <= idx <= len(models) else pick
        except ValueError:
            repo = pick

        emit_info(f"Pulling [{VIOLET}]{repo}[/] …")
        try:
            entry = gb_synapse.pull(repo)
            emit_ok(f"Pulled {entry.name}  ({entry.n_bytes / (1024 ** 3):.2f} GiB, {entry.quant})")
            emit_info(f"Use:  /model {entry.name}")
        except Exception as e:
            emit_err(f"Pull failed: {e}")

    return True


def cmd_fetch_model(args: str, _session, settings) -> bool:
    """/fetch-model huggingface [org/repo[:quant]]  — pull a HF model (asks for a
                                                       token first if none is set)
    /fetch-model ollama [model[:tag]]              — pull an Ollama model

    Give the model name on the command line for a one-shot fetch (recommended),
    or omit it to be prompted (huggingface) / see already-pulled models first
    (ollama). Either source lands in gb-synapse's manifest under EXACTLY the
    name you typed, so `/model <that same name>` switches to it — both are
    served identically, through gb-synapse.
    """
    try:
        gb_synapse = _import_gb_synapse()
    except ImportError as e:
        emit_err(f"Cannot import gb_synapse from {_GB_SRC}: {e}")
        emit_info(f"Fix: {gb_root_hint()}.")
        return True

    parts      = args.strip().split(None, 1)
    source_arg = parts[0].lower() if parts else ""
    model_arg  = parts[1].strip() if len(parts) > 1 else ""

    if source_arg in ("hf", "huggingface", "hugging-face"):
        _fetch_model_huggingface(gb_synapse, settings, model_arg)
    elif source_arg in ("ollama", "ol"):
        _fetch_model_ollama(gb_synapse, model_arg)
    else:
        console.print()
        console.print(f"[bold {VIOLET}]Fetch Model[/]")
        console.print(f"[{GRAY}]──────────────────────────────────────────────────[/]")
        emit_info("Usage: /fetch-model huggingface <org/repo[:quant]>")
        emit_info("       /fetch-model ollama <model[:tag]>")
    return True


def _fetch_model_huggingface(gb_synapse, settings: dict, repo: str = "") -> None:
    """Token (if not already set) → repo name (arg, else prompt) → pull.

    The manifest entry is named exactly `repo` — the string typed here — so
    `/model <that same string>` finds it directly, no second lookup or pull.
    """
    from greenboost_cli.environment.settings import save_settings

    token = _hf_token(settings) or gb_synapse.hf_token() or ""
    if not token:
        console.print()
        emit_info("A HuggingFace token is needed for gated/private repos "
                   "(public repos work without one).")
        try:
            import getpass
            new_token = getpass.getpass("  ❯ HuggingFace token (Enter to skip): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if new_token:
            try:
                gb_synapse.login(new_token)
            except Exception as e:
                emit_err(f"Could not save token: {e}")
                return
            settings["hf_token"] = new_token
            save_settings(settings)
            token = new_token
            emit_ok("Token saved.")

    if not repo:
        try:
            repo = input(f"\033[{AMBER}m  ❯ HuggingFace model (org/repo[:quant]): \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
    if not repo:
        emit_warn("No model given.")
        return

    emit_info(f"Pulling [{VIOLET}]{repo}[/] via gb-synapse …")
    try:
        entry = gb_synapse.pull(repo, name=repo)
        emit_ok(f"Pulled {entry.name}  ({entry.n_bytes / (1024 ** 3):.2f} GiB, {entry.quant})")
        emit_info(f"Serving via gb-synapse — switch with:  /model {entry.name}")
    except Exception as e:
        emit_err(f"Pull failed: {e}")


def _fetch_model_ollama(gb_synapse, name: str = "") -> None:
    """Pull `name` directly via `ollama pull` when given; otherwise show
    already-pulled Ollama models first, then prompt for one to pull."""
    if not name:
        emit_info("Indexing already-pulled Ollama models …")
        try:
            gb_synapse.index_ollama_models()
        except Exception as e:
            emit_warn(f"Could not index Ollama models: {e}")
        existing = [m for m in gb_synapse.list_models() if m.source == "ollama"]

        if existing:
            console.print()
            console.print(f"[bold {GRAY}]  {'MODEL':<35}  {'QUANT':<10}  SIZE[/]")
            console.print(f"[{GRAY}]  {'─' * 60}[/]")
            for m in existing:
                console.print(f"  [{GRAY}]{m.name[:34]:<35}  {m.quant:<10}  {m.n_bytes / (1024 ** 3):.1f} GiB[/]")
            console.print()

        try:
            name = input(f"\033[{AMBER}m  ❯ Ollama model to pull (Enter to skip): \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not name:
            return

    emit_info(f"Pulling [{VIOLET}]{name}[/] via ollama …")
    try:
        result = subprocess.run(["ollama", "pull", name])
    except FileNotFoundError:
        emit_err("ollama binary not found — install it: https://ollama.com")
        return
    if result.returncode != 0:
        emit_err(f"Failed to pull {name}")
        return
    emit_ok(f"Downloaded: {name}")
    try:
        gb_synapse.index_ollama_models()
        emit_info(f"Serving via gb-synapse — switch with:  /model {name}")
    except Exception as e:
        emit_warn(f"Pulled, but indexing into gb-synapse failed: {e}")


def _register() -> None:
    from greenboost_cli.terminal.commands import COMMAND_TABLE
    COMMAND_TABLE["download-models"] = cmd_download_models
    COMMAND_TABLE["fetch-model"] = cmd_fetch_model


_register()
