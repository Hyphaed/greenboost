"""gb clear-memory-pool — unload Ollama models from VRAM after batch work.

After contextual RAG ingest (LLM contextualize + embedding), the Ollama
server keeps models warm in VRAM for its keep_alive window (default 5 min).
On a 12 GB GPU this blocks subsequent work. This module:

  1. Exposes _ollama_unload(model) used internally after ingest phases.
  2. Exposes cmd_clear_memory_pool(argv) for the `gb clear-memory-pool` command.

Ollama unload mechanism: POST /api/generate with keep_alive=0 instructs the
server to release the model immediately rather than waiting for keep_alive.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _ollama_base_url() -> str:
    """Return the Ollama server base URL (respects OLLAMA_HOST env)."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return host.rstrip("/")


def _ollama_unload(model: str, *, base_url: str | None = None, silent: bool = False) -> bool:
    """Unload a model from Ollama's VRAM immediately.

    Sends keep_alive=0 to the /api/generate endpoint — the standard Ollama
    mechanism for immediate eviction.  Returns True on success, False on any
    network / server error.
    """
    if not model:
        return False
    url = (base_url or _ollama_base_url()) + "/api/generate"
    payload = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp.read()
        return True
    except urllib.error.URLError as exc:
        if not silent:
            sys.stderr.write(f"gb: clear-memory-pool: could not reach Ollama ({exc.reason})\n")
        return False
    except Exception as exc:      # noqa: BLE001
        if not silent:
            sys.stderr.write(f"gb: clear-memory-pool: {exc}\n")
        return False


def _ollama_list_loaded(*, base_url: str | None = None) -> list[dict]:
    """Return currently loaded models from Ollama's /api/ps endpoint."""
    url = (base_url or _ollama_base_url()) + "/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return data.get("models", [])
    except Exception:       # noqa: BLE001
        return []


# ── CLI handler ───────────────────────────────────────────────────────────────

def cmd_clear_memory_pool(argv: list[str]) -> int:
    """Handler for `gb clear-memory-pool [--model NAME] [--all] [--json]`."""
    p = argparse.ArgumentParser(
        prog="gb clear-memory-pool",
        description="Unload Ollama model(s) from VRAM immediately.",
        add_help=True,
    )
    p.add_argument(
        "--model", default=None,
        help="Model to unload (default: active model from gb settings).",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Unload ALL currently loaded models (reads /api/ps).",
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    base_url = _ollama_base_url()
    results: list[dict] = []

    if args.all:
        loaded = _ollama_list_loaded(base_url=base_url)
        if not loaded:
            if args.json:
                _emit_json({"unloaded": [], "note": "no models loaded"})
            else:
                print("  ·  no models currently loaded in Ollama")
            return 0
        for entry in loaded:
            m = entry.get("name", "")
            if not m:
                continue
            ok = _ollama_unload(m, base_url=base_url)
            results.append({"model": m, "ok": ok})
            if not args.json:
                status = "✓" if ok else "✗"
                print(f"  \033[{'32' if ok else '31'}m{status}\033[0m  unloaded {m}")
    else:
        model = args.model
        if not model:
            from greenboost_cli.environment.settings import load_settings   # noqa: PLC0415
            model = load_settings().get("model", "")
        if not model:
            _emit_err("no model specified; pass --model or configure gb settings")
            if args.json:
                _emit_json({"error": "no model specified", "unloaded": []})
            return 1
        ok = _ollama_unload(model, base_url=base_url)
        results.append({"model": model, "ok": ok})
        if not args.json:
            status = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
            verb = "unloaded" if ok else "failed to unload"
            print(f"  {status}  {verb} {model}")

    if args.json:
        _emit_json({"unloaded": [r["model"] for r in results if r["ok"]],
                    "failed":   [r["model"] for r in results if not r["ok"]]})
    return 0 if all(r["ok"] for r in results) else 1


def _emit_json(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def _emit_err(msg: str) -> None:
    sys.stderr.write(f"gb: {msg}\n")
