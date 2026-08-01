"""gb clear-memory-pool — LEGACY module, superseded (task #7 consolidation).

The `gb clear-memory-pool` command is gone: `cli_headless.py` now points
that verb at a one-release deprecation notice pointing to `sudo greenboost
clear memory-pool`, which itself delegates to the sibling greenboost repo's
gb_reclaim.py (classification + escalation, shared with the bash nuke —
see gb_reclaim.py's own docstring for why two similarly-named-but-different
"clear memory pool" commands existed in the first place).

_ollama_unload/_ollama_list_loaded's canonical home is now gb_reclaim.py
too (contextual_rag.py's ingest-phase VRAM release imports it from there via
greenboost_cli.gb_paths.gb_module, the same cross-repo convention
backend_cmds.py uses for gb_synapse). The copies below are kept only in case
an external caller still imports this module directly — do not add new
callers; import gb_reclaim instead.

Ollama unload mechanism: POST /api/generate with keep_alive=0 instructs the
server to release the model immediately rather than waiting for keep_alive.
"""
from __future__ import annotations

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
