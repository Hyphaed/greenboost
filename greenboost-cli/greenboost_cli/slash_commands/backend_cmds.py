"""
Backend slash commands: gb-synapse llama-server lifecycle + prompt cache.

Registered into commands.COMMAND_TABLE at module import.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

from greenboost_cli.terminal.theme import (
    console, emit_ok, emit_err, emit_warn, emit_info,
    VIOLET, GRAY, LIME, AMBER,
)

_OLLAMA_BASE = "http://localhost:11434"


# ── Ollama service coexistence ──────────────────────────────────────────────
#
# gb-synapse binds its own port (11435 by default, registry.py's "gb-synapse"
# entry — separate from Ollama's 11434 since the 2026-07 port migration), but
# both still compete for the same GPU VRAM. These manage the actual Ollama
# SYSTEMD SERVICE so it steps aside before gb-synapse (or any other GPU-heavy
# task, e.g. /gb-quant --serve) starts — unrelated to backend *choice*, which
# no longer exists.

def _ollama_service_running() -> bool:
    """Return True if the Ollama HTTP server is reachable."""
    try:
        with urllib.request.urlopen(_OLLAMA_BASE, timeout=2):
            return True
    except Exception:
        return False


def _ollama_unload_all() -> None:
    """Ask Ollama to unload all cached models (keep_alive=0) via its API.

    This frees VRAM and T2 DDR held by Ollama without stopping the process.
    Silently does nothing if Ollama is not running.
    """
    try:
        import json as _json
        req_list = urllib.request.Request(
            f"{_OLLAMA_BASE}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_list, timeout=3) as resp:
            data = _json.loads(resp.read())
        for m in data.get("models", []):
            name = m.get("name", "")
            if not name:
                continue
            body = _json.dumps({"model": name, "keep_alive": 0}).encode()
            unload_req = urllib.request.Request(
                f"{_OLLAMA_BASE}/api/generate",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(unload_req, timeout=5)
            except Exception:
                pass
    except Exception:
        pass


def _try_systemd_ollama(action: str) -> bool:
    """Try 'sudo -n systemctl <action> ollama' without prompting.

    Returns True if the command succeeded.  Falls back gracefully if sudo
    requires a password (non-interactive environments).
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "systemctl", action, "ollama"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def ollama_suspend_for_gpu_task(reason: str = "gb-synapse") -> None:
    """Suspend Ollama before another GPU task starts: unload models then try
    to stop the service. Emits a status line so the user knows what's
    happening. `reason` is included in the status message (e.g.
    "llama-server" or "gb-quant")."""
    if not _ollama_service_running():
        return
    emit_info(f"Suspending Ollama to free memory for {reason}…")
    _ollama_unload_all()
    stopped = _try_systemd_ollama("stop")
    if stopped:
        emit_ok("Ollama service stopped.")
    else:
        emit_info("Ollama models unloaded (stop requires sudo — run: sudo systemctl stop ollama)")


# ── /llamaserve — gb-synapse llama-server backend (prompt-cache via /llamacache) ─
#
# Process lifecycle consolidated into gb-synapse (2026-07-07): this used to
# spawn Ollama's vendored llama-server directly — a deliberate zero-build-
# step tradeoff (see git history for the original _build_llamacpp_cmd /
# _build_llamacpp_env / _ollama_cuda_run_dir). gb-synapse's serve() now owns
# ALL llama-server process spawning (it needs a from-source build anyway,
# for --rpc cluster support across host+feeder) and carries the same
# prompt-cache / P-core-threading / KV-quant flags this command used to
# build itself — see workflow/gb-synapse.md in the greenboost repo. This
# file is now a thin wrapper: import gb_synapse, call serve()/stop()/ps().
#
# Ollama wraps llama.cpp and hides its slot save/restore + prompt-cache API.
# Running llama-server directly (via gb-synapse) still gets two real caching
# layers "for free":
#   1. automatic in-memory prompt/context-checkpoint cache (--cache-prompt,
#      on by default) — reused across requests to the same running server
#      regardless of which greenboost-cli session sent them, as long as the
#      server process stays up.
#   2. disk-persisted slot save/restore (--slot-save-path, POST
#      /slots/{id}?action=save|restore) — survives a server restart, now
#      via gb_synapse_api.py's /slots passthrough (see /llamacache below).

# Path to the GreenBoost source tree (gb_synapse.py lives here) — same
# convention as quant_cmds.py's _GB_SRC.
from greenboost_cli.gb_paths import gb_py_root, gb_root_hint

_GB_SRC = gb_py_root()

_LLAMACPP_SLOT_DIR = Path("/var/lib/greenboost/synapse/slots")  # == gb_synapse.SLOT_DIR


def _ensure_gb_synapse_path() -> None:
    if str(_GB_SRC) not in sys.path:
        sys.path.insert(0, str(_GB_SRC))


def _import_gb_synapse():
    """Import gb_synapse from the sibling greenboost repo. Raises
    ImportError (with the checkout path in the message) if it's missing."""
    _ensure_gb_synapse_path()
    import gb_synapse
    return gb_synapse


def _llamaserve_model_name(settings: dict) -> str:
    """Strip the backend/ prefix greenboost-cli stores in settings["model"]."""
    model = settings.get("model", "")
    for prefix in ("gb-synapse/", "llamacpp/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def _resolve_gguf_path(model_id: str) -> str:
    """Resolve a model_id to a GGUF file path, for cache-key hashing only
    (server startup itself resolves models via gb_synapse's own manifest —
    see gb_synapse._resolve_model). Accepts an absolute path to an existing
    .gguf file, or an already-pulled Ollama model name/tag (resolved via
    `ollama show --modelfile`, the same files Ollama itself feeds to
    llama-server)."""
    p = Path(model_id)
    if p.exists():
        return str(p)
    try:
        result = subprocess.run(
            ["ollama", "show", model_id, "--modelfile"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("FROM ") and not line.startswith("FROM #"):
                    return line.split("FROM ", 1)[1].strip()
    except Exception:
        pass
    return model_id   # let llama-server fail loudly with a clear error


def _llamacpp_base_url(settings: dict) -> str:
    from greenboost_cli.inference.registry import BACKEND_REGISTRY
    return settings.get("llamacpp_base_url") or BACKEND_REGISTRY["gb-synapse"]["base_url"]


def _llamacpp_running_pid(settings: dict | None = None) -> int | None:
    """A real pid (from gb_synapse's run state) when a gb-synapse server is
    running. Delegates entirely to gb_synapse.ps() — there is no separate
    greenboost-cli-managed llama-server process anymore. Prefers the
    currently-configured model's server; falls back to the first running
    instance so status/cache commands still work with no settings passed."""
    try:
        gb_synapse = _import_gb_synapse()
    except ImportError:
        return None
    running = gb_synapse.ps()
    if not running:
        return None
    if settings:
        model = _llamaserve_model_name(settings)
        for st in running:
            if st["model"] == model:
                return st["llama_pid"]
    return running[0]["llama_pid"]


def llamacpp_server_status(settings: dict) -> str:
    pid = _llamacpp_running_pid(settings)
    if not pid:
        return "stopped"
    try:
        req = urllib.request.Request(_llamacpp_base_url(settings).rstrip("/") + "/models")
        with urllib.request.urlopen(req, timeout=2):
            return "running"
    except Exception:
        return "starting"


def cmd_llamaserve(args: str, _session, settings) -> bool:
    """Start, stop, or show status of gb-synapse.

    Usage:
      /llamaserve            — start with configured model (background)
      /llamaserve start      — same as above
      /llamaserve stop       — stop the running server
      /llamaserve status     — show whether it's running
      /llamaserve logs [N]   — tail last N lines of the log (default 40)
      /llamaserve restart    — stop then start
      /llamaserve <model_id> — start with a specific model (manifest name,
                                org/repo[:quant], or an Ollama model name)

    gb-synapse is the only backend — this talks to it directly, giving you
    disk-persisted prompt-cache reuse (see /llamacache) and cluster
    distribution across a connected feeder.
    """
    try:
        gb_synapse = _import_gb_synapse()
    except ImportError as e:
        emit_err(f"Cannot import gb_synapse from {_GB_SRC}: {e}")
        emit_info(f"Fix: {gb_root_hint()}.")
        return True

    subcmd = args.strip()
    model_for_status = _llamaserve_model_name(settings)

    if subcmd == "status":
        pid = _llamacpp_running_pid(settings)
        if pid:
            emit_ok(f"gb-synapse is running  (pid {pid})")
            console.print(f"  [{GRAY}]log:[/]  {gb_synapse.log_path(model_for_status)}")
            console.print(f"  [{GRAY}]slots:[/]  {_LLAMACPP_SLOT_DIR}")
        else:
            emit_warn("gb-synapse is not running.")
            emit_info("Start it:  /llamaserve")
        return True

    if subcmd.startswith("logs"):
        n = 40
        parts = subcmd.split(None, 1)
        if len(parts) > 1:
            try:
                n = int(parts[1])
            except ValueError:
                pass
        log_file = gb_synapse.log_path(model_for_status)
        if not log_file.exists():
            emit_warn(f"No log file at {log_file}")
            return True
        lines = log_file.read_text(errors="replace").splitlines()
        console.print(f"\n[{GRAY}]── gb-synapse log (last {n} lines) ───────────────────────────[/]")
        for ln in lines[-n:]:
            col = (LIME if "server is listening" in ln
                   else AMBER if "ERROR" in ln or "error" in ln.lower()
                   else GRAY)
            console.print(f"[{col}]{ln}[/]")
        console.print(f"[{GRAY}]── end ──────────────────────────────────────────────────────────[/]\n")
        return True

    if subcmd == "stop":
        if not gb_synapse.stop(model_for_status):
            emit_warn("gb-synapse is not running.")
        else:
            emit_ok("Stopped gb-synapse")
        return True

    if subcmd == "restart":
        cmd_llamaserve("stop", _session, settings)
        import time as _time; _time.sleep(1)
        cmd_llamaserve("start", _session, settings)
        return True

    if _llamacpp_running_pid(settings):
        emit_warn("gb-synapse is already running.  Use /llamaserve stop first.")
        return True

    model = model_for_status
    if subcmd and subcmd not in ("start",):
        model = subcmd
    if not model:
        emit_err("No model configured. Run /model <name> first, or /setup.")
        return True

    # Free VRAM held by Ollama before gb-synapse's llama-server claims it.
    ollama_suspend_for_gpu_task(reason="llama-server")

    try:
        state = gb_synapse.serve(
            model,
            ctx=int(settings.get("llamacpp_n_ctx") or 0),
            n_slots=int(settings.get("llamacpp_np") or -1),
            extra_args=settings.get("llamacpp_extra_args", ""))
    except Exception as e:
        emit_err(f"Could not start gb-synapse: {e}")
        return True

    emit_ok(f"gb-synapse started (pid {state.llama_pid})")
    console.print(f"  [{GRAY}]model:[/]  [{VIOLET}]{state.model}[/]")
    if state.ctx:
        console.print(f"  [{GRAY}]ctx:  [/]  [{GRAY}]{state.ctx:,} tokens"
                       f"{f', kv={state.kv_type}' if state.kv_type else ''}"
                       f"{f', ngl={state.n_gpu_layers}' if state.n_gpu_layers else ''}[/]")
    console.print(f"  [{GRAY}]url:  [/]  [{GRAY}]http://localhost:{state.port}/v1[/]")
    console.print(f"  [{GRAY}]log:  [/]  [{GRAY}]{gb_synapse.log_path(state.model)}[/]")
    if state.feeders:
        console.print(f"  [{GRAY}]feeders:[/] [{GRAY}]{', '.join(state.feeders)}"
                       f"  (tensor-split {state.tensor_split})[/]")
    console.print()
    emit_info("Check: /llamaserve status   (cross-restart cache persistence: see /llamacache help)")
    return True


# ── /llamacache — disk-persisted prompt-cache (slot save/restore) ──────────

def _llamacache_warmest_slot(base: str) -> int:
    """Pick the slot actually holding the longest cached prompt.

    llama-server load-balances requests across `-np` slots (4 by default);
    the warm context is NOT necessarily slot 0 — confirmed empirically: a
    test run landed entirely on slot 3 while slots 0-2 stayed idle the
    whole time. GET /slots only returns an `n_prompt_tokens` field on
    slots that have processed a request, so the max of that field across
    all slots is the one worth saving/restoring. Falls back to 0 if no
    slot has ever been used (nothing to save yet, harmless no-op)."""
    try:
        with urllib.request.urlopen(f"{base}/slots", timeout=10) as r:
            slots = json.loads(r.read())
        best_id, best_tokens = 0, -1
        for s in slots:
            n = s.get("n_prompt_tokens", -1)
            if n is not None and n > best_tokens:
                best_id, best_tokens = s.get("id", 0), n
        return best_id
    except Exception:
        return 0


def _llamacache_slot_action(action: str, key: str, settings: dict) -> "dict | None":
    """POST /slots/{id}?action=<save|restore|erase> against the running
    llama-server, auto-selecting the slot that actually holds the warm
    context (see _llamacache_warmest_slot). Returns the parsed JSON
    response, or None on failure (e.g. server not running) — caller emits
    the warning."""
    base = _llamacpp_base_url(settings).rsplit("/v1", 1)[0]
    slot_id = _llamacache_warmest_slot(base)
    url = f"{base}/slots/{slot_id}?action={action}"
    body = json.dumps({"filename": f"{key}.bin"}).encode() if action != "erase" else b"{}"
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _llamacache_key(settings: dict) -> str:
    """Default cache key — hash of (model, GGUF size, GGUF mtime) so:
    - Different models never collide.
    - A re-quantized GGUF with the same model name gets a fresh key, preventing
      a stale-format slot file from being reused as a cache hit.

    Falls back to hashing the model name alone when the GGUF path cannot be
    resolved (non-path model IDs, Ollama model not pulled, etc.) — harmless,
    preserves the original behaviour for non-file-backed models.
    """
    import hashlib
    model = settings.get("model", "default")
    blob_path = _resolve_gguf_path(model)
    try:
        p = Path(blob_path)
        if p.is_file():
            st = p.stat()
            key_str = f"{model}\0{st.st_size}\0{st.st_mtime_ns}"
            return hashlib.sha1(key_str.encode()).hexdigest()[:16]
    except Exception:
        pass
    return hashlib.sha1(model.encode()).hexdigest()[:16]


# ── Register into command table ────────────────────────────────────────────

def _register() -> None:
    from greenboost_cli.terminal.commands import COMMAND_TABLE
    COMMAND_TABLE["llamaserve"] = cmd_llamaserve


_register()
