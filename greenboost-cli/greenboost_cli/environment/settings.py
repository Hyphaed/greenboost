"""Settings management for GreenBoost CLI."""
from __future__ import annotations

import os
import json
from pathlib import Path

GB_HOME       = Path.home() / ".greenboost_cli"
SETTINGS_PATH = GB_HOME / "config.json"
HISTORY_PATH  = GB_HOME / "input_history.txt"
SESSIONS_PATH = GB_HOME / "sessions"

DEFAULT_SETTINGS: dict = {
    # Empty until the setup wizard (or /model) picks one from the gb-synapse
    # manifest — see wizard/setup.py.
    "model":           "",
    "max_tokens":       8192,    # output completion tokens (NOT context window)
    "context_window":   0,       # 0 = auto-detect (gb_synapse_ctx); set >0 to override
    # Native function-calling for the currently-served GGUF. Native FC support
    # varies a lot across GGUF quants/chat templates; set False for
    # older/injection-only models. See inference/injection.py.
    "gb_synapse_native_fc": True,
    # Context compaction thresholds (fractions of context_window):
    #   0.75 → soft start (48K of 64K): compact oldest history into structured memory
    #   0.875 → aggressive (56K of 64K): more aggressive compaction
    # Plan mode hands straight to implementation once the plan file is
    # written. Typing "implement the plan" carried no decision — the plan
    # was already on disk and the user asked for it by entering plan mode.
    "plan_auto_implement":    True,
    "auto_compact_pct":       0.75,
    "auto_compact_hard_pct":  0.875,
    "permission_mode": "auto",    # auto | accept-all | manual
    "verbose":         False,
    "qwen_thinking":   False,     # True = enable Qwen3 <think> blocks; False = think:false + /nothink
    "tool_format":     "auto",
    "active_project":  None,      # per-project brain
    "diffusion_model": "klein-fp8",
    "diffusion_output_dir": None,
    "design_assets_dir": None,
    "design_skill_dir": None,    # path to ui-ux-pro-max-skill CSV data (or use $GB_DESIGN_SKILL_DIR)
    "dashboard_port":  7821,
    "rag_embed_model": "jinaai/jina-embeddings-v2-base-code",
    "rag_top_k":       5,
    "rag_min_score":   0.1,
    "rag_auto_update_on_start": True,  # incremental RAG refresh for cwd project on REPL start
    "greenboost_turboquant": False,
    "greenboost_inject_context": True,
    # Auto-run `sudo -n greenboost turboquant on` at REPL startup when
    # /dev/greenboost is present.  Idempotent — no-op if already enabled.
    "gb_auto_turboquant": True,
    # Max system-context characters. Keeps the initial prefill small on
    # PCIe-bound overflow models. ~10 k chars ≈ 2500 tokens.
    "local_sys_ctx_chars": 10000,
    # gb-synapse (llama-server) tuning — see gb_synapse.serve()
    # 0 = no override: gb-synapse solves ctx jointly against the live VRAM
    # budget (gb_synapse._solve_ctx_and_layers). Set >0 to EXPLICITLY request
    # a window (e.g. 262144, the model's native size) — that is also what
    # gates the opt-in T2 KV extension when the shim is active; a nonzero
    # default here would silently turn "explicit request" into "always", so
    # 0 is the only safe default. 65536 was a guessed constant that was never
    # actually sent to the server (serve() ignored ctx=0 the same way) — it
    # only misled gb_synapse_ctx()'s client-side bookkeeping, now fixed to
    # probe the live server instead (see gb_synapse_ctx()).
    "llamacpp_n_ctx":       0,      # context window (0 = auto)
    "llamacpp_np":          1,      # parallel slots; 1 = no KV split for single-user
    "llamacpp_extra_args":  "",     # extra flags on top of gb-synapse's defaults
                                     # (--jinja is now always on — needed for correct
                                     # native tool-calling); e.g. "--reasoning-budget 0"
    "gb_t1_alert_pct": 90,
    "gb_t2_alert_pct": 85,
    "auto_rag":        True,   # feed every Q&A turn into the local RAG automatically
    # Skill auto-discovery: scan ~/.claude-accounts/*/skills/, ~/.claude/skills/,
    # ~/Dev/claude_workflow/commands/ in addition to settings["skills_dir"].
    "skills_auto_discover": True,
}


def load_settings() -> dict:
    GB_HOME.mkdir(exist_ok=True)
    SESSIONS_PATH.mkdir(exist_ok=True)
    (GB_HOME / "projects").mkdir(exist_ok=True)
    cfg = dict(DEFAULT_SETTINGS)

    if SETTINGS_PATH.exists():
        try:
            cfg.update(json.loads(SETTINGS_PATH.read_text()))
        except Exception:
            pass

    return cfg


def save_settings(cfg: dict) -> None:
    """Persist `cfg`, excluding underscore-prefixed keys (e.g. `_backend`,
    `_cancel_event`, `_loaded_skills`) — runtime-only state several call
    sites stash on the shared settings dict for convenience, not meant to
    survive a restart. Some of those values (like a threading.Event) aren't
    even JSON-serializable."""
    GB_HOME.mkdir(exist_ok=True)
    persisted = {k: v for k, v in cfg.items() if not k.startswith("_")}
    SETTINGS_PATH.write_text(json.dumps(persisted, indent=2))


_GB_SYNAPSE_CTX_CACHE: dict = {"val": 0, "ts": 0.0}
_GB_SYNAPSE_CTX_TTL_S = 10.0


def invalidate_gb_synapse_ctx_cache() -> None:
    """Force the next gb_synapse_ctx() call to re-probe the server. Call this
    after a 400 "exceeds the available context size" — the cached value is
    proven wrong at that point, waiting out the TTL just delays recovery."""
    _GB_SYNAPSE_CTX_CACHE["ts"] = 0.0


def gb_synapse_ctx(settings: dict) -> int:
    """Context-window size actually being served right now, for auto-compact
    thresholds and /status display.

    This used to return a hardcoded 65536 guess and never contact the
    server — every compaction threshold in the CLI was a fraction of a
    number that had nothing to do with reality, so compaction could never
    fire before a real 400 (confirmed live: server served ctx=7680, this
    returned 65536). Fixed to ask the server directly, short-TTL cached
    since every caller (repl.py toolbar + pre-send check, intelligence.py
    compaction, orchestrator.py overflow guard, /status) calls this often.

    Priority: live GET /slots (n_ctx, ground truth) → the gb-synapse
    run-state JSON's own `ctx` field (gb_synapse.ServerState, set at serve
    time) → settings["llamacpp_n_ctx"] if the user explicitly requested a
    window → 65536 as the last-resort fallback when nothing is running yet.
    """
    import time as _time
    now = _time.monotonic()
    if now - _GB_SYNAPSE_CTX_CACHE["ts"] < _GB_SYNAPSE_CTX_TTL_S and _GB_SYNAPSE_CTX_CACHE["val"]:
        return _GB_SYNAPSE_CTX_CACHE["val"]

    ctx = _probe_synapse_slots_ctx()
    if not ctx:
        ctx = _read_run_state_ctx(settings)
    if not ctx:
        configured = settings.get("llamacpp_n_ctx")
        ctx = int(configured) if configured else 0
    if not ctx:
        ctx = 65536

    _GB_SYNAPSE_CTX_CACHE["val"] = ctx
    _GB_SYNAPSE_CTX_CACHE["ts"] = now
    return ctx


def _probe_synapse_slots_ctx() -> int:
    """GET /slots on the gb-synapse proxy — raw llama.cpp passthrough,
    verified live to report the real per-slot n_ctx (e.g. 7680 today, wrong
    before this fix). Short timeout: this runs on the UI/turn-prep path and
    must never make a hung server stall the REPL."""
    try:
        import httpx
        port = os.environ.get("GB_SYNAPSE_PORT", "11369")
        r = httpx.get(f"http://127.0.0.1:{port}/slots", timeout=0.5)
        if r.status_code == 200:
            slots = r.json()
            if slots and isinstance(slots, list):
                n_ctx = slots[0].get("n_ctx")
                if n_ctx:
                    return int(n_ctx)
    except Exception:
        pass
    return 0


def _read_run_state_ctx(settings: dict) -> int:
    """Fallback when /slots isn't reachable (proxy down, port unknown) but a
    run-state JSON exists — gb_synapse.ServerState.ctx, written at serve time
    (see gb_synapse.py's _launch_proxy_and_record)."""
    try:
        model = settings.get("model") or ""
        if not model:
            return 0
        run_dir = Path("/run/greenboost/synapse")
        safe_name = model.replace("/", "_").replace(":", "_")
        state_path = run_dir / f"{safe_name}.json"
        if state_path.exists():
            data = json.loads(state_path.read_text())
            return int(data.get("ctx") or 0)
    except Exception:
        pass
    return 0
