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
    "llamacpp_n_ctx":       65536,  # context window
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


def gb_synapse_ctx(settings: dict) -> int:
    """Context-window size to assume for the currently-served gb-synapse
    model, for CLIENT-SIDE bookkeeping only (auto-compact thresholds,
    /status display) — never sent per-request. llama-server's actual
    context is fixed at server-start time via gb_synapse.serve(ctx=...).

    Priority: settings["llamacpp_n_ctx"] → 65536.
    """
    val = settings.get("llamacpp_n_ctx")
    return int(val) if val else 65536
