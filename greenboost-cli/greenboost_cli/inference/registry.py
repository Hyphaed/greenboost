"""Backend registry: gb-synapse config and credential lookup.

GreenBoost's gb-synapse is the ONLY inference backend greenboost-cli talks
to (see workflow/gb-synapse.md in the sibling ~/Dev/greenboost_all/greenboost
repo). Every model is a GGUF served locally, or cluster-distributed across a
connected feeder, by gb-synapse — HuggingFace-pulled or Ollama-indexed.
There is no multi-provider routing here anymore.
"""
from __future__ import annotations

import os

BACKEND_REGISTRY: dict[str, dict] = {
    "gb-synapse": {
        "type":        "openai",
        "api_key_env": None,
        "api_key":     "",
        "base_url":    "http://localhost:11434/v1",
        # Display/tab-complete only — the actual catalog is whatever's in the
        # manifest (`greenboost synapse list`). Run /llamaserve status to
        # probe the live server.
        "models":      [],
        "description": ("GreenBoost gb-synapse — HuggingFace-pulled + Ollama-indexed "
                         "GGUFs, cluster-distributed via llama.cpp RPC (--tensor-split "
                         "host+feeder). Serve with: /llamaserve <model>"),
    },
}


def get_credentials(backend_name: str, settings: dict) -> str:
    """Return API key for a backend. Always empty for gb-synapse (no auth
    needed) — kept as a stable call site rather than special-cased away."""
    backend = BACKEND_REGISTRY.get(backend_name, {})
    cfg_key = settings.get(f"{backend_name}_api_key", "")
    if cfg_key:
        return cfg_key
    env_var = backend.get("api_key_env")
    if env_var:
        return os.environ.get(env_var, "")
    return backend.get("api_key", "")
