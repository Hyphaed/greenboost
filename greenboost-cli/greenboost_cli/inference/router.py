"""Backend routing: gb-synapse-only entry point, streaming event types."""
from __future__ import annotations

from typing import Generator


# ── Streaming event types ──────────────────────────────────────────────────

class StreamFragment:
    """A chunk of streamed text from the model."""
    def __init__(self, text: str):
        self.text = text


class ReasoningFragment:
    """A chunk of extended thinking / reasoning from the model."""
    def __init__(self, text: str):
        self.text = text


class CompletedResponse:
    """A fully completed assistant turn, including any tool calls."""
    def __init__(self, text: str, tool_calls: list, in_tokens: int, out_tokens: int):
        self.text       = text
        self.tool_calls = tool_calls   # list of {id, name, input}
        self.in_tokens  = in_tokens
        self.out_tokens = out_tokens


# ── Provider routing helpers ───────────────────────────────────────────────
#
# There is only one backend (gb-synapse) — these two helpers exist so model
# strings can still be written as "gb-synapse/qwen3-coder" or bare
# "qwen3-coder" interchangeably, and so callers don't need a special case
# for "is there a prefix or not".

def resolve_backend(model: str) -> str:
    """Every model resolves to gb-synapse — there is no other backend."""
    return "gb-synapse"


def strip_prefix(model: str) -> str:
    """Remove a leading 'gb-synapse/' prefix, if present."""
    if model.startswith("gb-synapse/"):
        return model[len("gb-synapse/"):]
    return model


# ── Unified streaming entry point ──────────────────────────────────────────

def generate(
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    settings: dict,
) -> Generator:
    """
    Unified streaming entry point. Always talks to gb-synapse (OpenAI-
    compatible, served by llama-server behind gb_synapse_api.py's proxy).
    Yields: StreamFragment | ReasoningFragment | CompletedResponse
    """
    from greenboost_cli.inference.registry import BACKEND_REGISTRY, get_credentials
    from greenboost_cli.inference.adapters import _stream_openai_api, _stream_injected
    from greenboost_cli.inference.injection import should_inject_tools

    backend_name = "gb-synapse"
    model_name   = strip_prefix(model)
    backend      = BACKEND_REGISTRY["gb-synapse"]
    api_key      = get_credentials(backend_name, settings)
    base_url     = backend["base_url"]
    settings["_backend"] = backend_name

    if tool_schemas and should_inject_tools(settings, model_name):
        yield from _stream_injected(api_key, base_url, model_name, system, messages, tool_schemas, settings)
    else:
        yield from _stream_openai_api(api_key, base_url, model_name, system, messages, tool_schemas, settings)
