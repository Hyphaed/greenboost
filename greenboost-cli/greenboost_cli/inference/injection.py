"""Text-based tool injection for GGUFs served by gb-synapse.

The implementation now lives in the greenboost repo as `gb_synapse_tools.py`
(beside gb_synapse — tool-calling support is a property of the served model, so
it belongs with the serving layer). This module is a thin re-export, the same
`_GB_SRC` bootstrap pattern backend_cmds.py uses for gb_synapse. If the greenboost
checkout isn't importable a minimal local fallback keeps `should_inject_tools`
working so the CLI still runs.

Public API (unchanged):
  should_inject_tools, inject_tools_into_system, messages_to_openai_for_injection,
  parse_tool_calls_from_text, clean_text_from_tool_blocks
"""
from __future__ import annotations

import sys

from greenboost_cli.gb_paths import gb_py_root

_GB_SRC = gb_py_root()
if str(_GB_SRC) not in sys.path:
    sys.path.insert(0, str(_GB_SRC))

try:
    from gb_synapse_tools import (  # noqa: F401
        should_inject_tools,
        inject_tools_into_system,
        messages_to_openai_for_injection,
        parse_tool_calls_from_text,
        clean_text_from_tool_blocks,
    )
except Exception:  # pragma: no cover - greenboost checkout absent
    # Minimal fallback: never inject (assume native FC), pass text through.
    def should_inject_tools(settings: dict, model: str = "") -> bool:
        fmt = settings.get("tool_format", "auto")
        if fmt == "inject":
            return True
        if fmt == "native":
            return False
        return not settings.get("gb_synapse_native_fc", True)

    def inject_tools_into_system(system: str, tool_schemas: list, model: str = "") -> str:
        return system

    def messages_to_openai_for_injection(messages: list) -> list:
        return [{"role": m["role"], "content": m.get("content", "")}
                for m in messages if m.get("role") in ("user", "assistant")]

    def parse_tool_calls_from_text(text, tool_schemas=None, model: str = "") -> list:
        return []

    def clean_text_from_tool_blocks(text: str) -> str:
        return text
