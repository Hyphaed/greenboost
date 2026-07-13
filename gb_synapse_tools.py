# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_synapse_tools.py — text-based tool-call injection for GGUFs served by
gb-synapse that don't reliably emit native OpenAI-style tool_calls.

Whether a served GGUF can do native function-calling is a property of the
*model* (its chat template + quant), not of the client — so it belongs beside
the serving layer (gb_synapse), the same way llama-server orchestration was
lifted here. Any gb-synapse consumer (greenboost-cli, ai-forge's LLM steps, a
future agent) imports this instead of carrying its own copy.

Pure stdlib (json/re/subprocess/uuid); import-safe from any process, no torch.
greenboost-cli/greenboost_cli/inference/injection.py is a thin re-export wrapper
over this module (see backend_cmds.py's _import_gb_synapse pattern).

Public API:
  should_inject_tools(settings, model)             → bool
  inject_tools_into_system(system, schemas, model) → str
  messages_to_openai_for_injection(messages)       → list
  parse_tool_calls_from_text(text, schemas, model) → list[{id, name, input}]
  clean_text_from_tool_blocks(text)                → str
"""
from __future__ import annotations

import json
import re
import subprocess
import uuid

# PARSER values (from `ollama show <model> --modelfile`'s literal `PARSER`
# line) that use the XML tool-call format below instead of generic
# Hermes-JSON. "nemotron-3-nano" delegates its tool-call parsing entirely to
# Qwen3CoderParser per Ollama's own model/parsers/nemotron3nano.go — it is NOT
# a Qwen model, but it emits the identical <tool_call><function=NAME>
# <parameter=KEY> wire format, so it belongs in this set.
_XML_TOOLCALL_PARSERS = {"qwen3.5", "qwen3-coder", "nemotron-3-nano"}

# Name-substring fallback for when a model isn't Ollama-sourced (raw HF/vLLM
# pull) or `ollama show` fails — same families as _XML_TOOLCALL_PARSERS,
# matched by naming convention instead of the real resolved PARSER value.
_XML_TOOLCALL_NAME_HINTS = (
    "qwen35", "qwen36", "qwen3.5", "qwen3.6", "claude-coder",
    "nemotron-3-nano", "nemotron3nano",
)

_parser_name_cache: dict[str, str] = {}


def _ollama_parser_name(model: str) -> str:
    """Read the real PARSER value Ollama resolved for this model from its
    Modelfile (`ollama show <model> --modelfile` prints a literal `PARSER
    <name>` line) — a direct, non-fragile signal instead of guessing the
    tool-call format from the model name string. Cached per model name since
    this is called on every turn and a subprocess call per turn would add real
    per-request latency. Returns "" on any failure (not an Ollama model, ollama
    not running/installed, no PARSER line) so callers fall back to name-based
    heuristics."""
    if not model:
        return ""
    if model in _parser_name_cache:
        return _parser_name_cache[model]
    parser = ""
    try:
        result = subprocess.run(
            ["ollama", "show", model, "--modelfile"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            m = re.search(r"^PARSER\s+(\S+)", result.stdout, re.MULTILINE)
            if m:
                parser = m.group(1).strip()
    except Exception:
        pass
    _parser_name_cache[model] = parser
    return parser


def _is_qwen35_family(model: str) -> bool:
    """True for models that need the Qwen3.5/Qwen3-Coder XML tool-call format
    (<tool_call><function=NAME><parameter=KEY>value</parameter></function>
    </tool_call>) instead of the generic Hermes-JSON format.

    Two-tier check: prefer the real PARSER value Ollama resolved for this model
    (_ollama_parser_name) over guessing from the name string; falls back to
    name-substring matching only when that lookup fails (not an Ollama-sourced
    model, or ollama unavailable).

    Native tool-calling for the qwen3.5/qwen35moe architecture is a confirmed,
    still-open upstream llama.cpp bug as of 2026-07 (ggml-org/llama.cpp #19905,
    #19872: the Jinja template calls the `|items` filter on tool arguments
    passed as a string instead of a dict/map) — --jinja's own grammar sampling
    can't fix this, the bug is in template rendering itself."""
    parser = _ollama_parser_name(model)
    if parser:
        return parser in _XML_TOOLCALL_PARSERS
    m = model.lower()
    return any(tag in m for tag in _XML_TOOLCALL_NAME_HINTS)


def should_inject_tools(settings: dict, model: str = "") -> bool:
    """Return True if tool definitions should be injected into the system
    prompt instead of relying on native OpenAI-style function-calling.

    settings["tool_format"] explicit override ("inject"/"native") always wins.
    Next, the Qwen3.5/3.6 family (see _is_qwen35_family) always uses injection —
    their native template is broken upstream, not a matter of per-user trust.
    Otherwise settings["gb_synapse_native_fc"] (default True) controls whether
    the currently-served model is trusted to emit native tool_calls reliably —
    set it False for older/injection-only GGUFs.
    """
    fmt = settings.get("tool_format", "auto")
    if fmt == "inject":
        return True
    if fmt == "native":
        return False
    if _is_qwen35_family(model):
        return True
    return not settings.get("gb_synapse_native_fc", True)


# ── System prompt injection ────────────────────────────────────────────────

_INJECTION_PREAMBLE = """\


---
## Tool Use

You have access to the following tools. To call a tool, output **only** a fenced JSON block
(no extra commentary inside the block):

```json
{{"name": "<tool_name>", "arguments": {{"param1": "val1", "param2": "val2"}}}}
```

Alternatively you may use Hermes-style tags (accepted equally):

<tool_call>
{{"name": "<tool_name>", "arguments": {{...}}}}
</tool_call>

Call tools one at a time. After each tool call you will receive the result and may continue.
Do NOT include any text inside the JSON block — tool name and arguments only.

### Available Tools (JSON schema)

```json
{tools_json}
```

### Tool Descriptions

{tool_descriptions}
---
"""


def _schema_to_description(schema: dict) -> str:
    name  = schema["name"]
    desc  = schema.get("description", "")
    props = schema.get("input_schema", {}).get("properties", {})
    req   = set(schema.get("input_schema", {}).get("required", []))
    lines = [f"**{name}** — {desc}"]
    for pname, pdata in props.items():
        r   = " *(required)*" if pname in req else ""
        typ = pdata.get("type", "any")
        pd  = pdata.get("description", "")
        lines.append(f"  - `{pname}` ({typ}){r}: {pd}")
    return "\n".join(lines)


# ── Qwen3.5/3.6 native XML tool-call format ─────────────────────────────────
#
# Ollama's own "qwen3.5" renderer/parser wiring uses the Hermes-JSON format
# above (_INJECTION_PREAMBLE) — but per ollama/ollama#14493 (still open,
# 2026-07), that's a bug: Qwen 3.5 was trained on the Qwen3-Coder XML format
# (<function=name><parameter=key>value</parameter></function>). This gives
# gb-synapse the format Ollama *should* be using for this family.

_XML_INJECTION_PREAMBLE = """


# Tools

You have access to the following functions:

{tools_xml}

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""


def _xml_tool_definitions(tool_schemas: list) -> str:
    lines = ["<tools>"]
    for s in tool_schemas:
        props = s.get("input_schema", {}).get("properties", {})
        lines.append("<function>")
        lines.append(f"<name>{s['name']}</name>")
        if s.get("description"):
            lines.append(f"<description>{s['description']}</description>")
        lines.append("<parameters>")
        for pname, pdata in props.items():
            lines.append("<parameter>")
            lines.append(f"<name>{pname}</name>")
            ptype = pdata.get("type")
            if ptype:
                lines.append(f"<type>{ptype}</type>")
            if pdata.get("description"):
                lines.append(f"<description>{pdata['description']}</description>")
            lines.append("</parameter>")
        lines.append("</parameters>")
        lines.append("</function>")
    lines.append("</tools>")
    return "\n".join(lines)


def inject_tools_into_system(system: str, tool_schemas: list, model: str = "") -> str:
    """Return system prompt with tool definitions appended. Qwen3.5/3.6 family
    gets its native XML format (see above); everything else gets the generic
    Hermes-JSON format most other injection-only GGUFs expect."""
    if _is_qwen35_family(model):
        block = _XML_INJECTION_PREAMBLE.format(tools_xml=_xml_tool_definitions(tool_schemas))
        return system.rstrip() + block

    tools_json = json.dumps(
        [
            {
                "name":        s["name"],
                "description": s.get("description", ""),
                "parameters":  s.get("input_schema", {}),
            }
            for s in tool_schemas
        ],
        ensure_ascii=False,
        indent=2,
    )
    descriptions = "\n\n".join(_schema_to_description(s) for s in tool_schemas)
    block = _INJECTION_PREAMBLE.format(
        tools_json=tools_json,
        tool_descriptions=descriptions,
    )
    return system.rstrip() + block


# ── Message format for injection mode ─────────────────────────────────────

def messages_to_openai_for_injection(messages: list) -> list:
    """
    Convert neutral messages → OpenAI format adapted for injection mode.
    Tool results become user messages because Ollama won't accept role:tool
    when no tools were declared in the API request.
    """
    result = []
    for m in messages:
        role = m["role"]
        if role == "user":
            result.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            result.append({"role": "assistant", "content": m.get("content") or ""})
        elif role == "tool":
            name    = m.get("name", "tool")
            content = m.get("content", "")
            result.append({
                "role":    "user",
                "content": f"[Tool result — `{name}`]\n{content}",
            })
    return result


# ── Qwen3.5/3.6 native XML tool-call parsing ────────────────────────────────

_QWEN_FUNCTION_RE  = re.compile(r"<function=([^>]+)>([\s\S]*?)</function>")
_QWEN_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>([\s\S]*?)</parameter>")
# Fallback when the model doesn't close with the exact `</function>` tag —
# confirmed live (2026-07-13, satgeze/qwen36-35b-uncensored-1m): it emitted
# `</parameters>` (plural, no matching `<function>` open) instead, dropping
# the whole tool call silently since _QWEN_FUNCTION_RE requires a literal
# `</function>`. `raw` here is already bounded by the outer `<tool_call>...
# </tool_call>` match, so once the opening `<function=NAME>` tag is found,
# whatever comes after it — to any of the plausible near-miss closers, or
# just the end of `raw` — is a safe body to scan for <parameter=...> pairs.
_QWEN_FUNCTION_OPEN_RE = re.compile(r"<function=([^>]+)>")
_QWEN_FUNCTION_CLOSE_RE = re.compile(r"</function>|</parameters>")


def _coerce_qwen_value(raw: str, ptype: "str | None"):
    """Mirrors ollama's Qwen3CoderParser.parseValue() precedence: null →
    boolean → integer → number → array → object → string, trimming a single
    leading/trailing newline first (not a full strip — multi-line values are
    valid and their internal whitespace must survive)."""
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    if raw.lower() == "null":
        return None
    if not ptype:
        return raw
    if ptype == "boolean":
        return raw.lower() == "true"
    if ptype == "integer":
        try:
            return int(raw)
        except ValueError:
            return raw
    if ptype == "number":
        try:
            f = float(raw)
            return int(f) if f.is_integer() else f
        except ValueError:
            return raw
    if ptype in ("array", "object"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _parse_qwen_xml_tool_call(raw: str, tool_schemas: list) -> "dict | None":
    """Parse a single <tool_call>...</tool_call> block's native Qwen3.5/3.6
    content: <function=NAME><parameter=KEY>VALUE</parameter>...</function>.
    Ported from ollama's parseToolCall()/transformToXML() — regex-based rather
    than transform-to-XML-then-unmarshal since Python's stdlib XML parser adds
    no value here and this avoids the escaping edge cases that approach has to
    work around."""
    m = _QWEN_FUNCTION_RE.search(raw)
    if m:
        name = m.group(1).strip()
        body = m.group(2)
    else:
        # No exact `</function>` close — fall back to the open tag plus
        # whatever's left of `raw` (itself already bounded by the outer
        # <tool_call> match), trimmed at the first near-miss closer if one
        # is present so a stray `</parameters>` etc. doesn't leak into body.
        om = _QWEN_FUNCTION_OPEN_RE.search(raw)
        if not om:
            return None
        name = om.group(1).strip()
        rest = raw[om.end():]
        cm = _QWEN_FUNCTION_CLOSE_RE.search(rest)
        body = rest[:cm.start()] if cm else rest

    schema = next((s for s in tool_schemas if s.get("name") == name), None)
    props = (schema or {}).get("input_schema", {}).get("properties", {})

    args = {}
    for pm in _QWEN_PARAMETER_RE.finditer(body):
        pname = pm.group(1).strip()
        args[pname] = _coerce_qwen_value(pm.group(2), props.get(pname, {}).get("type"))

    return {"id": f"call_{uuid.uuid4().hex[:8]}", "name": name, "input": args, "_raw": raw}


# ── Tool call parsing ──────────────────────────────────────────────────────

def parse_tool_calls_from_text(text: str, tool_schemas: "list | None" = None, model: str = "") -> list[dict]:
    """
    Parse tool calls from model text output.  Handles formats:
      1. Fenced ```json {"name":...,"arguments":...} ```
      2. <tool_call>{"name":...,"arguments":...}</tool_call>   (Qwen3/Hermes)
      3. <|tool_call_begin|>...<|tool_call_end|>               (GLM-4)
      4. ✿FUNCTION✿...✿RESULT✿                                 (some GLM variants)
      5. Inline bare JSON (fallback, only when nothing found above)

    Qwen3.5/3.6 family uses a different, XML-flavored <tool_call> body
    (<function=NAME><parameter=KEY>VALUE</parameter></function>) — see
    _parse_qwen_xml_tool_call and the module-level comment above it.
    """
    if _is_qwen35_family(model):
        calls: list[dict] = []
        seen:  set[str]   = set()
        for m in re.finditer(r"<tool_call>([\s\S]*?)</tool_call>", text):
            _absorb(_parse_qwen_xml_tool_call(m.group(1).strip(), tool_schemas or []), calls, seen)
        return calls

    calls: list[dict] = []
    seen:  set[str]   = set()

    for m in re.finditer(r"<\|tool_call_begin\|>([\s\S]*?)<\|tool_call_end\|>", text):
        _absorb(_try_parse(m.group(1).strip()), calls, seen)

    for m in re.finditer(r"✿FUNCTION✿([\s\S]*?)✿RESULT✿", text):
        _absorb(_try_parse(m.group(1).strip()), calls, seen)

    for m in re.finditer(r"<tool_call>([\s\S]*?)</tool_call>", text):
        _absorb(_try_parse(m.group(1).strip()), calls, seen)

    for m in re.finditer(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text):
        raw = m.group(1).strip()
        if re.search(r'"(?:name|tool_name)"\s*:', raw):
            _absorb(_try_parse(raw), calls, seen)

    if not calls:
        for m in re.finditer(
            r'\{\s*"(?:name|tool_name)"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|input|parameters|args)"\s*:',
            text,
        ):
            _absorb(_try_parse_from_pos(text, m.start()), calls, seen)

    return calls


def _try_parse(raw: str) -> "dict | None":
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            repaired = re.sub(r",\s*([}\]])", r"\1", raw)
            data = json.loads(repaired)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    return _normalize(data, raw)


def _try_parse_from_pos(text: str, start: int) -> "dict | None":
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return _try_parse(text[start : i + 1])
    return None


def _normalize(data: dict, raw: str) -> "dict | None":
    name = (
        data.get("name")
        or data.get("tool_name")
        or data.get("function")
        or (data.get("function", {}) or {}).get("name")  # type: ignore[union-attr]
    )
    if not name or not isinstance(name, str):
        return None

    raw_input = (
        data.get("arguments")
        or data.get("input")
        or data.get("parameters")
        or data.get("args")
        or {}
    )
    if isinstance(raw_input, str):
        try:
            raw_input = json.loads(raw_input)
        except json.JSONDecodeError:
            raw_input = {"_raw": raw_input}
    if not isinstance(raw_input, dict):
        raw_input = {}

    return {
        "id":    data.get("id") or f"call_{uuid.uuid4().hex[:8]}",
        "name":  name,
        "input": raw_input,
        "_raw":  raw,
    }


def _absorb(call: "dict | None", calls: list, seen: set) -> None:
    if call is None:
        return
    key = call.get("_raw", "")
    if key in seen:
        return
    seen.add(key)
    calls.append({k: v for k, v in call.items() if k != "_raw"})


# ── Text cleaning ──────────────────────────────────────────────────────────

_STRIP_PATTERNS = [
    re.compile(r"<\|tool_call_begin\|>[\s\S]*?<\|tool_call_end\|>"),
    re.compile(r"✿FUNCTION✿[\s\S]*?✿RESULT✿"),
    re.compile(r"<tool_call>[\s\S]*?</tool_call>"),
    re.compile(r"```(?:json)?\s*\n?[\s\S]*?\n?```"),
]


def clean_text_from_tool_blocks(text: str) -> str:
    """Strip tool call blocks from model output, returning only prose text."""
    for pat in _STRIP_PATTERNS:
        text = pat.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
