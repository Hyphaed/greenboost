"""Streaming adapters and message format converters for gb-synapse (OpenAI-compatible)."""
from __future__ import annotations

import json
from typing import Generator

from greenboost_cli.inference.router import StreamFragment, ReasoningFragment, CompletedResponse


# ── Mid-stream disconnect guard ───────────────────────────────────────────────

_DISCONNECT_MARKERS = (
    "RemoteProtocolError",     # httpx/httpcore: peer closed without a complete body
    "IncompleteRead",
    "ChunkedEncodingError",
    "ConnectionResetError",
    "APIConnectionError",
)


class ContextOverflowError(RuntimeError):
    """The backend rejected the request because it exceeds the server's
    context window (llama-server: 400 "request (N tokens) exceeds the
    available context size (M tokens)"). Distinguished from a generic
    BadRequestError so orchestrator.py can force a compaction and retry once
    instead of surfacing a dead end — see execute_turn's overflow handling.
    Previously every 400 (this one included) became an identical generic
    RuntimeError with no retry, which is exactly the failure the owner hit
    live (8324 tokens against a 7680-token window).

    Carries the server's own numbers when it reported them. They matter
    because compaction CANNOT always fix this: the prompt's fixed overhead
    (system prompt + every connected MCP server's tool schemas) is paid on
    every request and no amount of history trimming touches it. Live
    2026-08-18: ten MCP servers, 238 tools, `n_prompt_tokens: 29507` against
    `n_ctx: 16384` for a thirty-word message and an essentially empty history
    , compacting freed a few hundred tokens, the retry failed identically, and
    the operator got a raw 400. Knowing prompt vs window lets the retry decide
    whether it is worth attempting at all, and lets the message name the real
    cause."""

    def __init__(self, message: str = "", *, prompt_tokens: int = 0,
                 n_ctx: int = 0) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.n_ctx = n_ctx


def _overflow_numbers(err_str: str) -> "tuple[int, int]":
    """(prompt_tokens, n_ctx) from llama-server's 400 body, or (0, 0).

    The body carries them as JSON fields AND in the prose; the fields are
    read first because they are unambiguous.
    """
    import re as _re
    pt = _re.search(r"'n_prompt_tokens'\s*:\s*(\d+)", err_str) or \
         _re.search(r'"n_prompt_tokens"\s*:\s*(\d+)', err_str) or \
         _re.search(r"request \((\d+) tokens\)", err_str)
    nc = _re.search(r"'n_ctx'\s*:\s*(\d+)", err_str) or \
         _re.search(r'"n_ctx"\s*:\s*(\d+)', err_str) or \
         _re.search(r"context size \((\d+) tokens\)", err_str)
    return (int(pt.group(1)) if pt else 0, int(nc.group(1)) if nc else 0)


def _is_context_overflow(err_str: str) -> bool:
    return "exceeds the available context size" in err_str


def guarded_stream(stream_obj, model: str, base_url: str):
    """Yield stream chunks, turning a mid-stream disconnect into an explanation.

    The engine dying *after* the request was accepted (still loading the model,
    OOM, an unsupported GGUF hyperparameter) reaches the client as nothing but
    a truncated chunked body — which surfaced as a raw
    `RemoteProtocolError: peer closed connection` traceback. gb-synapse knows
    what really happened, so ask it.
    """
    it = iter(stream_obj)
    while True:
        try:
            chunk = next(it)
        except StopIteration:
            return
        except Exception as exc:
            name = type(exc).__name__
            if not any(m in name or m in str(exc) for m in _DISCONNECT_MARKERS):
                raise
            reason = ""
            try:
                import gb_synapse
                reason = gb_synapse.failure_report(model)
            except Exception:
                pass
            raise RuntimeError(
                f"\ngb-synapse stopped responding while generating.\n\n"
                + (f"  Reason: {reason}\n\n" if reason else "")
                + f"  • /llamaserve status   — is the model loaded?\n"
                  f"  • /llamaserve logs     — the engine's own error\n"
                  f"  • /llamaserve restart  — bring it back up\n"
            ) from exc
        yield chunk


# ── Thinking-tag filter (for local models that emit <think>…</think>) ──────────

class _ThinkFilter:
    """State machine that splits <think>/<thinking> blocks from prose in a stream.

    Yields (text, is_thinking) pairs with minimal latency — only buffers a
    small lookahead window (~12 chars) to avoid splitting across tag boundaries.
    """

    _OPEN_TAGS  = ("<thinking>", "<think>")
    _CLOSE_TAGS = ("</thinking>", "</think>")
    _LOOKAHEAD  = 12   # max tag len to buffer

    def __init__(self) -> None:
        self._buf       = ""
        self._in_think  = False

    def feed(self, chunk: str) -> list[tuple[str, bool]]:
        self._buf += chunk
        results: list[tuple[str, bool]] = []
        while True:
            if not self._in_think:
                best = None
                for tag in self._OPEN_TAGS:
                    pos = self._buf.find(tag)
                    if pos >= 0 and (best is None or pos < best[0]):
                        best = (pos, tag)
                if best is None:
                    safe = self._buf[: -self._LOOKAHEAD] if len(self._buf) > self._LOOKAHEAD else ""
                    if safe:
                        results.append((safe, False))
                        self._buf = self._buf[len(safe):]
                    break
                pos, tag = best
                if pos > 0:
                    results.append((self._buf[:pos], False))
                self._buf = self._buf[pos + len(tag):]
                self._in_think = True
            else:
                best = None
                for tag in self._CLOSE_TAGS:
                    pos = self._buf.find(tag)
                    if pos >= 0 and (best is None or pos < best[0]):
                        best = (pos, tag)
                if best is None:
                    safe = self._buf[: -self._LOOKAHEAD] if len(self._buf) > self._LOOKAHEAD else ""
                    if safe:
                        results.append((safe, True))
                        self._buf = self._buf[len(safe):]
                    break
                pos, tag = best
                if pos > 0:
                    results.append((self._buf[:pos], True))
                self._buf = self._buf[pos + len(tag):]
                self._in_think = False
        return results

    def flush(self) -> list[tuple[str, bool]]:
        result = [(self._buf, self._in_think)] if self._buf else []
        self._buf = ""
        return result


# ── Tool-call XML filter (injection mode streams raw <tool_call> markup) ───────

class _ToolTagFilter:
    """Withholds Qwen/Hermes-style tool-call XML from the live stream, the
    same way _ThinkFilter withholds <think> blocks.

    Injection mode (gb_synapse_tools.should_inject_tools — the path this
    exact reference model uses) teaches the model to emit tool calls as raw
    XML in its own text output. Before this filter, that XML streamed
    straight to the terminal as it arrived: confirmed live,
    <tool_call>, <function=Glob>, <parameter=pattern>... stayed on screen
    for the whole turn. The post-hoc cleaner (clean_text_from_tool_blocks)
    and the finalize_response() erase only run AFTER the stream ends, and
    both can miss a pure-tool-call turn (empty cleaned text never triggers
    the markdown-erase gate) — suppressing the markup before it's ever
    written is the actual fix; the other two stay as a second line of
    defense for whatever slips past this one.

    Unlike a think block, an opened tool-call block that never sees its
    close tag is dropped entirely rather than surfaced on flush() — the
    Qwen parser (_parse_qwen_xml_tool_call) already tolerates an
    unterminated <tool_call>, so anything between an open marker and
    end-of-stream is guaranteed to be tool-call markup, never real prose.
    """

    _OPEN_TAGS = ("<tool_call>", "<tool_call ", "<function=", "<parameter=",
                 "<|tool_call|>", "<|tool_call_start|>", "✿FUNCTION✿")
    _CLOSE_TAGS = ("</tool_call>",)
    _LOOKAHEAD = 24   # longest open/close tag

    def __init__(self) -> None:
        self._buf     = ""
        self._in_tool = False

    def feed(self, chunk: str) -> str:
        """Returns the VISIBLE portion of `chunk` — tool-call markup withheld."""
        self._buf += chunk
        visible = ""
        while True:
            if not self._in_tool:
                best = None
                for tag in self._OPEN_TAGS:
                    pos = self._buf.find(tag)
                    if pos >= 0 and (best is None or pos < best[0]):
                        best = (pos, tag)
                if best is None:
                    safe = self._buf[: -self._LOOKAHEAD] if len(self._buf) > self._LOOKAHEAD else ""
                    if safe:
                        visible += safe
                        self._buf = self._buf[len(safe):]
                    break
                pos, tag = best
                if pos > 0:
                    visible += self._buf[:pos]
                self._buf = self._buf[pos + len(tag):]
                self._in_tool = True
            else:
                best = None
                for tag in self._CLOSE_TAGS:
                    pos = self._buf.find(tag)
                    if pos >= 0 and (best is None or pos < best[0]):
                        best = (pos, tag)
                if best is None:
                    # Still inside a tool block — nothing here is displayable;
                    # keep only a small lookahead tail in case the close tag
                    # is split across this chunk boundary.
                    if len(self._buf) > self._LOOKAHEAD:
                        self._buf = self._buf[-self._LOOKAHEAD:]
                    break
                pos, tag = best
                self._buf = self._buf[pos + len(tag):]
                self._in_tool = False
        return visible

    def flush(self) -> str:
        """End of stream: return buffered prose; drop anything still inside
        an opened-but-never-closed tool block."""
        result = "" if self._in_tool else self._buf
        self._buf = ""
        return result


# ── Message format converters ──────────────────────────────────────────────
#
# Internal "neutral" message format used throughout the codebase:
#   {"role": "user",      "content": "text"}
#   {"role": "assistant", "content": "text", "tool_calls": [
#       {"id": "...", "name": "...", "input": {...}}
#   ]}
#   {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}


def convert_to_openai(messages: list) -> list:
    """Convert neutral messages → OpenAI API format."""
    result = []
    for m in messages:
        role = m["role"]

        if role == "user":
            result.append({"role": "user", "content": m.get("content") or ""})

        elif role == "assistant":
            # llama-server (and some other OpenAI-compat servers) reject null
            # content — always use "" when there is no text, never None.
            msg: dict = {"role": "assistant", "content": m.get("content") or ""}
            tcs = m.get("tool_calls", [])
            if tcs:
                msg["tool_calls"] = [
                    {
                        "id":   tc["id"],
                        "type": "function",
                        "function": {
                            "name":      tc["name"],
                            "arguments": json.dumps(tc["input"], ensure_ascii=False),
                        },
                    }
                    for tc in tcs
                ]
            result.append(msg)

        elif role == "tool":
            result.append({
                "role":         "tool",
                "tool_call_id": m["tool_call_id"],
                "content":      m.get("content") or "(empty result)",
            })

    return result


def schemas_to_openai_functions(tool_schemas: list) -> list:
    """Convert Anthropic-style tool schemas to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["input_schema"],
            },
        }
        for t in tool_schemas
    ]


# ── Streaming implementations ──────────────────────────────────────────────


#: Marker key carrying WHY a tool call's arguments could not be parsed. The
#: dispatcher turns this into a message the model can act on. Its presence
#: means `_raw` holds the unparsed argument string.
TOOL_ARG_PARSE_ERROR_KEY = "_parse_error"


def parse_tool_arguments(raw: "str | None") -> "tuple[dict, str | None]":
    """Parse a tool call's `arguments` string into a dict.

    Returns (input_dict, error_or_None).

    Why this is not just `json.loads`
    ---------------------------------
    Local GGUF finetunes are far looser about JSON than a hosted model, and the
    two ways they get it wrong need OPPOSITE handling:

    1. **Literal control characters inside a string.** A model writing code
       emits a real newline instead of `\\n`. The arguments are COMPLETE and
       unambiguous, strict JSON just refuses them. `strict=False` accepts them,
       so this is repaired silently , there is nothing for the model to fix.

    2. **Truncation.** The response hit its token ceiling mid-argument. The
       content is INCOMPLETE, and it is tempting to salvage it by closing the
       open quotes and braces , do NOT. A salvaged `Write` writes half a file
       and reports success, which is worse than any error. Truncation is
       reported back so the model can retry with a smaller payload.

    Before 2026-08-19 every failure here collapsed to `{"_raw": ...}`, which
    reached the instrument as a missing required parameter. A truncated Godot
    script became `instrument 'Write' called with invalid parameters
    ('file_path')` , pointing the model at a parameter name that was never the
    problem, so it would retry the identical oversized call.
    """
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            # Case 1: complete, but with raw newlines/tabs inside strings.
            parsed = json.loads(raw, strict=False)
        except json.JSONDecodeError as e:
            # Case 2 and anything else: do not guess at the missing bytes.
            if _looks_truncated(raw):
                msg = (f"the arguments were cut off after {len(raw)} characters "
                       f"(unterminated JSON). The response most likely hit its "
                       f"token limit. Retry with a smaller payload , for a file "
                       f"write, create it in several smaller pieces.")
            else:
                msg = (f"the arguments were not valid JSON ({e.msg} at "
                       f"position {e.pos}).")
            return {"_raw": raw, TOOL_ARG_PARSE_ERROR_KEY: msg}, msg
    if not isinstance(parsed, dict):
        msg = (f"the arguments must be a JSON object, got "
               f"{type(parsed).__name__}.")
        return {"_raw": raw, TOOL_ARG_PARSE_ERROR_KEY: msg}, msg
    return parsed, None


def _looks_truncated(raw: str) -> bool:
    """True when `raw` ends mid-JSON rather than being merely malformed.

    Scans once, tracking string state and escapes, so a brace inside a string
    literal (very common in generated code) is not counted as structure.
    """
    depth = 0
    in_str = False
    esc = False
    for ch in raw:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
    return in_str or depth > 0


def _stream_openai_api(
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    settings: dict,
) -> Generator:
    """Stream from gb-synapse's OpenAI-compatible endpoint. Yields
    StreamFragment, then CompletedResponse."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

    oai_messages = [{"role": "system", "content": system}] + convert_to_openai(messages)

    kwargs: dict = {
        "model":    model,
        "messages": oai_messages,
        "stream":   True,
        # vLLM omits the final usage chunk unless explicitly asked (unlike
        # llama-server, which already sends it) — needed for real tok/s
        # instead of orchestrator.py's char-count/4 fallback estimate.
        "stream_options": {"include_usage": True},
    }
    if tool_schemas and not settings.get("no_tools"):
        kwargs["tools"] = schemas_to_openai_functions(tool_schemas)
        if not settings.get("disable_tool_choice"):
            kwargs["tool_choice"] = "auto"
    if settings.get("max_tokens"):
        kwargs["max_tokens"] = settings["max_tokens"]
    # No per-request context-window injection here: gb-synapse's llama-server
    # gets its context size fixed at startup (--ctx-size, see gb_synapse.serve),
    # unlike Ollama's API which needs options.num_ctx on every request.
    #
    # Extended thinking: the Qwen3.5/3.6 family's chat template supports a
    # server-side `enable_thinking` toggle (not just client-side _ThinkFilter
    # display-hiding below, which still pays the FULL generation cost for a
    # hidden reasoning block). Confirmed live (2026-07-13, CPU-only serve of
    # satgeze/qwen36-35b-uncensored-1m): a 2-turn agentic tool-call round trip
    # (decide-to-call → tool result → final answer) timed out past 300s with
    # thinking on; the identical round trip completed in ~5s with it off, via
    # `chat_template_kwargs: {"enable_thinking": false}`. Default OFF for the
    # agentic tool-calling path where speed/reliability matter more than a
    # visible chain-of-thought; `settings["enable_thinking"]=True` re-enables
    # it for anyone who wants deeper reasoning and can afford the latency.
    # Harmless no-op for templates that don't recognize the kwarg.
    if not settings.get("enable_thinking", False):
        kwargs["extra_body"] = {
            **settings.get("extra_body", {}),
            "chat_template_kwargs": {"enable_thinking": False},
        }

    text = ""
    tool_buf: dict = {}
    in_tok = out_tok = 0
    # Apply think-tag filtering when base_url is a local server (always true
    # for gb-synapse, but keep the check generic in case of a future remote
    # cluster proxy on a non-localhost address).
    _is_local = bool(base_url) and (
        base_url.startswith("http://localhost")
        or base_url.startswith("http://127.")
        or base_url.startswith("http://0.")
        or "localhost" in base_url
    )
    _think_filter = _ThinkFilter() if _is_local else None
    # Defense-in-depth on the native-FC path: real tool calls arrive via
    # delta.tool_calls, but a backend/template that still emits XML tool-call
    # markup as plain content text (this reference model's chat template does
    # exactly that in injection mode — see _stream_injected) must not leak it
    # here either.
    _tool_filter = _ToolTagFilter() if _is_local else None

    try:
        stream = client.chat.completions.create(**kwargs)
    except Exception as _conn_err:
        _e = str(_conn_err)
        if "num_gpu" in _e or "memory layout" in _e or "out of memory" in _e.lower():
            raise RuntimeError(
                f"\nModel '{model}' cannot fit in GPU memory.\n\n"
                f"  • Try /turboquant on — enables KV compression (GreenBoost T2/T3)\n"
                f"  • Pull a smaller quant: greenboost pull <repo>:<smaller-quant>\n"
                f"  • Check fit: greenboost recommend\n"
                f"  • Raw error: {_e[:120]}\n"
            ) from _conn_err
        _cls = type(_conn_err).__name__
        if "Connection refused" in _e or "ConnectError" in _cls or "APIConnectionError" in _cls or "ConnectionError" in _cls:
            raise RuntimeError(
                f"\nCannot connect to {base_url}\n\n"
                f"  • /llamaserve          — start gb-synapse\n"
                f"  • /llamaserve logs     — follow startup log\n"
                f"  • /llamaserve status   — check if the model is loaded\n"
            ) from _conn_err
        if "404" in _e or "not found" in _e.lower() or "NotFoundError" in type(_conn_err).__name__:
            raise RuntimeError(
                f"\nModel {model!r} not found on the backend at {base_url}.\n\n"
                f"  • Make sure gb-synapse is running with this model: /llamaserve status\n"
                f"  • Check available models:  curl {base_url}/models\n"
                f"  • Switch model:            /model <name>\n"
            ) from _conn_err
        if _is_context_overflow(_e):
            _pt, _nc = _overflow_numbers(str(_e))
            raise ContextOverflowError(str(_e), prompt_tokens=_pt,
                                       n_ctx=_nc) from _conn_err
        _status = getattr(_conn_err, "status_code", None)
        if _status is not None or "BadRequestError" in _cls or "InternalServerError" in _cls or "APIStatusError" in _cls:
            raise RuntimeError(
                f"\nBackend at {base_url} rejected the request"
                f"{f' (HTTP {_status})' if _status else ''}.\n\n"
                f"  • Raw error: {_e[:300]}\n"
            ) from _conn_err
        raise

    for chunk in guarded_stream(stream, model, base_url):
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage:
                in_tok  = chunk.usage.prompt_tokens
                out_tok = chunk.usage.completion_tokens
            continue

        choice = chunk.choices[0]
        delta  = choice.delta

        if delta.content:
            if _think_filter:
                for seg, is_think in _think_filter.feed(delta.content):
                    if is_think:
                        yield ReasoningFragment(seg)
                    else:
                        text += seg
                        visible = _tool_filter.feed(seg) if _tool_filter else seg
                        if visible:
                            yield StreamFragment(visible)
            else:
                text += delta.content
                yield StreamFragment(delta.content)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_buf:
                    tool_buf[idx] = {"id": "", "name": "", "args": ""}
                if tc.id:
                    tool_buf[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_buf[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_buf[idx]["args"] += tc.function.arguments

        if hasattr(chunk, "usage") and chunk.usage:
            in_tok  = chunk.usage.prompt_tokens  or in_tok
            out_tok = chunk.usage.completion_tokens or out_tok

    # Flush any remaining thinking-filter buffer
    if _think_filter:
        for seg, is_think in _think_filter.flush():
            if is_think:
                yield ReasoningFragment(seg)
            else:
                text += seg
                visible = _tool_filter.feed(seg) if _tool_filter else seg
                if visible:
                    yield StreamFragment(visible)
    if _tool_filter:
        tail = _tool_filter.flush()
        if tail:
            yield StreamFragment(tail)

    tool_calls = []
    for idx in sorted(tool_buf):
        v = tool_buf[idx]
        inp, _err = parse_tool_arguments(v["args"])
        tool_calls.append({"id": v["id"] or f"call_{idx}", "name": v["name"], "input": inp})

    # Fallback: some GGUF finetunes emit tool calls as text (<tool_call>…</tool_call>,
    # Hermes, fenced JSON) even when sent via native FC. If they didn't parse into
    # delta.tool_calls, fish them out of the prose.
    if not tool_calls and text:
        from greenboost_cli.inference.injection import (
            parse_tool_calls_from_text,
            clean_text_from_tool_blocks,
        )
        _parsed = parse_tool_calls_from_text(text, tool_schemas, model)
        if _parsed:
            tool_calls = _parsed
            _clean = clean_text_from_tool_blocks(text)
            try:
                from greenboost_cli.terminal.renderer import replace_text_buffer
                replace_text_buffer(_clean)
            except Exception:
                pass
            text = _clean

    yield CompletedResponse(text, tool_calls, in_tok, out_tok)


def _stream_injected(
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    settings: dict,
) -> Generator:
    """
    Stream without native function-calling; inject tool definitions into the
    system prompt and parse tool calls from the text output. Used for GGUFs
    served by gb-synapse that don't reliably emit native tool_calls (or any
    backend with tool_format=inject in settings).
    """
    from greenboost_cli.inference.injection import (
        inject_tools_into_system,
        messages_to_openai_for_injection,
        parse_tool_calls_from_text,
        clean_text_from_tool_blocks,
    )
    from openai import OpenAI
    injected_system = inject_tools_into_system(system, tool_schemas, model)
    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

    oai_messages = (
        [{"role": "system", "content": injected_system}]
        + messages_to_openai_for_injection(messages)
    )

    kwargs: dict = {
        "model":    model,
        "messages": oai_messages,
        "stream":   True,
        "stream_options": {"include_usage": True},
    }
    if settings.get("max_tokens"):
        kwargs["max_tokens"] = settings["max_tokens"]
    # See _stream_openai_api's matching comment: llama-server's context is
    # fixed at startup (no per-request injection needed), but extended
    # thinking DOES need a per-request server-side toggle — _ThinkFilter
    # below only hides the reasoning block from display, it doesn't stop the
    # model from generating it, which is the expensive part. This is the
    # tool-call-injection path (used for the Qwen3.5/3.6 family specifically,
    # see _is_qwen35_family/should_inject_tools) — same fix, same reasoning.
    if not settings.get("enable_thinking", False):
        kwargs["extra_body"] = {
            **settings.get("extra_body", {}),
            "chat_template_kwargs": {"enable_thinking": False},
        }

    full_text  = ""   # raw text (includes think tags + tool calls, used for parsing)
    prose_text = ""   # displayable prose (think-stripped + tool-stripped)
    in_tok = out_tok = 0
    think_filter = _ThinkFilter()
    tool_filter  = _ToolTagFilter()

    try:
        stream_obj = client.chat.completions.create(**kwargs)
    except Exception as _conn_err:
        err_str = str(_conn_err)
        if _is_context_overflow(err_str):
            # This is the path the reference model (qwen3.6, injection mode)
            # actually goes through. Before this check it had NO 400 handler
            # at all — even the generic one below — so a context-overflow 400
            # here reached the REPL as a raw, unwrapped openai.BadRequestError.
            _pt, _nc = _overflow_numbers(err_str)
            raise ContextOverflowError(err_str, prompt_tokens=_pt,
                                       n_ctx=_nc) from _conn_err
        _is_refused = (
            "Connection refused" in err_str
            or "ConnectError"    in type(_conn_err).__name__
            or "APIConnectionError" in type(_conn_err).__name__
        )
        _is_oom = (
            "out of memory" in err_str.lower()
            or "CUDA error" in err_str
            or "num_gpu" in err_str
            or "memory layout" in err_str
            or "model failed to load" in err_str.lower()
        )
        _is_not_found = (
            "404" in err_str
            or "not found" in err_str.lower()
            or "NotFoundError" in type(_conn_err).__name__
        )
        if _is_refused:
            _url = base_url or "http://localhost:11369"  # gb-synapse's own default (registry.py)
            raise RuntimeError(
                f"\nCannot connect to gb-synapse at {_url}\n\n"
                f"  • /llamaserve          — start the server\n"
                f"  • /llamaserve logs     — follow startup log\n"
                f"  • /llamaserve status   — check if it's ready\n"
            ) from _conn_err
        if _is_not_found:
            raise RuntimeError(
                f"\nModel {model!r} not found on the backend at {base_url}.\n\n"
                f"  • Pull it first:  greenboost pull <org/repo>[:quant]\n"
                f"  • Check available models:  curl {base_url}/models\n"
                f"  • Switch model:            /model <name>\n"
            ) from _conn_err
        if _is_oom:
            raise RuntimeError(
                f"\nModel '{model}' cannot be loaded (out of memory).\n\n"
                f"  • Pull a smaller quant: greenboost pull <repo>:<smaller-quant>\n"
                f"  • Check fit: greenboost recommend\n"
                f"  • Raw error: {err_str[:120]}\n"
            ) from _conn_err
        raise

    for chunk in guarded_stream(stream_obj, model, base_url):
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage:
                in_tok  = chunk.usage.prompt_tokens
                out_tok = chunk.usage.completion_tokens
            continue
        choice = chunk.choices[0]
        delta  = choice.delta
        if delta.content:
            full_text += delta.content
            for seg_text, is_thinking in think_filter.feed(delta.content):
                if is_thinking:
                    yield ReasoningFragment(seg_text)
                else:
                    prose_text += seg_text
                    # This is the path the reference model actually streams
                    # through — withhold <tool_call>/<function=.../<parameter=
                    # markup from the terminal as it arrives instead of only
                    # cleaning it up after the fact (full_text above still
                    # accumulates the RAW text unfiltered, for
                    # parse_tool_calls_from_text below).
                    visible = tool_filter.feed(seg_text)
                    if visible:
                        yield StreamFragment(visible)
        if hasattr(chunk, "usage") and chunk.usage:
            in_tok  = chunk.usage.prompt_tokens  or in_tok
            out_tok = chunk.usage.completion_tokens or out_tok

    # Flush any remaining buffered content
    for seg_text, is_thinking in think_filter.flush():
        if is_thinking:
            yield ReasoningFragment(seg_text)
        else:
            prose_text += seg_text
            visible = tool_filter.feed(seg_text)
            if visible:
                yield StreamFragment(visible)
    _tool_tail = tool_filter.flush()
    if _tool_tail:
        yield StreamFragment(_tool_tail)

    tool_calls = parse_tool_calls_from_text(full_text, tool_schemas, model)
    if tool_calls:
        # Tool call blocks were streamed as raw text; replace the buffer with
        # the cleaned prose so finalize_response() re-renders it correctly.
        clean_text = clean_text_from_tool_blocks(prose_text)
        from greenboost_cli.terminal.renderer import replace_text_buffer
        replace_text_buffer(clean_text)
    else:
        clean_text = prose_text.strip()

    yield CompletedResponse(clean_text, tool_calls, in_tok, out_tok)
