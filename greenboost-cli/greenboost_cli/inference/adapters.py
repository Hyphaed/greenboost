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
                        yield StreamFragment(seg)
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
                yield StreamFragment(seg)

    tool_calls = []
    for idx in sorted(tool_buf):
        v = tool_buf[idx]
        try:
            inp = json.loads(v["args"]) if v["args"] else {}
        except json.JSONDecodeError:
            inp = {"_raw": v["args"]}
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

    try:
        stream_obj = client.chat.completions.create(**kwargs)
    except Exception as _conn_err:
        err_str = str(_conn_err)
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
            _url = base_url or "http://localhost:11434"
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
                    yield StreamFragment(seg_text)
        if hasattr(chunk, "usage") and chunk.usage:
            in_tok  = chunk.usage.prompt_tokens  or in_tok
            out_tok = chunk.usage.completion_tokens or out_tok

    # Flush any remaining buffered content
    for seg_text, is_thinking in think_filter.flush():
        if is_thinking:
            yield ReasoningFragment(seg_text)
        else:
            prose_text += seg_text
            yield StreamFragment(seg_text)

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
