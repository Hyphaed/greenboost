# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, redacting capture for `cli_tool_call` dataflux telemetry.

NemoClaw audit, Phase 3d. This module is a near-verbatim adaptation (not a
from-scratch re-implementation, unlike the rest of the NemoClaw audit — see
`third_party/nemoclaw_patterns/NOTICE`) of the stdlib-only redaction core in
NemoClaw's `agents/langchain-deepagents-code/nemoclaw_observability.py`
(`_CaptureBudget`, `_bounded_string`, `_scrub_secret_values`,
`_redact_capture_key`, `_capture_jsonable`, plus the standalone/anchored/
truncated credential-pattern tuples and the aggregate-budget constants).
The secret-shaped-value redaction is subtle security code — porting it
near-verbatim is deliberate, not laziness; re-deriving these patterns from
scratch would risk missing an edge case the original already handles (a
truncated token at a byte boundary, an unpaired UTF-16 surrogate, a
credential-shaped dict key under three different naming conventions).

Trimmed relative to the source: NemoClaw's version also redacts LangGraph
checkpoint/interrupt state (`_STATE_CAPTURE_KEYS`) and captures LLM request
payloads for OTLP export (`_bounded_llm_request`, the lifecycle/exporter
classes) — none of that applies here. What's kept is exactly the part this
module needs: turn an arbitrary Python value (a tool call's `params` dict,
most concretely) into a JSON-safe, depth/size-bounded, credential-redacted
structure safe to append to `~/.local/share/greenboost/dataflux.jsonl`.

Without this, a naive `{"name": "Bash", "args": params}` dataflux emit risks
writing a token pasted into a shell command straight into that log.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

# ── Bounds ──────────────────────────────────────────────────────────────
# Same shape as instruments/handlers.py's _ctx_char_budget precedent
# (bound a result size, floor + ceiling) but these are FIXED, not
# context-scaled — a dataflux event's job is to be a durable, replayable
# audit record, not a model-facing result, so its size budget shouldn't
# shrink or grow with the currently served context window.
MAX_CAPTURE_DEPTH = 8
MAX_CAPTURE_ITEMS = 50
MAX_CAPTURE_NODES = 2_048
MAX_CAPTURE_STRING_CHARS = 8_000
MAX_CAPTURE_AGGREGATE_STRING_CHARS = 50_000
MAX_CAPTURE_JSON_CHARS = 50_000
MAX_CAPTURE_PREVIEW_CHARS = 16_000
_MIN_SAFE_JSON_INTEGER = -(1 << 63)
_MAX_SAFE_JSON_INTEGER = (1 << 64) - 1
_REDACTED_VALUE = "<redacted>"
_OUT_OF_RANGE_INTEGER = "<integer outside safe JSON range>"
# A dict carrying either of these keys is an opaque fallback encoding from
# some other layer (e.g. a base64/pickle escape hatch) — never inspect or
# export it, same reasoning as _opaque_capture_marker below.
_UNSAFE_SERIALIZATION_TAGS = {
    "__nv_fallback_str__",
    "__nv_pickle__",
}

_UNICODE_SURROGATE = re.compile(r"[\ud800-\udfff]")
_CAPTURE_KEY_ACRONYM_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_CAPTURE_KEY_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAPTURE_KEY_DELIMITER = re.compile(r"[^A-Za-z0-9]+")

_SENSITIVE_CAPTURE_KEYS = {
    "api_key", "auth", "authorization", "cookie", "credential",
    "credentials", "headers", "password", "proxy_authorization",
    "secret", "set_cookie", "token",
}

# ── Secret-shaped-value redaction ──────────────────────────────────────
# SECURITY — ported verbatim from NemoClaw's mirror of its own canonical
# TypeScript secret-pattern groups (src/lib/security/secret-patterns.ts).
# These are provider-specific token shapes (GitHub, OpenAI/Anthropic,
# Slack, AWS, HuggingFace, GitLab, npm, PyPI, Telegram, JWT, Tavily,
# LangSmith, PEM private keys) plus generic KEY/TOKEN/SECRET/PASSWORD-
# shaped assignments. Kept as-is rather than re-derived — a regex this
# specific is easy to get subtly wrong (byte-boundary truncation, one
# false-negative charset) and this one is already battle-tested.
_REDACTED_SECRET_VALUE = "<redacted-secret>"
# Python's \s also includes control separators ECMAScript excludes, so
# spell out the canonical whitespace set for parity with the TypeScript
# mirror these patterns were ported from.
_NON_WHITESPACE_SECRET_CHAR = (
    r"[^\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029"
    r"\u202f\u205f\u3000\ufeff'\"]"
)
_STANDALONE_SECRET_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"nvapi-[A-Za-z0-9_-]{10,}",
        r"nvcf-[A-Za-z0-9_-]{10,}",
        r"ghp_[A-Za-z0-9_-]{10,}",
        r"github_pat_[A-Za-z0-9_]{30,}",
        r"sk-proj-[A-Za-z0-9_-]{10,}",
        r"sk-ant-[A-Za-z0-9_-]{10,}",
        r"sk-[A-Za-z0-9_-]{20,}",
        r"(?:xox[bpas]|xapp)-[A-Za-z0-9-]{10,}",
        r"A(?:K|S)IA[A-Z0-9]{16}",
        r"hf_[A-Za-z0-9]{10,}",
        r"glpat-[A-Za-z0-9_-]{10,}",
        r"gsk_[A-Za-z0-9]{10,}",
        r"pypi-[A-Za-z0-9_-]{10,}",
        r"\bbot\d{8,10}:[A-Za-z0-9_-]{35}\b",
        r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b",
        r"\b[A-Za-z0-9]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}\b",
        r"tvly-[A-Za-z0-9_-]{10,}",
        r"lsv2_(?:pt|sk)_[A-Za-z0-9]{10,}(?:_[A-Za-z0-9]+)*",
        r"(?s)-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----",
    )
)
_ANCHORED_SECRET_PATTERNS = (
    re.compile(
        r"(Bearer[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029"
        r"\u202f\u205f\u3000\ufeff]+)[A-Za-z0-9_.+/=-]{10,}",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:^|[^A-Za-z0-9])(?:[A-Za-z0-9]{1,128}_"
        r"(?:KEY|TOKEN|SECRET|CREDENTIAL|PASSWORD|PASSWD|PASS)|"
        r"(?:X[-_])?API[-_]KEY|"
        r"TOKEN|SECRET|CREDENTIAL|PASSWORD|PASSWD|PASS)"
        r"['\"]?(?:[ \t]{0,32}[=:][ \t]{0,32}|[ \t]{1,32})['\"]?)"
        rf"{_NON_WHITESPACE_SECRET_CHAR}{{10,}}",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:^|[^A-Za-z0-9])"
        r"(?:[A-Za-z0-9]{1,128}(?:Token|Secret|Credential)|"
        r"[A-Za-z0-9]{0,128}(?:[Aa]ccess|[Rr]efresh|[Cc]lient|[Bb]earer|"
        r"[Aa]uth|[Aa][Pp][Ii]|[Pp]rivate|[Ss]igning|[Ss]ession|[Bb]ot|"
        r"[Aa]pp|[Rr]esolved)Key|"
        r"[A-Za-z0-9]{1,128}(?:Password|Passwd|Pass))"
        r"['\"]?(?:[ \t]{0,32}[=:][ \t]{0,32}|[ \t]{1,32})['\"]?)"
        rf"{_NON_WHITESPACE_SECRET_CHAR}{{10,}}",
    ),
    re.compile(
        r"((?:^|[^A-Za-z0-9])KEY['\"]?"
        r"(?:[ \t]{0,32}[=:][ \t]{0,32}|[ \t]{1,32})['\"]?)"
        rf"{_NON_WHITESPACE_SECRET_CHAR}{{10,}}",
    ),
)
_ANCHORED_SECRET_REPLACEMENT = rf"\g<1>{_REDACTED_SECRET_VALUE}"
_UNTERMINATED_PRIVATE_KEY_PATTERN = re.compile(
    r"(?s)-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----.*\Z"
)
_TRUNCATED_SECRET_PATTERNS = tuple(
    re.compile(pattern, flags)
    for pattern, flags in (
        (
            r"(?:nvapi-|nvcf-|ghp_|github_pat_|sk-proj-|sk-ant-|sk-|"
            r"(?:xox[bpas]|xapp)-|hf_|glpat-|gsk_|pypi-|tvly-|"
            r"lsv2_(?:pt|sk)_)[A-Za-z0-9_-]*\Z",
            0,
        ),
        (r"A(?:K|S)IA[A-Z0-9]*\Z", 0),
        (r"(?:bot)?\d{1,10}:[A-Za-z0-9_-]*\Z", 0),
        (
            r"[A-Za-z0-9]{1,24}\.[A-Za-z0-9_-]{0,6}"
            r"(?:\.[A-Za-z0-9_-]*)?\Z",
            0,
        ),
        (
            r"(?:Bearer[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029"
            r"\u202f\u205f\u3000\ufeff]+)"
            r"[A-Za-z0-9_.+/=-]*\Z",
            re.IGNORECASE,
        ),
        (
            r"(?:^|[^A-Za-z0-9])(?:[A-Za-z0-9]{1,128}_"
            r"(?:KEY|TOKEN|SECRET|CREDENTIAL|PASSWORD|PASSWD|PASS)|"
            r"(?:X[-_])?API[-_]KEY|"
            r"TOKEN|SECRET|CREDENTIAL|PASSWORD|PASSWD|PASS)"
            r"['\"]?(?:[ \t]{0,32}[=:][ \t]{0,32}|[ \t]{1,32})['\"]?"
            rf"{_NON_WHITESPACE_SECRET_CHAR}*\Z",
            re.IGNORECASE,
        ),
        (
            r"(?:^|[^A-Za-z0-9])"
            r"(?:[A-Za-z0-9]{1,128}(?:Token|Secret|Credential)|"
            r"[A-Za-z0-9]{0,128}(?:[Aa]ccess|[Rr]efresh|[Cc]lient|"
            r"[Bb]earer|[Aa]uth|[Aa][Pp][Ii]|[Pp]rivate|[Ss]igning|"
            r"[Ss]ession|[Bb]ot|[Aa]pp|[Rr]esolved)Key|"
            r"[A-Za-z0-9]{1,128}(?:Password|Passwd|Pass))"
            r"['\"]?(?:[ \t]{0,32}[=:][ \t]{0,32}|[ \t]{1,32})['\"]?"
            rf"{_NON_WHITESPACE_SECRET_CHAR}*\Z",
            0,
        ),
        (
            r"(?:^|[^A-Za-z0-9])KEY['\"]?"
            r"(?:[ \t]{0,32}[=:][ \t]{0,32}|[ \t]{1,32})['\"]?"
            rf"{_NON_WHITESPACE_SECRET_CHAR}*\Z",
            0,
        ),
    )
)


def _scrub_secret_values(value: str, *, source_was_truncated: bool = False) -> str:
    """Best-effort redaction of recognized credential-shaped tokens in text."""
    scrubbed = value
    for pattern in _STANDALONE_SECRET_PATTERNS:
        scrubbed = pattern.sub(_REDACTED_SECRET_VALUE, scrubbed)
    for pattern in _ANCHORED_SECRET_PATTERNS:
        scrubbed = pattern.sub(_ANCHORED_SECRET_REPLACEMENT, scrubbed)
    # Bounding (below) can cut a private-key block before its END marker.
    # Once a BEGIN marker is present, redact the remaining bounded segment
    # rather than emit a partial key body.
    scrubbed = _UNTERMINATED_PRIVATE_KEY_PATTERN.sub(_REDACTED_SECRET_VALUE, scrubbed)
    if source_was_truncated:
        for pattern in _TRUNCATED_SECRET_PATTERNS:
            scrubbed = pattern.sub(_REDACTED_SECRET_VALUE, scrubbed)
    return scrubbed


class _CaptureBudget:
    """Bound aggregate traversal and repeated container expansion across
    one whole `_capture_jsonable` call — the per-node/per-container caps
    below are local; this is the global ceiling that stops a wide, shallow
    structure (many short strings, or many small siblings) from adding up
    to something unbounded even though no single node exceeds its own cap."""

    def __init__(self) -> None:
        self.remaining_nodes = MAX_CAPTURE_NODES
        self.remaining_string_chars = MAX_CAPTURE_AGGREGATE_STRING_CHARS
        self.seen_containers: "set[int]" = set()

    def claim_node(self) -> bool:
        if self.remaining_nodes <= 0:
            return False
        self.remaining_nodes -= 1
        return True

    def claim_container(self, value: Any) -> bool:
        """False on a cycle or a container already visited via another
        reference (id() re-use) — both would otherwise recurse forever or
        double-count the same data against the budget."""
        identity = id(value)
        if identity in self.seen_containers:
            return False
        self.seen_containers.add(identity)
        return True


def _bounded_string(
    value: str, budget: "_CaptureBudget | None" = None, *, scrub_secrets: bool = False,
) -> str:
    limit = min(len(value), MAX_CAPTURE_STRING_CHARS)
    if budget is not None:
        limit = min(limit, budget.remaining_string_chars)
        budget.remaining_string_chars -= limit
    bounded_source = value if limit == len(value) else value[:limit]
    if scrub_secrets:
        bounded_source = _scrub_secret_values(
            bounded_source, source_was_truncated=limit < len(value)
        )
    bounded = (
        bounded_source
        if limit == len(value)
        else f"{bounded_source}...[truncated {len(value) - limit} chars]"
    )
    # A JSONL sink requires valid UTF-8. Replace unpaired UTF-16 surrogates
    # without rejecting the value or mutating it in place.
    return _UNICODE_SURROGATE.sub("�", bounded)


def _redact_capture_key(key: Any) -> bool:
    """True if `key` names a field whose VALUE should be replaced with
    `_REDACTED_VALUE` outright rather than captured at all — a credential-
    shaped key under any of several real-world naming conventions
    (snake_case, camelCase, PascalCase, or a bare suffix like `_token`)."""
    if type(key) is not str:
        return True
    segmented = _CAPTURE_KEY_ACRONYM_BOUNDARY.sub("_", key.strip())
    normalized = _CAPTURE_KEY_DELIMITER.sub(
        "_", _CAPTURE_KEY_CAMEL_BOUNDARY.sub("_", segmented)
    ).strip("_").lower()
    segments = set(normalized.split("_"))
    return (
        normalized in _SENSITIVE_CAPTURE_KEYS
        or bool(
            segments
            & {
                "auth", "authentication", "authorization", "bearer",
                "cookie", "credential", "credentials", "header",
                "password", "secret", "token",
            }
        )
        or ("key" in segments and bool(segments & {"access", "api", "private", "signing"}))
        or normalized.endswith("_api_key")
        or normalized.endswith("_access_key")
        or normalized.endswith("_headers")
        or normalized in {"pass", "passwd"}
        or normalized.endswith("_pass")
        or normalized.endswith("_passwd")
        or normalized.endswith("_password")
        or normalized.endswith("_private_key")
        or normalized.endswith("_secret")
        or normalized.endswith("_token")
    )


def _opaque_capture_marker(_value: Any) -> "dict[str, str]":
    # Keep this marker constant. Even a type-name lookup can invoke
    # attacker-owned metaclass behavior (`type(value).__name__` runs
    # arbitrary `__getattr__`/`__class__` machinery on a crafted object),
    # and the concrete class name is not useful audit data anyway.
    return {"_omitted_type": "opaque"}


def _unique_capture_key(candidate: str, captured: "dict[str, Any]") -> str:
    """Keep redacted or bounded mapping keys distinct without exposing
    the original (a truncated key colliding with a real one would
    otherwise silently overwrite it in the captured dict)."""
    if candidate not in captured:
        return candidate
    for index in range(2, MAX_CAPTURE_ITEMS + 2):
        suffix = f"#{index}"
        unique = f"{candidate[: MAX_CAPTURE_STRING_CHARS - len(suffix)]}{suffix}"
        if unique not in captured:
            return unique
    return f"_duplicate_key_{len(captured)}"


def _capture_jsonable(value: Any, *, depth: int = 0, budget: "_CaptureBudget | None" = None) -> Any:
    """Bound an arbitrary Python value to something JSON-safe, redacting
    credential-shaped keys along the way. depth/item/node/string caps and
    a cycle guard all apply — see the module docstring for why."""
    if budget is None:
        budget = _CaptureBudget()
    if depth >= MAX_CAPTURE_DEPTH:
        return {"_omitted_at_depth": MAX_CAPTURE_DEPTH}
    if not budget.claim_node():
        return {"_truncated_by_budget": True}
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if _MIN_SAFE_JSON_INTEGER <= value <= _MAX_SAFE_JSON_INTEGER:
            return value
        return _OUT_OF_RANGE_INTEGER
    if type(value) is float:
        return value if math.isfinite(value) else "<non-finite float>"
    if type(value) is str:
        return _bounded_string(value, budget, scrub_secrets=True)
    if type(value) in (bytes, bytearray):
        return f"<{len(value)} bytes>"
    if type(value) is dict:
        if any(tag in value for tag in _UNSAFE_SERIALIZATION_TAGS):
            return _opaque_capture_marker(value)
        if not budget.claim_container(value):
            return {"_omitted_reference": "shared_or_cycle"}
        captured: "dict[str, Any]" = {}
        omitted_items = 0
        inspected_items = 0
        for key, item in value.items():
            if inspected_items >= MAX_CAPTURE_ITEMS:
                break
            inspected_items += 1
            if type(key) is not str:
                omitted_items += 1
                continue
            bounded_key = _unique_capture_key(
                _bounded_string(key, budget, scrub_secrets=True), captured
            )
            captured[bounded_key] = (
                _REDACTED_VALUE
                if _redact_capture_key(key)
                else _capture_jsonable(item, depth=depth + 1, budget=budget)
            )
        truncated_items = len(value) - inspected_items
        if truncated_items > 0:
            captured["_truncated_items"] = truncated_items
        if omitted_items > 0:
            captured["_omitted_non_string_keys"] = omitted_items
        return captured
    if type(value) in (list, tuple):
        if not budget.claim_container(value):
            return {"_omitted_reference": "shared_or_cycle"}
        captured_items: "list[Any]" = []
        inspected_items = 0
        for item in value:
            if inspected_items >= MAX_CAPTURE_ITEMS or budget.remaining_nodes <= 0:
                break
            inspected_items += 1
            captured_items.append(_capture_jsonable(item, depth=depth + 1, budget=budget))
        if len(value) > inspected_items:
            captured_items.append({"_truncated_items": len(value) - inspected_items})
        return captured_items
    # Any other type (a custom object, a set, ...) — never introspect it
    # further; see _opaque_capture_marker's own comment for why.
    return _opaque_capture_marker(value)


def _finalize_capture(captured: Any, original: Any) -> Any:
    try:
        encoded = json.dumps(captured, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except Exception:
        return {"_truncated": True, **_opaque_capture_marker(original)}
    if len(encoded) <= MAX_CAPTURE_JSON_CHARS:
        return captured
    return {
        "_truncated": True,
        **_opaque_capture_marker(original),
        "preview": encoded[:MAX_CAPTURE_PREVIEW_CHARS],
    }


def bounded_capture(value: Any) -> Any:
    """Public entry point: turn `value` (typically a tool call's `params`
    dict) into a JSON-safe, depth/size-bounded, credential-redacted
    structure. Never raises — worst case on an exotic input is an opaque
    marker, never an exception that would break the caller's dataflux
    emit."""
    try:
        budget = _CaptureBudget()
        return _finalize_capture(_capture_jsonable(value, budget=budget), value)
    except Exception:
        return {"_omitted_type": "opaque", "_truncated": True}
