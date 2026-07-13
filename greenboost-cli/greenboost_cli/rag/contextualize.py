"""gb contextualize — batch chunk contextualizer using the local LLM.

Reads JSON from stdin:
  {"document": "<full doc text>", "chunks": ["chunk1", "chunk2", ...]}

For each chunk, calls the local model with the full document in a stable
system-prompt prefix (maximizing vLLM server-side KV-cache reuse across
all chunks of the same document) and the chunk as the user message.

Returns (--json):
  {"contexts": ["ctx1", "ctx2", ...], "model": "<model>", "chunk_count": N}

Returns (human):
  Prints [N] <first 120 chars of context> for each chunk.

Usage:
  echo '{"document": "...", "chunks": ["..."]}' | gb contextualize [--json]
  echo '...' | gb contextualize --model ollama/qwen3:8b --json

This is the Anthropic Contextual Retrieval recipe (context_retrieval.md:66-77):
  For each chunk, Claude responds with 50-100 tokens that situate the chunk
  within the whole document. That context is prepended to the chunk BEFORE
  embedding and BEFORE BM25 indexing — not at search time.
"""
from __future__ import annotations

import argparse
import json
import sys

# How many characters of the document to pass to the model.
# At ~4 chars/token this is ~8 k tokens — safe for most local 8 k+ context models.
# Very large documents are centre-truncated so we keep the beginning and ending
# (title, summary, conclusion) which carry the most identifying metadata.
_MAX_DOC_CHARS = 32_000

# The situating instruction appended after <document>...</document> in the
# system prompt.  Kept here (not in the user turn) so it is part of the stable
# cached prefix and doesn't cost extra tokens per chunk call.
_SYSTEM_SUFFIX = (
    "\n\nFor each user message I will give you a chunk from the document above "
    "enclosed in <chunk>…</chunk> tags. "
    "Reply with 1-3 sentences that situate the chunk within the document "
    "(company name, time period, topic section, etc.) to help a search engine "
    "retrieve the right chunk. "
    "Answer ONLY with the situating context — no preamble, no explanation."
)


def _collect_text(gen) -> str:
    """Drain a router.generate() generator, return the final text."""
    from greenboost_cli.inference.router import StreamFragment, CompletedResponse  # noqa: PLC0415
    parts: list[str] = []
    for event in gen:
        if isinstance(event, StreamFragment):
            parts.append(event.text)
        elif isinstance(event, CompletedResponse):
            if event.text and not parts:
                return event.text.strip()
    return "".join(parts).strip()


def contextualize_chunks(
    document: str,
    chunks: list[str],
    model: str | None = None,
    max_doc_chars: int = _MAX_DOC_CHARS,
) -> list[str]:
    """Python API: contextualize a list of chunk texts, return context strings.

    Returns a list of context strings (same length as chunks). Falls back to
    empty strings if the model or inference fails. Does NOT write to stdout/stderr.

    Unlike Ollama's keep_alive=0, gb-synapse has no soft "evict from VRAM but
    stay ready to reload" primitive — its llama-server stays up until
    explicitly stopped. Deliberately NOT auto-stopping the server here: the
    user's interactive session may be using the same model concurrently.
    """
    if not chunks:
        return []

    if len(document) > max_doc_chars:
        half = max_doc_chars // 2
        doc_trunc = document[:half] + "\n[…truncated…]\n" + document[-half:]
    else:
        doc_trunc = document

    from greenboost_cli.environment.settings import load_settings   # noqa: PLC0415
    from greenboost_cli.inference.router import generate            # noqa: PLC0415

    settings = load_settings()
    model = model or settings.get("model", "")
    if not model:
        return [""] * len(chunks)

    raw_system = f"<document>\n{doc_trunc}\n</document>{_SYSTEM_SUFFIX}"

    ctx_settings: dict = {
        **settings, "temperature": 0.2, "top_p": 0.8,
        "repeat_penalty": 1.05, "num_ctx": 16384, "enable_thinking": False,
    }

    contexts: list[str] = []
    for chunk in chunks:
        user_msg = f"<chunk>\n{chunk}\n</chunk>"
        messages = [{"role": "user", "content": user_msg}]
        try:
            gen = generate(
                model=model,
                system=raw_system,
                messages=messages,
                tool_schemas=[],
                settings=ctx_settings,
            )
            ctx = _collect_text(gen)
        except Exception:    # noqa: BLE001
            ctx = ""
        contexts.append(ctx)

    return contexts


def cmd_contextualize(argv: list[str]) -> int:
    """Handler for `gb contextualize ...`."""
    p = argparse.ArgumentParser(
        prog="gb contextualize",
        description="Batch-contextualize document chunks for Contextual Retrieval.",
        add_help=True,
    )
    p.add_argument(
        "--model", default=None,
        help="Model to use (default: active model from gb settings).",
    )
    p.add_argument(
        "--max-doc-chars", type=int, default=_MAX_DOC_CHARS,
        metavar="N",
        help=f"Truncate document to N chars (default {_MAX_DOC_CHARS}).",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = p.parse_args(argv)

    # ── Read + validate stdin ─────────────────────────────────────────────────
    raw = sys.stdin.read()
    if not raw.strip():
        _emit_err("no input on stdin")
        if args.json:
            _emit_json({"error": "empty stdin", "contexts": []})
        return 1

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _emit_err(f"invalid JSON — {exc}")
        if args.json:
            _emit_json({"error": f"invalid JSON: {exc}", "contexts": []})
        return 1

    document: str = payload.get("document", "")
    chunks: list[str] = payload.get("chunks", [])
    if not document or not chunks:
        _emit_err("'document' and 'chunks' fields are required")
        if args.json:
            _emit_json({"error": "'document' and 'chunks' required", "contexts": []})
        return 1

    # ── Truncate document (centre-cut: keep head + tail) ─────────────────────
    max_chars = max(1024, args.max_doc_chars)
    if len(document) > max_chars:
        half = max_chars // 2
        doc_trunc = document[:half] + "\n[…truncated…]\n" + document[-half:]
        sys.stderr.write(
            f"gb: contextualize: document truncated {len(document)} → {max_chars} chars\n"
        )
    else:
        doc_trunc = document

    # ── Load model name (for display only) ───────────────────────────────────
    from greenboost_cli.environment.settings import load_settings           # noqa: PLC0415
    settings = load_settings()
    model = args.model or settings.get("model", "")
    if not model:
        msg = "no model configured; run: gb /config model=<name>"
        _emit_err(msg)
        if args.json:
            _emit_json({"error": msg, "contexts": []})
        return 1

    if not args.json:
        print(f"  \033[2m·\033[0m  contextualizer: {model}  [gb-synapse]")
        n = len(chunks)
        # progress ticker — contextualize_chunks doesn't print; mimic it
        print(f"  contextualizing {n} chunks…", end="\r", flush=True)

    # ── Run contextualizer ────────────────────────────────────────────────────
    contexts = contextualize_chunks(
        document, chunks,
        model=model,
        max_doc_chars=max(1024, args.max_doc_chars),
    )

    if not args.json:
        print()  # clear progress line

    # ── Emit ──────────────────────────────────────────────────────────────────
    if args.json:
        _emit_json({"contexts": contexts, "model": model, "chunk_count": n})
    else:
        for i, ctx in enumerate(contexts):
            preview = ctx[:120] + ("…" if len(ctx) > 120 else "")
            print(f"  [{i}] {preview}")
    return 0


def _emit_json(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def _emit_err(msg: str) -> None:
    sys.stderr.write(f"gb: {msg}\n")
