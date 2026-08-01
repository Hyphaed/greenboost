"""Contextual Retrieval document RAG engine — native greenboost-cli integration.

Implements the Anthropic Contextual Retrieval recipe:
  1. Parse   — markitdown_adapter (Docling lazy upgrade if installed)
  2. Chunk   — header-aware split (~600 words / ~800 tokens), code-block safe
  3. Contextualize — greenboost_cli.rag.contextualize.contextualize_chunks()
               (no subprocess — direct Python API, always Haiku or configured model)
  4. Embed   — qwen3-embedding:4b via Ollama REST API, 1024-dim Matryoshka truncation
  5. BM25    — rank_bm25.BM25Okapi, enriched text
  6. RRF     — Reciprocal Rank Fusion k=60 (same as knowledge-rag/semble)
  7. Rerank  — CrossEncoder ms-marco-MiniLM-L-6-v2 (CPU-only)

VRAM management (RTX 5070 12 GB):
  Phase 1 contextualize: Haiku cloud (no VRAM) or local → unloaded after
  Phase 2 embed:         qwen3-embedding:4b (~3-4 GB) → unloaded after
  Phase 3 rerank:        CrossEncoder CPU-only
  Never two models in VRAM simultaneously.

State layout:
  ~/.greenboost_cli/contextual_rag/<project>/
    chunks.json     — list of chunk records
    embeddings.npy  — float32 (n_chunks × dims)
    bm25.pkl        — BM25Okapi instance
    meta.json       — {model, dims, time, doc_count, chunk_count}

Public API:
  ingest_document(path, project)   — parse + chunk + contextualize + index
  ingest_folder(folder, project)   — ingest all supported files in a directory
  search(query, project, top_k)    — hybrid+RRF+rerank → list[Hit]
  status(project)                  — print index stats
  clear(project)                   — delete the index
  format_hits(hits, query)         — format for Claude context
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

_BASE_DIR = Path.home() / ".greenboost_cli" / "contextual_rag"

_EMBED_MODEL = "qwen3-embedding:4b"
_EMBED_DIM   = 1024          # Matryoshka truncation
_EMBED_BATCH = 32

_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_RRF_K             = 60
_CHUNK_TARGET_WORDS = 600
_CHUNK_MIN_WORDS   = 50

# The contextualizer model — Haiku for cost/speed + Anthropic prompt caching.
# Override via env var when a local model is preferred.
_CONTEXTUALIZER_MODEL = os.environ.get(
    "CONTEXTUAL_RAG_MODEL", "claude-haiku-4-5-20251001"
)

_SUPPORTED_EXTS = {
    ".md", ".markdown", ".txt", ".rst",
    ".pdf", ".docx", ".html", ".htm",
    ".tex",
}


def _ollama_base() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


# ── ANSI helpers (match lib/ui.sh) ───────────────────────────────────────────

def _ok(msg: str)   -> None: print(f"  \033[32m✓\033[0m  {msg}")
def _warn(msg: str) -> None: print(f"  \033[33m!\033[0m  {msg}", file=sys.stderr)
def _err(msg: str)  -> None: print(f"  \033[31m✗\033[0m  {msg}", file=sys.stderr)
def _info(msg: str) -> None: print(f"  \033[2m·\033[0m  {msg}")
def _kv(k: str, v: str) -> None: print(f"  \033[2m{k:<18}\033[0m  {v}")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Hit:
    source:       str
    heading_path: str
    score:        float
    snippet:      str
    ordinal:      int = 0


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_document(path: Path) -> str:
    """Convert a document to markdown. Returns empty string on failure."""
    ext = path.suffix.lower()

    if ext in (".md", ".markdown", ".txt", ".rst"):
        try:
            return path.read_text(errors="replace")
        except OSError as e:
            _warn(f"read {path.name}: {e}")
            return ""

    # Try Docling first (lazy import — optional heavy dep)
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        converter = DocumentConverter()
        result = converter.convert(str(path))
        md = result.document.export_to_markdown()
        if md.strip():
            return md
    except ImportError:
        pass
    except Exception as exc:
        _warn(f"docling failed for {path.name}: {exc}")

    # Fallback: markitdown_adapter (always available in greenboost-cli)
    try:
        from greenboost_cli.converters.markitdown_adapter import convert  # noqa: PLC0415
        md = convert(str(path), feed_rag=False)
        if isinstance(md, str) and md.strip():
            return md
    except Exception as exc:
        _warn(f"markitdown failed for {path.name}: {exc}")

    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


# ── Chunk ─────────────────────────────────────────────────────────────────────

def chunk_markdown(
    md: str,
    source: str = "",
    target_words: int = _CHUNK_TARGET_WORDS,
    min_words: int = _CHUNK_MIN_WORDS,
) -> list[dict]:
    """Split markdown into header-aware chunks, tracking heading path."""
    HEADER_RE    = re.compile(r"^(#{1,6})\s+(.+)$")
    CODE_FENCE_RE = re.compile(r"^```")
    chunks: list[dict] = []
    heading_stack: dict[int, str] = {}

    def _heading_path() -> str:
        return " > ".join(heading_stack[l] for l in sorted(heading_stack)) if heading_stack else ""

    def _word_count(text: str) -> int:
        return len(text.split())

    def _make_chunk(text: str, heading: str) -> dict | None:
        t = text.strip()
        if _word_count(t) < min_words:
            return None
        return {"text": t, "source": source, "heading_path": heading,
                "ordinal": len(chunks), "enriched": ""}

    lines = md.splitlines(keepends=True)
    current_lines: list[str] = []
    in_code_block = False

    def _flush(h_path: str) -> None:
        nonlocal current_lines
        block = "".join(current_lines).strip()
        if not block:
            current_lines = []
            return
        if _word_count(block) <= target_words * 1.5:
            c = _make_chunk(block, h_path)
            if c:
                chunks.append(c)
        else:
            paras = re.split(r"\n{2,}", block)
            buf = ""
            for para in paras:
                if _word_count(buf) + _word_count(para) <= target_words:
                    buf = (buf + "\n\n" + para).strip() if buf else para
                else:
                    if buf:
                        c = _make_chunk(buf, h_path)
                        if c:
                            chunks.append(c)
                    buf = para
            if buf:
                c = _make_chunk(buf, h_path)
                if c:
                    chunks.append(c)
        current_lines = []

    prev_heading_path = ""
    for line in lines:
        stripped = line.rstrip()
        if CODE_FENCE_RE.match(stripped):
            in_code_block = not in_code_block
        m = HEADER_RE.match(stripped) if not in_code_block else None
        if m:
            _flush(prev_heading_path)
            level = len(m.group(1))
            text  = m.group(2).strip()
            for k in list(heading_stack):
                if k >= level:
                    del heading_stack[k]
            heading_stack[level] = text
            prev_heading_path = _heading_path()
            current_lines = [line]
        else:
            current_lines.append(line)
    _flush(prev_heading_path)

    for i, c in enumerate(chunks):
        c["ordinal"] = i
    return chunks


# ── Contextualize ─────────────────────────────────────────────────────────────

def contextualize(
    document: str,
    chunks: list[dict],
    *,
    model: str = _CONTEXTUALIZER_MODEL,
) -> list[dict]:
    """Prepend situating context to each chunk using contextualize_chunks().

    Mutates chunks in-place (sets 'enriched'), returns the same list.
    Falls back gracefully if inference fails.
    """
    if not chunks:
        return chunks

    from greenboost_cli.rag.contextualize import contextualize_chunks  # noqa: PLC0415

    _info(f"contextualizing {len(chunks)} chunks via {model} …")
    try:
        contexts = contextualize_chunks(
            document,
            [c["text"] for c in chunks],
            model=model,
        )
    except Exception as exc:    # noqa: BLE001
        _warn(f"contextualize failed: {exc} — using raw chunks")
        contexts = []

    for i, c in enumerate(chunks):
        ctx = contexts[i].strip() if i < len(contexts) else ""
        c["enriched"] = f"{ctx}\n\n{c['text']}" if ctx else c["text"]

    return chunks


# ── Ollama embedding ──────────────────────────────────────────────────────────

def _ollama_embed(
    texts: list[str],
    model: str = _EMBED_MODEL,
    dim: int = _EMBED_DIM,
) -> np.ndarray:
    """Embed texts via Ollama /api/embed; Matryoshka truncation + L2-normalise."""
    import urllib.error
    import urllib.request

    url = _ollama_base() + "/api/embed"
    all_vecs: list[list[float]] = []

    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start : start + _EMBED_BATCH]
        payload = json.dumps({"model": model, "input": batch}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama embed failed (is Ollama running? model '{model}' pulled?): {exc}"
            ) from exc
        vecs = data.get("embeddings", [])
        if len(vecs) != len(batch):
            raise RuntimeError(f"Ollama returned {len(vecs)} embeddings for {len(batch)} texts")
        all_vecs.extend(vecs)

    mat = np.array(all_vecs, dtype=np.float32)
    if mat.shape[1] > dim:
        mat = mat[:, :dim]
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / norms


# ── Index persistence ─────────────────────────────────────────────────────────

def _project_dir(project: str) -> Path:
    d = _BASE_DIR / re.sub(r"[^\w.-]", "_", project)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_index(project: str) -> dict | None:
    d = _project_dir(project)
    chunks_f, emb_f, bm25_f, meta_f = (
        d / "chunks.json", d / "embeddings.npy", d / "bm25.pkl", d / "meta.json"
    )
    if not all(f.exists() for f in (chunks_f, emb_f, bm25_f, meta_f)):
        return None
    try:
        chunks = json.loads(chunks_f.read_text())
        embeddings = np.load(str(emb_f))
        with open(bm25_f, "rb") as fh:
            bm25 = pickle.load(fh)
        meta = json.loads(meta_f.read_text())
        return {"chunks": chunks, "embeddings": embeddings, "bm25": bm25, "meta": meta}
    except Exception as exc:
        _warn(f"failed to load index for '{project}': {exc}")
        return None


def _save_index(
    project: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    bm25: Any,
    meta: dict,
) -> None:
    d = _project_dir(project)
    (d / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    np.save(str(d / "embeddings.npy"), embeddings.astype(np.float32))
    with open(d / "bm25.pkl", "wb") as fh:
        pickle.dump(bm25, fh)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _get_reranker():
    from sentence_transformers import CrossEncoder  # type: ignore
    return CrossEncoder(_RERANK_MODEL)


def _build_index(project: str, chunks: list[dict]) -> None:
    """Embed enriched chunks, build BM25, merge with existing, persist."""
    if not chunks:
        return

    # _ollama_unload moved to the sibling greenboost repo's gb_reclaim.py
    # (task #7 consolidation , it's also gb_reclaim's own graceful-unload
    # step now, one implementation instead of two). Same cross-repo import
    # convention as slash_commands/backend_cmds.py's _import_gb_synapse.
    from greenboost_cli.gb_paths import gb_module  # noqa: PLC0415
    _ollama_unload = gb_module("gb_reclaim")._ollama_unload

    enriched_texts = [c["enriched"] or c["text"] for c in chunks]

    _info(f"embedding {len(chunks)} chunks via {_EMBED_MODEL} (dim={_EMBED_DIM}) …")
    try:
        embeddings = _ollama_embed(enriched_texts)
    except RuntimeError as exc:
        _err(str(exc))
        _err("Hint: ollama pull qwen3-embedding:4b")
        return
    _ollama_unload(_EMBED_MODEL, silent=True)
    _info(f"unloaded {_EMBED_MODEL} from VRAM")

    _info("building BM25 index …")
    from rank_bm25 import BM25Okapi  # type: ignore  # noqa: PLC0415
    corpus_tokens = [_tokenize_bm25(t) for t in enriched_texts]

    existing = _load_index(project)
    if existing:
        old_chunks  = existing["chunks"]
        old_emb     = existing["embeddings"]
        old_corpus  = [_tokenize_bm25(c["enriched"] or c["text"]) for c in old_chunks]
        all_chunks  = old_chunks + chunks
        all_emb     = np.vstack([old_emb, embeddings])
        merged_bm25 = BM25Okapi(old_corpus + corpus_tokens)
        for i, c in enumerate(all_chunks):
            c["ordinal"] = i
        meta = {
            "model":       _EMBED_MODEL,
            "dims":        int(all_emb.shape[1]),
            "time":        time.time(),
            "doc_count":   len({c["source"] for c in all_chunks}),
            "chunk_count": len(all_chunks),
        }
        _save_index(project, all_chunks, all_emb, merged_bm25, meta)
    else:
        meta = {
            "model":       _EMBED_MODEL,
            "dims":        int(embeddings.shape[1]),
            "time":        time.time(),
            "doc_count":   len({c["source"] for c in chunks}),
            "chunk_count": len(chunks),
        }
        _save_index(project, chunks, embeddings, BM25Okapi(corpus_tokens), meta)


# ── RRF ───────────────────────────────────────────────────────────────────────

def _rrf_fuse(
    sem_order: list[int],
    bm25_order: list[int],
    k: int = _RRF_K,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(sem_order):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_order):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_document(path_str: str, project: str = "default") -> bool:
    """Parse, chunk, contextualize, embed, and index one document."""
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        _err(f"not found: {path}")
        return False
    if path.suffix.lower() not in _SUPPORTED_EXTS:
        _warn(f"unsupported extension '{path.suffix}' — skipping {path.name}")
        return False

    _info(f"parsing  {path.name} …")
    doc_text = parse_document(path)
    if not doc_text.strip():
        _warn(f"no content extracted from {path.name}")
        return False

    _info(f"chunking {path.name} …")
    chunks = chunk_markdown(doc_text, source=str(path))
    if not chunks:
        _warn(f"no chunks from {path.name}")
        return False
    _info(f"  {len(chunks)} chunks")

    chunks = contextualize(doc_text, chunks)
    _build_index(project, chunks)
    _ok(f"{path.name}  →  {len(chunks)} chunks indexed  [{project}]")
    return True


def ingest_folder(folder_str: str, project: str = "default") -> dict:
    """Ingest all supported files in a folder (non-recursive)."""
    folder = Path(folder_str).expanduser().resolve()
    if not folder.is_dir():
        _err(f"not a directory: {folder}")
        return {"indexed": 0, "skipped": 0, "errors": 0}

    files = [f for f in sorted(folder.iterdir())
             if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS]
    if not files:
        _warn(f"no supported files in {folder}")
        return {"indexed": 0, "skipped": 0, "errors": 0}

    indexed = errors = 0
    for f in files:
        ok = ingest_document(str(f), project=project)
        if ok:
            indexed += 1
        else:
            errors += 1

    print()
    _ok(f"folder done — {indexed} docs indexed, {errors} errors  [{project}]")
    return {"indexed": indexed, "skipped": 0, "errors": errors}


def search(
    query: str,
    project: str = "default",
    top_k: int = 10,
    candidates: int = 20,
) -> list[Hit]:
    """Hybrid BM25 + vector search with RRF fusion and cross-encoder reranking."""
    index = _load_index(project)
    if index is None:
        return []

    chunks:     list[dict]   = index["chunks"]
    embeddings: np.ndarray   = index["embeddings"]
    bm25                     = index["bm25"]
    n = len(chunks)
    if n == 0:
        return []

    embed_model = index["meta"].get("model", _EMBED_MODEL)
    embed_dim   = index["meta"].get("dims",  _EMBED_DIM)
    q_mat = _ollama_embed([query], model=embed_model, dim=embed_dim)
    q_vec = q_mat[0]

    sem_scores = embeddings @ q_vec
    sem_order  = np.argsort(sem_scores)[::-1].tolist()

    q_tokens  = _tokenize_bm25(query)
    bm25_raw  = bm25.get_scores(q_tokens)
    bm25_order = np.argsort(bm25_raw)[::-1].tolist()

    n_cand = min(candidates, n)
    fused  = _rrf_fuse(sem_order[:n_cand], bm25_order[:n_cand])
    top_idx = [ci for ci, _ in fused[:n_cand]]

    try:
        reranker = _get_reranker()
        pairs     = [(query, chunks[i]["enriched"] or chunks[i]["text"]) for i in top_idx]
        re_scores = reranker.predict(pairs).tolist()
        reranked  = sorted(zip(top_idx, re_scores), key=lambda x: x[1], reverse=True)
        final     = reranked[:top_k]
    except Exception as exc:    # noqa: BLE001
        _warn(f"reranker failed, using RRF order: {exc}")
        final = [(i, s) for i, s in fused[:top_k]]

    return [
        Hit(
            source=chunks[i]["source"],
            heading_path=chunks[i].get("heading_path", ""),
            score=round(float(s), 4),
            snippet=chunks[i]["enriched"] or chunks[i]["text"],
            ordinal=chunks[i].get("ordinal", i),
        )
        for i, s in final
    ]


def status(project: str = "default") -> None:
    """Print index statistics."""
    idx = _load_index(project)
    d = _project_dir(project)
    if idx is None:
        _warn(f"no index found for project '{project}'")
        _kv("path", str(d))
        return
    meta = idx["meta"]
    _kv("project",   project)
    _kv("path",      str(d))
    _kv("model",     meta.get("model", "?"))
    _kv("dims",      str(meta.get("dims", "?")))
    _kv("chunks",    str(meta.get("chunk_count", len(idx["chunks"]))))
    _kv("documents", str(meta.get("doc_count", "?")))
    ts = meta.get("time")
    if ts:
        import datetime
        _kv("indexed", datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"))


def clear(project: str = "default") -> None:
    """Delete the index for a project."""
    d = _project_dir(project)
    removed = []
    for fname in ("chunks.json", "embeddings.npy", "bm25.pkl", "meta.json"):
        f = d / fname
        if f.exists():
            f.unlink()
            removed.append(fname)
    if removed:
        _ok(f"cleared {len(removed)} index files for '{project}'")
    else:
        _warn(f"nothing to clear for '{project}'")


def format_hits(hits: list[Hit], query: str) -> str:
    """Format hits for Claude's context (matches gb's format_for_claude style)."""
    if not hits:
        return f"No contextual RAG results for: {query}"
    lines = [f"Contextual RAG — query: {query!r}\n"]
    for i, h in enumerate(hits, 1):
        heading = f"  [{h.heading_path}]" if h.heading_path else ""
        lines.append(f"[{i}] {h.source}{heading}  (score {h.score})")
        lines.append(h.snippet[:800] + ("…" if len(h.snippet) > 800 else ""))
        lines.append("")
    return "\n".join(lines)
