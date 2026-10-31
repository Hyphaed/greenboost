"""Code-aware RAG engine.

Uses PyTorch tensors + sentence-transformers dot-product search (no FAISS required).
Persists index to ~/.greenboost_cli/rag/ as embeddings.npy + metadata.json.
"""
from __future__ import annotations

import json
import os
import sys
import re
import threading
import time
import uuid
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from greenboost_cli.environment.settings import GB_HOME

# Serializes load→mutate→save of the shared embeddings/metadata store across the
# REPL's auto-RAG writer, the startup auto-update thread, and the in-process
# dashboard server.  RLock so a holder may call into other locked helpers.
_store_lock = threading.RLock()
# Guards lazy model construction (startup thread may race the first query).

RAG_DIR          = GB_HOME / "rag"
# The live index is ONE directory per generation (rag/generations/<id>/ holding
# embeddings.npy, metadata.json, manifest.json) and rag/current is a symlink to the
# active one. Switching generations is a single atomic rename, so vectors and chunk
# text can never come from different writes (they used to be two os.replace calls,
# and a crash between them left new vectors paired with old text). These two names
# are kept for the callers that stat or delete them; the engine itself derives the
# live paths from RAG_DIR at call time via _emb_path()/_meta_path().
EMBEDDINGS_FILE  = RAG_DIR / "current" / "embeddings.npy"
METADATA_FILE    = RAG_DIR / "current" / "metadata.json"
CHUNKER_VERSION  = "1"          # bump when _chunk_* output changes: old vectors then no longer match new chunks
#: Generations kept on disk: the active one plus one previous, for recovery.
#: A full store is ~850 MB here, so this bounds the cost at ~1.7 GB, not "one per save".
KEEP_GENERATIONS = 2
FOLDERS_FILE     = RAG_DIR / "indexed_folders.yaml"
WEB_SOURCES_FILE = RAG_DIR / "web_sources.json"   # registry of fetchable web URLs

DEFAULT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".yaml", ".yml", ".toml",
    ".json", ".md", ".txt", ".env.example",
}

EXCLUDE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".cache",
    ".pytest_cache", "eggs", ".eggs",
}


def _read_hf_token() -> str | None:
    p = Path.home() / ".cache" / "huggingface" / "token"
    if p.exists():
        return p.read_text().strip() or None
    return None


# ── Embedding model (lazy-loaded) ─────────────────────────────────────────────
#
# Placement policy (owner decision 2026-10-30, measured on this box):
#   * Searching (a query, a skill match, a small batch) runs on the CPU. One
#     query costs ~60-90 ms with 8-16 threads and touches no VRAM, so the served
#     35B model keeps the whole card. Qwen3-Embedding-0.6B needs ~1.2 GB; with
#     the 35B loaded there is less than that free.
#   * Bulk indexing (a batch of GB_EMBED_BULK_MIN texts or more, and /rag
#     reembed) runs on the GPU when the card has room, then RELEASES it: on the
#     CPU the same work is ~2-4 s per 1,500-char chunk, which is days for the
#     297k-chunk store, and it would take cores the 35B's CPU experts use.
#   * GB_EMBED_DEVICE=cuda restores "everything on the GPU"; =cpu forces the CPU.
# This is a measured placement against the GPU alternative, not a silent
# fallback, and the CPU path is the exception the owner approved for embeddings.

_models: dict = {}                    # device -> loaded SentenceTransformer
_models_lock = threading.Lock()
_model = None                         # most recently loaded model (introspection only)
_embed_device = None
EMBED_MODEL   = os.environ.get("GB_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_FALLBACK = "all-MiniLM-L6-v2"


def _resolve_device() -> str:
    """The CUDA device check, kept for the GPU path: no CUDA is a real fault."""
    global _embed_device
    if _embed_device is not None:
        return _embed_device
    try:
        import torch
        if torch.cuda.is_available():
            _embed_device = "cuda"
        else:
            raise RuntimeError(
                "torch.cuda.is_available() returned False , check CUDA driver and "
                "torch+cuXXX version match. Bulk embedding needs the GPU; searching "
                "(GB_EMBED_DEVICE=auto) does not.")
    except ImportError:
        _embed_device = "cpu"
    return _embed_device


_fallback_warned = False


def _warn_embed_fallback(err: Exception) -> None:
    """Say, once, that search is running on the weaker fallback embedder.

    This fallback was silent: under transformers 5 the old primary code embedder
    can't load at all (its remote code imports a function transformers removed),
    so every RAG/skill search ran on MiniLM with no word to anyone. Measured on
    200 functions of this repo (docstring -> body): MiniLM top-1 0.52, against
    0.81 for Qwen3-Embedding-0.6B.
    """
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    reason = " ".join(str(err).split())[:140]
    msg = (f"Code search is using the small fallback embedder ({EMBED_FALLBACK}), "
           f"because {EMBED_MODEL} would not load ({reason}). Search still works, "
           f"but it finds the right function first far less often in our "
           f"measurement. Nothing is lost: the index records which model built it. "
           f"Check GB_EMBED_MODEL and that sentence-transformers imports, then restart.")
    try:
        from greenboost_cli.terminal.theme import emit_warn
        emit_warn(msg)
    except Exception:
        print(f"[greenboost-cli] {msg}", file=sys.stderr)


#: Qwen3-Embedding is trained to read "Instruct: <task>\nQuery: <query>" on the SEARCH side
#: only. The prompt the model ships is about web search, and measured no better than no
#: instruction at all (top-1 0.79 vs 0.79 on 200 functions of this repo); a code-oriented
#: one gave 0.84 and MRR 0.892 vs 0.859. Documents are never prefixed, so an existing index
#: stays valid. GB_EMBED_QUERY_PROMPT overrides the code task ("none" = the model's own).
_QUERY_TASKS = {
    "code": "Given a question about a codebase or its documentation, retrieve the code or "
            "text that answers it",
    "skill": "Given a user's request, retrieve the description of the skill that applies to it",
}


def _query_prompt(model, task: str) -> dict:
    """encode() kwargs that put the right instruction in front of a query ({} = none)."""
    if "query" not in (getattr(model, "prompts", None) or {}):
        return {}                                    # models without a query prompt (MiniLM, ...)
    override = os.environ.get("GB_EMBED_QUERY_PROMPT", "").strip()
    if override.lower() == "none":
        return {"prompt_name": "query"}
    text = override if (override and task == "code") else _QUERY_TASKS.get(task, _QUERY_TASKS["code"])
    return {"prompt": f"Instruct: {text}\nQuery: "}


class EmbedDeferred(RuntimeError):
    """A batch too big to embed on the CPU was refused; run it when the GPU is free."""


def _cpu_bulk_cap() -> int:
    """Most chunks one call may embed on the CPU (0 = no limit). Qwen3-Embedding-0.6B
    costs ~2.3 s per 1,500-char chunk there (measured), so a few hundred is minutes and
    thousands is hours."""
    try:
        return max(0, int(os.environ.get("GB_EMBED_CPU_MAX_CHUNKS", "300")))
    except ValueError:
        return 300


def _policy() -> str:
    p = os.environ.get("GB_EMBED_DEVICE", "auto").lower()
    return p if p in ("auto", "cpu", "cuda") else "auto"


def _bulk_min() -> int:
    try:
        return max(1, int(os.environ.get("GB_EMBED_BULK_MIN", "64")))
    except ValueError:
        return 64


def _cpu_threads() -> int:
    """About a third of the logical CPUs: enough for ~90 ms queries (measured at 8
    of 24), few enough that the served model's CPU experts keep their cores."""
    return max(2, (os.cpu_count() or 8) // 3)


def _gpu_has_room_for(model) -> bool:
    """Weights x1.25 for activations, plus the 10% VRAM headroom RULE #1 keeps."""
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        need = int(sum(p.numel() * p.element_size() for p in model.parameters()) * 1.25)
        return free - need >= int(total * 0.10)
    except Exception:
        return False


def _load(device: str, name: str | None = None):
    """Load the primary embedder on `device`, else the fallback on the same device.

    `name` loads exactly that model (the one an existing index was built with)
    and changes no module state."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as _e:
        _cause = str(_e)
        if "torch" in _cause.lower() or "torchao" in _cause.lower() or "int1" in _cause:
            raise ImportError(
                f"sentence-transformers import failed due to a torch/torchao version mismatch: {_cause}\n"
                "Fix: upgrade torch in the active env:\n"
                "  pip install --upgrade 'torch>=2.0' --index-url https://download.pytorch.org/whl/cu130"
            ) from _e
        raise ImportError(
            f"sentence-transformers not installed or failed to import ({_cause}).\n"
            "Run: pip install 'greenboost-cli[rag]'"
        ) from _e

    global _active_embed_model, _model
    import warnings
    warnings.filterwarnings("ignore", message=".*optimum.*", category=UserWarning)
    hf_token = os.environ.get("HF_TOKEN") or _read_hf_token()
    # On the CPU, float32: transformers 5 keeps the checkpoint's bfloat16, which the
    # CPU runs ~3x slower (283 ms vs ~90 ms per query measured here).
    mk = {}
    if device == "cpu":
        try:
            import torch
            mk = {"model_kwargs": {"dtype": torch.float32}}
        except Exception:
            pass
    if name:
        return SentenceTransformer(name, device=device, token=hf_token, **mk)
    try:
        m = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device, token=hf_token, **mk)
        _active_embed_model = EMBED_MODEL
    except Exception as _primary_err:
        m = SentenceTransformer(EMBED_FALLBACK, device=device, token=hf_token, **mk)
        _active_embed_model = EMBED_FALLBACK       # recorded in the manifest, so it can't pass as the primary
        _warn_embed_fallback(_primary_err)
    try:        # Qwen3-Embedding ships max_seq_length 32768: bound it (4,000-char chunks are ~1,000 tokens)
        m.max_seq_length = min(int(m.max_seq_length or 1024), 1024)
    except Exception:
        pass
    _model = m
    return m


def _get_model(device: str | None = None, name: str | None = None):
    """The embedder on `device` ("cpu" | "cuda"); None = the GPU path.
    `name` = a specific model, cached under (device, name)."""
    device = device or _resolve_device()
    key = (device, name) if name else device
    with _models_lock:
        m = _models.get(key)
        if m is not None:
            return m
        if device == "cuda":
            # Apply GreenBoost env before loading model onto GPU
            try:
                from greenboost_cli.greenboost.gb_torch import apply_gb_torch_env
                apply_gb_torch_env()
            except ImportError:
                pass
        else:
            try:
                import torch
                # never ABOVE what a caller already chose: the RAG auto-update thread
                # sets 1 on purpose (a background reindex once froze input for a minute)
                torch.set_num_threads(min(torch.get_num_threads(), _cpu_threads()))
            except Exception:
                pass
        m = _load(device, name)
        _models[key] = m
        return m


def release_gpu_embedder() -> None:
    """Hand the GPU back (the served model needs it). Safe to call any time."""
    global _model
    with _models_lock:
        m = _models.pop("cuda", None)
        if m is not None and _model is m:
            _model = None            # _load() also parks the newest model here; that reference alone kept ~1.2 GB alive
    if m is not None:
        del m
        try:
            import torch
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def _pick_device(n_texts: int, bulk: bool | None) -> str:
    p = _policy()
    if p == "cpu":
        return "cpu"
    if p == "cuda":
        return "cuda"
    if bulk is None:
        bulk = n_texts >= _bulk_min()
    if not bulk:
        return "cpu"
    # Bulk: the GPU if there is room for it right now, else the CPU (slow, but a
    # batch this size is bounded; /rag reembed refuses instead, see reembed_store).
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        if _models.get("cuda") is not None:
            return "cuda"
        return "cuda" if _gpu_has_room_for(_get_model("cpu")) else "cpu"
    except Exception:
        return "cpu"


#: Texts embedded per slice when a progress callback is given. Slices are taken from the
#: length-sorted order (as encode() does internally), so padding stays low.
_PROGRESS_SLICE = 256


def _embed_with_progress(model, texts: list[str], kw: dict, device: str, progress) -> np.ndarray:
    """Embed in slices, calling progress(done, total, eta_seconds | None, device) after each.

    The ETA is weighted by characters, not by count: slices are length-sorted, so the long
    ones come last and a plain count-based countdown would run optimistic then stall."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    total_chars = sum(len(t) for t in texts) or 1
    out = None
    done = chars = 0
    t0 = time.time()
    progress(0, len(texts), None, device)
    for s in range(0, len(order), _PROGRESS_SLICE):
        idx = order[s:s + _PROGRESS_SLICE]
        vecs = model.encode([texts[i] for i in idx], convert_to_tensor=True, show_progress_bar=False, **kw)
        arr = vecs.float().cpu().numpy().astype(np.float32)
        if out is None:
            out = np.empty((len(texts), arr.shape[1]), dtype=np.float32)
        out[idx] = arr
        done += len(idx)
        chars += sum(len(texts[i]) for i in idx)
        elapsed = max(time.time() - t0, 1e-6)
        eta = (total_chars - chars) / (chars / elapsed) if chars else None
        progress(done, len(texts), eta, device)
    return out


def _embed(texts: list[str], *, query: bool = False, bulk: bool | None = None,
           model_name: str | None = None, cpu_max: int | None = None,
           task: str = "code", progress=None) -> np.ndarray:
    """Embed `texts`. `query=True` applies the model's own query prompt when it
    ships one (Qwen3-Embedding's "Instruct: ... Query:" form), which it was
    trained to expect on the search side only. `bulk=None` decides by batch size."""
    # An explicit model (search against an index built by an older embedder) is
    # always a small CPU job: the vectors must come from the model that made the index.
    device = "cpu" if model_name else _pick_device(len(texts), bulk)
    if device == "cpu" and cpu_max and len(texts) > cpu_max:
        raise EmbedDeferred(
            f"{len(texts):,} chunks need embedding and the GPU has no room for the embedder "
            f"right now (the CPU would take about {len(texts) * 2.3 / 3600:.1f} h and slow the "
            f"model). Nothing was changed and search still works. Free the card (/exit "
            f"unloads the model) and run /rag-update again, or set "
            f"GB_EMBED_CPU_MAX_CHUNKS=0 to allow the CPU.")
    model = _get_model(device, model_name)
    kw = _query_prompt(model, task) if query else {}
    vecs = None
    try:
        if progress is not None and len(texts) > _PROGRESS_SLICE:
            out = _embed_with_progress(model, texts, kw, device, progress)
        else:
            # Never show tqdm's progress bar: it writes raw \r-based output to stderr,
            # bypassing prompt_toolkit's patch_stdout, which corrupts the live REPL
            # screen (cursor-position-report misreads show up as stray Escape input).
            vecs = model.encode(texts, convert_to_tensor=True, show_progress_bar=False, **kw)
            out = vecs.float().cpu().numpy().astype(np.float32)
    finally:
        # A GPU embedder is held only while there is bulk work: the served model
        # gets its card back the moment this batch is done. This frame's own references
        # go FIRST: release_gpu_embedder() empties the allocator cache, and weights that
        # a local variable still holds are not freed yet, so ~1.2 GB stayed allocated to
        # this process for the rest of the session (measured: "GPU: python3 1.5G" after a
        # /rag-update), which cut the model server's expert-cache room and its decode speed.
        model = vecs = None
        if device == "cuda" and _policy() == "auto":
            release_gpu_embedder()
    return out


# ── Code-aware chunker ────────────────────────────────────────────────────────

CHUNK_PATTERNS = {
    ".py":   re.compile(r"^(def |class |async def )", re.MULTILINE),
    ".js":   re.compile(r"^(function |class |const \w+ = |export )", re.MULTILINE),
    ".ts":   re.compile(r"^(function |class |const \w+ = |export |interface |type )", re.MULTILINE),
    ".tsx":  re.compile(r"^(function |class |const \w+ = |export |interface |type )", re.MULTILINE),
    ".jsx":  re.compile(r"^(function |class |const \w+ = |export )", re.MULTILINE),
    ".go":   re.compile(r"^(func |type |var |const )", re.MULTILINE),
    ".rs":   re.compile(r"^(fn |pub fn |struct |impl |enum |trait |type )", re.MULTILINE),
    ".java": re.compile(r"^(\s*(public|private|protected|static|final)\s+)", re.MULTILINE),
    ".rb":   re.compile(r"^(def |class |module )", re.MULTILINE),
}

WINDOW_SIZE = 30
WINDOW_STEP = 25
MIN_TOKENS  = 10


def _chunk_code(content: str, ext: str) -> list[tuple[int, int, str]]:
    lines = content.splitlines()
    if not lines:
        return []

    pattern = CHUNK_PATTERNS.get(ext)
    if pattern:
        boundaries = [0]
        for m in pattern.finditer(content):
            line_no = content[:m.start()].count("\n")
            if line_no > 0 and line_no not in boundaries:
                boundaries.append(line_no)
        boundaries.append(len(lines))

        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end   = boundaries[i + 1]
            chunk = "\n".join(lines[start:end]).strip()
            if (end - start) > WINDOW_SIZE * 2:
                sub = _sliding_window(lines[start:end], start)
                chunks.extend(sub)
            elif chunk and len(chunk.split()) >= MIN_TOKENS:
                chunks.append((start + 1, end, chunk))
        return chunks

    return _sliding_window(lines, 0)


def _sliding_window(lines: list[str], offset: int) -> list[tuple[int, int, str]]:
    chunks = []
    for i in range(0, max(1, len(lines) - WINDOW_SIZE + 1), WINDOW_STEP):
        segment = lines[i:i + WINDOW_SIZE]
        text = "\n".join(segment).strip()
        if text and len(text.split()) >= MIN_TOKENS:
            chunks.append((offset + i + 1, offset + i + len(segment), text))
    return chunks


def _chunk_markdown(content: str) -> list[tuple[int, int, str]]:
    """Split markdown on headers and paragraphs, with sliding-window fallback."""
    lines = content.splitlines()
    if not lines:
        return []

    header_re = re.compile(r"^#{1,4}\s")
    chunks: list[tuple[int, int, str]] = []
    start = 0
    current: list[str] = []

    def _flush(start_line: int, end_line: int, acc: list[str]) -> None:
        text = "\n".join(acc).strip()
        if len(text.split()) >= 3:  # prose sections can be short
            chunks.append((start_line + 1, end_line, text[:4000]))

    for i, line in enumerate(lines):
        if header_re.match(line) and current:
            _flush(start, i, current)
            start = i
            current = [line]
        else:
            current.append(line)

    _flush(start, len(lines), current)

    if not chunks:
        return _sliding_window(lines, 0)

    # Sub-chunk any section that's too large
    final: list[tuple[int, int, str]] = []
    for s, e, text in chunks:
        if (e - s) > WINDOW_SIZE * 2:
            sub_lines = text.splitlines()
            final.extend(_sliding_window(sub_lines, s - 1))
        else:
            final.append((s, e, text))
    return final


# ── Storage helpers ───────────────────────────────────────────────────────────

class RagIndexMismatch(RuntimeError):
    """The index on disk can't be trusted to match what's being asked of it
    (different embedding model, different dimensions, vectors and chunks that
    disagree). Raised instead of quietly returning wrong neighbours."""


#: The model that actually produced the vectors written in this process. Set by
#: _get_model(): the fallback model has a different dimension, and writing its
#: vectors under the primary model's name is how an index gets silently mixed.
_active_embed_model: str = EMBED_MODEL


def _live_dir() -> Path:
    return RAG_DIR / "current"


def _emb_path() -> Path:
    return _live_dir() / "embeddings.npy"


def _meta_path() -> Path:
    return _live_dir() / "metadata.json"


def _manifest_path() -> Path:
    return _live_dir() / "manifest.json"


def read_manifest() -> dict:
    """The active generation's manifest, or {} (no store, or an unreadable one)."""
    try:
        return json.loads(_manifest_path().read_text())
    except (OSError, ValueError):
        return {}


def _switch_current(gid: str) -> None:
    """Point rag/current at generation `gid` with one atomic rename."""
    tmp = RAG_DIR / "current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(Path("generations") / gid)       # relative: the rag dir stays relocatable
    cur = RAG_DIR / "current"
    if cur.exists() and not cur.is_symlink():
        # A real directory where the symlink belongs (made by hand, or by a tool
        # that predates it). A rename can't replace a directory with a symlink, so
        # move it aside as a generation; the next prune removes it.
        os.rename(cur, RAG_DIR / "generations" / f"g-adopted-{uuid.uuid4().hex[:6]}")
    os.replace(tmp, cur)


def _migrate_legacy() -> None:
    """Adopt a pre-generations store (rag/embeddings.npy + rag/metadata.json) as
    the first generation. Hard links, not copies: no second 850 MB, and every step
    is idempotent, so a crash part-way is finished by the next call."""
    legacy_e, legacy_m = RAG_DIR / "embeddings.npy", RAG_DIR / "metadata.json"
    if (RAG_DIR / "current").exists() or not (legacy_e.exists() and legacy_m.exists()):
        return
    gid = "g-legacy"
    gdir = RAG_DIR / "generations" / gid
    gdir.mkdir(parents=True, exist_ok=True)
    for src, name in ((legacy_e, "embeddings.npy"), (legacy_m, "metadata.json")):
        dst = gdir / name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
    try:
        arr = np.load(str(gdir / "embeddings.npy"), mmap_mode="r")       # header only
        rows, dims = int(arr.shape[0]), int(arr.shape[1]) if arr.ndim == 2 else 0
    except Exception:
        rows, dims = None, None
    (gdir / "manifest.json").write_text(json.dumps({
        "generation": gid, "created_at": time.time(), "legacy": True,
        "n_chunks": rows, "dims": dims, "dtype": "float32",
        "embed_model": None,            # unknown: it predates the manifest
        "chunker_version": None, "metadata_sha256": None,
    }, indent=2))
    _switch_current(gid)
    for src in (legacy_e, legacy_m):
        try:
            src.unlink()
        except OSError:
            pass


def _load_store() -> tuple[np.ndarray | None, list[dict]]:
    _migrate_legacy()
    if _emb_path().exists() and _meta_path().exists():
        embeddings = np.load(str(_emb_path()))
        with open(_meta_path()) as f:
            metadata = json.load(f)
        if embeddings.ndim == 2 and len(metadata) != embeddings.shape[0] and embeddings.size:
            raise RagIndexMismatch(
                f"the index is inconsistent: {embeddings.shape[0]} vectors but "
                f"{len(metadata)} chunks. Rebuild it with /rag clear, then re-index.")
        return embeddings, metadata
    return None, []


def _guard_append(existing_emb, new_emb) -> None:
    """Refuse to mix vectors from a different model or dimension into this index.
    Same-dimension models would otherwise combine without any error at all."""
    if existing_emb is None or new_emb is None or not getattr(existing_emb, "size", 0):
        return
    if existing_emb.shape[1] != new_emb.shape[1]:
        raise RagIndexMismatch(
            f"the index holds {existing_emb.shape[1]}-dim vectors but the embedding model "
            f"'{_active_embed_model}' produces {new_emb.shape[1]}. Run /rag-reembed to rebuild it with the new model.")
    built_with = read_manifest().get("embed_model")
    if built_with and built_with != _active_embed_model:
        raise RagIndexMismatch(
            f"the index was built with '{built_with}' but this process is embedding with "
            f"'{_active_embed_model}'. Mixing them would return wrong neighbours. "
            f"Run /rag-reembed to rebuild it with the new model (searches keep working "
            f"on the old index until then), or set GB_EMBED_MODEL='{built_with}'.")


#: Tiny sidecar holding only what count-only callers need. Written beside the
#: store on every save; regenerated on demand if absent or stale.
STATS_FILE = RAG_DIR / "stats.json"


def store_stats() -> dict:
    """Chunk and file counts, WITHOUT parsing the whole store.

    `_load_store()` is called from ten places, and several of them want nothing
    but `len(metadata)` — the startup banner, `/doctor`, the MCP status line.
    On this box that meant parsing a 413 MB JSON (measured: 964 MB peak RSS,
    ~2 s) to print "272,080 chunks", at every single startup.

    Falls back to the full parse ONLY when the sidecar is missing or older than
    the store it describes, and writes the sidecar on the way out so the cost is
    paid once rather than every time.

    Returns {"chunks": int, "files": int} — zeros when there is no store at all,
    which is a real answer, not a failure.
    """
    _migrate_legacy()
    if not _meta_path().exists():
        return {"chunks": 0, "files": 0}
    fp = _store_fingerprint()
    try:
        if STATS_FILE.exists():
            data = json.loads(STATS_FILE.read_text())
            if (isinstance(data.get("chunks"), int)
                    and isinstance(data.get("files"), int)
                    and data.get("fingerprint") == fp):
                return {"chunks": data["chunks"], "files": data["files"]}
    except (OSError, ValueError):
        pass    # unreadable or malformed sidecar — fall through and rebuild it
    try:
        with open(_meta_path()) as f:
            metadata = json.load(f)
    except (OSError, ValueError):
        return {"chunks": 0, "files": 0}
    stats = {
        "chunks": len(metadata),
        "files": len({m.get("file") for m in metadata if isinstance(m, dict)}),
    }
    _write_stats(stats)
    return stats


def _store_fingerprint() -> "str | None":
    """Identity of the store this sidecar claims to describe: size + mtime.

    Mtime ORDERING alone is not enough, and trusting it produced a real wrong
    answer: a sidecar written by something else was newer than a 413 MB store
    it had never read, so the banner confidently reported 3 chunks instead of
    272,080. "Newer" says nothing about "describes THIS file". A fingerprint
    that must match makes any foreign or hand-edited sidecar self-invalidating,
    and costs one stat() to check.
    """
    try:
        st = _meta_path().stat()
        return f"{st.st_size}:{int(st.st_mtime_ns)}"
    except OSError:
        return None


def _write_stats(stats: dict) -> None:
    """Best-effort sidecar write; a failure here only costs a re-parse later.

    Stamps the store's fingerprint so the sidecar can only ever be believed
    about the exact file it was computed from.
    """
    try:
        RAG_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(stats)
        payload["fingerprint"] = _store_fingerprint()
        tmp = STATS_FILE.with_name(STATS_FILE.name + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, STATS_FILE)
    except OSError:
        pass


def _prune_generations(keep: "set[str]") -> None:
    """Bound disk use: the active generation plus the newest others, up to
    KEEP_GENERATIONS. Also sweeps a generation a crashed save left behind."""
    gens = RAG_DIR / "generations"
    import shutil
    try:
        all_dirs = [d for d in gens.iterdir() if d.is_dir()]
    except OSError:
        return
    # A generation without a manifest is a save that crashed before finishing.
    # It must never count as a "previous" generation, or it would evict the real one.
    complete = sorted((d for d in all_dirs if (d / "manifest.json").exists()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
    survivors = set(keep) | {d.name for d in complete[:KEEP_GENERATIONS]}
    for d in all_dirs:
        if d.name not in survivors:
            shutil.rmtree(d, ignore_errors=True)


def _save_store(embeddings: np.ndarray, metadata: list[dict]) -> None:
    """Write a complete new generation, verify it, then switch to it atomically.

    Nothing live is touched until the new generation is whole: a crash at any
    point before the final rename leaves the previous generation active and
    untouched (the half-written one is swept up on the next save). Readers
    that already hold `current` keep a consistent pair for as long as they need.
    """
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    with _store_lock:
        _migrate_legacy()
        gid = f"g-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        gdir = RAG_DIR / "generations" / gid
        gdir.mkdir(parents=True, exist_ok=True)
        try:
            with open(gdir / "embeddings.npy", "wb") as f:
                np.save(f, embeddings)
            sha = hashlib.sha256()
            with open(gdir / "metadata.json", "w") as f:
                for piece in json.JSONEncoder(indent=2).iterencode(metadata):
                    f.write(piece)
                    sha.update(piece.encode())
            try:        # the compact index search reads; best effort, search rebuilds it lazily if absent
                _build_light(gdir, metadata)
            except Exception:
                pass
            rows = int(np.load(str(gdir / "embeddings.npy"), mmap_mode="r").shape[0]) if embeddings.size else 0
            if embeddings.size and rows != len(metadata):
                raise RagIndexMismatch(
                    f"refusing to save an inconsistent index: {rows} vectors for {len(metadata)} chunks")
            (gdir / "manifest.json").write_text(json.dumps({
                "generation": gid, "created_at": time.time(),
                "n_chunks": len(metadata),
                "dims": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                "dtype": str(embeddings.dtype),
                "embed_model": _active_embed_model,
                "chunker_version": CHUNKER_VERSION,
                "metadata_sha256": sha.hexdigest(),
                "embeddings_bytes": (gdir / "embeddings.npy").stat().st_size,
            }, indent=2))
        except BaseException:
            import shutil
            shutil.rmtree(gdir, ignore_errors=True)         # never leave a half generation behind
            raise
        try:
            previous = os.readlink(RAG_DIR / "current").rsplit("/", 1)[-1]
        except OSError:
            previous = ""
        try:
            _switch_current(gid)                             # the one atomic step
        except BaseException:
            import shutil
            shutil.rmtree(gdir, ignore_errors=True)          # finished, but never became live
            raise
        _prune_generations(keep={gid, previous})
    # Refresh the count sidecar in the same breath, so the cheap path is never
    # stale after a write. Written AFTER the switch: a sidecar describing a
    # store that failed to land would be worse than none.
    _write_stats({
        "chunks": len(metadata),
        "files": len({m.get("file") for m in metadata if isinstance(m, dict)}),
    })


def _load_folders() -> list[dict]:
    if not FOLDERS_FILE.exists():
        return []
    with open(FOLDERS_FILE) as f:
        return yaml.safe_load(f) or []


def _save_folders(folders: list[dict]) -> None:
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    with open(FOLDERS_FILE, "w") as f:
        yaml.dump(folders, f, default_flow_style=False)


def resolve_folder_entry(cwd: Path | None = None) -> dict | None:
    """Return the registered folder entry that equals or contains `cwd`.

    Picks the longest (most specific) match.  Synthetic entries created by
    index_text() (e.g. ``qa:…`` or URLs) are skipped — only real directories
    on disk are considered.
    """
    cwd = (cwd or Path.cwd()).resolve()
    best: dict | None = None
    best_len = -1
    for entry in _load_folders():
        raw = entry.get("folder", "")
        p = Path(raw)
        if not p.is_absolute() or not p.is_dir():
            continue
        p = p.resolve()
        if cwd == p or cwd.is_relative_to(p):
            if len(str(p)) > best_len:
                best, best_len = entry, len(str(p))
    return best


# ── Index ──────────────────────────────────────────────────────────────────────

#: Files that are data or output, not something a person searches for: lockfiles (6.9k
#: chunks of the live index, 2.3%), minified bundles and source maps.
_NOISE_NAMES = {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock", "uv.lock",
                "Cargo.lock", "composer.lock", "Gemfile.lock", "go.sum"}
_NOISE_SUFFIXES = (".lock", ".min.js", ".min.css", ".map")


def _is_noise(f: Path) -> bool:
    """True for a path the RAG index should not hold: an excluded or build-output
    directory anywhere in it (build-synapse/, cmake-build-*, .mcp-cache: 7.8k chunks of
    duplicated output), or a lockfile/minified/map file. Files already indexed that now
    match are dropped by the next /rag-update, which needs no embedding."""
    for part in f.parts[:-1]:                 # directories only: a FILE called build-x.md is source
        if part in EXCLUDE_DIRS or part == ".mcp-cache" or part.startswith(("build-", "cmake-build")):
            return True
    return f.name in _NOISE_NAMES or f.name.endswith(_NOISE_SUFFIXES)


def _scan_folder_files(folder: Path, exts: set[str] | None = None) -> list[Path]:
    """Walk a folder, returning indexable files (see _is_noise)."""
    exts = exts or DEFAULT_EXTENSIONS
    return [f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in exts and not _is_noise(f)]


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _free_cuda_cache() -> None:
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


REEMBED_PART = 4096     # chunks per checkpoint part: a crash loses at most this much work


def reembed_store(progress=None, batch_size: int = 32) -> dict:
    """Re-embed every chunk of the live store with the CURRENT embedder, on the GPU.

    The chunk text is kept in the store's metadata, so no source file is read.
    Vectors are written to a new generation by the same atomic path every save
    uses: the live index keeps answering searches until the swap, and a crash
    or Ctrl+C leaves it untouched. Work is checkpointed in parts under
    rag/reembed.tmp so a re-run resumes instead of starting over.

    Refuses (rather than falling back to the CPU) when the card has no room:
    at ~2-4 s per chunk the CPU needs days for a 297k-chunk store, and it would
    take the cores the served model's experts use. progress(done, total, eta_s).
    """
    import shutil
    emb, meta = _load_store()
    if emb is None or not meta:
        return {"status": "empty", "chunks": 0}
    fp0 = _store_fingerprint()
    built_with = read_manifest().get("embed_model")
    if built_with == EMBED_MODEL:
        return {"status": "current", "chunks": len(meta), "embed_model": built_with}

    cpu_model = _get_model("cpu")
    if not _gpu_has_room_for(cpu_model):
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            have = f"{free / 2**30:.1f} of {total / 2**30:.1f} GiB free"
        except Exception:
            have = "no CUDA device"
        raise RuntimeError(
            f"Not enough free VRAM to re-embed the index on the GPU ({have}). Nothing "
            f"was changed and the current index still works. The CPU would take days "
            f"for {len(meta):,} chunks. Stop the served model (or wait until it is "
            f"unloaded), then run /rag-reembed again; it resumes where it stopped.")
    release_gpu_embedder()
    model = _get_model("cuda")
    try:
        texts = [m.get("text", "") or " " for m in meta]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))     # short first: little padding
        tmp = RAG_DIR / "reembed.tmp"
        plan = {"model": EMBED_MODEL, "n": len(texts), "part": REEMBED_PART,
                "sig": hashlib.sha256(("".join(texts[i][:64] for i in order[::997])).encode()).hexdigest()[:16]}
        plan_f = tmp / "plan.json"
        if tmp.exists() and (not plan_f.exists() or json.loads(plan_f.read_text()) != plan):
            shutil.rmtree(tmp, ignore_errors=True)         # a different job's leftovers
        tmp.mkdir(parents=True, exist_ok=True)
        plan_f.write_text(json.dumps(plan))
        n_parts = (len(order) + REEMBED_PART - 1) // REEMBED_PART
        t0, done_now = time.time(), 0
        for p in range(n_parts):
            part_f = tmp / f"p-{p:06d}.npy"
            idx = order[p * REEMBED_PART:(p + 1) * REEMBED_PART]
            if part_f.exists():
                continue
            # Longest chunks come last (sorted short-first), so activation memory climbs
            # through the run: a real run was at 10.2 of 11.6 GiB at 60%. On an
            # out-of-memory error, free the allocator's cache and halve the batch
            # instead of dying; the smaller batch stays for the rest of the run.
            while True:
                try:
                    vecs = model.encode([texts[i] for i in idx], batch_size=batch_size,
                                        convert_to_tensor=True, show_progress_bar=False)
                    break
                except Exception as e:
                    if not _is_oom(e) or batch_size <= 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    _free_cuda_cache()
            np.save(str(part_f) + ".tmp.npy", vecs.float().cpu().numpy().astype(np.float32))
            os.replace(str(part_f) + ".tmp.npy", part_f)
            _free_cuda_cache()                      # hand cached blocks back between parts
            done_now += len(idx)
            if progress:
                rate = done_now / max(time.time() - t0, 1e-6)
                progress(min((p + 1) * REEMBED_PART, len(order)), len(order),
                         (len(order) - (p + 1) * REEMBED_PART) / max(rate, 1e-6))
        parts = [np.load(str(tmp / f"p-{p:06d}.npy")) for p in range(n_parts)]
        sorted_vecs = np.concatenate(parts, axis=0)
        out = np.empty_like(sorted_vecs)
        out[np.asarray(order)] = sorted_vecs               # back to the store's own order
        if _store_fingerprint() != fp0:
            raise RuntimeError(
                "The index changed while it was being re-embedded (something was added or "
                "removed), so the new vectors no longer line up. Nothing was swapped; run "
                "/rag-reembed again.")
        _save_store(out, meta)                             # new generation, atomic swap
        shutil.rmtree(tmp, ignore_errors=True)
        return {"status": "done", "chunks": len(meta), "dims": int(out.shape[1]),
                "embed_model": _active_embed_model, "seconds": int(time.time() - t0)}
    finally:
        model = None                    # the local reference would keep the weights alive past the release
        release_gpu_embedder()


def index_folder(
    folder: Path,
    project: str | None = None,
    extensions: set[str] | None = None,
    progress=None,
) -> dict:
    """Walk folder, chunk code files, embed, append to shared store."""
    from datetime import datetime

    folder  = folder.resolve()
    project = project or folder.name
    exts    = extensions or DEFAULT_EXTENSIONS

    files = _scan_folder_files(folder, exts)

    if not files:
        return {"indexed": 0, "skipped": 0, "project": project, "folder": str(folder)}

    all_texts: list[str] = []
    all_meta: list[dict] = []
    skipped = 0

    for f in files:
        try:
            content = f.read_text(errors="ignore")
        except Exception:
            skipped += 1
            continue
        chunks = _chunk_code(content, f.suffix.lower())
        for start, end, text in chunks:
            all_texts.append(text[:4000])
            all_meta.append({
                "file": str(f),
                "project": project,
                "lines": [start, end],
                "text": text[:4000],
            })

    if not all_texts:
        return {"indexed": 0, "skipped": skipped, "project": project, "folder": str(folder)}

    new_embeddings = _embed(all_texts, cpu_max=_cpu_bulk_cap(), progress=progress)

    # Separator-safe prefix so /a/proj does not match /a/proj2.
    prefix = str(folder) + os.sep
    with _store_lock:
        existing_emb, existing_meta = _load_store()
        _guard_append(existing_emb, new_embeddings)
        if existing_emb is not None and len(existing_meta) > 0:
            keep_idx = [
                i for i, m in enumerate(existing_meta)
                if m["file"] != str(folder) and not m["file"].startswith(prefix)
            ]
            if keep_idx:
                existing_emb = existing_emb[keep_idx]
                existing_meta = [existing_meta[i] for i in keep_idx]
                combined_emb  = np.concatenate([existing_emb, new_embeddings], axis=0)
                combined_meta = existing_meta + all_meta
            else:
                combined_emb  = new_embeddings
                combined_meta = all_meta
        else:
            combined_emb  = new_embeddings
            combined_meta = all_meta

        _save_store(combined_emb, combined_meta)

        folders = _load_folders()
        folders = [f for f in folders if f.get("folder") != str(folder)]
        folders.append({
            "folder": str(folder),
            "project": project,
            "file_count": len(files),
            "chunk_count": len(all_texts),
            "last_indexed": datetime.now().isoformat(timespec="seconds"),
        })
        _save_folders(folders)

    return {"indexed": len(all_texts), "skipped": skipped,
            "project": project, "folder": str(folder)}


def index_text(
    text: str,
    source_name: str,
    project: str | None = None,
) -> dict:
    """Index arbitrary text (markdown, prose, converted docs) into the RAG store.

    Uses markdown-aware chunking for .md/.txt sources; code chunking for others.
    De-duplicates by source_name on re-index.
    """
    from datetime import datetime

    project = project or "documents"
    ext = Path(source_name).suffix.lower() if "." in Path(source_name).name else ".md"

    if ext in {".md", ".txt", ".rst", ""}:
        chunks = _chunk_markdown(text)
    else:
        chunks = _chunk_code(text, ext)

    if not chunks:
        return {"indexed": 0, "skipped": 0, "source": source_name, "project": project}

    all_texts = [c[2][:4000] for c in chunks]
    all_meta  = [
        {
            "file":    source_name,
            "project": project,
            "lines":   [c[0], c[1]],
            "text":    c[2][:4000],
        }
        for c in chunks
    ]

    new_embeddings = _embed(all_texts, cpu_max=_cpu_bulk_cap())

    with _store_lock:
        existing_emb, existing_meta = _load_store()
        _guard_append(existing_emb, new_embeddings)
        if existing_emb is not None and len(existing_meta) > 0:
            keep_idx = [i for i, m in enumerate(existing_meta) if m["file"] != source_name]
            if keep_idx:
                existing_emb  = existing_emb[keep_idx]
                existing_meta = [existing_meta[i] for i in keep_idx]
                combined_emb  = np.concatenate([existing_emb, new_embeddings], axis=0)
                combined_meta = existing_meta + all_meta
            else:
                combined_emb  = new_embeddings
                combined_meta = all_meta
        else:
            combined_emb  = new_embeddings
            combined_meta = all_meta

        _save_store(combined_emb, combined_meta)

        # Track in folders file as a synthetic "folder" entry
        folders = _load_folders()
        folders = [f for f in folders if f.get("folder") != source_name]
        folders.append({
            "folder":       source_name,
            "project":      project,
            "file_count":   1,
            "chunk_count":  len(all_texts),
            "last_indexed": datetime.now().isoformat(timespec="seconds"),
        })
        _save_folders(folders)

    return {"indexed": len(all_texts), "skipped": 0, "source": source_name, "project": project}


# ── Incremental update ──────────────────────────────────────────────────────────

def _empty_update_result(project: str, folder: str, forced: bool = False) -> dict:
    return {
        "project": project, "folder": folder,
        "scanned": 0, "reindexed_files": 0, "removed_files": 0,
        "unchanged_files": 0, "chunks_added": 0, "chunks_removed": 0,
        "forced": forced,
    }


def update_folder(
    folder: Path,
    project: str | None = None,
    force: bool = False,
    verbose: bool = False,
    progress=None,
) -> dict:
    """Incrementally re-index a registered folder.

    Only files whose mtime is newer than the folder's ``last_indexed`` timestamp
    are re-chunked and re-embedded; chunks of files that no longer exist on disk
    are purged; unchanged files keep their existing chunks.  ``force=True`` (or
    an unregistered folder) falls back to a full :func:`index_folder` rebuild.

    Returns a counts dict (see :func:`_empty_update_result`).  When nothing has
    changed it returns early **without loading the embedding model**.
    """
    from datetime import datetime

    folder  = folder.resolve()
    str_folder = str(folder)

    entry = next((e for e in _load_folders() if e.get("folder") == str_folder), None)

    # No prior incremental baseline → full index.
    if force or entry is None or not entry.get("last_indexed"):
        res = index_folder(folder, project, progress=progress)
        out = _empty_update_result(res.get("project", project or folder.name),
                                   str_folder, forced=True)
        out["scanned"] = res.get("indexed", 0)
        out["reindexed_files"] = res.get("indexed", 0)  # chunk count proxy for full rebuild
        out["chunks_added"] = res.get("indexed", 0)
        if verbose:
            print(f"  [{out['project']}] full rebuild: {res.get('indexed', 0)} chunks")
        return out

    project = entry.get("project") or project or folder.name
    try:
        last_ts = datetime.fromisoformat(entry["last_indexed"]).timestamp()
    except (ValueError, KeyError):
        last_ts = 0.0

    scan_start = datetime.now()
    files   = _scan_folder_files(folder)
    current = {str(f) for f in files}

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    changed = [f for f in files if _mtime(f) > last_ts]

    prefix = str_folder + os.sep
    with _store_lock:
        emb, meta = _load_store()
        indexed = {m["file"] for m in meta
                   if m["file"] == str_folder or m["file"].startswith(prefix)}
        removed = indexed - current

        # Nothing to do — bump the timestamp and skip the model entirely.
        if not changed and not removed:
            entry["last_indexed"] = scan_start.isoformat(timespec="seconds")
            folders = [e for e in _load_folders() if e.get("folder") != str_folder]
            folders.append(entry)
            _save_folders(folders)
            out = _empty_update_result(project, str_folder)
            out["scanned"] = len(files)
            out["unchanged_files"] = len(files)
            if verbose:
                print(f"  [{project}] up to date — {len(files)} files unchanged")
            return out

    # Chunk changed files (outside the lock — file I/O + CPU only).
    new_texts: list[str] = []
    new_meta:  list[dict] = []
    for f in changed:
        try:
            content = f.read_text(errors="ignore")
        except Exception:
            continue
        for start, end, text in _chunk_code(content, f.suffix.lower()):
            new_texts.append(text[:4000])
            new_meta.append({
                "file": str(f), "project": project,
                "lines": [start, end], "text": text[:4000],
            })

    new_embeddings = _embed(new_texts, cpu_max=_cpu_bulk_cap(), progress=progress) if new_texts else None

    with _store_lock:
        emb, meta = _load_store()           # re-load: a Q&A turn may have written
        _guard_append(emb, new_embeddings)
        stale = {str(f) for f in changed} | removed
        keep_idx = [i for i, m in enumerate(meta) if m["file"] not in stale]
        chunks_removed = len(meta) - len(keep_idx)

        if emb is not None and keep_idx:
            kept_emb  = emb[keep_idx]
            kept_meta = [meta[i] for i in keep_idx]
        else:
            kept_emb, kept_meta = None, []

        if new_embeddings is not None:
            if kept_emb is not None:
                combined_emb  = np.concatenate([kept_emb, new_embeddings], axis=0)
                combined_meta = kept_meta + new_meta
            else:
                combined_emb, combined_meta = new_embeddings, new_meta
        else:
            combined_emb  = kept_emb if kept_emb is not None else np.zeros((0, 0), dtype=np.float32)
            combined_meta = kept_meta

        _save_store(combined_emb, combined_meta)

        chunk_count = sum(1 for m in combined_meta
                          if m["file"] == str_folder or m["file"].startswith(prefix))
        folders = [e for e in _load_folders() if e.get("folder") != str_folder]
        folders.append({
            "folder": str_folder, "project": project,
            "file_count": len(files), "chunk_count": chunk_count,
            "last_indexed": scan_start.isoformat(timespec="seconds"),
        })
        _save_folders(folders)

    out = _empty_update_result(project, str_folder)
    out.update({
        "scanned": len(files),
        "reindexed_files": len(changed),
        "removed_files": len(removed),
        "unchanged_files": len(files) - len(changed),
        "chunks_added": len(new_texts),
        "chunks_removed": chunks_removed,
    })
    if verbose:
        print(f"  [{project}] {len(changed)} re-indexed · {len(removed)} removed · "
              f"{out['unchanged_files']} unchanged  (+{len(new_texts)}/-{chunks_removed} chunks)")
    return out


def update_all(
    force: bool = False,
    verbose: bool = False,
    include_web: bool = True,
    progress=None,
) -> dict:
    """Incrementally update every registered directory folder (and web sources).

    Synthetic non-directory entries (qa:/URL) are skipped here; web sources are
    refreshed separately via :func:`update_web_sources`.
    """
    results: list[dict] = []
    for entry in _load_folders():
        raw = entry.get("folder", "")
        p = Path(raw)
        if not p.is_absolute() or not p.is_dir():
            continue
        try:
            results.append(update_folder(p, project=entry.get("project"),
                                         force=force, verbose=verbose, progress=progress))
        except Exception as e:
            if verbose:
                print(f"  ✗  {raw}: {e}")
            results.append({**_empty_update_result(entry.get("project", "?"), str(p)),
                            "error": str(e)})

    web = update_web_sources(verbose=verbose) if include_web else None

    return {
        "folders": results,
        "web": web,
        "reindexed_files": sum(r.get("reindexed_files", 0) for r in results),
        "removed_files":   sum(r.get("removed_files", 0) for r in results),
        "chunks_added":    sum(r.get("chunks_added", 0) for r in results),
        "chunks_removed":  sum(r.get("chunks_removed", 0) for r in results),
    }


# ── BM25 helper ───────────────────────────────────────────────────────────────

def _bm25_score(query_terms: set[str], text: str) -> float:
    """Lightweight BM25-inspired term-overlap score (no extra storage needed)."""
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for t in query_terms if t in text_lower)
    return hits / len(query_terms)


# ── Compact index (what a search needs without parsing every chunk's text) ─────
#
# metadata.json holds every chunk's full text: 449 MiB for the live 297k-chunk index. A
# search used to json.load all of it (0.85 s of a 1.1 s search, ~1.2 GB of transient RAM)
# to score vectors that take 0.03 s. A generation is immutable, so the parts a search needs
# are derived once and kept beside it, memory-mapped: per-chunk file/project/lines ids, and
# the texts as one byte blob plus offsets, read only for the handful of hits.

_LIGHT_FILES = ("light_rows.npz", "light_tables.json", "texts.bin", "text_offsets.npy")
_light_cache: dict = {}


class _Light:
    def __init__(self, live: Path, n: int):
        rows = np.load(str(live / "light_rows.npz"))
        self.f_idx, self.p_idx, self.lines = rows["f_idx"], rows["p_idx"], rows["lines"]
        tables = json.loads((live / "light_tables.json").read_text())
        self.files, self.projects = tables["files"], tables["projects"]
        self.offsets = np.load(str(live / "text_offsets.npy"), mmap_mode="r")
        self.blob = np.memmap(str(live / "texts.bin"), dtype=np.uint8, mode="r") if self.offsets[-1] else b""
        self.n = n

    def text(self, i: int) -> str:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return bytes(self.blob[a:b]).decode("utf-8", "replace") if b > a else ""


def _build_light(live: Path, metadata: list) -> None:
    """Write the compact index for the generation at `live`. Atomic: every file is
    written under a temp name and renamed, and the marker goes last."""
    n = len(metadata)
    files: dict = {}
    projects: dict = {}
    f_idx = np.empty(n, dtype=np.int32)
    p_idx = np.empty(n, dtype=np.int32)
    lines = np.zeros((n, 2), dtype=np.int32)
    offsets = np.zeros(n + 1, dtype=np.int64)
    tmp = {name: live / (name + ".tmp") for name in _LIGHT_FILES}
    with open(tmp["texts.bin"], "wb") as blob:
        pos = 0
        for i, m in enumerate(metadata):
            f_idx[i] = files.setdefault(m["file"], len(files))
            p_idx[i] = projects.setdefault(m.get("project", ""), len(projects))
            ln = m.get("lines") or [0, 0]
            lines[i] = (ln[0], ln[1])
            raw = (m.get("text") or "").encode("utf-8")
            blob.write(raw)
            pos += len(raw)
            offsets[i + 1] = pos
    with open(tmp["light_rows.npz"], "wb") as f:
        np.savez(f, f_idx=f_idx, p_idx=p_idx, lines=lines)
    with open(tmp["text_offsets.npy"], "wb") as f:
        np.save(f, offsets)
    tmp["light_tables.json"].write_text(json.dumps({"files": list(files), "projects": list(projects)}))
    for name in _LIGHT_FILES:
        os.replace(tmp[name], live / name)
    (live / "light.ok").write_text(str(n))             # last: its presence means all of the above is whole


def _load_light(n_vectors: int) -> "_Light":
    live = _live_dir()
    key = (os.path.realpath(live), n_vectors)
    hit = _light_cache.get("k")
    if hit and hit[0] == key:
        return hit[1]
    marker = live / "light.ok"
    try:
        ready = marker.read_text().strip() == str(n_vectors) and all((live / f).exists() for f in _LIGHT_FILES)
    except OSError:
        ready = False
    if not ready:
        with open(_meta_path()) as f:
            metadata = json.load(f)                    # one-time, per generation
        if len(metadata) != n_vectors:
            raise RagIndexMismatch(
                f"the index is inconsistent: {n_vectors} vectors but {len(metadata)} chunks. "
                f"Rebuild it with /rag clear, then re-index.")
        _build_light(live, metadata)
    light = _Light(live, n_vectors)
    _light_cache["k"] = (key, light)
    return light


# ── Search ─────────────────────────────────────────────────────────────────────

def search(
    query: str,
    project: str | None = None,
    top_k: int = 5,
    min_score: float = 0.1,
    path_prefix: str | None = None,
) -> list[dict]:
    import os

    _migrate_legacy()
    if not (_emb_path().exists() and _meta_path().exists()):
        return []
    # mmap: the matrix stays on disk and is paged in as the matmul reads it.
    embeddings = np.load(str(_emb_path()), mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return []
    light = _load_light(embeddings.shape[0])

    # Until /rag-reembed finishes, the live index may have been built by an older
    # embedder; its queries must come from that same model or the neighbours are wrong.
    _built = read_manifest().get("embed_model")
    query_vec = _embed([query], query=True,
                       model_name=_built if _built and _built != EMBED_MODEL else None)[0]
    if query_vec.shape[0] != embeddings.shape[1]:
        raise RagIndexMismatch(
            f"the index holds {embeddings.shape[1]}-dim vectors but '{_active_embed_model}' "
            f"produces {query_vec.shape[0]}. Run /rag-reembed to rebuild it.")
    scores = np.ascontiguousarray(embeddings @ np.asarray(query_vec, dtype=np.float32))

    keep = None
    if project:
        keep = np.array([p == project for p in light.projects])[light.p_idx]
    if path_prefix:
        # Keep only chunks whose file lives under path_prefix (separator-safe).
        prefix_with_sep = path_prefix.rstrip(os.sep) + os.sep
        under = np.array([f.startswith(prefix_with_sep) for f in light.files])[light.f_idx]
        keep = under if keep is None else (keep & under)
    if keep is not None:
        scores = np.where(keep, scores, 0.0)

    k = min(top_k * 2, len(scores))
    top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
    top = top[np.argsort(-scores[top])]

    query_terms = set(query.lower().split())
    results = []
    seen_files: dict[str, int] = {}
    for idx in top.tolist():
        score = float(scores[idx])
        if score < min_score:
            break
        file_path = light.files[light.f_idx[idx]]
        if seen_files.get(file_path, 0) >= 2:
            continue
        seen_files[file_path] = seen_files.get(file_path, 0) + 1
        text = light.text(idx)
        bm25 = _bm25_score(query_terms, text)
        hybrid = round(0.75 * score + 0.25 * bm25, 3)
        results.append({
            "file":    file_path,
            "project": light.projects[light.p_idx[idx]],
            "lines":   [int(light.lines[idx][0]), int(light.lines[idx][1])],
            "score":   hybrid,
            "text":    text,
        })
        if len(results) >= top_k * 2:
            break

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def format_for_claude(results: list[dict], query: str) -> str:
    """Format search results as context the model can use directly."""
    if not results:
        return f"=== RAG Search: \"{query}\" ===\nNo relevant results found.\n==="

    total_tokens  = sum(len(r["text"].split()) for r in results)
    project_names = list({r["project"] for r in results})
    project_str   = ", ".join(project_names)

    lines = [
        f'=== RAG Search: "{query}" ===',
        f"{len(results)} results  ·  ~{total_tokens} tokens  ·  project: {project_str}",
        "",
    ]
    for r in results:
        try:
            rel = Path(r["file"]).relative_to(Path.cwd())
        except ValueError:
            rel = Path(r["file"])
        start, end = r["lines"]
        lines.append(f"── {rel}:{start}-{end}  (score: {r['score']}) ──")
        for code_line in r["text"].splitlines()[:30]:
            lines.append(code_line)
        lines.append("")

    lines.append("=" * 40)
    lines.append("Reference these by path:line — do not re-read the full files.")
    return "\n".join(lines)


def results_to_json(results: list[dict], query: str) -> dict:
    """Serialize search() results into a stable JSON shape for headless callers.

    Mirrors format_for_claude but emits structured data: gb_bridge (optimal-claude)
    parses this directly. Keeps `text` capped — full chunks are large and noisy
    for downstream callers that only need a snippet.
    """
    payload_results = []
    for r in results:
        text = r.get("text", "") or ""
        snippet = text[:600]
        lines = r.get("lines", [0, 0])
        payload_results.append({
            "file":       r.get("file", ""),
            "project":    r.get("project", ""),
            "line_start": lines[0] if len(lines) > 0 else 0,
            "line_end":   lines[1] if len(lines) > 1 else 0,
            "score":      r.get("score", 0.0),
            "snippet":    snippet,
            "truncated":  len(text) > len(snippet),
        })
    return {
        "query":   query,
        "count":   len(results),
        "results": payload_results,
    }


def get_context_summary(folder: str | None = None) -> str:
    """Return a brief RAG index summary for context_builder (no query, no embedding).

    When *folder* is given, counts are scoped to chunks whose file lives under
    that directory prefix.  When not given or the folder has no indexed chunks,
    returns a note that no folder-scoped index is available.
    """
    import os
    _, metadata = _load_store()
    if not metadata:
        return ""

    if folder:
        prefix = folder.rstrip(os.sep) + os.sep
        scoped = [m for m in metadata if m["file"].startswith(prefix)]
        if not scoped:
            return (
                f"\n- RAG index: no chunks indexed for current folder ({folder})."
                " Use /rag-add to index it.\n"
            )
        n_chunks = len(scoped)
        n_files  = len({m["file"] for m in scoped})
        return (
            f"\n- RAG index (current folder): {n_chunks} chunks from {n_files} files."
            " Use /rag-inject <query> to prepend relevant code.\n"
        )

    # No folder scope — global summary (legacy fallback, shouldn't appear in normal flow)
    n_chunks = len(metadata)
    n_files  = len({m["file"] for m in metadata})
    folders  = _load_folders()
    projects = list({f.get("project", "") for f in folders if f.get("project")})
    proj_str = ", ".join(projects[:3]) + ("…" if len(projects) > 3 else "")
    return (
        f"\n- RAG index available: {n_chunks} chunks from {n_files} files"
        f" (projects: {proj_str}). Use /rag-inject <query> to prepend relevant code.\n"
    )


# ── CLI helpers ────────────────────────────────────────────────────────────────

def print_status() -> None:
    embeddings, metadata = _load_store()
    folders  = _load_folders()
    n_chunks = len(metadata) if metadata else 0
    n_files  = len({m["file"] for m in metadata}) if metadata else 0
    db_mb    = EMBEDDINGS_FILE.stat().st_size / 1_048_576 if EMBEDDINGS_FILE.exists() else 0.0

    print(f"  RAG index at: {RAG_DIR}")
    print(f"  Chunks: {n_chunks}  ·  Files: {n_files}  ·  Index size: {db_mb:.1f} MB")
    print()
    if folders:
        print("  Indexed folders:")
        for f in folders:
            print(f"    {f['project']:<20}  {f['folder']}")
            print(f"    {'':20}  {f['chunk_count']} chunks  ·  {f['last_indexed'][:10]}")
    else:
        print("  No folders indexed yet. Run: /rag-add <folder>")

    web = _load_web_sources()
    if web:
        print()
        print(f"  Web sources ({len(web)} registered — /rag-update to refresh):")
        for entry in web[:8]:
            print(f"    {entry['url'][:60]}  [{entry.get('project','?')}]  {entry.get('last_indexed','?')[:10]}")
        if len(web) > 8:
            print(f"    … and {len(web) - 8} more")


# ── Q&A auto-feed ─────────────────────────────────────────────────────────────

def feed_qa_turn(
    user_msg: str,
    assistant_reply: str,
    project: str | None = None,
) -> None:
    """Index a single Q&A turn into the RAG store (called after each response).

    Skips very short exchanges to avoid noise. Uses a stable source_name so
    repeated indexing de-duplicates rather than accumulates duplicates.
    """
    from datetime import datetime

    user_msg       = (user_msg or "").strip()
    assistant_reply = (assistant_reply or "").strip()

    if len(user_msg) < 10 or len(assistant_reply) < 20:
        return

    ts    = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = f"qa:{ts}"
    doc   = f"Q: {user_msg}\n\nA: {assistant_reply}"

    try:
        index_text(doc, source_name=label, project=project or "qa_history")
    except Exception:
        pass


# ── Web source registry ────────────────────────────────────────────────────────

def _load_web_sources() -> list[dict]:
    if not WEB_SOURCES_FILE.exists():
        return []
    try:
        return json.loads(WEB_SOURCES_FILE.read_text())
    except Exception:
        return []


def _save_web_sources(sources: list[dict]) -> None:
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    WEB_SOURCES_FILE.write_text(json.dumps(sources, indent=2))


def register_web_source(url: str, project: str | None = None) -> None:
    """Record a URL so /rag-update can re-fetch and re-index it later."""
    from datetime import datetime

    sources = _load_web_sources()
    existing = next((s for s in sources if s["url"] == url), None)
    if existing:
        existing["project"]      = project or existing.get("project", "web")
        existing["last_indexed"] = datetime.now().isoformat(timespec="seconds")
    else:
        sources.append({
            "url":          url,
            "project":      project or "web",
            "last_indexed": datetime.now().isoformat(timespec="seconds"),
        })
    _save_web_sources(sources)


def update_web_sources(
    project_filter: str | None = None,
    verbose: bool = True,
) -> dict:
    """Re-fetch and re-index all registered web URLs.

    Returns summary dict: {"updated": N, "failed": N, "skipped": N}.
    """
    from datetime import datetime

    sources = _load_web_sources()
    if not sources:
        if verbose:
            print("  No web sources registered. Use /convert <url> to add some.")
        return {"updated": 0, "failed": 0, "skipped": 0}

    if project_filter:
        targets = [s for s in sources if s.get("project") == project_filter]
    else:
        targets = sources

    updated = failed = skipped = 0

    for entry in targets:
        url     = entry["url"]
        project = entry.get("project", "web")
        try:
            from greenboost_cli.converters.markitdown_adapter import convert
            if verbose:
                print(f"  Fetching: {url[:70]} …")
            md = convert(url, feed_rag=True, project=project)
            if md.strip():
                entry["last_indexed"] = datetime.now().isoformat(timespec="seconds")
                updated += 1
                if verbose:
                    print(f"    ✓  {len(md):,} chars indexed → [{project}]")
            else:
                skipped += 1
                if verbose:
                    print(f"    ·  Empty response — skipped")
        except Exception as e:
            failed += 1
            if verbose:
                print(f"    ✗  {e}")

    _save_web_sources(sources)
    return {"updated": updated, "failed": failed, "skipped": skipped}
