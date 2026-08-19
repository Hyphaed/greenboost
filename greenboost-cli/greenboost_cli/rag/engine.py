"""Code-aware RAG engine.

Uses PyTorch tensors + sentence-transformers dot-product search (no FAISS required).
Persists index to ~/.greenboost_cli/rag/ as embeddings.npy + metadata.json.
"""
from __future__ import annotations

import json
import os
import re
import threading
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
_model_lock = threading.Lock()

RAG_DIR          = GB_HOME / "rag"
EMBEDDINGS_FILE  = RAG_DIR / "embeddings.npy"
METADATA_FILE    = RAG_DIR / "metadata.json"
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

_model = None
_embed_device = None
EMBED_MODEL   = "jinaai/jina-embeddings-v2-base-code"
EMBED_FALLBACK = "all-MiniLM-L6-v2"


def _resolve_device() -> str:
    global _embed_device
    if _embed_device is not None:
        return _embed_device
    try:
        import torch
        if torch.cuda.is_available():
            # GreenBoost: T2/T3 tiers are always available — CPU spillover is
            # forbidden.  Trust torch.cuda.is_available() and skip the probe
            # tensor, which can spuriously fail under T1 VRAM pressure when
            # vLLM is loading and the CUDA context hasn't been created yet.
            _embed_device = "cuda"
        else:
            # No CUDA driver / runtime — this is a genuine incompatibility.
            raise RuntimeError(
                "torch.cuda.is_available() returned False — "
                "check CUDA driver and torch+cuXXX version match. "
                "CPU spillover is not permitted when GreenBoost is active."
            )
    except ImportError:
        _embed_device = "cpu"
    return _embed_device


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:        # another thread won the race
            return _model
        # Apply GreenBoost env before loading model onto GPU
        try:
            from greenboost_cli.greenboost.gb_torch import apply_gb_torch_env
            apply_gb_torch_env()
        except ImportError:
            pass

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

        import warnings
        warnings.filterwarnings("ignore", message=".*optimum.*", category=UserWarning)
        device = _resolve_device()
        hf_token = os.environ.get("HF_TOKEN") or _read_hf_token()
        try:
            _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True,
                                         device=device, token=hf_token)
        except Exception:
            # Fallback model, still on CUDA — never drop to CPU silently.
            _model = SentenceTransformer(EMBED_FALLBACK, device=device, token=hf_token)
    return _model


def _embed(texts: list[str]) -> np.ndarray:
    model = _get_model()
    try:
        # Never show tqdm's progress bar: it writes raw \r-based output to stderr,
        # bypassing prompt_toolkit's patch_stdout, which corrupts the live REPL
        # screen (cursor-position-report misreads show up as stray Escape input).
        vecs = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return vecs.cpu().numpy().astype(np.float32)
    except RuntimeError as e:
        if "no kernel image" in str(e) or "CUDA error" in str(e):
            # Re-raise: CPU spillover is not permitted — GreenBoost T2/T3 should
            # absorb any memory pressure.  Surface the error so it can be fixed.
            raise
        raise


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

def _load_store() -> tuple[np.ndarray | None, list[dict]]:
    if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
        embeddings = np.load(str(EMBEDDINGS_FILE))
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        return embeddings, metadata
    return None, []


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
    if not METADATA_FILE.exists():
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
        with open(METADATA_FILE) as f:
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
        st = METADATA_FILE.stat()
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


def _save_store(embeddings: np.ndarray, metadata: list[dict]) -> None:
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic writes: a concurrent reader (e.g. standalone `gb web` process) never
    # observes a torn file, and a crash mid-write leaves the old store intact.
    emb_tmp  = EMBEDDINGS_FILE.with_name(EMBEDDINGS_FILE.name + ".tmp")
    meta_tmp = METADATA_FILE.with_name(METADATA_FILE.name + ".tmp")
    # Pass a file object so numpy does not append ".npy" to the temp name.
    with open(emb_tmp, "wb") as f:
        np.save(f, embeddings)
    os.replace(emb_tmp, EMBEDDINGS_FILE)
    with open(meta_tmp, "w") as f:
        json.dump(metadata, f, indent=2)
    os.replace(meta_tmp, METADATA_FILE)
    # Refresh the count sidecar in the same breath, so the cheap path is never
    # stale after a write. Written AFTER the replace: a sidecar describing a
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

def _scan_folder_files(folder: Path, exts: set[str] | None = None) -> list[Path]:
    """Walk a folder, returning indexable files (honouring EXCLUDE_DIRS)."""
    exts = exts or DEFAULT_EXTENSIONS
    files: list[Path] = []
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            if not any(part in EXCLUDE_DIRS for part in f.parts):
                files.append(f)
    return files


def index_folder(
    folder: Path,
    project: str | None = None,
    extensions: set[str] | None = None,
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

    new_embeddings = _embed(all_texts)

    # Separator-safe prefix so /a/proj does not match /a/proj2.
    prefix = str(folder) + os.sep
    with _store_lock:
        existing_emb, existing_meta = _load_store()
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

    new_embeddings = _embed(all_texts)

    with _store_lock:
        existing_emb, existing_meta = _load_store()
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
        res = index_folder(folder, project)
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

    new_embeddings = _embed(new_texts) if new_texts else None

    with _store_lock:
        emb, meta = _load_store()           # re-load: a Q&A turn may have written
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
                                         force=force, verbose=verbose))
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


# ── Search ─────────────────────────────────────────────────────────────────────

def search(
    query: str,
    project: str | None = None,
    top_k: int = 5,
    min_score: float = 0.1,
    path_prefix: str | None = None,
) -> list[dict]:
    import os
    import torch

    embeddings, metadata = _load_store()
    if embeddings is None or len(metadata) == 0:
        return []

    query_vec = _embed([query])[0]
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    q_tensor   = torch.tensor(query_vec,  dtype=torch.float32)
    scores     = torch.mv(emb_tensor, q_tensor)

    if project:
        mask = torch.tensor(
            [1.0 if m["project"] == project else 0.0 for m in metadata],
            dtype=torch.float32,
        )
        scores = scores * mask

    if path_prefix:
        # Keep only chunks whose file lives under path_prefix (separator-safe).
        prefix_with_sep = path_prefix.rstrip(os.sep) + os.sep
        mask = torch.tensor(
            [1.0 if m["file"].startswith(prefix_with_sep) else 0.0 for m in metadata],
            dtype=torch.float32,
        )
        scores = scores * mask

    k = min(top_k * 2, len(metadata))
    top_scores, top_indices = torch.topk(scores, k)

    query_terms = set(query.lower().split())
    results = []
    seen_files: dict[str, int] = {}
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        if score < min_score:
            break
        meta      = metadata[idx]
        file_path = meta["file"]
        if seen_files.get(file_path, 0) >= 2:
            continue
        seen_files[file_path] = seen_files.get(file_path, 0) + 1
        bm25 = _bm25_score(query_terms, meta["text"])
        hybrid = round(0.75 * score + 0.25 * bm25, 3)
        results.append({
            "file":    file_path,
            "project": meta["project"],
            "lines":   meta["lines"],
            "score":   hybrid,
            "text":    meta["text"],
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
