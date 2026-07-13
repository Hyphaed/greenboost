"""GreenBoost log RAG — snapshot CLI outputs and search semantically across versions."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from greenboost_cli.environment.settings import GB_HOME

LOG_RAG_DIR  = GB_HOME / "log_rag"
LOG_EMB_FILE = LOG_RAG_DIR / "embeddings.npy"
LOG_META_FILE = LOG_RAG_DIR / "metadata.json"
LOG_SNAPS_DIR = LOG_RAG_DIR / "snapshots"

_ANSI = re.compile(r'\x1b\[[0-9;]*[mK]')


def _strip(text: str) -> str:
    return _ANSI.sub("", text)


def _run(cmd: list[str], timeout: int = 12) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return _strip(r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]"
    except Exception as exc:
        return f"[error: {exc}]"


def _load_store() -> tuple[np.ndarray | None, list[dict]]:
    if LOG_EMB_FILE.exists() and LOG_META_FILE.exists():
        try:
            emb = np.load(str(LOG_EMB_FILE))
            meta = json.loads(LOG_META_FILE.read_text())
            return emb, meta
        except Exception:
            pass
    return None, []


def _save_store(embeddings: np.ndarray, metadata: list[dict]) -> None:
    LOG_RAG_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(LOG_EMB_FILE), embeddings)
    LOG_META_FILE.write_text(json.dumps(metadata, indent=2))


def _chunk(text: str, window: int = 20, stride: int = 10) -> list[str]:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    chunks = []
    for i in range(0, max(1, len(lines) - window + 1), stride):
        chunks.append("\n".join(lines[i: i + window]))
    if not chunks:
        chunks = ["\n".join(lines)]
    return chunks


def snapshot_and_index(label: str = "") -> dict:
    """Run greenboost status/logs/nvtx-logs, save snapshot, embed into log RAG index."""
    LOG_SNAPS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{label}.txt" if label else f"{ts}.txt"
    snap_path = LOG_SNAPS_DIR / filename

    parts = []
    for section, cmd in [
        ("status",    ["greenboost", "status"]),
        ("logs",      ["greenboost", "logs"]),
        ("nvtx-logs", ["greenboost", "nvtx-logs", "--tail", "200", "--llm"]),
    ]:
        parts.append(f"=== {section} ===\n{_run(cmd, timeout=12)}")
    content = "\n\n".join(parts)
    snap_path.write_text(content)

    chunks = _chunk(content)
    if not chunks:
        return {"snapshot": filename, "chunks": 0}

    try:
        from greenboost_cli.rag.engine import _embed
        new_emb = _embed(chunks)
        existing_emb, existing_meta = _load_store()

        chunk_meta = [
            {"text": c, "snapshot": filename, "label": label, "ts": ts, "chunk_idx": i}
            for i, c in enumerate(chunks)
        ]

        if existing_emb is not None and existing_meta:
            combined_emb = np.vstack([existing_emb, new_emb])
            combined_meta = existing_meta + chunk_meta
        else:
            combined_emb = new_emb
            combined_meta = chunk_meta

        _save_store(combined_emb, combined_meta)
        return {"snapshot": filename, "chunks": len(chunks)}
    except Exception as exc:
        return {"snapshot": filename, "chunks": 0, "error": str(exc)}


def search_logs(query: str, top_k: int = 20) -> list[dict]:
    """Semantic search over all log snapshots."""
    emb, meta = _load_store()
    if emb is None or not meta:
        return []
    try:
        from greenboost_cli.rag.engine import _embed
        q_emb = _embed([query])
        scores = (emb @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            if float(scores[i]) < 0.1:
                continue
            m = meta[i]
            results.append({
                "text":     m["text"],
                "score":    float(scores[i]),
                "snapshot": m["snapshot"],
                "label":    m.get("label", ""),
                "ts":       m.get("ts", ""),
            })
        return results
    except Exception:
        return []


def list_snapshots() -> list[dict]:
    """Return snapshot metadata sorted newest-first."""
    if not LOG_SNAPS_DIR.exists():
        return []
    snaps = []
    for p in sorted(LOG_SNAPS_DIR.glob("*.txt"), reverse=True):
        name = p.name
        base = name.removesuffix(".txt")
        parts = base.split("_", 2)
        ts_str = "_".join(parts[:2]) if len(parts) >= 2 else base
        label = parts[2] if len(parts) >= 3 else ""
        try:
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            ts = ts_str
        snaps.append({
            "name":    name,
            "label":   label,
            "ts":      ts,
            "size_kb": round(p.stat().st_size / 1024, 1),
        })
    return snaps


def delete_snapshot(name: str) -> bool:
    """Remove snapshot file and its embeddings from the store."""
    snap_path = LOG_SNAPS_DIR / name
    if snap_path.exists():
        snap_path.unlink()

    emb, meta = _load_store()
    if emb is None or not meta:
        return True

    keep = [i for i, m in enumerate(meta) if m.get("snapshot") != name]
    if len(keep) == len(meta):
        return True

    if keep:
        _save_store(emb[np.array(keep)], [meta[i] for i in keep])
    else:
        if LOG_EMB_FILE.exists():
            LOG_EMB_FILE.unlink()
        if LOG_META_FILE.exists():
            LOG_META_FILE.unlink()
    return True
