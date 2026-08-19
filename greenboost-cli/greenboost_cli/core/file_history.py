"""What the agent changed, and how to put it back.

`/undo` already existed and removes the last conversation exchange. That is a
different thing from undoing an EDIT, and the difference matters most in the
case this CLI is built for: a `/nonstop` run works unattended for hours, and
the question the next morning is not "what did it say" but "what did it write,
and can I put one of those files back".

Before every Write and Edit the previous content is snapshotted here, so the
pre-agent state of a file is always recoverable even after twenty edits to it.
Snapshots are per session, so one night's run cannot bury another's.

Design notes worth keeping:

- **The snapshot is taken BEFORE the write, not after.** A post-write snapshot
  records what the agent produced, which is the one version you can always get
  back by looking at the file. What is irrecoverable is what was there first.
- **A file that did not exist is recorded too**, as a zero-byte snapshot marked
  `created`. Reverting then means deleting it, and without that record "revert"
  would silently leave the agent's new file in place.
- **Content-addressed by path, versioned by sequence** (`<pathhash>@vN`),
  matching the layout the reference harness uses: a flat directory survives
  paths that a nested mirror could not represent (absolute paths, symlinks,
  two files differing only by case).
- **Never raises into the caller.** A failed snapshot must not fail the edit ,
  it degrades to "this file is not revertable", which is reported honestly by
  `/changes` rather than pretended away.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

#: Files larger than this are recorded as metadata only. A 200 MB weights file
#: is not something a text-editing agent should be silently duplicating on
#: every touch, and the failure mode of trying is a full disk mid-run.
MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024


def _root() -> Path:
    from greenboost_cli.environment.settings import GB_HOME
    session = os.environ.get("GB_SESSION") or str(os.getpid())
    d = Path(GB_HOME) / "file-history" / session
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class Snapshot:
    path: str
    version: int
    ts: float
    created: bool          # the file did not exist before this edit
    too_large: bool        # content not stored, only the fact of the change
    size: int

    def to_dict(self) -> dict:
        return asdict(self)


def _index_path() -> Path:
    return _root() / "index.json"


def _load_index() -> dict:
    p = _index_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_index(idx: dict) -> None:
    try:
        _index_path().write_text(json.dumps(idx, indent=1), encoding="utf-8")
    except OSError:
        pass


#: Total bytes one session's snapshot store may hold. greenboost-cli is meant
#: to run unattended for DAYS, so "one snapshot per write" is not a bounded
#: quantity , a long run editing the same files repeatedly would grow this
#: without limit and fill the disk out from under the run it is auditing.
#: Oldest versions are evicted first, so the recent, likely-to-be-reverted
#: history survives and the ancient history does not.
MAX_STORE_BYTES = 256 * 1024 * 1024


def _prune(root) -> None:
    """Keep the store under MAX_STORE_BYTES, oldest blob first.

    Never touches index.json: an evicted version stays listed, marked
    unrevertable, because "we no longer hold that content" is information the
    user needs , silently forgetting a file was changed is worse than
    admitting the content is gone.
    """
    try:
        blobs = [(p.stat().st_mtime, p.stat().st_size, p)
                 for p in root.glob("*@v*") if p.is_file()]
        total = sum(b[1] for b in blobs)
        if total <= MAX_STORE_BYTES:
            return
        for _, size, p in sorted(blobs):
            p.unlink(missing_ok=True)
            total -= size
            if total <= MAX_STORE_BYTES * 0.8:   # hysteresis: prune in batches
                break
    except OSError:
        pass


def snapshot(file_path) -> "Snapshot | None":
    """Record the CURRENT content of `file_path` before it is overwritten."""
    try:
        p = Path(file_path).resolve()
        root = _root()
        idx = _load_index()
        key = _key(p)
        entry = idx.setdefault(key, {"path": str(p), "versions": []})
        version = len(entry["versions"]) + 1

        exists = p.exists()
        size = p.stat().st_size if exists else 0
        too_large = exists and size > MAX_SNAPSHOT_BYTES
        blob = root / f"{key}@v{version}"
        if exists and not too_large:
            blob.write_bytes(p.read_bytes())
        elif not exists:
            blob.write_bytes(b"")

        snap = Snapshot(path=str(p), version=version, ts=time.time(),
                        created=not exists, too_large=too_large, size=size)
        entry["versions"].append(snap.to_dict())
        _save_index(idx)
        _prune(root)
        return snap
    except Exception:
        return None          # never break an edit over its own audit trail


def changed_files() -> list:
    """Every file this session touched, oldest change first."""
    out = []
    for entry in _load_index().values():
        vs = entry.get("versions") or []
        if vs:
            out.append({"path": entry["path"], "edits": len(vs),
                        "first_ts": vs[0]["ts"],
                        "created": bool(vs[0].get("created")),
                        "revertable": not vs[0].get("too_large")})
    return sorted(out, key=lambda e: e["first_ts"])


def revert(file_path, version: int = 1) -> str:
    """Restore `file_path` to a recorded version. Default 1 = pre-agent state."""
    try:
        p = Path(file_path).resolve()
    except OSError:
        return f"Error: cannot resolve {file_path}"
    idx = _load_index()
    entry = idx.get(_key(p))
    if not entry or not entry.get("versions"):
        return (f"Error: no recorded changes to {p} in this session , "
                f"/changes lists what can be reverted")
    versions = entry["versions"]
    if version < 1 or version > len(versions):
        return (f"Error: {p} has {len(versions)} recorded version(s), "
                f"asked for v{version}")
    meta = versions[version - 1]
    if meta.get("too_large"):
        return (f"Error: {p} was {meta['size']:,} bytes , too large to snapshot, "
                f"so its previous content was never stored. Use git if the file "
                f"is tracked.")
    blob = _root() / f"{_key(p)}@v{version}"
    if not blob.exists():
        return (f"Error: v{version} of {p} is no longer stored , this session's "
                f"snapshot store passed its size limit and evicted the oldest "
                f"versions. The change is still recorded; the content is not. "
                f"Use git if the file is tracked.")
    try:
        if meta.get("created"):
            # It did not exist before the agent made it: reverting is deleting.
            if p.exists():
                p.unlink()
            return f"Deleted {p} , it did not exist before this session"
        p.write_bytes(blob.read_bytes())
        return f"Restored {p} to v{version} ({meta['size']:,} bytes)"
    except OSError as e:
        return f"Error: could not restore {p} , {e}"
