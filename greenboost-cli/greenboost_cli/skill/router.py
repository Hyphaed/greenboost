"""Embedding-based skill router.

Given a skills directory (Claude Code convention: each skill is a folder
containing SKILL.md with YAML frontmatter), score each skill's description
against a user-turn embedding and return the top-k matches.

The embedding model is the same one rag/engine.py already loads, so no
extra dependency cost. Per-skill embeddings are cached on disk keyed by
a hash of the manifest contents; the cache invalidates automatically
when a skill is added, removed, or its description is edited.

Cache location: ~/.greenboost_cli/skills/{cache_key}.npz
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from greenboost_cli.environment.settings import GB_HOME

SKILLS_CACHE_DIR = GB_HOME / "skills"


@dataclass
class SkillEntry:
    name: str
    description: str
    path: str          # absolute path to the SKILL.md file
    triggers: list[str]


@dataclass
class SkillHit:
    name: str
    score: float
    description: str
    path: str
    reason: str        # short explanation: "matched: query, embedding, trigger"


# ── Manifest parsing ─────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_skill_md(skill_md: Path) -> SkillEntry | None:
    """Parse a SKILL.md file's YAML frontmatter. Returns None if unparseable."""
    try:
        text = skill_md.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None

    fm_text = m.group(1)

    # Light YAML — handle name/description/triggers without pulling pyyaml here
    # (rag/engine.py already imports yaml, so we could, but stay minimal).
    name = None
    description_parts: list[str] = []
    triggers: list[str] = []

    in_description = False
    in_triggers = False
    for line in fm_text.splitlines():
        stripped = line.rstrip()
        if not stripped:
            in_description = False
            in_triggers = False
            continue
        if stripped.startswith("name:"):
            name = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            in_description = False
            in_triggers = False
        elif stripped.startswith("description:"):
            rest = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if rest:
                description_parts.append(rest)
            in_description = True
            in_triggers = False
        elif stripped.startswith("triggers:"):
            in_description = False
            in_triggers = True
            rest = stripped.split(":", 1)[1].strip()
            if rest and rest != "[]":
                # Inline list like [a, b]
                if rest.startswith("[") and rest.endswith("]"):
                    triggers.extend(
                        t.strip().strip('"').strip("'")
                        for t in rest[1:-1].split(",")
                        if t.strip()
                    )
        elif in_description and stripped.startswith((" ", "\t")):
            description_parts.append(stripped.strip())
        elif in_triggers and stripped.lstrip().startswith("-"):
            triggers.append(stripped.lstrip().lstrip("- ").strip().strip('"').strip("'"))
        else:
            in_description = False
            in_triggers = False

    if not name:
        # Fall back to parent directory name
        name = skill_md.parent.name
    description = " ".join(description_parts).strip()
    if not description:
        return None

    return SkillEntry(
        name=name,
        description=description,
        path=str(skill_md),
        triggers=triggers,
    )


def discover_skills(skills_dir: Path) -> list[SkillEntry]:
    """Walk skills_dir and return all parseable SKILL.md entries."""
    if not skills_dir.is_dir():
        return []
    entries: list[SkillEntry] = []
    for skill_md in skills_dir.glob("*/SKILL.md"):
        e = _parse_skill_md(skill_md)
        if e is not None:
            entries.append(e)
    # Also support nested layout: skills/<group>/<name>/SKILL.md
    for skill_md in skills_dir.glob("*/*/SKILL.md"):
        e = _parse_skill_md(skill_md)
        if e is not None and not any(x.path == e.path for x in entries):
            entries.append(e)
    return entries


def _parse_plain_md(md_file: Path) -> SkillEntry | None:
    """Fallback parser for plain .md files without YAML frontmatter.

    Used for ~/Dev/claude_workflow/commands/*.md pattern.
    name = filename stem, description = first non-empty non-header line.
    """
    try:
        text = md_file.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    if not text or text.startswith("---"):
        return None

    name = md_file.stem
    desc = next(
        (l.strip().lstrip("#").strip() for l in text.splitlines() if l.strip()),
        "",
    )
    if not desc:
        return None
    return SkillEntry(name=name, description=desc, path=str(md_file), triggers=[])


#: skills dir (str) -> plugin name, filled by discover_all_skill_dirs() so a
#: plugin's skills surface as "plugin:skill", the way Claude Code names them.
_PLUGIN_PREFIX: dict[str, str] = {}


def discover_skills_in_dir(skills_dir: Path) -> list[SkillEntry]:
    """Discover SKILL.md entries AND plain .md files (claude_workflow/commands style)."""
    entries = _discover_skills_in_dir(skills_dir)
    prefix = _PLUGIN_PREFIX.get(str(skills_dir))
    if prefix:
        for e in entries:
            if ":" not in e.name:
                e.name = f"{prefix}:{e.name}"
    return entries


def _discover_skills_in_dir(skills_dir: Path) -> list[SkillEntry]:
    if not skills_dir.is_dir():
        return []
    entries = discover_skills(skills_dir)
    existing = {e.path for e in entries}
    # Plain *.md files directly inside the dir (not SKILL.md subdirs)
    for md_file in skills_dir.glob("*.md"):
        if str(md_file) in existing:
            continue
        e = _parse_plain_md(md_file)
        if e is not None:
            entries.append(e)
    return entries


def _plugin_skill_dirs(cwd: Path) -> list[tuple[Path, str]]:
    """(skills dir, plugin name) for every installed Claude Code plugin."""
    import json
    import os
    roots = [Path.home() / ".claude"]
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    if cfg:
        roots.insert(0, Path(cfg).expanduser())
    out: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for root in roots:
        try:
            data = json.loads((root / "plugins" / "installed_plugins.json").read_text())
        except Exception:
            continue
        for key, installs in (data.get("plugins") or {}).items():
            plugin = key.split("@", 1)[0]
            for inst in installs if isinstance(installs, list) else []:
                if inst.get("scope") == "project":
                    pp = inst.get("projectPath")
                    if not pp or (Path(pp) != cwd and Path(pp) not in cwd.parents):
                        continue
                path = inst.get("installPath")
                if not path:
                    continue
                d = Path(path) / "skills"
                if d.is_dir() and str(d) not in seen:
                    seen.add(str(d))
                    out.append((d, plugin))
    return out


def discover_all_skill_dirs(settings: dict) -> list[Path]:
    """Return all directories that should be scanned for skills.

    Order = precedence (first skill with a given name wins), the order Claude
    Code uses: project > account > global > plugin.
    1. User-configured skills_dir
    2. <project>/.claude/skills (cwd and its ancestors up to $HOME)
    3. $CLAUDE_CONFIG_DIR/skills (the active Claude Code account)
    4. ~/.claude-accounts/*/skills/  (other per-account installed skills)
    5. ~/.claude/skills/             (global Claude Code skills)
    6. ~/Dev/claude_workflow/commands/ (plain .md files, if auto_discover enabled)
    7. installed plugins' skills/, named "plugin:skill"
    """
    import os
    dirs: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        if p.is_dir() and str(p) not in seen:
            dirs.append(p)
            seen.add(str(p))

    # 1. User-configured
    raw = settings.get("skills_dir") if settings else None
    if raw:
        _add(Path(raw).expanduser())

    # 2. Project skills: nearest first
    cwd = Path.cwd().resolve()
    home = Path.home()
    p = cwd
    for _ in range(10):
        _add(p / ".claude" / "skills")
        if p == home or p.parent == p:
            break
        p = p.parent

    # 3. Active Claude Code account
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    if cfg:
        _add(Path(cfg).expanduser() / "skills")

    # 4. Claude Code per-account skills
    accounts_dir = Path.home() / ".claude-accounts"
    if accounts_dir.is_dir():
        for account_dir in sorted(accounts_dir.iterdir()):
            _add(account_dir / "skills")

    # 5. Global Claude Code skills
    _add(Path.home() / ".claude" / "skills")

    # 6. claude_workflow/commands — plain .md files
    if (settings or {}).get("skills_auto_discover", True):
        _add(Path.home() / "Dev" / "claude_workflow" / "commands")

    # 7. Plugins
    for d, plugin in _plugin_skill_dirs(cwd):
        _PLUGIN_PREFIX[str(d)] = plugin
        _add(d)

    return dirs


#: project file -> words that make a skill relevant to it. Ranking only: no skill
#: is ever hidden by this, it just decides who gets the limited index budget.
_PROJECT_MARKERS = {
    "pyproject.toml": ("python", "pytest", "django", "fastapi"),
    "requirements.txt": ("python", "pytest"),
    "package.json": ("javascript", "typescript", "node", "react", "vue", "angular", "frontend"),
    "Cargo.toml": ("rust", "cargo"),
    "go.mod": ("golang", " go "),
    "CMakeLists.txt": ("c++", "cmake", "cuda"),
    "Makefile": ("c/c++", "kernel", "cuda"),
    "build.gradle": ("kotlin", "java", "android"),
    "Package.swift": ("swift", "ios"),
    "capacitor.config.ts": ("capacitor", "ionic", "mobile"),
}


def _project_words(cwd: Path) -> tuple[str, ...]:
    words: list[str] = []
    for marker, ws in _PROJECT_MARKERS.items():
        if (cwd / marker).exists():
            words.extend(ws)
    if any(cwd.glob("*.cu")) or any(cwd.glob("*/*.cu")):
        words.append("cuda")
    return tuple(words)


def _rank_for_project(entries: list, cwd: Path) -> list:
    """Stable re-order: skills that mention the project's stack first.

    Stable, so precedence (project > account > global > plugin) still breaks
    ties, and deterministic for a given cwd, so the prompt prefix doesn't move.
    """
    words = _project_words(cwd)
    if not words:
        return entries
    def score(e) -> int:
        text = f" {e.name} {e.description} ".lower()
        return sum(1 for w in words if w in text)
    return sorted(entries, key=lambda e: -score(e))


def skill_index(settings: dict | None = None, max_chars: int = 3000) -> str:
    """Compact "name: description" list for the STATIC part of the system prompt.

    Claude Code shows the model its available skills up front; greenboost-cli
    made it call Skill() blind. Bounded by max_chars (a small context pays for
    every character on every request) and built in precedence order, so the
    text is identical from turn to turn and the prompt cache survives.
    """
    try:
        entries = discover_skills_multi(discover_all_skill_dirs(settings or {}))
    except Exception:
        return ""
    if not entries:
        return ""
    entries = _rank_for_project(entries, Path.cwd())
    lines: list[str] = []
    used = 0
    entries_skipped = 0
    for e in entries:
        # hyperresearch-<N>-<step> skills are internal stages the router skill
        # chains to itself; listing sixteen of them spent the whole budget.
        if re.match(r"^hyperresearch-\d", e.name):
            entries_skipped += 1
            continue
        # YAML folded/literal markers ("description: >") leak into the parsed text.
        desc = " ".join(e.description.split()).lstrip(">|").strip()
        line = f"- {e.name}: {desc[:100]}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    rest = len(entries) - entries_skipped - len(lines)
    head = ("\n# Available skills\nInvoke with the Skill tool by name before doing "
            "work a skill covers.\n")
    tail = f"\n(+{rest} more: call Skill with no name to list them)" if rest > 0 else ""
    return head + "\n".join(lines) + tail + "\n"


def discover_skills_multi(dirs: list[Path]) -> list[SkillEntry]:
    """Discover skills across multiple directories, deduplicating by name."""
    all_entries: list[SkillEntry] = []
    seen_names: set[str] = set()
    for d in dirs:
        for e in discover_skills_in_dir(d):
            if e.name not in seen_names:
                all_entries.append(e)
                seen_names.add(e.name)
    return all_entries


def match_skills_multi(
    turn: str,
    dirs: list[Path],
    top_k: int = 3,
    min_score: float = 0.20,
    require_trigger: bool = True,
) -> list[SkillEntry]:
    """Match skills across multiple directories using embedding + trigger gating."""
    entries = discover_skills_multi(dirs)
    if not entries:
        return []

    embeddings = _load_or_build_embeddings(entries)
    if embeddings.size == 0:
        return []

    from greenboost_cli.rag.engine import _embed
    q_vec = _embed([turn], query=True, task="skill")[0]
    embeddings = _aligned(embeddings, q_vec, entries)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    e_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = e_norm @ q_norm

    turn_lower = turn.lower()
    hits: list[tuple[float, SkillEntry]] = []
    for entry, sim in zip(entries, sims.tolist()):
        trigger_bonus = 0.0
        for t in entry.triggers:
            if t and _trigger_match(turn_lower, t):
                trigger_bonus = 0.15
                break
        score = 0.85 * float(sim) + trigger_bonus
        if score < min_score:
            continue
        hits.append((score, entry))

    hits.sort(key=lambda x: x[0], reverse=True)

    matched: list[SkillEntry] = []
    for score, entry in hits:
        if entry.triggers:
            if any(_trigger_match(turn_lower, t) for t in entry.triggers):
                matched.append(entry)
        elif not require_trigger:
            matched.append(entry)
        if len(matched) >= top_k:
            break
    return matched


# ── Embedding cache ───────────────────────────────────────────────────────────

def _manifest_signature(entries: list[SkillEntry]) -> str:
    h = hashlib.sha256()
    # The embedder is part of the key: vectors from another model are not
    # comparable (and usually not even the same width).
    try:
        from greenboost_cli.rag.engine import EMBED_MODEL
        h.update(EMBED_MODEL.encode("utf-8"))
    except Exception:
        pass
    h.update(b"\x01")
    for e in sorted(entries, key=lambda x: x.name):
        h.update(e.name.encode("utf-8"))
        h.update(b"\x00")
        h.update(e.description.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _load_or_build_embeddings(entries: list[SkillEntry]) -> np.ndarray:
    """Return an (N, D) embedding matrix aligned with `entries`, caching to disk."""
    SKILLS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sig = _manifest_signature(entries)
    cache_file = SKILLS_CACHE_DIR / f"{sig}.npz"

    if cache_file.exists():
        try:
            data = np.load(str(cache_file))
            embeddings = data["embeddings"]
            if embeddings.shape[0] == len(entries):
                return embeddings
        except Exception:
            pass

    # Build fresh — reuse the same model the RAG uses
    from greenboost_cli.rag.engine import _embed
    texts = [f"{e.name}: {e.description}" for e in entries]
    embeddings = _embed(texts)

    # Clean stale caches in this dir before saving
    for old in SKILLS_CACHE_DIR.glob("*.npz"):
        if old.name != cache_file.name:
            try:
                old.unlink()
            except OSError:
                pass
    np.savez(str(cache_file), embeddings=embeddings)
    return embeddings


def _aligned(embeddings: np.ndarray, q_vec: np.ndarray, entries: list[SkillEntry]) -> np.ndarray:
    """Cached skill vectors must be the same width as the live query vector.

    They differ after the embedder changes, or when the primary embedder fails
    and the fallback answers. The mismatch raised a shape error that
    _auto_load_skills swallowed, so skills silently stopped loading. Rebuild
    with the model that is actually live instead.
    """
    if embeddings.ndim == 2 and embeddings.shape[1] == q_vec.shape[0]:
        return embeddings
    for old in SKILLS_CACHE_DIR.glob("*.npz"):
        try:
            old.unlink()
        except OSError:
            pass
    return _load_or_build_embeddings(entries)


# ── Routing ────────────────────────────────────────────────────────────────────

def route(
    query: str,
    skills_dir: Path,
    top_k: int = 5,
    min_score: float = 0.20,
) -> list[SkillHit]:
    """Score skills against query; return top-k SkillHits.

    Scoring blends:
      - cosine similarity between query embedding and skill description embedding (weight 0.85)
      - trigger keyword overlap bonus (weight 0.15, max +0.15)
    """
    entries = discover_skills(skills_dir)
    if not entries:
        return []

    embeddings = _load_or_build_embeddings(entries)
    if embeddings.size == 0:
        return []

    from greenboost_cli.rag.engine import _embed
    q_vec = _embed([query], query=True, task="skill")[0]
    embeddings = _aligned(embeddings, q_vec, entries)

    # Normalise for cosine
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    e_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = e_norm @ q_norm  # shape (N,)

    q_lower = query.lower()
    hits: list[SkillHit] = []
    for entry, sim in zip(entries, sims.tolist()):
        trigger_bonus = 0.0
        trigger_match = ""
        for t in entry.triggers:
            if t and t.lower() in q_lower:
                trigger_bonus = 0.15
                trigger_match = t
                break
        score = 0.85 * float(sim) + trigger_bonus
        if score < min_score:
            continue
        reason_parts = [f"sim={sim:.2f}"]
        if trigger_match:
            reason_parts.append(f"trigger='{trigger_match}'")
        hits.append(SkillHit(
            name=entry.name,
            score=round(score, 3),
            description=entry.description[:240],
            path=entry.path,
            reason=", ".join(reason_parts),
        ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def route_to_json(query: str, skills_dir: Path, top_k: int = 5,
                  min_score: float = 0.20) -> dict:
    """Serialised route() output for headless callers."""
    hits = route(query, skills_dir, top_k=top_k, min_score=min_score)
    return {
        "query": query,
        "skills_dir": str(skills_dir),
        "count": len(hits),
        "skills": [asdict(h) for h in hits],
    }


# ── Trigger-gated matching for auto-loading ───────────────────────────────────

def _trigger_match(turn: str, trigger: str) -> bool:
    """Return True if `trigger` (substring or regex) matches `turn`.

    A trigger entry is treated as a regex if it looks like one (contains
    regex metacharacters); otherwise it's a case-insensitive substring match.
    """
    if not trigger:
        return False
    turn_lower = turn.lower()
    t = trigger.strip()
    # Heuristic: looks like a regex if it has typical metacharacters
    if any(ch in t for ch in r".*+?^$[](){}|\\"):
        try:
            return re.search(t, turn, re.IGNORECASE) is not None
        except re.error:
            return t.lower() in turn_lower
    return t.lower() in turn_lower


def match_skills(
    turn: str,
    skills_dir: Path,
    top_k: int = 3,
    min_score: float = 0.20,
    require_trigger: bool = True,
) -> list[SkillEntry]:
    """Skill auto-load gate: embedding similarity AND trigger match.

    Combines `route()`'s top hits with a strict trigger gate: a skill only
    auto-loads if at least one of its declared `triggers` matches the user's
    turn (substring or regex). When `require_trigger=False`, skills without
    any triggers fall back to pure-embedding scoring.

    Returns the full SkillEntry (including absolute SKILL.md path) so the
    caller can read the body and inject it into the system prompt.
    """
    hits = route(turn, skills_dir, top_k=top_k * 3, min_score=min_score)
    if not hits:
        return []

    # Build a name → SkillEntry lookup so we can return the rich entry
    entries = discover_skills(skills_dir)
    by_name = {e.name: e for e in entries}

    matched: list[SkillEntry] = []
    for h in hits:
        entry = by_name.get(h.name)
        if entry is None:
            continue
        if entry.triggers:
            if any(_trigger_match(turn, t) for t in entry.triggers):
                matched.append(entry)
        elif not require_trigger:
            matched.append(entry)
        if len(matched) >= top_k:
            break
    return matched


_SKILL_BODY_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def load_skill_body(skill_md: Path, max_chars: int = 4000) -> str:
    """Read SKILL.md and return the body (post-frontmatter), capped at max_chars."""
    try:
        text = skill_md.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    body = _SKILL_BODY_RE.sub("", text, count=1).strip()
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n[...truncated...]"
    return body
