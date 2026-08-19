"""Memory that recalls itself: one fact per file, surfaced when it applies.

Before this module, memory here was `prime-goals.yaml` plus a history log,
surfaced only when a human typed `/memory`. Nothing brought a past lesson back
at the moment it mattered, which is the only moment it is worth anything.

Three things decide what gets recalled, in this order:

1. **Rules always.** A `type: rule` memory is a correction the user already
   made once. Those are the ratchet , the set of mistakes the agent no longer
   repeats only grows if every rule is present on every turn. They are never
   crowded out by relevance scoring.
2. **Scope matches.** A memory with `scope: greenboost_cuda_shim.c` surfaces
   when a path under that scope is touched, and stays quiet otherwise. On a
   repo spanning a kernel module, a CUDA shim, a network daemon and two Python
   layers, per-subsystem memory that arrives with the subsystem is worth more
   than any amount of general context.
3. **Query relevance, on a budget.** Whatever room is left goes to the
   memories whose name and description best match the turn.

**Deliberately no model call.** The reference implementation of this idea
spends a side-model call per turn asking a fast model to pick five memories
from a manifest. On a box generating at ~5 tok/s that is not a rounding error,
it is a second inference in the critical path of every single turn. Scoring is
lexical over descriptions the author wrote to be matched. If measurement later
says lexical scoring misses too much, a model call can be added for ties , but
it starts off, because a memory system that makes every turn slower is a
memory system the user turns off.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Recall budget, in characters, when the caller does not set one. Small on
#: purpose: memory competes with the actual conversation for the window.
DEFAULT_BUDGET_CHARS = 4_000

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)
_WORD = re.compile(r"[a-z0-9_]{3,}")

#: Types, in the order they are allowed to consume budget.
_TYPES = ("rule", "project", "user", "reference")


@dataclass
class Memory:
    name: str
    description: str
    type: str = "reference"
    scope: str = ""
    path: Path | None = None
    body: str = ""
    mtime: float = 0.0

    @property
    def age_days(self) -> int:
        import time
        return int((time.time() - self.mtime) / 86400) if self.mtime else 0

    def render(self) -> str:
        head = f"- **{self.name}**"
        if self.description:
            head += f" , {self.description}"
        # Age is rendered, not hidden. A memory is a record of what was true
        # when it was written; how long ago that was is what tells the reader
        # how hard to check it before acting.
        if self.age_days >= 7:
            head += f"  _(recorded {self.age_days}d ago)_"
        body = self.body.strip()
        return f"{head}\n  {body}" if body else head


def memory_dir(project_dir: Path | None = None) -> Path:
    """`<project>/memory/`, created on demand."""
    if project_dir is None:
        from greenboost_cli.memory.brain import project_dir as _pd
        project_dir = _pd()
    d = Path(project_dir) / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse(path: Path) -> Memory | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _FRONTMATTER.match(text)
    meta: dict = {}
    body = text
    if m:
        body = text[m.end():]
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip().strip('"').strip("'")
    return Memory(
        name=meta.get("name") or path.stem,
        description=meta.get("description", ""),
        type=(meta.get("type") or "reference").lower(),
        scope=meta.get("scope", ""),
        path=path,
        body=body.strip(),
        mtime=path.stat().st_mtime if path.exists() else 0.0,
    )


def scan(project_dir: Path | None = None) -> list:
    """Every memory, cheapest-first: frontmatter and body of small files."""
    d = memory_dir(project_dir)
    out = []
    for p in sorted(d.glob("*.md")):
        if p.name.upper() == "MEMORY.MD":
            continue                      # the index, not a memory
        mem = _parse(p)
        if mem is not None:
            out.append(mem)
    return out


#: What is NOT worth a memory, because it can be derived from the project
#: itself , and on a local model, every recalled character competes with the
#: conversation for the window, so noise here is not free.
DERIVABLE_PATTERNS = (
    (r"\b(is|lives|are) (in|at|under) [\w./-]+\.(py|c|h|ts|md|json|yaml)\b",
     "where code lives , reading the file answers that"),
    (r"\b(function|class|method|variable) [\w_]+\b.*\b(does|is|returns)\b",
     "what a symbol does , the code is authoritative"),
    (r"\bcommit\b|\bgit log\b|\bchanged (on|in) \d",
     "git history , `git log` is authoritative"),
    (r"\bfixed by\b|\bthe fix (was|is)\b",
     "a fix recipe , the fix is in the code and the commit message has the why"),
)


def derivable_reason(text: str) -> str:
    """Why this is not worth remembering, or "" if it is.

    The rule is not "save less", it is save the part that could not have been
    worked out from the project. When something looks derivable, the useful
    question is what was SURPRISING about it , that part is worth keeping.
    """
    import re as _re
    low = (text or "").lower()
    for pattern, reason in DERIVABLE_PATTERNS:
        if _re.search(pattern, low):
            return reason
    return ""


def write_memory(name: str, description: str, body: str, *,
                 mtype: str = "reference", scope: str = "",
                 project_dir: Path | None = None) -> Path:
    """Create or replace one memory. Returns its path."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "memory"
    p = memory_dir(project_dir) / f"{slug}.md"
    fm = [f"name: {name}", f"description: {description}", f"type: {mtype}"]
    if scope:
        fm.append(f"scope: {scope}")
    p.write_text("---\n" + "\n".join(fm) + "\n---\n\n" + body.strip() + "\n",
                 encoding="utf-8")
    return p


def record_rule(text: str, *, description: str = "",
                project_dir: Path | None = None) -> Path:
    """Codify a correction as a standing rule (the ratchet).

    A rule is the one memory type that is recalled unconditionally, so this is
    the mechanism by which a mistake the user corrects once stops recurring.
    """
    name = (description or text)[:60].strip().rstrip(".")
    return write_memory(name, description or text[:120], text,
                        mtype="rule", project_dir=project_dir)


#: Words that match everything and therefore mean nothing. Without this, a
#: query containing "the" scored against every memory whose description also
#: contained "the" , which is all of them.
_STOPWORDS = frozenset("""
the and for with that this from into then than when what why how are was were
you your our its has have had not but all any can could should would will
about out get got use used using make made does did done here there where
which while some more most such own same too very just now new
""".split())


def _tokens(s: str) -> set:
    return {w for w in _WORD.findall((s or "").lower()) if w not in _STOPWORDS}


def _scope_matches(mem: Memory, touched) -> bool:
    if not mem.scope:
        return False
    scope = mem.scope.strip()
    for t in touched or ():
        t = str(t)
        if scope in t or os.path.basename(t) == os.path.basename(scope):
            return True
    return False


def recall(query: str = "", touched_paths=None,
           budget_chars: int = DEFAULT_BUDGET_CHARS,
           project_dir: Path | None = None, memories=None) -> list:
    """Select the memories worth spending this turn's budget on."""
    mems = list(memories) if memories is not None else scan(project_dir)
    if not mems:
        return []
    q = _tokens(query)
    chosen, spent, seen = [], 0, set()

    def take(m) -> bool:
        nonlocal spent
        if m.name in seen:
            return False
        cost = len(m.render())
        if chosen and spent + cost > budget_chars:
            return False
        seen.add(m.name)
        chosen.append(m)
        spent += cost
        return True

    for m in mems:                                    # 1. rules, always
        if m.type == "rule":
            take(m)
    for m in mems:                                    # 2. scope hits
        if _scope_matches(m, touched_paths):
            take(m)
    scored = []                                       # 3. relevance, on budget
    for m in mems:
        if m.name in seen or not q:
            continue
        overlap = len(q & _tokens(f"{m.name} {m.description}"))
        if overlap:
            scored.append((overlap, _TYPES.index(m.type) if m.type in _TYPES else 9, m))
    for _, _, m in sorted(scored, key=lambda t: (-t[0], t[1], t[2].name)):
        if not take(m):
            break
    return chosen


def render_block(memories) -> str:
    """The injectable block, or "" when there is nothing worth injecting."""
    if not memories:
        return ""
    lines = [m.render() for m in memories]
    return ("<recalled-memory>\n"
            "Things you already learned about this project. They are recalled "
            "because they apply to this turn, not as general background:\n"
            + "\n".join(lines)
            + "\n\nThese record what was true when they were written. Before "
              "acting on one, check it against the current state , if a "
              "memory names a file, function or flag, confirm it still exists. "
              "When a memory and what you can see disagree, what you can see "
              "wins: say so, and correct the memory rather than working "
              "around it.\n"
              "</recalled-memory>")
