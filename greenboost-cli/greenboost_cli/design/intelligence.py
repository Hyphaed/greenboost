"""Design intelligence engine — BM25 over ui-ux-pro-max-skill CSV databases."""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

_DEFAULT_SKILL_DIR = Path.home() / "Dev" / "claude_workflow_sources" / "ui_design" / "ui-ux-pro-max-skill"


def _resolve_skill_dir() -> Path:
    """Return the skill data directory, trying env var then default path."""
    env = os.environ.get("GB_DESIGN_SKILL_DIR", "")
    if env:
        return Path(env)
    return _DEFAULT_SKILL_DIR


SKILL_DIR = _resolve_skill_dir()

DB_FILES = {
    "styles":     SKILL_DIR / "styles.csv",
    "colors":     SKILL_DIR / "colors.csv",
    "typography": SKILL_DIR / "typography.csv",
    "products":   SKILL_DIR / "products.csv",
    "charts":     SKILL_DIR / "charts.csv",
    "ux":         SKILL_DIR / "ux-guidelines.csv",
    "reasoning":  SKILL_DIR / "ui-reasoning.csv",
    "landing":    SKILL_DIR / "landing.csv",
}


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokenized  = [_tokenize(doc) for doc in corpus]
        self.n          = len(corpus)
        self.avgdl      = sum(len(t) for t in self.tokenized) / max(self.n, 1)
        self.df: dict[str, int] = {}
        for tokens in self.tokenized:
            for tok in set(tokens):
                self.df[tok] = self.df.get(tok, 0) + 1

    def score(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_terms = _tokenize(query)
        scores = []
        for i, tokens in enumerate(self.tokenized):
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            dl = len(tokens)
            s  = 0.0
            for term in query_terms:
                tf = tf_map.get(term, 0)
                idf = math.log((self.n - self.df.get(term, 0) + 0.5) /
                               (self.df.get(term, 0) + 0.5) + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                s += idf * (tf * (self.k1 + 1)) / max(denom, 1e-9)
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


_db:          dict[str, list[dict]] = {}
_bm25:        dict[str, BM25]       = {}
_loaded_from: Path | None           = None


def _skill_dir_from_settings(settings: dict | None) -> Path | None:
    if settings:
        override = settings.get("design_skill_dir", "")
        if override:
            return Path(override)
    return None


def _ensure_loaded(skill_dir: Path | None = None) -> None:
    global _db, _bm25, _loaded_from
    effective = skill_dir or SKILL_DIR
    if _db and _loaded_from == effective:
        return
    _db.clear()
    _bm25.clear()
    db_files = {
        "styles":     effective / "styles.csv",
        "colors":     effective / "colors.csv",
        "typography": effective / "typography.csv",
        "products":   effective / "products.csv",
        "charts":     effective / "charts.csv",
        "ux":         effective / "ux-guidelines.csv",
        "reasoning":  effective / "ui-reasoning.csv",
        "landing":    effective / "landing.csv",
    }
    for key, path in db_files.items():
        rows = _load_csv(path)
        _db[key] = rows
        if rows:
            texts      = [" ".join(str(v) for v in row.values()) for row in rows]
            _bm25[key] = BM25(texts)
    _loaded_from = effective


def search(
    query: str,
    domains: list[str] | None = None,
    top_k: int = 3,
    settings: dict | None = None,
) -> dict[str, list[dict]]:
    _ensure_loaded(_skill_dir_from_settings(settings))
    domains = domains or list(_db.keys())
    results: dict[str, list[dict]] = {}
    for domain in domains:
        if domain not in _bm25 or not _db[domain]:
            continue
        hits = _bm25[domain].score(query, top_k)
        results[domain] = [_db[domain][i] for i, score in hits if score > 0]
    return results


def generate_design_system(prompt: str, settings: dict | None = None) -> dict[str, Any]:
    _ensure_loaded(_skill_dir_from_settings(settings))
    results = search(
        prompt,
        domains=["styles", "colors", "typography", "products", "reasoning", "ux", "landing"],
        top_k=2,
        settings=settings,
    )
    return {
        "prompt":          prompt,
        "style":           results.get("styles", [{}])[0],
        "colors":          results.get("colors", [{}])[:2],
        "typography":      results.get("typography", [{}])[0],
        "reasoning":       results.get("reasoning", [{}])[0],
        "ux_rules":        results.get("ux", [{}])[:3],
        "landing_pattern": results.get("landing", [{}])[0],
    }


def format_design_system(ds: dict[str, Any]) -> str:
    lines = [f"# Design System: {ds['prompt']}", "", "## Visual Style"]
    for k, v in ds.get("style", {}).items():
        if v:
            lines.append(f"- **{k}**: {v}")

    lines += ["", "## Color Palettes"]
    for palette in ds.get("colors", []):
        for k, v in palette.items():
            if v:
                lines.append(f"- **{k}**: {v}")

    lines += ["", "## Typography"]
    for k, v in ds.get("typography", {}).items():
        if v:
            lines.append(f"- **{k}**: {v}")

    lines += ["", "## Industry Reasoning"]
    for k, v in ds.get("reasoning", {}).items():
        if v:
            lines.append(f"- **{k}**: {v}")

    lines += ["", "## UX Guidelines"]
    for rule in ds.get("ux_rules", []):
        for k, v in rule.items():
            if v:
                lines.append(f"- {v}")
            break

    lines += ["", "## Landing Page Pattern"]
    for k, v in ds.get("landing_pattern", {}).items():
        if v:
            lines.append(f"- **{k}**: {v}")

    return "\n".join(lines)


def is_available(settings: dict | None = None) -> bool:
    """Return True if the skill data directory exists.

    Checks (in order): settings["design_skill_dir"], $GB_DESIGN_SKILL_DIR,
    then the default developer path.
    """
    if settings:
        override = settings.get("design_skill_dir", "")
        if override and Path(override).exists():
            return True
    return _resolve_skill_dir().exists()
