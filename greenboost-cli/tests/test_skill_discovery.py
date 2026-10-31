"""Skills: project and plugin roots, Claude Code precedence, a stable prompt index."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from greenboost_cli.skill import router as R


def _skill(root: Path, name: str, desc: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\nbody of {name}\n")


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    proj = home / "proj"
    proj.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.chdir(proj)
    R._PLUGIN_PREFIX.clear()
    return home, proj


def test_project_skill_beats_global_skill_of_same_name(env):
    home, proj = env
    _skill(proj / ".claude" / "skills", "deploy", "project version")
    _skill(home / ".claude" / "skills", "deploy", "global version")
    entries = R.discover_skills_multi(R.discover_all_skill_dirs({"skills_auto_discover": False}))
    assert [e.description for e in entries if e.name == "deploy"] == ["project version"]


def test_plugin_skills_are_namespaced_and_scoped(env):
    home, proj = env
    inst = home / "plug"
    _skill(inst / "skills", "bfl-api", "api help")
    (home / ".claude" / "plugins").mkdir(parents=True)
    (home / ".claude" / "plugins" / "installed_plugins.json").write_text(json.dumps({"plugins": {
        "flux@bfl": [{"scope": "user", "installPath": str(inst)},
                     {"scope": "project", "installPath": str(inst), "projectPath": "/elsewhere"}]}}))
    entries = R.discover_skills_multi(R.discover_all_skill_dirs({"skills_auto_discover": False}))
    assert [e.name for e in entries] == ["flux:bfl-api"]


def test_index_is_bounded_stable_and_skips_internal_steps(env):
    home, proj = env
    for i in range(60):
        _skill(home / ".claude" / "skills", f"skill-{i:02d}", "does a thing " * 20)
    _skill(home / ".claude" / "skills", "hyperresearch-3-contradiction-graph", "internal step")
    a = R.skill_index({"skills_auto_discover": False}, max_chars=1000)
    b = R.skill_index({"skills_auto_discover": False}, max_chars=1000)
    assert a == b                                   # identical => prompt cache survives
    assert len(a) < 1400 and "hyperresearch-3" not in a and "more: call Skill" in a


def test_index_strips_yaml_fold_marker(env):
    home, _ = env
    d = home / ".claude" / "skills" / "folded"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: folded\ndescription: >\n  the real text\n---\nx\n")
    assert "folded: the real text" in R.skill_index({"skills_auto_discover": False})


def test_index_prefers_skills_matching_the_project_stack(env):
    home, proj = env
    (proj / "Cargo.toml").write_text("[package]\n")
    for i in range(30):
        _skill(home / ".claude" / "skills", f"a-filler-{i:02d}", "unrelated cooking tips " * 5)
    _skill(home / ".claude" / "skills", "z-rust-helper", "Writes idiomatic Rust with cargo")
    out = R.skill_index({"skills_auto_discover": False}, max_chars=400)
    assert "z-rust-helper" in out                                   # would sort last otherwise
    assert out == R.skill_index({"skills_auto_discover": False}, max_chars=400)


def test_skill_vectors_are_rebuilt_when_the_embedder_width_changes(env, tmp_path, monkeypatch):
    import numpy as np
    from greenboost_cli.rag import engine as E
    home, _ = env
    _skill(home / ".claude" / "skills", "alpha", "does alpha things")
    monkeypatch.setattr(R, "SKILLS_CACHE_DIR", tmp_path / "cache")
    calls = []
    def fake_embed(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 8), dtype="float32")        # new embedder: 8-wide
    monkeypatch.setattr(E, "_embed", fake_embed)
    entries = R.discover_skills_multi(R.discover_all_skill_dirs({"skills_auto_discover": False}))
    (tmp_path / "cache").mkdir()
    np.savez(str(tmp_path / "cache" / "old.npz"), embeddings=np.ones((len(entries), 4), dtype="float32"))
    stale = np.ones((len(entries), 4), dtype="float32")         # 4-wide cache vs 8-wide query
    fresh = R._aligned(stale, np.ones(8, dtype="float32"), entries)
    assert fresh.shape == (len(entries), 8) and calls
