"""Markdown subagents: discovery, tool allowlists, /agent name, Delegate agent=."""
from greenboost_cli.agents import custom, subagent


def _mk(tmp_path, name, text, where=".greenboost/agents"):
    d = tmp_path / where; d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(text)


def test_discovery_parses_front_matter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _mk(tmp_path, "reviewer", "---\ndescription: reviews diffs\ntools: Read, Grep, Nope\n---\nYou review.")
    a = custom.discover(tmp_path)["reviewer"]
    assert a.description == "reviews diffs" and a.tools == ["Read", "Grep", "Nope"]
    assert a.body == "You review."


def test_unknown_tools_are_dropped_not_granted():
    known, unknown = custom.usable_tools(["Read", "Nope"])
    assert known == ["Read"] and unknown == ["Nope"]
    assert custom.usable_tools(None) == (None, [])


def test_delegate_with_agent_applies_body_and_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _mk(tmp_path, "scout", "---\ndescription: d\ntools: Read\n---\nBe brief.")
    seen = {}
    monkeypatch.setattr(subagent, "delegate", lambda prompt, **kw: seen.update(prompt=prompt, **kw) or "ok")
    assert subagent.delegate_named("find x", agent="scout") == "ok"
    assert seen["tools"] == ["Read"] and seen["prompt"].startswith("Be brief.") and "find x" in seen["prompt"]
    assert seen["label"] == "scout"


def test_delegate_unknown_agent_lists_the_real_ones(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _mk(tmp_path, "scout", "x")
    out = subagent.delegate_named("t", agent="nope")
    assert out.startswith("Error") and "scout" in out


def test_commands_registered():
    from greenboost_cli.terminal.commands import COMMAND_TABLE
    import greenboost_cli.slash_commands.agent_cmds  # noqa: F401
    assert "agents" in COMMAND_TABLE and "agent" in COMMAND_TABLE


def test_only_greenboost_folders_are_searched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    for d in (".claude/agents", ".greenboost_cli/agents"):
        (home / d).mkdir(parents=True)
    (home / ".claude/agents/other.md").write_text("x")
    (home / ".greenboost_cli/agents/mine.md").write_text("x")
    _mk(tmp_path, "proj", "x", where=".claude/agents")
    monkeypatch.setattr(custom.Path, "home", classmethod(lambda cls: home))
    assert set(custom.discover(tmp_path)) == {"mine"}
