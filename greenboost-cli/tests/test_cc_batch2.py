"""Markdown commands, /gpu verdicts, output sanitising."""
from greenboost_cli import custom_commands as CC
from greenboost_cli.terminal import cc_features as C


def test_markdown_command_expands_arguments(tmp_path):
    d = tmp_path / ".greenboost" / "commands"; d.mkdir(parents=True)
    (d / "fix.md").write_text("---\ndescription: fix it\n---\nFix this: $ARGUMENTS\n")
    assert CC.expand("fix", "the bug", tmp_path) == "Fix this: the bug"
    assert CC.discover(tmp_path)["fix"][1] == "fix it"


def test_arguments_appended_when_no_placeholder(tmp_path):
    d = tmp_path / ".greenboost" / "commands"; d.mkdir(parents=True)
    (d / "a.md").write_text("Do a thing.")
    assert CC.expand("a", "now", tmp_path) == "Do a thing.\n\nnow"


def test_shipped_commands_exist_and_project_overrides(tmp_path):
    assert {"security-review", "review", "explain"} <= set(CC.discover(tmp_path))
    d = tmp_path / ".greenboost" / "commands"; d.mkdir(parents=True)
    (d / "review.md").write_text("mine")
    assert CC.expand("review", "", tmp_path) == "mine"


def test_unknown_command_is_none(tmp_path):
    assert CC.expand("nope", "", tmp_path) is None


def test_sanitize_strips_escapes_and_controls():
    assert C.sanitize("a\x1b[2Jb\x1b]0;title\x07c\x08d\n\te") == "abcd\n\te"


def test_gpu_verdict():
    ok = {"total": 12000, "used": 1000, "free": 11000, "apps": []}
    assert C.gpu_verdict(ok) == "ok"
    busy = {"total": 12000, "used": 9000, "free": 3000, "apps": [(1, "FLUX generation", 8000)]}
    v = C.gpu_verdict(busy)
    assert "FLUX generation" in v and "nothing is killed" in v
    mine = {"total": 12000, "used": 9000, "free": 3000, "apps": [(2, "gb-synapse model (llama-server)", 8000)]}
    assert "spill" in C.gpu_verdict(mine)
    desk = {"total": 12000, "used": 3000, "free": 9000, "apps": [(3, "chrome", 400), (4, "slack", 200)]}
    assert "desktop apps" in C.gpu_verdict(desk) and "held by" not in C.gpu_verdict(desk)


def test_new_commands_registered():
    from greenboost_cli.terminal.commands import COMMAND_TABLE
    for n in ("gpu", "hooks", "vim", "think", "commands"):
        assert n in COMMAND_TABLE


def test_output_style_appends_to_system_prompt(tmp_path, monkeypatch):
    from greenboost_cli import output_styles as OS
    monkeypatch.chdir(tmp_path)
    assert OS.style_text({}) == ""
    t = OS.style_text({"output_style": "concise"})
    assert "Output style: concise" in t and "Lead with the result" in t
    d = tmp_path / ".greenboost" / "output-styles"; d.mkdir(parents=True)
    (d / "concise.md").write_text("mine")
    assert "mine" in OS.style_text({"output_style": "concise"})
    assert OS.style_text({"output_style": "nope"}) == ""


def test_think_sets_both_settings():
    s = {}
    C.cmd_think("on", None, s)
    assert s["enable_thinking"] is True and s["qwen_thinking"] is True


def test_studio_status_reports_summary(monkeypatch):
    import json
    import greenboost_cli.mcp.client as MC

    class Fake:
        def __init__(self, *a, **k): pass
        def connect(self): return True
        def call_tool(self, name, args):
            return json.dumps({"summary": "FLUX: yes; Qwen: no"}) if name == "gpu_status" else "ERROR: boom"
        def close(self): pass

    monkeypatch.setattr(MC, "MCPStdioClient", Fake)
    st = C.studio_status(5)
    assert st["summary"] == "FLUX: yes; Qwen: no" and st["projects"] is None and "boom" in st["projects_error"]


def test_tab_is_not_stolen_from_completion():
    """ctrl+i IS Tab in a terminal; binding it to a command silently disabled completion."""
    import inspect
    from greenboost_cli.terminal import repl as R
    src = inspect.getsource(R)
    assert '_pt_kb.add("c-i")' not in src and '_pt_kb.add("tab"' not in src
    assert '_pt_kb.add("c-q")' in src and '_pt_kb.add("c-t")' in src


def test_picker_takes_and_returns_the_terminal(monkeypatch):
    """A slash-command picker must not read the keyboard alongside the live input box."""
    from greenboost_cli.terminal import repl as R, wizard_prompt as W
    calls = []
    monkeypatch.setattr(R, "_pt_suspend_hook", lambda: calls.append("hook"))
    monkeypatch.setattr(W, "_picker", lambda *a, **k: calls.append("picker") or [2])
    assert W.run_picker([{"label": "a", "description": ""}]) == [2]
    assert calls == ["hook", "picker"] and R._wizard_free.is_set() and R._wizard_depth == 0


def test_nested_wizards_keep_the_terminal_until_the_outer_one_ends(monkeypatch):
    from greenboost_cli.terminal import repl as R
    monkeypatch.setattr(R, "_pt_suspend_hook", None)
    R._suspend_pt_for_wizard(); R._suspend_pt_for_wizard()
    R._resume_pt_after_wizard()
    assert not R._wizard_free.is_set() and R._wizard_depth == 1
    R._resume_pt_after_wizard()
    assert R._wizard_free.is_set() and R._wizard_depth == 0


def test_prompt_waits_for_a_loading_model():
    states = iter(["starting", "starting", "running"])
    waited = []
    ok = C.await_model_ready({}, on_wait=lambda: waited.append(1), status_fn=lambda s: next(states), sleep=lambda x: None)
    assert ok and waited == [1]


def test_wait_can_be_cancelled_and_never_hangs():
    assert C.await_model_ready({}, cancelled=lambda: True, status_fn=lambda s: "starting", sleep=lambda x: None) is False
    assert C.await_model_ready({}, status_fn=lambda s: "stopped") is True
    assert C.await_model_ready({}, timeout=0.0, status_fn=lambda s: "starting", sleep=lambda x: None) is True


def test_http_502_is_explained_as_still_loading():
    head, hints = C.friendly_turn_error("\nBackend at http://localhost:11369/v1 rejected the request (HTTP 502).\n Raw error: Error code: 502")
    assert "still loading" in head and any("/retry" in h[1] for h in hints)
