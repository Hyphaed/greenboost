"""Smoke tests for pure-function logic in greenboost-cli.

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path


# ── _ThinkFilter ──────────────────────────────────────────────────────────────

def test_thinkfilter_passthrough():
    from greenboost_cli.inference.adapters import _ThinkFilter
    tf = _ThinkFilter()
    results = tf.feed("Hello world")
    results += tf.flush()
    texts = "".join(t for t, _ in results)
    assert "Hello" in texts
    for _, is_t in results:
        assert not is_t


def test_thinkfilter_full_block():
    from greenboost_cli.inference.adapters import _ThinkFilter
    tf = _ThinkFilter()
    out = tf.feed("<thinking>inner</thinking>done")
    out += tf.flush()
    think = "".join(t for t, is_t in out if is_t)
    prose = "".join(t for t, is_t in out if not is_t)
    assert "inner" in think
    assert "done" in prose


def test_thinkfilter_split_across_chunks():
    from greenboost_cli.inference.adapters import _ThinkFilter
    tf = _ThinkFilter()
    out  = tf.feed("<thi")
    out += tf.feed("nking>idea</thinking> result")
    out += tf.flush()
    think = "".join(t for t, is_t in out if is_t)
    prose = "".join(t for t, is_t in out if not is_t)
    assert "idea" in think
    assert "result" in prose


def test_thinkfilter_short_tag():
    from greenboost_cli.inference.adapters import _ThinkFilter
    tf = _ThinkFilter()
    out  = tf.feed("<think>brief</think>after")
    out += tf.flush()
    think = "".join(t for t, is_t in out if is_t)
    prose = "".join(t for t, is_t in out if not is_t)
    assert "brief" in think
    assert "after" in prose


# ── handle_read / handle_edit ─────────────────────────────────────────────────

def test_handle_read_basic(tmp_path):
    from greenboost_cli.instruments.handlers import handle_read
    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\nline3\n")
    result = handle_read(str(f))
    assert "line1" in result
    assert "line2" in result


def test_handle_read_missing():
    from greenboost_cli.instruments.handlers import handle_read
    result = handle_read("/nonexistent/path/file.txt")
    assert "Error" in result or "not found" in result.lower()


def test_handle_read_limit(tmp_path):
    from greenboost_cli.instruments.handlers import handle_read
    f = tmp_path / "many.txt"
    f.write_text("\n".join(f"line{i}" for i in range(20)))
    result = handle_read(str(f), limit=5)
    assert "line0" in result
    assert "line6" not in result


def test_handle_edit_basic(tmp_path):
    from greenboost_cli.instruments.handlers import handle_edit
    f = tmp_path / "edit_me.txt"
    f.write_text("foo bar baz")
    result = handle_edit(str(f), "bar", "QUX")
    assert "Replaced" in result
    assert f.read_text() == "foo QUX baz"


def test_handle_edit_not_found(tmp_path):
    from greenboost_cli.instruments.handlers import handle_edit
    f = tmp_path / "no_match.txt"
    f.write_text("aaa bbb")
    result = handle_edit(str(f), "zzz", "X")
    assert "Error" in result


def test_handle_edit_ambiguous(tmp_path):
    from greenboost_cli.instruments.handlers import handle_edit
    f = tmp_path / "dup.txt"
    f.write_text("x x x")
    result = handle_edit(str(f), "x", "y")
    assert "Error" in result or "appears" in result


def test_handle_edit_replace_all(tmp_path):
    from greenboost_cli.instruments.handlers import handle_edit
    f = tmp_path / "dup2.txt"
    f.write_text("x x x")
    handle_edit(str(f), "x", "y", replace_all=True)
    assert f.read_text() == "y y y"


# ── brain.read_recent_history edge cases ─────────────────────────────────────

def test_read_recent_history_empty_file(tmp_path):
    from greenboost_cli.memory.brain import read_recent_history
    (tmp_path / "history.md").write_text("")
    result = read_recent_history(tmp_path)
    assert result == "(no history yet)"


def test_read_recent_history_no_file(tmp_path):
    from greenboost_cli.memory.brain import read_recent_history
    result = read_recent_history(tmp_path)
    assert result == "(no history yet)"


def test_read_recent_history_has_entries(tmp_path):
    from greenboost_cli.memory.brain import read_recent_history, append_history
    append_history(tmp_path, "First note", "note")
    append_history(tmp_path, "Second note", "milestone")
    result = read_recent_history(tmp_path)
    assert "First note" in result
    assert "Second note" in result


# ── handle_glob ───────────────────────────────────────────────────────────────

def test_handle_glob_finds_files(tmp_path):
    from greenboost_cli.instruments.handlers import handle_glob
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    (tmp_path / "c.txt").write_text("z")
    result = handle_glob("*.py", str(tmp_path))
    assert "a.py" in result
    assert "b.py" in result
    assert "c.txt" not in result


def test_handle_glob_no_match(tmp_path):
    from greenboost_cli.instruments.handlers import handle_glob
    result = handle_glob("*.xyz", str(tmp_path))
    assert "No files matched" in result


# ── is_readonly_command safety ────────────────────────────────────────────────

def test_readonly_safe_commands():
    from greenboost_cli.instruments.safety import is_readonly_command
    assert is_readonly_command("ls -la")
    assert is_readonly_command("git log --oneline -10")
    assert is_readonly_command("git status")
    assert is_readonly_command("find . -name '*.py'")
    assert is_readonly_command("find /tmp -maxdepth 2 -type f")
    assert is_readonly_command("grep 'a < b' file.py")


def test_readonly_unsafe_commands():
    from greenboost_cli.instruments.safety import is_readonly_command
    assert not is_readonly_command("rm -rf /")
    assert not is_readonly_command("ls; rm -rf /")
    assert not is_readonly_command("echo foo > /etc/passwd")
    # Interpreters must require explicit approval — a single call with no
    # chain operator can still execute arbitrary code (HIGH security risk).
    assert not is_readonly_command("python3 -c 'x = 5 < 10'")
    assert not is_readonly_command("python script.py")
    assert not is_readonly_command("node index.js")
    assert not is_readonly_command("ruby hack.rb")
    assert not is_readonly_command("perl -e 'print 1'")
    # find with destructive flags must require approval.
    assert not is_readonly_command("find . -delete")
    assert not is_readonly_command("find . -exec rm {} \\;")


# ── is_autonomous_safe ────────────────────────────────────────────────────────

def test_autonomous_safe_coding_commands():
    from greenboost_cli.instruments.safety import is_autonomous_safe
    # Test runners — should be auto-approved
    assert is_autonomous_safe("pytest tests/")
    assert is_autonomous_safe("python -m pytest")
    assert is_autonomous_safe("npm test")
    assert is_autonomous_safe("npm run build")
    assert is_autonomous_safe("cargo test")
    assert is_autonomous_safe("go test ./...")
    # Package managers
    assert is_autonomous_safe("pip install requests")
    assert is_autonomous_safe("npm install")
    assert is_autonomous_safe("poetry install")
    # Linters / formatters
    assert is_autonomous_safe("black src/")
    assert is_autonomous_safe("ruff check .")
    assert is_autonomous_safe("git add -p")
    assert is_autonomous_safe("git commit -m 'fix: update'")
    assert is_autonomous_safe("git pull")
    # Regular read-only commands still work
    assert is_autonomous_safe("ls -la")
    assert is_autonomous_safe("find . -name '*.py'")


def test_autonomous_blocks_destructive():
    from greenboost_cli.instruments.safety import is_autonomous_safe
    # Hard-blocked even in autonomous mode
    assert not is_autonomous_safe("git push")
    assert not is_autonomous_safe("git push --force origin main")
    assert not is_autonomous_safe("rm -rf /tmp/foo")
    assert not is_autonomous_safe("sudo rm -rf .")
    assert not is_autonomous_safe("rm -r src/")
    # Chain operators blocked
    assert not is_autonomous_safe("pytest && git push")
    assert not is_autonomous_safe("git commit -m x; git push")
    assert not is_autonomous_safe("npm test | tee log.txt")
    # find with destructive flags
    assert not is_autonomous_safe("find . -delete")
    assert not is_autonomous_safe("find . -exec rm {} \\;")


# ── compress_text ─────────────────────────────────────────────────────────────

def test_compress_text_small_passthrough():
    from greenboost_cli.workflow.intelligence import compress_text
    text = "Short text"
    assert compress_text(text) == text


def test_compress_text_no_headers_truncates():
    from greenboost_cli.workflow.intelligence import compress_text
    long = "word " * 2000
    result = compress_text(long, target_chars=100)
    assert len(result) <= 115  # at most target + ellipsis overhead
    assert "[…]" in result


def test_compress_text_headers_preserved():
    from greenboost_cli.workflow.intelligence import compress_text
    text = "## Section A\n" + ("word " * 500) + "\n## Section B\n" + ("word " * 500)
    result = compress_text(text, target_chars=200)
    assert "## Section A" in result
    assert "## Section B" in result
    assert len(result) < len(text)


# ── handle_write ──────────────────────────────────────────────────────────────

def test_handle_write_basic(tmp_path):
    from greenboost_cli.instruments.handlers import handle_write
    f = tmp_path / "out.txt"
    result = handle_write(str(f), "hello\nworld\n")
    assert "Wrote" in result
    assert f.read_text() == "hello\nworld\n"


def test_handle_write_creates_parent_dirs(tmp_path):
    from greenboost_cli.instruments.handlers import handle_write
    f = tmp_path / "nested" / "dir" / "file.txt"
    result = handle_write(str(f), "data")
    assert "Wrote" in result
    assert f.read_text() == "data"


# ── _summarize_turn_pair ──────────────────────────────────────────────────────

def test_summarize_turn_pair_basic():
    from greenboost_cli.workflow.intelligence import _summarize_turn_pair
    user = {"role": "user", "content": "What is 2+2?"}
    asst = {"role": "assistant", "content": "The answer is 4."}
    summary = _summarize_turn_pair(user, asst)
    assert "2+2" in summary
    assert "4" in summary


def test_summarize_turn_pair_with_tool_calls():
    from greenboost_cli.workflow.intelligence import _summarize_turn_pair
    user = {"role": "user", "content": "List files"}
    asst = {
        "role": "assistant",
        "content": "I'll list the files.",
        "tool_calls": [{"name": "Bash", "id": "x", "input": {"command": "ls"}}],
    }
    summary = _summarize_turn_pair(user, asst)
    assert "Bash" in summary


# ── _compress_context force mode ──────────────────────────────────────────────

def test_compress_context_force(tmp_path, monkeypatch):
    from greenboost_cli.workflow.intelligence import _compress_context
    from greenboost_cli.core.session import ConversationSession

    session = ConversationSession()
    for i in range(12):
        session.messages.append({"role": "user",      "content": f"Q{i}: " + "a" * 100})
        session.messages.append({"role": "assistant", "content": f"A{i}: " + "b" * 100})

    before = len(session.messages)
    _compress_context(session, {}, force=True)
    after = len(session.messages)
    assert after < before


def test_compress_context_skips_small_session():
    from greenboost_cli.workflow.intelligence import _compress_context
    from greenboost_cli.core.session import ConversationSession

    session = ConversationSession()
    for i in range(3):
        session.messages.append({"role": "user",      "content": f"Q{i}"})
        session.messages.append({"role": "assistant", "content": f"A{i}"})

    before = len(session.messages)
    _compress_context(session, {}, force=True)
    assert len(session.messages) == before  # fewer than 10 — no change


# ── resolve_backend — gb-synapse is the only backend ─────────────────────────

def test_resolve_backend_always_gb_synapse():
    from greenboost_cli.inference.router import resolve_backend
    assert resolve_backend("qwen3-coder") == "gb-synapse"
    assert resolve_backend("gb-synapse/qwen3-coder") == "gb-synapse"
    assert resolve_backend("org/repo:Q4_K_M") == "gb-synapse"


def test_resolve_backend_empty():
    from greenboost_cli.inference.router import resolve_backend
    assert resolve_backend("") == "gb-synapse"


# ── strip_prefix ──────────────────────────────────────────────────────────────

def test_strip_prefix_removes_gb_synapse_prefix():
    from greenboost_cli.inference.router import strip_prefix
    assert strip_prefix("gb-synapse/qwen3-coder") == "qwen3-coder"


def test_strip_prefix_no_prefix():
    from greenboost_cli.inference.router import strip_prefix
    assert strip_prefix("qwen3-coder") == "qwen3-coder"


def test_strip_prefix_preserves_ollama_namespace():
    # A bare Ollama namespace/model:tag has no "gb-synapse/" prefix to strip.
    from greenboost_cli.inference.router import strip_prefix
    assert strip_prefix("mirage335/some-model:latest") == "mirage335/some-model:latest"


# ── _compress_context with tool messages ─────────────────────────────────────

def test_compress_context_absorbs_tool_messages():
    from greenboost_cli.workflow.intelligence import _compress_context
    from greenboost_cli.core.session import ConversationSession

    session = ConversationSession()
    for i in range(6):
        session.messages.append({"role": "user",      "content": f"Q{i}"})
        session.messages.append({"role": "assistant", "content": f"A{i}", "tool_calls": []})
        session.messages.append({"role": "tool",      "tool_call_id": f"id{i}", "name": "Bash", "content": f"result{i}"})

    _compress_context(session, {}, force=True)
    # Tool messages should be absorbed — summary messages should be first
    roles = [m["role"] for m in session.messages[:2]]
    assert roles == ["user", "assistant"]  # summary pair at the front


# ── llamacpp_server_status (gb-synapse) ──────────────────────────────────────

def test_llamacpp_status_stopped_when_not_running(monkeypatch):
    import greenboost_cli.slash_commands.backend_cmds as bc
    monkeypatch.setattr(bc, "_llamacpp_running_pid", lambda settings=None: None)
    assert bc.llamacpp_server_status({"model": "test-model"}) == "stopped"


def test_llamacpp_status_starting_when_unreachable(monkeypatch):
    import greenboost_cli.slash_commands.backend_cmds as bc
    monkeypatch.setattr(bc, "_llamacpp_running_pid", lambda settings=None: 12345)
    monkeypatch.setattr(bc, "_llamacpp_base_url", lambda settings: "http://localhost:19999/v1")
    assert bc.llamacpp_server_status({"model": "test-model"}) == "starting"


# ── design/intelligence.search with nonexistent settings dir ─────────────────

def test_design_search_nonexistent_settings_dir():
    from greenboost_cli.design.intelligence import search
    results = search("hero layout", settings={"design_skill_dir": "/nonexistent/path/xyz"})
    assert isinstance(results, dict)
    # No crash — all domains empty since the dir doesn't exist
    assert all(isinstance(v, list) for v in results.values())


# ── tasks/tracker.py ─────────────────────────────────────────────────────────

def test_task_add_and_list(tmp_path, monkeypatch):
    import greenboost_cli.tasks.tracker as _t
    monkeypatch.setattr(_t, "TASKS_DIR", tmp_path)
    task = _t.add_task("myproject", "Write tests", description="cover edge cases")
    assert task.id.startswith("t")
    assert task.subject == "Write tests"
    assert task.status == "pending"
    tasks = _t.list_tasks("myproject")
    assert len(tasks) == 1
    assert tasks[0].id == task.id


def test_task_update_status(tmp_path, monkeypatch):
    import greenboost_cli.tasks.tracker as _t
    monkeypatch.setattr(_t, "TASKS_DIR", tmp_path)
    task = _t.add_task("proj", "Do something")
    updated = _t.update_task("proj", task.id, "in_progress")
    assert updated is not None
    assert updated.status == "in_progress"
    # Confirm persisted
    tasks = _t.list_tasks("proj")
    assert tasks[0].status == "in_progress"


def test_task_update_invalid_status(tmp_path, monkeypatch):
    import greenboost_cli.tasks.tracker as _t
    monkeypatch.setattr(_t, "TASKS_DIR", tmp_path)
    _t.add_task("proj", "task")
    try:
        _t.update_task("proj", "t001-0000", "nonexistent")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_task_delete(tmp_path, monkeypatch):
    import greenboost_cli.tasks.tracker as _t
    monkeypatch.setattr(_t, "TASKS_DIR", tmp_path)
    task = _t.add_task("proj", "To delete")
    assert _t.delete_task("proj", task.id) is True
    assert _t.list_tasks("proj") == []
    assert _t.delete_task("proj", task.id) is False  # already gone


def test_task_default_active_form(tmp_path, monkeypatch):
    import greenboost_cli.tasks.tracker as _t
    monkeypatch.setattr(_t, "TASKS_DIR", tmp_path)
    task = _t.add_task("proj", "Refactor router")
    # active_form defaults to subject when not provided
    assert task.active_form == "Refactor router"


# ── planning/plan.py ─────────────────────────────────────────────────────────

def test_create_and_read_plan(tmp_path, monkeypatch):
    import greenboost_cli.planning.plan as _p
    monkeypatch.setattr(_p, "PLANS_DIR", tmp_path)
    entry = _p.create_plan("Fix the authentication bug")
    assert entry.id
    assert entry.path.exists()
    body = _p.read_plan(entry.id)
    assert "Fix the authentication bug" in body
    assert "## Steps" in body


def test_plan_short_id_uniqueness():
    from greenboost_cli.planning.plan import short_id
    ids = {short_id() for _ in range(10)}
    assert len(ids) == 10  # no collisions


def test_list_plans(tmp_path, monkeypatch):
    import greenboost_cli.planning.plan as _p
    monkeypatch.setattr(_p, "PLANS_DIR", tmp_path)
    _p.create_plan("Plan A")
    _p.create_plan("Plan B")
    plans = _p.list_plans()
    assert len(plans) == 2
    # sorted newest-first — both have the same mtime so order is not strict,
    # but both must appear
    ids = {e.id for e in plans}
    assert len(ids) == 2


def test_read_nonexistent_plan(tmp_path, monkeypatch):
    import greenboost_cli.planning.plan as _p
    monkeypatch.setattr(_p, "PLANS_DIR", tmp_path)
    assert _p.read_plan("doesnotexist") == ""


# ── cluster/config.py ────────────────────────────────────────────────────────

def test_cluster_config_roundtrip(tmp_path, monkeypatch):
    import greenboost_cli.cluster.config as _cc
    user_cfg = tmp_path / "cluster.conf"
    monkeypatch.setattr(_cc, "USER_CONFIG", user_cfg)
    monkeypatch.setattr(_cc, "SYS_CONFIG", tmp_path / "nonexistent.conf")
    from greenboost_cli.cluster.config import Peer, ClusterConfig, save_config, load_config
    cfg = ClusterConfig(peers=[Peer("10.0.0.1", 9740, "peer1", "ubuntu")],
                        cluster_extra_mem_gb=16)
    save_config(cfg)
    loaded = load_config()
    assert len(loaded.peers) == 1
    assert loaded.peers[0].host == "10.0.0.1"
    assert loaded.peers[0].hostname == "peer1"
    assert loaded.cluster_extra_mem_gb == 16


def test_cluster_config_find(tmp_path, monkeypatch):
    import greenboost_cli.cluster.config as _cc
    monkeypatch.setattr(_cc, "USER_CONFIG", tmp_path / "cluster.conf")
    monkeypatch.setattr(_cc, "SYS_CONFIG", tmp_path / "nonexistent.conf")
    from greenboost_cli.cluster.config import Peer, ClusterConfig, save_config, load_config
    cfg = ClusterConfig(peers=[Peer("192.168.1.5", 9740, "omen", "ferran")])
    save_config(cfg)
    loaded = load_config()
    assert loaded.find("192.168.1.5") is not None
    assert loaded.find("omen") is not None
    assert loaded.find("nonexistent") is None


# ── execute_turn_sync signature guard ────────────────────────────────────────

def test_execute_turn_sync_signature():
    """Regression: factory.py was calling execute_turn_sync with wrong kwargs.
    This test pins the parameter names so future renames fail loudly.
    """
    import inspect
    from greenboost_cli.core.orchestrator import execute_turn_sync
    params = list(inspect.signature(execute_turn_sync).parameters)
    assert params[0] == "user_message", f"expected 'user_message', got '{params[0]}'"
    assert params[1] == "session",      f"expected 'session',      got '{params[1]}'"
    assert params[2] == "settings",     f"expected 'settings',     got '{params[2]}'"
    assert params[3] == "system_context", f"expected 'system_context', got '{params[3]}'"


# ── generate_ui_asset signature guard ────────────────────────────────────────

def test_generate_ui_asset_signature():
    """Regression: dashboard was calling generate_ui_asset(prompt=, ...) missing
    required output_path arg. Pin the signature to catch future drifts.
    """
    import inspect
    from greenboost_cli.diffusion.pipeline import generate_ui_asset
    params = inspect.signature(generate_ui_asset).parameters
    assert "asset_type" in params
    assert "output_path" in params
    assert "custom_prompt" in params
    # 'prompt' is not a valid kwarg — make sure it was never re-added
    assert "prompt" not in params


# ── skill/router.py ──────────────────────────────────────────────────────────

def test_discover_skills_empty_dir(tmp_path):
    from greenboost_cli.skill.router import discover_skills
    assert discover_skills(tmp_path) == []


def test_discover_skills_finds_skill(tmp_path):
    from greenboost_cli.skill.router import discover_skills, _parse_skill_md
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    md = skill_dir / "SKILL.md"
    md.write_text(
        "---\nname: my-skill\ndescription: Test skill for unit tests.\ntriggers:\n  - test\n---\n\nBody text."
    )
    entries = discover_skills(tmp_path)
    assert len(entries) == 1
    assert entries[0].name == "my-skill"
    assert entries[0].triggers == ["test"]


def test_load_skill_body(tmp_path):
    from greenboost_cli.skill.router import load_skill_body
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: x\ndescription: y\n---\n\nInstructions here.\nMore text.")
    body = load_skill_body(md)
    assert "Instructions here." in body
    assert "---" not in body  # frontmatter stripped


def test_trigger_match_substring():
    from greenboost_cli.skill.router import _trigger_match
    assert _trigger_match("how do I refactor this code", "refactor") is True
    assert _trigger_match("write tests please", "refactor") is False


def test_trigger_match_regex():
    from greenboost_cli.skill.router import _trigger_match
    assert _trigger_match("fix bug in auth.py", r"fix.*bug") is True
    assert _trigger_match("add feature", r"fix.*bug") is False


# ── security.py ───────────────────────────────────────────────────────────────

def test_validate_path_home_allowed(tmp_path):
    from greenboost_cli.security import validate_path
    result = validate_path(str(tmp_path))
    assert result == str(tmp_path)


def test_validate_path_tmp_allowed():
    from greenboost_cli.security import validate_path
    result = validate_path("/tmp/somefile.txt")
    assert result == "/tmp/somefile.txt"


def test_validate_path_outside_rejected():
    from greenboost_cli.security import validate_path
    import pytest
    with pytest.raises(ValueError, match="outside allowed"):
        validate_path("/etc/passwd")


def test_validate_path_url_passthrough():
    from greenboost_cli.security import validate_path
    url = "https://example.com/doc.pdf"
    assert validate_path(url, allow_url=True) == url


def test_validate_path_absolute_outside_rejected():
    from greenboost_cli.security import validate_path
    import pytest
    with pytest.raises(ValueError, match="outside allowed"):
        validate_path("/var/log/syslog")


def test_cap_truncates():
    from greenboost_cli.security import cap
    assert cap("hello world", 5) == "hello"
    assert cap("abc", 100) == "abc"


def test_safe_project_strips_separators():
    from greenboost_cli.security import safe_project
    assert "/" not in safe_project("my/project")
    assert "\\" not in safe_project("my\\project")


def test_safe_project_caps_length():
    from greenboost_cli.security import safe_project, _MAX_PROJECT_LEN
    long_name = "x" * (_MAX_PROJECT_LEN + 50)
    assert len(safe_project(long_name)) == _MAX_PROJECT_LEN


# ── prompt_queue.py ───────────────────────────────────────────────────────────

def test_prompt_queue_enqueue_dequeue():
    from greenboost_cli.terminal.prompt_queue import PromptQueue
    q = PromptQueue()
    assert len(q) == 0
    item = q.enqueue("hello")
    assert item.id == 1
    assert len(q) == 1
    out = q.dequeue()
    assert out is not None and out.text == "hello"
    assert len(q) == 0
    assert q.dequeue() is None


def test_prompt_queue_edit():
    from greenboost_cli.terminal.prompt_queue import PromptQueue
    q = PromptQueue()
    q.enqueue("first")
    q.enqueue("second")
    assert q.edit(1, "updated") is True
    assert q.snapshot()[0].text == "updated"
    assert q.edit(99, "nope") is False


def test_prompt_queue_delete():
    from greenboost_cli.terminal.prompt_queue import PromptQueue
    q = PromptQueue()
    q.enqueue("a")
    q.enqueue("b")
    assert q.delete(1) is True
    assert len(q) == 1
    assert q.snapshot()[0].text == "b"
    assert q.delete(99) is False


def test_prompt_queue_clear():
    from greenboost_cli.terminal.prompt_queue import PromptQueue
    q = PromptQueue()
    q.enqueue("x"); q.enqueue("y"); q.enqueue("z")
    removed = q.clear()
    assert removed == 3
    assert len(q) == 0


# ── token_tracker.py ─────────────────────────────────────────────────────────

def test_token_tracker_record_and_totals(tmp_path):
    from greenboost_cli.memory.token_tracker import record, get_totals
    record(tmp_path, api_tokens=100, local_tokens=50, session_id="s1")
    record(tmp_path, api_tokens=200, local_tokens=0, session_id="s2")
    t = get_totals(tmp_path)
    assert t["total_api"] == 300
    assert t["total_local"] == 50
    assert t["today_api"] == 300
    assert t["today_local"] == 50


def test_token_tracker_empty_dir(tmp_path):
    from greenboost_cli.memory.token_tracker import get_totals
    t = get_totals(tmp_path)
    assert t["total_api"] == 0
    assert t["today_local"] == 0


# ── injection.py — should_inject_tools, per-model not per-backend ────────────

def test_injection_native_fc_default_uses_native():
    from greenboost_cli.inference.injection import should_inject_tools
    assert should_inject_tools({"model": "gb-synapse/qwen3-coder"}) is False


def test_injection_native_fc_false_forces_injection():
    from greenboost_cli.inference.injection import should_inject_tools
    settings = {"model": "gb-synapse/some-older-gguf", "gb_synapse_native_fc": False}
    assert should_inject_tools(settings) is True


# ── BM25 design search ────────────────────────────────────────────────────────

def test_bm25_scores_matching_doc_higher():
    from greenboost_cli.design.intelligence import BM25
    corpus = ["dark minimalist design", "bright colorful playful style", "clean typography"]
    bm25 = BM25(corpus)
    hits = bm25.score("minimalist design", top_k=3)
    top_idx = hits[0][0]
    assert top_idx == 0  # first doc should score highest


def test_bm25_empty_corpus():
    from greenboost_cli.design.intelligence import BM25
    bm25 = BM25([])
    hits = bm25.score("anything", top_k=5)
    assert hits == []


def test_design_search_returns_empty_on_missing_dir(tmp_path):
    from greenboost_cli.design.intelligence import search
    results = search("button styles", settings={"design_skill_dir": str(tmp_path / "nonexistent")})
    assert results == {}


# ── cmd_clear resets bash CWD ─────────────────────────────────────────────────

def test_cmd_clear_resets_bash_cwd():
    import greenboost_cli.instruments.handlers as _h
    from greenboost_cli.terminal.commands import cmd_clear
    from greenboost_cli.core.session import ConversationSession

    _h._bash_cwd = "/some/directory"
    session = ConversationSession()
    session.messages = [{"role": "user", "content": "hi"}]
    cmd_clear("", session, {})
    assert _h._bash_cwd == ""
    assert session.messages == []


# ── injection.py — tool_format explicit override ─────────────────────────────

def test_injection_tool_format_inject_forces_injection():
    from greenboost_cli.inference.injection import should_inject_tools
    settings = {"model": "gb-synapse/qwen3", "tool_format": "inject"}
    assert should_inject_tools(settings) is True


def test_injection_tool_format_native_forces_native():
    from greenboost_cli.inference.injection import should_inject_tools
    settings = {"model": "gb-synapse/qwen3", "tool_format": "native", "gb_synapse_native_fc": False}
    assert should_inject_tools(settings) is False


# ── handle_shell CWD tracking ─────────────────────────────────────────────────

def test_handle_shell_cd_persists(tmp_path):
    import greenboost_cli.instruments.handlers as _h
    _h._bash_cwd = ""
    _h.handle_shell(f"cd {tmp_path}")
    assert _h._bash_cwd == str(tmp_path)
    result = _h.handle_shell("pwd")
    assert str(tmp_path) in result
    _h._bash_cwd = ""  # reset after test


def test_handle_shell_invalid_cwd_falls_back_to_cwd(tmp_path):
    import greenboost_cli.instruments.handlers as _h
    _h._bash_cwd = "/nonexistent/dir/xyz"
    result = _h.handle_shell("echo hello")
    assert "hello" in result
    _h._bash_cwd = ""  # reset


def test_handle_shell_uses_bash_not_dash(tmp_path):
    # Real incident, 2026-08-10: subprocess.run(..., shell=True) with no
    # executable= set defaults to /bin/sh, which on Ubuntu is dash — no
    # brace expansion. `mkdir -p X/{a,b,c}` silently "succeeded" against a
    # literal `X/{a,b,c}` path instead of erroring, producing garbage
    # directories the model never saw as a failure. Pin bash explicitly so
    # this can't regress silently.
    import greenboost_cli.instruments.handlers as _h
    _h.handle_shell(f"mkdir -p {tmp_path}/braces/{{a,b,c}}")
    for name in ("a", "b", "c"):
        assert (tmp_path / "braces" / name).is_dir()
    assert not (tmp_path / "braces" / "{a,b,c}").exists()


# ── security.validate_path traversal ─────────────────────────────────────────

def test_validate_path_traversal_rejected():
    from greenboost_cli.security import validate_path
    import pytest
    # /proc is outside home and /tmp — a resolved traversal attack target
    with pytest.raises(ValueError, match="outside allowed"):
        validate_path("/proc/1/cmdline")


# ── pdf2md code fence ordering ──────────────────────────────────────────────

def test_pdf2md_code_fence_closes_before_next_chunk():
    from greenboost_cli.pdf.pdf2md import _render_markdown
    chunks = [
        {"type": "code", "text": "x = 1"},
        {"type": "paragraph", "text": "After code."},
    ]
    out = _render_markdown(chunks)
    code_close = out.find("```\n")
    paragraph_start = out.find("After code.")
    # The closing fence must appear before the paragraph text
    assert code_close != -1, "closing ``` not found"
    assert paragraph_start != -1, "paragraph text not found"
    assert code_close < paragraph_start, (
        "closing ``` appears after paragraph text — code fence ordering bug"
    )


def test_pdf2md_consecutive_code_chunks_single_fence():
    from greenboost_cli.pdf.pdf2md import _render_markdown
    chunks = [
        {"type": "code", "text": "line 1"},
        {"type": "code", "text": "line 2"},
        {"type": "paragraph", "text": "done"},
    ]
    out = _render_markdown(chunks)
    # Should have exactly one opening and one closing fence
    assert out.count("```") == 2, f"expected exactly 2 ``` fences, got:\n{out}"


# ── LoopGuardTriggered — orchestrator loop guards ─────────────────────────────

_STUB_SETTINGS = {"model": "gb-synapse/qwen3-coder", "permission_mode": "accept-all"}


def test_loop_guard_repeat():
    """Repeat-limit guard fires when the same tool+args is called ≥3× in a row."""
    from greenboost_cli.core.orchestrator import execute_turn, LoopGuardTriggered
    from greenboost_cli.core.session import ConversationSession
    from greenboost_cli.inference.router import CompletedResponse

    session = ConversationSession()

    # Stub generate: always returns the same Bash call (will repeat ≥3 times)
    _tool_call = {"id": "c1", "name": "Bash", "input": {"command": "echo hi"}}
    _calls = [0]

    def _fake_generate(**kw):
        _calls[0] += 1
        yield CompletedResponse(
            text="", tool_calls=[_tool_call],
            in_tokens=0, out_tokens=0,
        )

    import greenboost_cli.core.orchestrator as _orch
    _orig = _orch.generate
    try:
        _orch.generate = _fake_generate
        events = list(execute_turn("do it", session, _STUB_SETTINGS, "system"))
    finally:
        _orch.generate = _orig

    guards = [e for e in events if isinstance(e, LoopGuardTriggered)]
    assert guards, "LoopGuardTriggered must be emitted"
    assert guards[0].reason == "repeat"


def test_loop_guard_max_turns():
    """Turn-cap guard fires after max_tool_turns iterations."""
    from greenboost_cli.core.orchestrator import execute_turn, LoopGuardTriggered
    from greenboost_cli.core.session import ConversationSession
    from greenboost_cli.inference.router import CompletedResponse

    session = ConversationSession()
    _counter = [0]

    def _fake_generate(**kw):
        _counter[0] += 1
        # Different args each time so the repeat guard doesn't trigger first
        yield CompletedResponse(
            text="", tool_calls=[{
                "id": f"c{_counter[0]}", "name": "Bash",
                "input": {"command": f"echo {_counter[0]}"},
            }],
            in_tokens=0, out_tokens=0,
        )

    import greenboost_cli.core.orchestrator as _orch
    _orig = _orch.generate
    try:
        _orch.generate = _fake_generate
        # max_tool_turns=3 so we hit the cap fast
        settings = {**_STUB_SETTINGS, "max_tool_turns": 3}
        events = list(execute_turn("do it", session, settings, "system"))
    finally:
        _orch.generate = _orig

    guards = [e for e in events if isinstance(e, LoopGuardTriggered)]
    assert guards, "LoopGuardTriggered must be emitted"
    assert guards[0].reason == "max_turns"


# ── strip_prefix namespace preservation ──────────────────────────────────────

def test_strip_prefix_preserves_namespace_after_gb_synapse_prefix():
    from greenboost_cli.inference.router import strip_prefix
    # Only the leading "gb-synapse/" is stripped; an Ollama-style namespace
    # after it must survive intact.
    assert (strip_prefix("gb-synapse/rafw007/qwen36-a3b-claude-coder:latest")
            == "rafw007/qwen36-a3b-claude-coder:latest")


def test_strip_prefix_no_slash():
    from greenboost_cli.inference.router import strip_prefix
    assert strip_prefix("llama3")             == "llama3"
    assert strip_prefix("qwen36-coder:studio") == "qwen36-coder:studio"
