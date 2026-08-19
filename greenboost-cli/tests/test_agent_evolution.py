"""The AE-* agent-evolution work: context, tools, memory, loop.

Anchored to the 14-day dataflux baseline that motivated it: a turn reusing
>=99% of its prompt prefix reached first token in ~5.5s, one in the 50-90%
band took ~140s, and 44.5 minutes across 32 turns went to time-to-first-token
alone. Everything here is in service of not paying that twice.
"""
import pytest

from greenboost_cli.core.session import ConversationSession


# ── AE-1: the benchmark scores what it says it scores ───────────────────────

def _task(**kw):
    from greenboost_cli.bench.agent_eval import EvalTask
    kw.setdefault("id", "t"); kw.setdefault("prompt", "p")
    return EvalTask(**kw)


def test_hallucinated_tool_is_detected_from_the_result_not_the_name():
    from greenboost_cli.bench.agent_eval import score_task, HALLUCINATION_MARKER
    s = score_task(_task(), {"tool_calls": [
        {"name": "Nope", "result": f"{HALLUCINATION_MARKER} Nope"},
        {"name": "Read", "result": "ok"}]})
    assert s.hallucinated == 1
    assert s.grounding == 0.5


def test_a_dimension_a_task_never_exercised_is_not_scored():
    """Averaging a free 1.0 into every task inflates every later comparison."""
    from greenboost_cli.bench.agent_eval import score_task
    s = score_task(_task(max_tool_calls=0), {"tool_calls": []})
    assert s.completion is None and s.tool_selection is None and s.efficiency is None


def test_bare_mcp_name_matches_its_prefixed_form():
    from greenboost_cli.bench.agent_eval import score_task
    s = score_task(_task(expect_tools=("dataflux_events",)),
                   {"tool_calls": [{"name": "mcp__gb__dataflux_events", "result": "ok"}]})
    assert s.tool_selection == 1.0


def test_a_forbidden_tool_zeroes_selection_even_when_the_right_one_was_used():
    from greenboost_cli.bench.agent_eval import score_task
    s = score_task(_task(expect_tools=("Read",), forbid_tools=("Bash",)),
                   {"tool_calls": [{"name": "Read", "result": "ok"},
                                   {"name": "Bash", "result": "ok"}]})
    assert s.tool_selection == 0.0


def test_an_errored_run_scores_zero_rather_than_being_dropped():
    """A change that makes runs crash must not look like one that made them fast."""
    from greenboost_cli.bench.agent_eval import score_task
    s = score_task(_task(expect_substrings=("x",)),
                   {"summary": "x", "tool_calls": [], "error": "boom"})
    assert s.overall == 0.0


def test_baseline_buckets_cache_hit_against_ttft():
    from greenboost_cli.bench.agent_eval.baseline import cache_baseline
    evs = [{"kind": "prompt_cache", "hit_pct": 99.8, "ttft_ms": 5000},
           {"kind": "prompt_cache", "hit_pct": 0.0, "ttft_ms": 165000}]
    b = cache_baseline(evs)
    assert b.n_turns == 2 and b.zero_hit_turns == 1
    assert b.by_bucket[">=99%"]["median_ttft_s"] == 5.0
    assert b.by_bucket["<50%"]["median_ttft_s"] == 165.0


# ── AE-2: compaction must not move the prefix ───────────────────────────────

def _long_session():
    s = ConversationSession()
    s.messages = [{"role": "user", "content": "ORIGINAL TASK: ship the health check"},
                  {"role": "assistant", "content": "Understood."}]
    for i in range(20):
        s.messages.append({"role": "user", "content": f"step {i}"})
        s.messages.append({"role": "assistant", "content": f"Done: step {i}"})
    return s


def test_the_leading_messages_survive_compaction_byte_identical():
    from greenboost_cli.workflow.intelligence import _compress_context
    s = _long_session()
    head_before = s.messages[0]["content"]
    _compress_context(s, {}, force=True)
    assert s.messages[0]["content"] == head_before


def test_a_second_compaction_extends_the_memory_instead_of_rewriting_it():
    from greenboost_cli.workflow.intelligence import _compress_context, _MEMORY_MARKER
    s = _long_session()
    _compress_context(s, {}, force=True)
    mem1 = next(m["content"] for m in s.messages
                if str(m.get("content", "")).startswith(_MEMORY_MARKER))
    for i in range(20, 40):
        s.messages.append({"role": "user", "content": f"step {i}"})
        s.messages.append({"role": "assistant", "content": f"Done: step {i}"})
    head_before = s.messages[0]["content"]
    _compress_context(s, {}, force=True)
    mem2 = next(m["content"] for m in s.messages
                if str(m.get("content", "")).startswith(_MEMORY_MARKER))
    assert mem2.startswith(mem1.rstrip())      # old bytes preserved
    assert s.messages[0]["content"] == head_before


def test_compaction_still_shrinks_the_session():
    from greenboost_cli.workflow.intelligence import _compress_context
    s = _long_session()
    before = len(s.messages)
    _compress_context(s, {}, force=True)
    assert len(s.messages) < before


def test_a_short_session_is_not_grown_by_compaction():
    from greenboost_cli.workflow.intelligence import _compress_context
    s = ConversationSession()
    s.messages = [{"role": "user", "content": "hi"}]
    _compress_context(s, {}, force=True)
    assert len(s.messages) == 1


# ── AE-3: the plan is pinned, and pinned means LAST ─────────────────────────

@pytest.fixture
def todos():
    from greenboost_cli.instruments import handlers
    handlers._session_todos[:] = [
        {"content": "write the check", "status": "in_progress"},
        {"content": "add a test", "status": "pending"}]
    yield
    handlers._session_todos.clear()


def test_no_todos_means_no_pinned_block(todos):
    from greenboost_cli.instruments import handlers
    from greenboost_cli.core.orchestrator import _with_pinned_todos
    handlers._session_todos.clear()
    msgs = [{"role": "user", "content": "hi"}]
    assert _with_pinned_todos(msgs) == msgs


def test_the_pinned_plan_goes_last_so_it_costs_no_cache(todos):
    from greenboost_cli.core.orchestrator import _with_pinned_todos
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    out = _with_pinned_todos(msgs)
    assert out[:2] == msgs                      # nothing before it moved
    assert out[-1]["content"].startswith("<pinned-plan>")
    assert "2 item(s) still open" in out[-1]["content"]


def test_the_pinned_plan_is_never_written_into_history(todos):
    from greenboost_cli.core.orchestrator import _with_pinned_todos
    msgs = [{"role": "user", "content": "a"}]
    _with_pinned_todos(msgs)
    assert len(msgs) == 1


# ── AE-4: one server's verbosity cannot spend the window ────────────────────

def test_a_huge_mcp_result_is_truncated_with_an_actionable_notice():
    from greenboost_cli.mcp.client import _extract_content
    out = _extract_content([{"type": "text", "text": "x" * 500_000}])
    assert len(out) < 30_000
    assert "truncated" in out and "Narrow the call" in out


def test_a_small_mcp_result_is_untouched():
    from greenboost_cli.mcp.client import _extract_content
    assert _extract_content([{"type": "text", "text": "hello"}]) == "hello"


# ── AE-9/10/11: memory recalls itself ──────────────────────────────────────

@pytest.fixture
def memdir(tmp_path):
    return tmp_path


def test_rules_are_recalled_no_matter_what_the_query_is(memdir):
    from greenboost_cli.memory import store
    store.record_rule("Never push; the developer pushes.", project_dir=memdir)
    store.write_memory("kv cache", "how KV size is measured", "b", project_dir=memdir)
    for q in ("", "anything at all", "kv cache"):
        assert any(m.type == "rule" for m in store.recall(q, project_dir=memdir))


def test_a_scoped_memory_is_silent_until_its_subsystem_is_touched(memdir):
    from greenboost_cli.memory import store
    store.write_memory("shim collision", "INIT-phase OOM in the shim", "b",
                       scope="greenboost_cuda_shim.c", project_dir=memdir)
    assert store.recall("unrelated question", project_dir=memdir) == []
    hit = store.recall("unrelated question",
                       touched_paths=["/x/greenboost_cuda_shim.c"], project_dir=memdir)
    assert [m.name for m in hit] == ["shim collision"]


def test_stopwords_do_not_match_everything(memdir):
    """"the" in a query must not pull in every memory containing "the"."""
    from greenboost_cli.memory import store
    store.write_memory("shim collision", "OOM collision in the CUDA shim", "b",
                       project_dir=memdir)
    assert store.recall("why is the kv cache so big", project_dir=memdir) == []


def test_recall_respects_its_context_budget(memdir):
    from greenboost_cli.memory import store
    for i in range(40):
        store.write_memory(f"cache note {i}", f"about the cache number {i}",
                           "x" * 200, project_dir=memdir)
    got = store.recall("cache", budget_chars=1_000, project_dir=memdir)
    assert 0 < len(got) < 40


# ── AE-12: delegation exists, and cannot recurse ───────────────────────────

def test_delegate_is_a_real_instrument():
    from greenboost_cli.instruments.schemas import INSTRUMENT_DEFINITIONS
    from greenboost_cli.instruments.dispatcher import _DISPATCH
    assert any(d["name"] == "Delegate" for d in INSTRUMENT_DEFINITIONS)
    assert "Delegate" in _DISPATCH


def test_delegation_refuses_past_the_depth_limit(monkeypatch):
    from greenboost_cli.agents.subagent import delegate, DEPTH_ENV, MAX_DEPTH
    monkeypatch.setenv(DEPTH_ENV, str(MAX_DEPTH))
    out = delegate("do a subtask")
    assert out.startswith("Error: delegation refused")


# ── AE-13: a Stop hook can refuse the stop ─────────────────────────────────

def test_no_stop_hooks_configured_means_the_run_may_stop(monkeypatch):
    from greenboost_cli.instruments import hooks
    monkeypatch.setattr(hooks, "_hooks_cache", {})
    assert hooks.run_stop_hooks("done") == (True, "")


def test_a_refusing_stop_hook_returns_its_reason(monkeypatch):
    from greenboost_cli.instruments import hooks
    monkeypatch.setattr(hooks, "_hooks_cache",
                        {"Stop": [{"command": "echo '{\"continue\": false, "
                                             "\"reason\": \"tests are red\"}'"}]})
    may_stop, reason = hooks.run_stop_hooks("done")
    assert may_stop is False and "tests are red" in reason


# ── AE-14: the trace explains the failure ──────────────────────────────────

def test_a_repeated_call_is_named_as_going_in_circles():
    from greenboost_cli.core.autonomy import AutonomyState
    from greenboost_cli.core.trajectory import diagnose
    s = AutonomyState()
    for _ in range(4):
        s.record("tool", name="Bash")
    kinds = [f.kind for f in diagnose(s)]
    assert "repeated_call" in kinds


def test_an_invented_tool_in_telemetry_is_surfaced():
    from greenboost_cli.core.autonomy import AutonomyState
    from greenboost_cli.core.trajectory import diagnose
    s = AutonomyState()
    s.record("tool", name="Read")
    out = diagnose(s, events=[{"kind": "agent_tool_schema_miss",
                               "outcome": "unknown_instrument",
                               "requested": "mcp__nope__x"}])
    assert any(f.kind == "invented_tool" for f in out)


def test_a_clean_run_says_so_instead_of_inventing_a_cause():
    from greenboost_cli.core.autonomy import AutonomyState
    from greenboost_cli.core.trajectory import diagnose
    s = AutonomyState()
    s.record("tool", name="Read"); s.record("tool", name="Grep")
    assert [f.kind for f in diagnose(s)] == ["no_obvious_failure"]


# ── Microcompaction: the cheap edit that postpones the expensive one ────────

def _tool_heavy_session(n=10, size=5000):
    s = ConversationSession()
    for i in range(n):
        s.messages.append({"role": "assistant", "content": f"reading {i}"})
        s.messages.append({"role": "tool", "name": "Read", "content": "X" * size})
    return s


def test_microcompact_preserves_message_count_roles_and_order():
    """This is the whole reason it is cheaper than compaction."""
    from greenboost_cli.workflow.intelligence import _microcompact
    s = _tool_heavy_session()
    before = [(m["role"], m.get("name")) for m in s.messages]
    assert _microcompact(s) > 0
    assert [(m["role"], m.get("name")) for m in s.messages] == before


def test_microcompact_keeps_the_most_recent_results_whole():
    from greenboost_cli.workflow.intelligence import _microcompact, _CLEARED, _MICROCOMPACT_KEEP
    s = _tool_heavy_session()
    _microcompact(s)
    tools = [m for m in s.messages if m["role"] == "tool"]
    assert all(not m["content"].startswith(_CLEARED) for m in tools[-_MICROCOMPACT_KEEP:])
    assert all(m["content"].startswith(_CLEARED) for m in tools[:-_MICROCOMPACT_KEEP])


def test_microcompact_leaves_small_results_alone():
    from greenboost_cli.workflow.intelligence import _microcompact
    s = _tool_heavy_session(n=10, size=50)
    assert _microcompact(s) == 0


def test_microcompact_never_touches_a_tool_it_cannot_re_run():
    from greenboost_cli.workflow.intelligence import _microcompact, _CLEARED
    s = ConversationSession()
    for _ in range(10):
        s.messages.append({"role": "tool", "name": "AskUserQuestion", "content": "Y" * 5000})
    _microcompact(s)
    assert all(not m["content"].startswith(_CLEARED) for m in s.messages)


def test_microcompact_is_idempotent():
    from greenboost_cli.workflow.intelligence import _microcompact
    s = _tool_heavy_session()
    _microcompact(s)
    assert _microcompact(s) == 0


def test_cold_cache_clears_aggressively_and_a_warm_one_is_left_alone():
    import time
    from greenboost_cli.workflow.intelligence import microcompact_if_cold
    warm = _tool_heavy_session()
    assert microcompact_if_cold(warm, time.time()) == 0
    cold = _tool_heavy_session()
    assert microcompact_if_cold(cold, time.time() - 1200) > 0


# ── Memory hygiene: age, drift, and what is not worth keeping ───────────────

def test_an_old_memory_shows_its_age(memdir):
    import os, time
    from greenboost_cli.memory import store
    p = store.write_memory("kv measured", "KV size is measured", "x", project_dir=memdir)
    os.utime(p, (time.time() - 30 * 86400,) * 2)
    assert "recorded 30d ago" in store.scan(project_dir=memdir)[0].render()


def test_a_fresh_memory_does_not_shout_its_age(memdir):
    from greenboost_cli.memory import store
    store.write_memory("kv measured", "KV size is measured", "x", project_dir=memdir)
    assert "recorded" not in store.scan(project_dir=memdir)[0].render()


def test_recalled_memory_carries_the_drift_caveat(memdir):
    """A memory naming a flag that was since renamed must not be acted on."""
    from greenboost_cli.memory import store
    store.write_memory("a", "b", "c", project_dir=memdir)
    block = store.render_block(store.scan(project_dir=memdir))
    assert "what you can see wins" in block


def test_derivable_facts_are_flagged_and_real_ones_are_not():
    from greenboost_cli.memory.store import derivable_reason
    assert derivable_reason("the parser is in gb_quant.py")
    assert derivable_reason("it was fixed by adding a null check")
    assert not derivable_reason(
        "the shim needs GB_VRAM_FRONTLOAD=1 or T2 fills before T1")


# ── Read-before-edit: the quiet overwrite ──────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_read_state():
    from greenboost_cli.instruments import handlers
    handlers._read_state.clear()
    yield
    handlers._read_state.clear()


def test_an_edit_after_a_clean_read_still_works(tmp_path):
    from greenboost_cli.instruments import handlers as H
    f = tmp_path / "x.py"
    f.write_text("alpha\nbeta\n")
    H.handle_read(str(f))
    assert "Replaced" in H.handle_edit(str(f), "alpha", "ALPHA")
    assert "ALPHA" in f.read_text()


def test_an_edit_is_refused_when_the_file_changed_since_the_read(tmp_path):
    """The dangerous case: old_string still matches, but the file moved on."""
    from greenboost_cli.instruments import handlers as H
    f = tmp_path / "x.py"
    f.write_text("alpha\nbeta\n")
    H.handle_read(str(f))
    f.write_text("alpha\nbeta\ngamma\n")          # someone else wrote
    out = H.handle_edit(str(f), "alpha", "ALPHA")
    assert out.startswith("Error:") and "changed since you read it" in out
    assert "ALPHA" not in f.read_text()           # nothing was applied


def test_a_file_this_session_never_read_is_not_blocked(tmp_path):
    """Factory workers and subagents write files they never read , that path
    must keep working; the guard is about stale reads, not about policy."""
    from greenboost_cli.instruments import handlers as H
    f = tmp_path / "x.py"
    f.write_text("alpha\n")
    assert "Replaced" in H.handle_edit(str(f), "alpha", "ALPHA")


def test_re_reading_clears_the_staleness(tmp_path):
    from greenboost_cli.instruments import handlers as H
    f = tmp_path / "x.py"
    f.write_text("alpha\n")
    H.handle_read(str(f))
    f.write_text("alpha\nbeta\n")
    assert H.handle_edit(str(f), "alpha", "A").startswith("Error:")
    H.handle_read(str(f))                          # look again
    assert "Replaced" in H.handle_edit(str(f), "alpha", "A")


# ── Background commands: a 90s build must not cost a whole turn ─────────────

def test_a_backgrounded_command_returns_immediately_with_an_id():
    import time
    from greenboost_cli.instruments.dispatcher import dispatch
    t0 = time.monotonic()
    out = dispatch("Bash", {"command": "sleep 2", "run_in_background": True})
    assert time.monotonic() - t0 < 1.0          # did not wait for the sleep
    assert "started in the background as" in out
    jid = out.split("as ")[1].split("]")[0]
    dispatch("TaskStop", {"task_id": jid})


def test_output_is_incremental_then_reports_the_exit_code():
    import time
    from greenboost_cli.instruments import background as B
    jid = B.start("echo one; sleep 0.5; echo two")
    time.sleep(0.2)
    first = B.output(jid)
    assert "one" in first and "still running" in first
    assert "two" not in first                   # not printed yet
    time.sleep(0.8)
    second = B.output(jid)
    assert "two" in second and "exit code 0" in second
    assert "one" not in second                  # already delivered, not repeated


def test_stopping_a_job_actually_kills_it():
    import time
    from greenboost_cli.instruments import background as B
    jid = B.start("sleep 30")
    time.sleep(0.1)
    assert "stopped" in B.stop(jid)
    time.sleep(0.3)
    assert B._JOBS[jid].running is False


def test_an_unknown_job_id_says_which_ones_exist():
    from greenboost_cli.instruments import background as B
    out = B.output("definitely-not-a-job")
    assert out.startswith("Error:") and "known:" in out


def test_a_failing_background_job_reports_its_exit_code():
    import time
    from greenboost_cli.instruments import background as B
    jid = B.start("exit 42")
    time.sleep(0.4)
    assert "exit code 42" in B.output(jid)


# ── Status rows say what the step did ──────────────────────────────────────

def test_a_step_with_no_tools_keeps_the_generic_label():
    from greenboost_cli.terminal.repl import step_phase_label
    assert step_phase_label([]) == "Processing"
    assert step_phase_label(None) == "Processing"


def test_each_tool_maps_to_a_verb_about_the_project():
    from greenboost_cli.terminal.repl import step_phase_label
    assert step_phase_label(["Write"]) == "Writing"
    assert step_phase_label(["Read"]) == "Reading"
    assert step_phase_label(["Bash"]) == "Running"


def test_repeats_collapse_and_the_label_stays_short():
    """Four writes are one 'Writing', and a step touching everything must not
    produce a label longer than the information it carries."""
    from greenboost_cli.terminal.repl import step_phase_label
    assert step_phase_label(["Write"] * 4) == "Writing"
    assert step_phase_label(["Read", "Edit"]) == "Reading · Editing"
    assert step_phase_label(["Read", "Edit", "Bash", "TodoWrite"]).count("·") == 1


def test_an_mcp_call_is_named_by_its_server():
    from greenboost_cli.terminal.repl import step_phase_label
    assert step_phase_label(["mcp__greenboost-dataflux__dataflux_summary"]) \
        == "Asking greenboost-dataflux"


def test_an_unrecognised_tool_does_not_produce_an_empty_label():
    from greenboost_cli.terminal.repl import step_phase_label
    assert step_phase_label(["SomethingNew"]) == "Processing"


# ── File history: undo an EDIT, not just a conversation turn ───────────────

@pytest.fixture
def hist(monkeypatch, tmp_path):
    monkeypatch.setenv("GB_SESSION", f"test-{tmp_path.name}")
    monkeypatch.setattr("greenboost_cli.environment.settings.GB_HOME", tmp_path)
    from greenboost_cli.core import file_history
    monkeypatch.setattr(file_history, "_root",
                        lambda: (tmp_path / "fh").resolve().mkdir(parents=True, exist_ok=True)
                        or (tmp_path / "fh").resolve())
    return file_history


def test_the_pre_agent_version_is_what_gets_restored(hist, tmp_path):
    """The snapshot is taken BEFORE the write , what the agent produced is
    always recoverable by reading the file; what was there first is not."""
    f = tmp_path / "a.txt"
    f.write_text("ORIGINAL")
    hist.snapshot(f); f.write_text("EDIT 1")
    hist.snapshot(f); f.write_text("EDIT 2")
    assert "Restored" in hist.revert(f)
    assert f.read_text() == "ORIGINAL"


def test_reverting_a_file_the_agent_created_deletes_it(hist, tmp_path):
    f = tmp_path / "new.txt"
    hist.snapshot(f)                    # did not exist yet
    f.write_text("agent made this")
    assert "Deleted" in hist.revert(f)
    assert not f.exists()


def test_a_later_version_can_be_targeted(hist, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("v1")
    hist.snapshot(f); f.write_text("v2")
    hist.snapshot(f); f.write_text("v3")
    hist.revert(f, 2)
    assert f.read_text() == "v2"


def test_changes_lists_what_was_touched(hist, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    hist.snapshot(f); hist.snapshot(f)
    rows = hist.changed_files()
    assert len(rows) == 1 and rows[0]["edits"] == 2 and rows[0]["revertable"]


def test_reverting_an_untouched_file_says_so_instead_of_guessing(hist, tmp_path):
    out = hist.revert(tmp_path / "never-seen.txt")
    assert out.startswith("Error:") and "no recorded changes" in out


def test_a_huge_file_is_recorded_but_honestly_marked_unrevertable(hist, tmp_path, monkeypatch):
    monkeypatch.setattr(hist, "MAX_SNAPSHOT_BYTES", 10)
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * 500)
    hist.snapshot(f)
    assert hist.changed_files()[0]["revertable"] is False
    assert "too large" in hist.revert(f)


def test_a_snapshot_failure_never_breaks_the_edit(hist, tmp_path, monkeypatch):
    monkeypatch.setattr(hist, "_load_index", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert hist.snapshot(tmp_path / "a.txt") is None      # degrades, not raises


def test_write_and_edit_actually_take_snapshots(hist, tmp_path):
    from greenboost_cli.instruments import handlers as H
    f = tmp_path / "wired.txt"
    H.handle_write(str(f), "first")
    H.handle_read(str(f))
    H.handle_edit(str(f), "first", "second")
    paths = [r["path"] for r in hist.changed_files()]
    assert str(f.resolve()) in paths
    assert hist.changed_files()[0]["edits"] == 2          # the write and the edit


# ── Unattended-For-Days rule: nothing may grow forever, nothing may block ──

def test_the_journal_is_bounded_and_says_how_much_it_dropped():
    """One entry per tool call is unbounded over a multi-day run."""
    from greenboost_cli.core.autonomy import AutonomyState, MAX_JOURNAL_ENTRIES
    s = AutonomyState()
    for _ in range(MAX_JOURNAL_ENTRIES + 500):
        s.record("tool", name="Bash")
    assert len(s.journal) <= MAX_JOURNAL_ENTRIES
    assert s.journal_dropped > 0          # disclosed, not silently forgotten


def test_the_read_state_map_is_bounded(tmp_path):
    from greenboost_cli.instruments import handlers as H
    H._read_state.clear()
    f = tmp_path / "f.txt"
    f.write_text("x")
    for i in range(H.MAX_READ_STATE + 200):
        H._read_state[f"/fake/path/{i}"] = (0.0, 0)
    H._note_read(f)
    assert len(H._read_state) <= H.MAX_READ_STATE
    H._read_state.clear()


def test_finished_background_jobs_are_reaped_but_running_ones_are_not():
    import time
    from greenboost_cli.instruments import background as B
    B._JOBS.clear()
    keep = B.start("sleep 30")                     # still running
    for _ in range(5):
        B.start("true")
    time.sleep(0.4)
    B.MAX_FINISHED_JOBS, orig = 2, B.MAX_FINISHED_JOBS
    try:
        B.start("true"); time.sleep(0.2); B._reap()
        assert keep in B._JOBS                      # a running job is never evicted
        assert len([j for j in B._JOBS.values() if not j.running]) <= 3
    finally:
        B.MAX_FINISHED_JOBS = orig
        B.stop(keep); B._JOBS.clear()


def test_the_file_history_store_is_capped(tmp_path, monkeypatch):
    from greenboost_cli.core import file_history as FH
    root = tmp_path / "fh"; root.mkdir()
    monkeypatch.setattr(FH, "_root", lambda: root)
    monkeypatch.setattr(FH, "MAX_STORE_BYTES", 2000)
    f = tmp_path / "a.bin"
    for _ in range(12):
        f.write_bytes(b"x" * 500)
        FH.snapshot(f)
    assert sum(p.stat().st_size for p in root.glob("*@v*")) <= 2000
    assert FH.changed_files()[0]["edits"] == 12      # every edit still recorded


def test_a_question_answers_itself_when_nobody_is_there():
    """Asking is fine and expected. A question outliving the person it was
    asked of is not , 5 minutes, then the safe option, on the record."""
    from greenboost_cli.terminal.wizard_prompt import (
        UNATTENDED_ANSWER_AFTER_S, run_question_wizard, TIMEOUT)
    import inspect
    assert UNATTENDED_ANSWER_AFTER_S == 300.0
    assert inspect.signature(run_question_wizard).parameters["timeout_s"].default \
        == UNATTENDED_ANSWER_AFTER_S
    assert TIMEOUT == "TIMEOUT"


def test_getch_reports_a_timeout_instead_of_blocking(monkeypatch):
    """pytest replaces stdin with an object that has no fileno, so the terminal
    primitives are stubbed , the assertion is about the timeout branch, which
    must return before any blocking read is attempted."""
    import types
    from greenboost_cli.terminal import wizard_prompt as W
    monkeypatch.setattr(W.sys, "stdin", types.SimpleNamespace(fileno=lambda: 0))
    monkeypatch.setattr(W.select, "select", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(W.termios, "tcgetattr", lambda fd: None)
    monkeypatch.setattr(W.termios, "tcsetattr", lambda *a: None)
    monkeypatch.setattr(W.tty, "setraw", lambda fd: None)
    monkeypatch.setattr(W.os, "read", lambda *a: (_ for _ in ()).throw(
        AssertionError("must not block-read after a timeout")))
    assert W._getch(timeout=0.01) == W.TIMEOUT


# ── GB-1: name the chunk that cost the prefix ──────────────────────────────

@pytest.fixture
def pfx():
    from greenboost_cli.core import prefix
    prefix.reset()
    yield prefix
    prefix.reset()


_BASE = dict(system="SYS", tools="TOOLS", history="H1", memory="", plan="P", user="u1")


def test_the_first_turn_is_not_a_shift(pfx):
    """No previous request to compare against is not the same as 'nothing
    changed', and must not be reported as either."""
    v = pfx.observe(_BASE, emit=False)
    assert v["first_turn"] is True and v["first_changed"] is None


def test_a_new_user_turn_costs_nothing_and_stays_silent(pfx):
    pfx.observe(_BASE, emit=False)
    v = pfx.observe({**_BASE, "user": "u2"}, emit=False)
    assert v["first_changed"] == "user"
    assert v["invalidated_chars"] == 0
    assert pfx.explain(v) == ""          # the healthy case says nothing


def test_a_moved_system_block_invalidates_everything(pfx):
    pfx.observe(_BASE, emit=False)
    v = pfx.observe({**_BASE, "system": "SYS2", "user": "u2"}, emit=False)
    assert v["first_changed"] == "system"
    assert v["stable_prefix_chars"] == 0
    assert "prefix shifted at 'system'" in pfx.explain(v)


def test_the_earliest_change_is_the_one_reported(pfx):
    """Two chunks moved; the diagnosis is the earlier one, because it is the
    one that threw away the rest."""
    pfx.observe(_BASE, emit=False)
    v = pfx.observe({**_BASE, "history": "H2", "plan": "P2", "user": "u2"}, emit=False)
    assert v["first_changed"] == "history"
    assert set(v["changed"]) >= {"history", "plan", "user"}


def test_a_late_change_keeps_the_earlier_chunks_reusable(pfx):
    pfx.observe(_BASE, emit=False)
    v = pfx.observe({**_BASE, "plan": "P2", "user": "u2"}, emit=False)
    assert v["first_changed"] == "plan"
    assert v["stable_prefix_chars"] == len("SYS") + len("TOOLS") + len("H1")


def test_an_unchanged_prompt_reports_no_shift(pfx):
    pfx.observe(_BASE, emit=False)
    v = pfx.observe(dict(_BASE), emit=False)
    assert v["first_changed"] is None and pfx.explain(v) == ""
