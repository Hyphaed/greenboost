"""`gb factory-run` — the verb that actually processes the queue.

Before it existed the headless plane could only ENQUEUE: `factory-submit`
persists a task to factory.db, but nothing dequeues it unless a REPL session or
a hand-written Python driver calls `AIFactory.start()`. An autonomous run could
not be expressed as `gb` commands at all, which is the tool gap CLAUDE.md's
MCP-Tool-Gaps rule says to close rather than route around.

Every test here drives a STUB factory injected over `factory_cli._get_factory`.
That is deliberate: `get_factory()` returns a process-wide singleton backed by
the real ~/.greenboost_cli/factory.db, so a test that touched it would spawn
worker threads against whatever queue the developer's machine happens to be
running — during this module's own development that hung the suite against a
live build. Tests must not be able to reach a real queue.
"""
from __future__ import annotations

import pytest

from greenboost_cli.cli_headless import HEADLESS_SUBCOMMANDS
from greenboost_cli.workflow import factory_cli
from greenboost_cli.workflow.factory_cli import FACTORY_SUBCOMMANDS, cmd_factory_run


class StubFactory:
    """Minimal stand-in exposing only what cmd_factory_run touches."""

    def __init__(self, agents=None, queue_depth=0, busy=False):
        self._agents = dict(agents or {})
        self.queue_depth = queue_depth
        self.busy = busy
        self.started_with: dict = {}
        self.stopped = False

    def snapshot(self) -> dict:
        return {
            "agents": {
                n: {
                    "current_task": "working" if self.busy else "idle",
                    "total_tasks": 1,
                    "failed_tasks": 0,
                    **v,
                }
                for n, v in self._agents.items()
            },
            "queue_depth": self.queue_depth,
        }

    def add_agent(self, name, model="", gpu_id=0, skills=None) -> None:
        self._agents[name] = {"model": model}

    def start(self, workers=2, sleep=False) -> None:
        self.started_with = {"workers": workers, "sleep": sleep}

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def stub(monkeypatch):
    f = StubFactory()
    monkeypatch.setattr(factory_cli, "_get_factory", lambda: f)
    return f


def test_registered_in_both_dispatch_tables() -> None:
    # A handler in one table but missing from the other is unreachable:
    # cli_headless gates on HEADLESS_SUBCOMMANDS before it looks the handler
    # up, so drift between the two silently disables the verb.
    assert "factory-run" in FACTORY_SUBCOMMANDS
    assert "factory-run" in HEADLESS_SUBCOMMANDS
    assert set(FACTORY_SUBCOMMANDS) <= set(HEADLESS_SUBCOMMANDS)


def test_help_exits_clean() -> None:
    with pytest.raises(SystemExit) as e:
        cmd_factory_run(["--help"])
    assert e.value.code == 0


def test_refuses_to_start_without_an_agent(stub, capsys) -> None:
    """Nothing could run, so fail loudly rather than spin worker threads that
    will never pick anything up."""
    assert cmd_factory_run(["--json"]) == 2
    assert stub.started_with == {}, "must not start workers when there is no agent"
    assert "no agents configured" in capsys.readouterr().err


def test_drains_an_empty_queue_and_exits(stub, capsys) -> None:
    """An empty queue with idle agents must return promptly — that is what
    makes the verb scriptable rather than something you background and hope."""
    rc = cmd_factory_run(
        ["--agent", "t-agent", "--model", "test-model", "--poll", "0.5",
         "--timeout", "20", "--json"]
    )
    assert rc == 0
    assert stub.stopped is True
    assert "queue_depth" in capsys.readouterr().out


def test_creates_the_named_agent_with_its_model(stub) -> None:
    cmd_factory_run(["--agent", "builder", "--model", "some-model",
                     "--poll", "0.5", "--timeout", "20", "--json"])
    assert stub._agents["builder"]["model"] == "some-model"


def test_times_out_rather_than_hanging_on_a_busy_agent(monkeypatch, capsys) -> None:
    """A wedged agent must surface as a non-zero exit, not an indefinite block."""
    f = StubFactory(agents={"a": {}}, queue_depth=3, busy=True)
    monkeypatch.setattr(factory_cli, "_get_factory", lambda: f)
    rc = cmd_factory_run(["--poll", "0.5", "--timeout", "2", "--json"])
    assert rc == 1
    assert f.stopped is True


def test_worker_count_is_clamped(stub) -> None:
    """One local model serves every worker, so an unbounded --workers just
    multiplies contention on the same GPU."""
    cmd_factory_run(["--agent", "a", "--model", "m", "--workers", "999",
                     "--poll", "0.5", "--timeout", "20", "--json"])
    assert stub.started_with["workers"] == 8

    stub.started_with = {}
    cmd_factory_run(["--agent", "a", "--model", "m", "--workers", "0",
                     "--poll", "0.5", "--timeout", "20", "--json"])
    assert stub.started_with["workers"] == 1


def test_watch_mode_keeps_running_until_timeout(stub) -> None:
    """--watch must NOT exit on an empty queue; it waits for more work."""
    rc = cmd_factory_run(["--agent", "a", "--model", "m", "--watch",
                          "--poll", "0.5", "--timeout", "2", "--json"])
    assert rc == 1, "watch mode should run to the timeout, not drain-exit"
