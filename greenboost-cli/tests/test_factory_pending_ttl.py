"""An interrupted factory must not ambush the next project's session.

`FactoryDB.delete_task()` only runs AFTER `_run_task()` returns, so a factory
whose process exits mid-task leaves its row in `pending_tasks`. `load_pending()`
then re-queued it unconditionally in whatever project started the next factory,
carrying its ORIGINAL `metadata["cwd"]`.

Observed live 2026-08-17: a finished "edit src/game.ts" task from one project was
resurrected 37 minutes later at the start of an unrelated project's very first
run, sorted ahead of that session's own task by `created_at ASC`, and took the
single local model with it. Nothing was corrupted (its stale cwd pointed back at
its own project) but the new run was blocked behind work nobody had asked for,
and the only clue was a foreign task_id in the dataflux `cli_tool_call` trail.
"""
from __future__ import annotations

import time

import pytest

from greenboost_cli.workflow.factory import FactoryDB, PENDING_TASK_TTL_S, Task


@pytest.fixture
def db(tmp_path):
    return FactoryDB(tmp_path / "factory.db")


def _queue(db: FactoryDB, task_id: str, age_s: float, cwd: str = "/tmp/p") -> None:
    db.save_task(Task(
        priority=1,
        created_at=time.time() - age_s,
        task_id=task_id,
        prompt=f"task {task_id}",
        metadata={"cwd": cwd},
    ))


def test_fresh_tasks_are_restored(db) -> None:
    _queue(db, "fresh0000001", age_s=30)
    assert [t.task_id for t in db.load_pending(ttl_s=3600)] == ["fresh0000001"]


def test_stale_tasks_are_dropped(db) -> None:
    _queue(db, "stale0000001", age_s=40 * 60)
    assert db.load_pending(ttl_s=600) == []


def test_a_dropped_task_is_reported_not_silent(db) -> None:
    """A task vanishing without explanation is its own debugging problem."""
    _queue(db, "stale0000001", age_s=40 * 60)
    db.load_pending(ttl_s=600)
    expired = getattr(db, "expired_on_load", [])
    assert len(expired) == 1
    assert expired[0]["task_id"] == "stale0000001"
    assert expired[0]["age_s"] >= 2400


def test_a_dropped_task_is_really_gone_from_the_db(db) -> None:
    """It must not resurface on the next start either."""
    _queue(db, "stale0000001", age_s=40 * 60)
    db.load_pending(ttl_s=600)
    assert db.load_pending(ttl_s=0) == []


def test_stale_and_fresh_are_separated(db) -> None:
    """The exact production shape: one stale row from another project sitting
    in front of this session's own task."""
    _queue(db, "stale0000001", age_s=40 * 60, cwd="/home/u/other-project")
    _queue(db, "mine00000001", age_s=10, cwd="/home/u/this-project")
    restored = db.load_pending(ttl_s=600)
    assert [t.task_id for t in restored] == ["mine00000001"]
    assert restored[0].cwd == "/home/u/this-project"


def test_ttl_zero_keeps_everything(db) -> None:
    """Opt-out must still work for callers that want full resume semantics."""
    _queue(db, "ancient00001", age_s=10 * 24 * 3600)
    assert len(db.load_pending(ttl_s=0)) == 1


def test_default_ttl_is_a_sane_window() -> None:
    # Long enough to resume a genuinely interrupted session, short enough that
    # yesterday's abandoned task never runs today.
    assert 600 <= PENDING_TASK_TTL_S <= 24 * 3600
