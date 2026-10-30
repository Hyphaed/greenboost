"""Factory task lifecycle: queued / running / interrupted, leases, startup
reconciliation, and verified resume. The 2026-10-02 finding: a crash made a
running task indistinguishable from a queued one, so it silently re-ran."""
import os
import subprocess
import time

import pytest

from greenboost_cli.workflow import factory as F


@pytest.fixture
def db(tmp_path):
    return F.FactoryDB(tmp_path / "factory.db")


def _task(tid="t1", cwd=""):
    return F.Task(priority=5, task_id=tid, prompt="edit things", metadata={"cwd": cwd} if cwd else {})


def _repo(path):
    path.mkdir()
    for cmd in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(path), *cmd], check=True)
    (path / "a.txt").write_text("1")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True)


def test_saved_task_is_queued_and_a_claim_marks_it_running(db):
    db.save_task(_task())
    assert [t.task_id for t in db.load_pending()] == ["t1"]
    assert db.claim("t1", f"{os.getpid()}:aaaa", "") == 1
    assert db.load_pending() == []                          # running tasks are not handed out again


def test_a_task_whose_owner_died_becomes_interrupted_not_requeued(db):
    db.save_task(_task())
    db.claim("t1", "999999999:dead", "")                    # a pid that can't exist
    assert db.load_pending() == []
    assert [i["task_id"] for i in db.interrupted_on_load] == ["t1"]
    assert db.interrupted()[0]["attempt"] == 1


def test_a_live_owner_with_a_fresh_lease_is_left_alone(db):
    db.save_task(_task())
    db.claim("t1", f"{os.getpid()}:live", "")               # another factory in this very process tree
    db.load_pending()
    assert db.interrupted() == []


def test_a_live_owner_with_an_expired_lease_is_treated_as_gone(db):
    db.save_task(_task())
    db.claim("t1", f"{os.getpid()}:stuck", "")
    with db._conn() as c:
        c.execute("UPDATE pending_tasks SET heartbeat=?", (time.time() - F.LEASE_S - 5,))
    db.load_pending()
    assert [i["task_id"] for i in db.interrupted()] == ["t1"]


def test_heartbeat_keeps_the_lease_fresh(db):
    owner = f"{os.getpid()}:hb"
    db.save_task(_task())
    db.claim("t1", owner, "")
    with db._conn() as c:
        c.execute("UPDATE pending_tasks SET heartbeat=?", (time.time() - F.LEASE_S - 5,))
    db.heartbeat(owner)
    db.load_pending()
    assert db.interrupted() == []


def test_a_retry_survives_finish_and_keeps_its_attempt_count(db):
    owner = f"{os.getpid()}:r"
    db.save_task(_task())
    db.claim("t1", owner, "")
    db.save_task(_task())                                   # the retry path re-saves the same id
    db.finish_task("t1", owner)                             # the worker loop's cleanup
    pending = db.load_pending()
    assert [t.task_id for t in pending] == ["t1"]           # not deleted out from under the retry
    assert db.claim("t1", owner, "") == 2


def test_finish_removes_only_a_task_this_owner_still_holds(db):
    db.save_task(_task("a")); db.save_task(_task("b"))
    db.claim("a", "1:x", ""); db.claim("b", "2:y", "")
    db.finish_task("a", "1:x"); db.finish_task("b", "1:x")
    ids = {r["task_id"] for r in db._conn().execute("SELECT task_id FROM pending_tasks")}
    assert ids == {"b"}


def test_an_old_database_without_the_new_columns_is_migrated(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.executescript("""CREATE TABLE pending_tasks (task_id TEXT PRIMARY KEY, priority INTEGER NOT NULL,
        created_at REAL NOT NULL, prompt TEXT NOT NULL, agent_name TEXT NOT NULL DEFAULT '',
        autonomous INTEGER NOT NULL DEFAULT 1, delegated_by TEXT NOT NULL DEFAULT '',
        delegation_depth INTEGER NOT NULL DEFAULT 0, metadata TEXT NOT NULL DEFAULT '{}');
        INSERT INTO pending_tasks (task_id, priority, created_at, prompt) VALUES ('old', 5, %f, 'x');""" % time.time())
    c.commit(); c.close()
    d = F.FactoryDB(path)
    assert [t.task_id for t in d.load_pending()] == ["old"]
    assert d.claim("old", f"{os.getpid()}:m", "") == 1


# ── resume verifies the workspace instead of trusting it ──────────────────────

def _factory(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "FACTORY_DB", tmp_path / "f.db")
    monkeypatch.setattr(F, "FACTORY_LOG", tmp_path / "f.log")
    return F.AIFactory()


def _interrupt(fac, repo):
    fac.db.save_task(_task("t1", str(repo)))
    fac.db.claim("t1", "999999999:dead", str(repo))
    fac.db.load_pending()


def test_resume_requeues_when_the_workspace_is_unchanged(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; _repo(repo)
    fac = _factory(tmp_path, monkeypatch); _interrupt(fac, repo)
    assert fac.resume_task("t1") == {"status": "queued", "task_id": "t1"}
    assert fac.db.interrupted() == []


def test_resume_refuses_when_the_tree_has_uncommitted_changes(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; _repo(repo)
    fac = _factory(tmp_path, monkeypatch); _interrupt(fac, repo)
    (repo / "a.txt").write_text("changed by someone")
    res = fac.resume_task("t1")
    assert res["status"] == "refused" and "uncommitted" in res["reason"]
    assert (repo / "a.txt").read_text() == "changed by someone"          # nothing was touched
    assert fac.resume_task("t1", force=True)["status"] == "queued"


def test_resume_refuses_when_the_repo_moved_to_another_commit(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; _repo(repo)
    fac = _factory(tmp_path, monkeypatch); _interrupt(fac, repo)
    (repo / "b.txt").write_text("new")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "later"], check=True)
    assert "started at" in fac.resume_task("t1")["reason"]


def test_resume_refuses_when_the_workspace_is_gone(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; _repo(repo)
    fac = _factory(tmp_path, monkeypatch); _interrupt(fac, repo)
    import shutil; shutil.rmtree(repo)
    assert "no longer exists" in fac.resume_task("t1")["reason"]


def test_discard_and_listing(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; _repo(repo)
    fac = _factory(tmp_path, monkeypatch); _interrupt(fac, repo)
    assert fac.list_tasks("interrupted")["interrupted"][0]["task_id"] == "t1"
    assert fac.discard_task("t1") is True and fac.interrupted_tasks() == []
    assert fac.resume_task("t1")["status"] == "error"
