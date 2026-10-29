"""SEC-01: a caller-supplied `confirm` is never authorization. Fakes and tmp dirs only; nothing real is run."""
import json
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import gb_actuation as ga

TOK = "s3cret-token-0123456789abcdef"


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    import gb_cluster
    monkeypatch.setattr(gb_cluster, "shim_env", lambda workload, enabled, base_env: {})
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")
    monkeypatch.setenv("GB_INFERENCE_ENV", str(tmp_path / "inference.env"))
    monkeypatch.setattr(ga, "INFERENCE_ENV", tmp_path / "inference.env")
    tf = tmp_path / "a2a.token"
    tf.write_text(TOK + "\n")
    tf.chmod(0o600)
    monkeypatch.setenv("GB_A2A_TOKEN_FILE", str(tf))
    return tf


def test_sec01_confirm_true_without_a_token_is_a_dry_run():
    r = ga.run_under_greenboost(["/bin/echo", "x"], confirm=True)        # the request of the audit: confirm true, no token
    assert r["gate"]["allowed"] is False and "dry_run" in r and "exit_code" not in r
    assert "proves nothing" in r["gate"]["reason"]


@pytest.mark.parametrize("call", [
    lambda: ga.tier_actuate("pool_cap_mb", 4096, confirm=True),
    lambda: ga.set_quant_policy(quality="fp8", confirm=True),
    lambda: ga.reclaim_run(confirm=True),
    lambda: ga.serve_and_repoint("m", confirm=True),
    lambda: ga.cluster_dispatch_plan(confirm=True),
])
def test_every_actuating_verb_denies_by_default(call, tmp_path):
    r = call()
    assert r["gate"]["allowed"] is False
    assert not (tmp_path / "inference.env").exists()


def test_a_valid_token_context_authorizes_and_ends_with_the_context():
    with ga.actuation_auth(TOK) as ok:
        assert ok and ga.actuation_gate(True)["allowed"] is True
        assert ga.actuation_gate(False)["allowed"] is False              # still needs confirm
    assert ga.actuation_gate(True)["allowed"] is False                   # context closed


def test_wrong_empty_or_missing_token_authorizes_nothing():
    for bad in ("wrong", "", None, TOK + "x", TOK[:-1]):
        with ga.actuation_auth(bad) as ok:
            assert ok is False and ga.actuation_gate(True)["allowed"] is False


def test_the_token_is_compared_in_constant_time(monkeypatch):
    seen = []
    real = ga.hmac.compare_digest
    monkeypatch.setattr(ga.hmac, "compare_digest", lambda a, b: (seen.append(1), real(a, b))[1])
    with ga.actuation_auth("nope"):
        pass
    assert seen


def test_the_gate_still_needs_the_server_flag_even_with_a_token(monkeypatch):
    monkeypatch.setenv("GB_ORCH_ACTUATE", "0")
    with ga.actuation_auth(TOK):
        g = ga.actuation_gate(True)
    assert g["allowed"] is False and "GB_ORCH_ACTUATE" in g["reason"]


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o666, 0o604])
def test_a_token_file_with_loose_mode_is_refused(env, mode):
    env.chmod(mode)
    with ga.actuation_auth(TOK) as ok:
        assert ok is False


def test_a_symlinked_or_missing_token_file_is_refused(env, tmp_path, monkeypatch):
    ln = tmp_path / "link.token"
    ln.symlink_to(env)
    monkeypatch.setenv("GB_A2A_TOKEN_FILE", str(ln))
    with ga.actuation_auth(TOK) as ok:
        assert ok is False
    monkeypatch.setenv("GB_A2A_TOKEN_FILE", str(tmp_path / "absent"))
    with ga.actuation_auth(TOK) as ok:
        assert ok is False


def test_a_token_file_owned_by_someone_else_is_refused(env, monkeypatch):
    real = os.fstat
    class S:                      # fake stat: owned by uid 4242
        def __init__(self, st): self.st_mode, self.st_uid = st.st_mode, 4242
    monkeypatch.setattr(ga.os, "fstat", lambda fd: S(real(fd)))
    with ga.actuation_auth(TOK) as ok:
        assert ok is False
    monkeypatch.setenv("GB_OWNER_UID", "4242")                           # unless the owner is configured
    with ga.actuation_auth(TOK) as ok:
        assert ok is True


def test_a_short_token_in_the_file_is_refused(env):
    env.write_text("abc\n")
    with ga.actuation_auth("abc") as ok:
        assert ok is False


def test_authorization_does_not_leak_to_another_thread():
    out = {}
    with ga.actuation_auth(TOK):
        t = threading.Thread(target=lambda: out.update(g=ga.actuation_gate(True)))
        t.start(); t.join()
        assert ga.actuation_gate(True)["allowed"] is True
    assert out["g"]["allowed"] is False


def test_an_authorized_verb_really_applies_and_dry_run_still_writes_nothing(tmp_path):
    f = tmp_path / "inference.env"
    with ga.actuation_auth(TOK):
        r = ga.set_quant_policy(quality="fp8", confirm=False)
        assert r["gate"]["allowed"] is False and not f.exists()
        r = ga.set_quant_policy(quality="fp8", confirm=True)
    assert r["gate"]["allowed"] is True and f.exists()


def test_interactive_path_needs_a_person_on_the_terminal(tmp_path):
    fake_tty = tmp_path / "tty"
    fake_tty.write_text("yes\n")
    with ga.interactive_auth("tier_actuate", "x", tty_path=str(fake_tty)) as ok:
        assert ok and ga.actuation_gate(True)["allowed"] is True
    fake_tty.write_text("y\n")
    with ga.interactive_auth("tier_actuate", "x", tty_path=str(fake_tty)) as ok:
        assert ok is False
    with ga.interactive_auth("tier_actuate", "x", tty_path=str(tmp_path / "no-tty")) as ok:
        assert ok is False and ga.actuation_gate(True)["allowed"] is False


def test_root_does_not_serve_run_under_greenboost_nor_reclaim_all(monkeypatch):
    table = {"run_under_greenboost": 1, "reclaim_run": 2, "shim_env": 3}
    assert "run_under_greenboost" not in ga._served_verbs(table, root=True)
    assert "run_under_greenboost" in ga._served_verbs(table, root=False)
    monkeypatch.setenv("GB_ACTUATION_ROOT_ALL", "1")
    assert "run_under_greenboost" in ga._served_verbs(table, root=True)
    monkeypatch.delenv("GB_ACTUATION_ROOT_ALL")
    import gb_reclaim
    monkeypatch.setattr(gb_reclaim, "plan_reclaim", lambda **k: {"targets": []})
    monkeypatch.setattr(ga, "_is_root", lambda: True)
    with ga.actuation_auth(TOK):
        assert ga.reclaim_run(scope="all", confirm=True)["gate"]["allowed"] is False
        assert "dry_run" in ga.reclaim_run(scope="all", confirm=True)
