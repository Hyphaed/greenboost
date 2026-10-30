#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""A2A gateway authorization (SEC-01): an actuating verb runs only inside
gb_actuation.actuation_auth(<Bearer token>) checked against the owner's 0600
token file. `confirm` alone, a wrong token or no token is a dry run that says why."""
import http.client
import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import gb_a2a
import gb_actuation

TOKEN = "t" * 32


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    tf = tmp_path / "a2a.token"
    tf.write_text(TOKEN + "\n")
    os.chmod(tf, 0o600)
    monkeypatch.setenv("GB_A2A_TOKEN_FILE", str(tf))
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")
    monkeypatch.delenv("GB_A2A_TOKEN", raising=False)
    ran = []

    def fake_verb(confirm=False):
        gate = gb_actuation.actuation_gate(confirm)
        if gate["allowed"]:
            ran.append("run")          # the fake runner
        return {"gate": gate}

    monkeypatch.setitem(gb_actuation.VERBS, "fake_verb", fake_verb)
    monkeypatch.setattr(gb_a2a, "_RATE_PER_MIN", 10_000)
    monkeypatch.setattr(gb_a2a, "_emit_request", lambda *a, **k: None)
    srv = gb_a2a.ThreadingHTTPServer(("127.0.0.1", 0), gb_a2a._Handler)
    port = srv.server_address[1]
    monkeypatch.setattr(gb_a2a, "DEFAULT_BIND", f"127.0.0.1:{port}")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield port, ran, tf
    srv.shutdown()


def _call(port, confirm, auth=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Content-Type": "application/json"}
    if auth is not None:
        h["Authorization"] = auth
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "fake_verb", "params": {"confirm": confirm}})
    c.request("POST", "/", body, h)
    r = c.getresponse()
    return r.status, json.loads(r.read())


def test_valid_token_executes(gateway):
    port, ran, _ = gateway
    st, body = _call(port, True, f"Bearer {TOKEN}")
    assert st == 200 and body["result"]["gate"]["allowed"] is True
    assert "auth" not in body["result"] and ran == ["run"]


@pytest.mark.parametrize("auth", [None, "Bearer wrong-token-wrong-token-xxxxxx", f"Basic {TOKEN}", "Bearer "])
def test_missing_or_invalid_token_is_a_dry_run(gateway, auth):
    port, ran, _ = gateway
    st, body = _call(port, True, auth)
    assert st == 200 and body["result"]["gate"]["allowed"] is False
    assert "dry run" in body["result"]["auth"] and ran == []


def test_confirm_true_alone_is_a_dry_run(gateway):
    port, ran, _ = gateway
    _, body = _call(port, True)
    assert body["result"]["gate"]["allowed"] is False and ran == []


def test_valid_token_without_confirm_is_a_dry_run(gateway):
    port, ran, _ = gateway
    _, body = _call(port, False, f"Bearer {TOKEN}")
    assert body["result"]["gate"]["allowed"] is False and ran == []


def test_unsafe_token_file_never_authorizes(gateway):
    port, ran, tf = gateway
    os.chmod(tf, 0o644)
    _, body = _call(port, True, f"Bearer {TOKEN}")
    assert body["result"]["gate"]["allowed"] is False and ran == []
