#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse_api's Bearer-token auth middleware and the
non-loopback-without-a-token startup refusal.

Real hazard this closes: gb_synapse.py always launched this proxy on
0.0.0.0 with no auth — reachable from anything with network access to the
host. gb_a2a.py (the A2A gateway) already enforces "non-loopback bind
requires a token or refuse to start"; this proxy violated that same rule
until now.
"""
import asyncio

import pytest

pytest.importorskip("aiohttp")

import gb_synapse_api as api


class _FakeRequest:
    def __init__(self, headers=None, path="/health", remote="203.0.113.1"):
        self.headers = headers or {}
        self.path = path
        self.remote = remote


async def _ok_handler(request):
    return "handler ran"


def _run(coro):
    return asyncio.run(coro)


# ── _resolve_token() ─────────────────────────────────────────────────────

def test_resolve_token_empty_when_unset(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_TOKEN", raising=False)
    monkeypatch.setattr(api, "_TOKEN_FILE", api.Path("/nonexistent/synapse_token"))
    assert api._resolve_token() == ""


def test_resolve_token_env_wins_over_file(monkeypatch, tmp_path):
    token_file = tmp_path / "synapse_token"
    token_file.write_text("file-token\n")
    monkeypatch.setattr(api, "_TOKEN_FILE", token_file)
    monkeypatch.setattr(api, "_trust_validate_root_file", lambda path, max_size=None: "trusted")
    monkeypatch.setenv("GB_SYNAPSE_TOKEN", "env-token")
    assert api._resolve_token() == "env-token"


def test_resolve_token_falls_back_to_file(monkeypatch, tmp_path):
    token_file = tmp_path / "synapse_token"
    token_file.write_text("file-token\n")
    monkeypatch.setattr(api, "_TOKEN_FILE", token_file)
    # TCB-U1: mock the trust validation since test tmp_path is not root-owned
    monkeypatch.setattr(api, "_trust_validate_root_file", lambda path, max_size=None: "trusted")
    monkeypatch.delenv("GB_SYNAPSE_TOKEN", raising=False)
    assert api._resolve_token() == "file-token"


# ── _is_loopback() ───────────────────────────────────────────────────────

def test_is_loopback():
    assert api._is_loopback("127.0.0.1")
    assert api._is_loopback("localhost")
    assert api._is_loopback("::1")
    assert not api._is_loopback("0.0.0.0")
    assert not api._is_loopback("192.168.1.5")


# ── _auth_middleware() ───────────────────────────────────────────────────

def test_auth_middleware_passes_through_when_no_token_configured(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_TOKEN", "")
    monkeypatch.setattr(api, "_TOKEN_FILE", api.Path("/nonexistent/synapse_token"))
    result = _run(api._auth_middleware(_FakeRequest(), _ok_handler))
    assert result == "handler ran"


def test_auth_middleware_rejects_missing_header_when_token_set(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_TOKEN", "s3cr3t")
    resp = _run(api._auth_middleware(_FakeRequest(), _ok_handler))
    assert resp.status == 401


def test_auth_middleware_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_TOKEN", "s3cr3t")
    req = _FakeRequest(headers={"Authorization": "Bearer wrong"})
    resp = _run(api._auth_middleware(req, _ok_handler))
    assert resp.status == 401


def test_auth_middleware_accepts_correct_token(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_TOKEN", "s3cr3t")
    req = _FakeRequest(headers={"Authorization": "Bearer s3cr3t"})
    result = _run(api._auth_middleware(req, _ok_handler))
    assert result == "handler ran"


def test_auth_middleware_no_health_bypass(monkeypatch):
    """Unlike a typical health-check exemption, /health still requires the
    token when one is configured — an unauthenticated liveness probe still
    tells a LAN attacker the endpoint exists and is serving."""
    monkeypatch.setenv("GB_SYNAPSE_TOKEN", "s3cr3t")
    req = _FakeRequest(path="/health")
    resp = _run(api._auth_middleware(req, _ok_handler))
    assert resp.status == 401


def test_auth_middleware_emits_dataflux_event_on_rejection(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_TOKEN", "s3cr3t")
    captured = []
    monkeypatch.setattr("gb_dataflux.emit", lambda event: captured.append(event))
    req = _FakeRequest(headers={}, path="/api/tags", remote="198.51.100.7")
    _run(api._auth_middleware(req, _ok_handler))
    assert len(captured) == 1
    assert captured[0]["kind"] == "synapse_auth"
    assert captured[0]["status"] == "rejected"
    assert captured[0]["path"] == "/api/tags"
    assert captured[0]["peer"] == "198.51.100.7"
