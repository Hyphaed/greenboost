"""Local-first enforcement must recognise a LAN feeder AND catch a cloud endpoint.

The executable half of CLAUDE.md's Local-First Must-Rule. Two failure modes, and
both matter: flagging the owner's own feeder would make the rule unusable and it
would be switched off, while missing a hosted endpoint defeats the entire stack
for that request.
"""
from __future__ import annotations

import pytest

import gb_semantics as gsem


@pytest.mark.parametrize("host", [
    "localhost", "127.0.0.1", "::1", "0.0.0.0",
    "192.168.0.12",         # a feeder on the owner's own LAN IS local
    "10.0.0.5", "172.16.4.4", "172.31.255.1",
    "omen.local", "nas.lan", "build.internal",
])
def test_local_hosts_are_local(host):
    assert gsem._is_local_host(host) is True, host


@pytest.mark.parametrize("host", [
    "integrate.api.nvidia.com",   # NemoClaw's own router pool points here
    "api.openai.com", "api.anthropic.com", "api.together.xyz",
    "generativelanguage.googleapis.com", "bedrock-runtime.us-east-1.amazonaws.com",
    "172.15.0.1",   # just outside 172.16/12
    "172.32.0.1",   # just above 172.16/12
    "8.8.8.8",
])
def test_remote_hosts_are_not_local(host):
    assert gsem._is_local_host(host) is False, host


def test_empty_host_is_treated_as_local():
    """An unset bind means the default loopback, not an unknown remote."""
    assert gsem._is_local_host("") is True
    assert gsem._is_local_host(None) is True


def test_cloud_base_url_env_is_caught(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "ps", lambda: [])
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    r = gsem._res_inference_endpoint_is_local(None, None)
    assert r["value"] is False
    assert any("api.openai.com" in x for x in r["remote_endpoints"])


def test_local_proxy_env_is_not_flagged(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "ps", lambda: [])
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11369/v1")
    r = gsem._res_inference_endpoint_is_local(None, None)
    assert r["value"] is True
    assert r["remote_endpoints"] == []


def test_remote_serve_bind_is_caught(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "ps",
                        lambda: [{"model": "m", "bind": "203.0.113.9"}])
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    r = gsem._res_inference_endpoint_is_local(None, None)
    assert r["value"] is False


def test_unknown_state_is_not_reported_as_compliant(monkeypatch):
    """Silence must never read as compliance."""
    import gb_synapse
    def _boom():
        raise RuntimeError("cannot reach run-state")
    monkeypatch.setattr(gb_synapse, "ps", _boom)
    r = gsem._res_inference_endpoint_is_local(None, None)
    assert r["value"] is None

    monkeypatch.setattr(gsem, "resolve", lambda name, **kw: {"value": None})
    matched, _ = gsem._seg_cloud_inference_in_use()
    assert matched is None, "unresolvable endpoints must not read as 'no violation'"
