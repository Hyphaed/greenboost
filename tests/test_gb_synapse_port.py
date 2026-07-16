#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse.py port migration (:11434 -> :11435, GB_SYNAPSE_PORT).

CPU-only. No GGUF, no CUDA, no real HF network calls.
"""
import importlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_synapse as gs


@pytest.fixture
def reload_gb_synapse(monkeypatch):
    """Reload gb_synapse after mutating GB_SYNAPSE_PORT, then restore the
    module to its default-env state so later tests importing it at module
    scope see the expected DEFAULT_PORT again."""
    yield
    monkeypatch.delenv("GB_SYNAPSE_PORT", raising=False)
    importlib.reload(gs)


def test_default_port_no_env(monkeypatch, reload_gb_synapse):
    monkeypatch.delenv("GB_SYNAPSE_PORT", raising=False)
    importlib.reload(gs)
    assert gs.DEFAULT_PORT == 11435


def test_default_port_env_override(monkeypatch, reload_gb_synapse):
    monkeypatch.setenv("GB_SYNAPSE_PORT", "9999")
    importlib.reload(gs)
    assert gs.DEFAULT_PORT == 9999


class _FakeProc:
    def __init__(self):
        self._polled = False

    def poll(self):
        return None  # never exits — forces the timeout path


def test_wait_proxy_ready_hints_ollama_on_11434(monkeypatch):
    """When the squatted port is the legacy 11434, the failure message should
    name raw ollama.service specifically (still the recurring real case)."""
    def _raise_conn_refused(*a, **k):
        raise OSError("not listening")

    monkeypatch.setattr(urllib.request, "urlopen", _raise_conn_refused)
    reason = gs._wait_proxy_ready(_FakeProc(), 11434, "test", grace_s=0.1)
    assert reason is not None
    assert "ollama.service" in reason
    assert "11434" in reason


def test_wait_proxy_ready_generic_port(monkeypatch):
    def _raise_conn_refused(*a, **k):
        raise OSError("not listening")

    monkeypatch.setattr(urllib.request, "urlopen", _raise_conn_refused)
    reason = gs._wait_proxy_ready(_FakeProc(), 55555, "test", grace_s=0.1)
    assert reason is not None
    assert "55555" in reason
    assert "ollama" not in reason
