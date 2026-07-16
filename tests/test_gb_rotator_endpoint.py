#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_rotator's endpoint resolution (port migration :11434 -> :11435,
GB_SYNAPSE_PORT): FORGE_OLLAMA_URL env -> gb-synapse's own port -> raw
Ollama's legacy :11434 as last-resort fallback.

CPU-only. No real network calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_rotator as rot


def test_candidate_urls_default_order(monkeypatch):
    monkeypatch.delenv("FORGE_OLLAMA_URL", raising=False)
    monkeypatch.delenv("GB_SYNAPSE_PORT", raising=False)
    assert rot._candidate_urls() == [
        "http://127.0.0.1:11435",
        "http://127.0.0.1:11434",
    ]


def test_candidate_urls_forge_env_first(monkeypatch):
    monkeypatch.setenv("FORGE_OLLAMA_URL", "http://10.0.0.5:11435")
    assert rot._candidate_urls()[0] == "http://10.0.0.5:11435"


def test_candidate_urls_no_duplicate_when_env_matches_synapse_port(monkeypatch):
    monkeypatch.setenv("FORGE_OLLAMA_URL", "http://127.0.0.1:11435")
    urls = rot._candidate_urls()
    assert urls.count("http://127.0.0.1:11435") == 1


def test_resolve_ollama_url_returns_first_alive(monkeypatch):
    monkeypatch.delenv("FORGE_OLLAMA_URL", raising=False)
    monkeypatch.setattr(rot, "_candidate_urls",
                        lambda: ["http://127.0.0.1:1", "http://127.0.0.1:2"])

    import urllib.request

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(url, timeout=5):
        if "127.0.0.1:2" in url:
            return _FakeResp()
        raise OSError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    assert rot._resolve_ollama_url() == "http://127.0.0.1:2"


def test_resolve_ollama_url_none_when_nothing_alive(monkeypatch):
    import urllib.request

    def _fake_urlopen(url, timeout=5):
        raise OSError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    assert rot._resolve_ollama_url() is None
