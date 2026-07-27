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


def test_resolve_ollama_url_probes_health_for_synapse_candidate(monkeypatch):
    """gb-synapse's own candidate must be probed via /health (tied to the
    real upstream engine), not /api/ps — the old shared probe used to pass
    against gb-synapse's dead-engine /api/ps stub before that stub was
    fixed to be honest."""
    monkeypatch.delenv("FORGE_OLLAMA_URL", raising=False)
    monkeypatch.setattr(rot, "_candidate_urls",
                        lambda: ["http://127.0.0.1:11435", "http://127.0.0.1:11434"])

    import urllib.request

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = []

    def _fake_urlopen(url, timeout=5):
        seen.append(url)
        if url.endswith(":11435/health"):
            return _FakeResp()
        raise OSError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    assert rot._resolve_ollama_url() == "http://127.0.0.1:11435"
    assert seen == ["http://127.0.0.1:11435/health"]  # never tried /api/ps on this one


def test_resolve_ollama_url_probes_api_ps_for_raw_ollama_fallback(monkeypatch):
    monkeypatch.delenv("FORGE_OLLAMA_URL", raising=False)
    monkeypatch.setattr(rot, "_candidate_urls",
                        lambda: ["http://127.0.0.1:11435", "http://127.0.0.1:11434"])

    import urllib.request

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = []

    def _fake_urlopen(url, timeout=5):
        seen.append(url)
        if url.endswith(":11434/api/ps"):
            return _FakeResp()
        raise OSError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    assert rot._resolve_ollama_url() == "http://127.0.0.1:11434"
    assert "http://127.0.0.1:11435/health" in seen
    assert "http://127.0.0.1:11434/api/ps" in seen


def test_engine_built_true_when_llama_cpp_engine_installed(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "engine_installed", lambda: True)
    assert rot._engine_built() is True


def test_engine_built_true_when_only_torch_env_present(monkeypatch):
    """A torch-only host (no llama.cpp build) must still count as
    'built' — otherwise overnight rotation silently falls through to the
    ollama keep_alive path, which the proxy has no handling for at all."""
    import gb_synapse
    import gb_synapse_backends
    monkeypatch.setattr(gb_synapse, "engine_installed", lambda: False)
    monkeypatch.setattr(gb_synapse_backends, "_torch_env_dir", lambda: "/opt/torch-env")
    assert rot._engine_built() is True


def test_engine_built_false_when_neither_present(monkeypatch):
    import gb_synapse
    import gb_synapse_backends
    monkeypatch.setattr(gb_synapse, "engine_installed", lambda: False)
    monkeypatch.setattr(gb_synapse_backends, "_torch_env_dir", lambda: None)
    assert rot._engine_built() is False
