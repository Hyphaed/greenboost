#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_aviary.py's certification-endpoint default port fix.

Before this fix, niah_certify()/smoke_gate() both defaulted to raw Ollama's
legacy :11434 , contradicting the standing rule that gb-synapse (default
:11435) is THE Ollama replacement. Certifying against the wrong port
measures a different serving path than the one that actually runs the
model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_aviary


def test_default_synapse_url_uses_gb_synapse_port(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "DEFAULT_PORT", 11435)
    assert gb_aviary._default_synapse_url() == "http://127.0.0.1:11435"


def test_default_synapse_url_reflects_env_override(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "DEFAULT_PORT", 9999)
    assert gb_aviary._default_synapse_url() == "http://127.0.0.1:9999"


def test_default_synapse_url_falls_back_when_gb_synapse_unimportable(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "gb_synapse":
            raise ImportError("simulated")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert gb_aviary._default_synapse_url() == "http://127.0.0.1:11435"


def test_niah_certify_default_url_param_is_none_not_11434():
    import inspect
    sig = inspect.signature(gb_aviary.niah_certify)
    assert sig.parameters["url"].default is None


def test_smoke_gate_default_url_param_is_none_not_11434():
    import inspect
    sig = inspect.signature(gb_aviary.smoke_gate)
    assert sig.parameters["url"].default is None


def test_smoke_gate_resolves_default_url_when_omitted(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "DEFAULT_PORT", 11435)

    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"hello world foo bar baz qux"}}]}'

    def _fake_urlopen(req, timeout=30):
        captured["url"] = req.full_url
        return _FakeResp()

    monkeypatch.setattr(gb_aviary.urllib.request, "urlopen", _fake_urlopen)
    gb_aviary.smoke_gate("some-model")
    assert captured["url"].startswith("http://127.0.0.1:11435")
