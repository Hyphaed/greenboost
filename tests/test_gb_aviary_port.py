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
    assert gb_aviary._default_synapse_url() == "http://127.0.0.1:11369"


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

def _smoke_with_content(monkeypatch, content: str) -> dict:
    """Run smoke_gate against a canned completion."""
    import json as _json

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return _json.dumps(
                {"choices": [{"message": {"content": content}}]}).encode()

    monkeypatch.setattr(gb_aviary.urllib.request, "urlopen",
                        lambda req, timeout=30: _FakeResp())
    monkeypatch.setattr(gb_aviary, "_emit", lambda *a, **k: None)
    return gb_aviary.smoke_gate("some-model", url="http://127.0.0.1:1")


def test_a_correct_but_terse_answer_is_not_reported_as_collapse(monkeypatch):
    """The exact live output that exposed this, 2026-08-20.

    The gate asked "Say hello and name three colors." and the model answered
    "Hello! Three colors: red, blue, green." — right, and six whitespace
    tokens against a flat `len(toks) < 8 -> FAIL`. Both f16 and q4_0 KV
    produced that byte-identical string and both were failed, so the gate was
    grading its own prompt. It would have blocked a q4_0 KV configuration
    measured 1.48x faster at long prompt with 15/15 needle recall.
    """
    out = _smoke_with_content(monkeypatch, "Hello! Three colors: red, blue, green.")
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["verdict"] != "FAIL", "a right answer must never read as collapse"
    assert "too short to score repetition" in out["reason"]


def test_no_content_at_all_is_still_a_failure(monkeypatch):
    """Short-but-present and empty are different findings. Only one is the
    model's fault, and it keeps FAIL."""
    out = _smoke_with_content(monkeypatch, "")
    assert out["verdict"] == "FAIL"
    assert out["reason"] == "no content returned"


def test_real_repetition_collapse_still_fails(monkeypatch):
    """The behaviour the gate exists for must survive the fix."""
    out = _smoke_with_content(monkeypatch, "red blue green " * 12)
    assert out["verdict"] == "FAIL"
    assert out["reason"] == "repetition collapse"
    assert out["max6gram"] >= 4


def test_the_prompt_asks_for_enough_text_to_score(monkeypatch):
    """Prevent the regression at its source: a prompt whose ideal answer is
    shorter than the six-gram window makes the length branch unavoidable."""
    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"a b c d e f g h i"}}]}'

    def _fake_urlopen(req, timeout=30):
        captured["body"] = req.data.decode()
        return _FakeResp()

    monkeypatch.setattr(gb_aviary.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(gb_aviary, "_emit", lambda *a, **k: None)
    gb_aviary.smoke_gate("some-model", url="http://127.0.0.1:1")
    assert "short sentence" in captured["body"]
