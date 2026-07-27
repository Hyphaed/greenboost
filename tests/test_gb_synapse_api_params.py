#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse_api's Ollama/TGI parameter forwarding and the
Ollama-shaped duration/count fields on streaming final frames.

Before this fix, _ollama_opts/_tgi_params forwarded only
temperature/top_p/(num_predict|max_new_tokens) — `stop`, `seed`, `top_k`,
`repeat_penalty`, `min_p`, `presence_penalty`, `frequency_penalty` were all
silently dropped, so e.g. a client relying on stop sequences got runaway
generation. And streaming final frames carried no eval_count/
prompt_eval_count/*_duration fields at all, so any client computing tok/s
from them (LangChain, OpenWebUI, ai-forge) got zeros.
"""
import pytest

pytest.importorskip("aiohttp")

import gb_synapse_api as api


def test_ollama_opts_forwards_full_param_set():
    opts = {
        "temperature": 0.7, "top_p": 0.9, "num_predict": 128,
        "stop": ["\n", "###"], "seed": 42,
        "presence_penalty": 0.1, "frequency_penalty": 0.2,
        "top_k": 40, "repeat_penalty": 1.1, "min_p": 0.05,
        "num_ctx": 8192,  # intentionally NOT forwarded — see comment in source
    }
    out = api._ollama_opts(opts)
    assert out["temperature"] == 0.7
    assert out["top_p"] == 0.9
    assert out["max_tokens"] == 128
    assert out["stop"] == ["\n", "###"]
    assert out["seed"] == 42
    assert out["presence_penalty"] == 0.1
    assert out["frequency_penalty"] == 0.2
    assert out["top_k"] == 40
    assert out["repeat_penalty"] == 1.1
    assert out["min_p"] == 0.05
    assert "num_ctx" not in out
    assert "n_ctx" not in out


def test_ollama_opts_empty_input_forwards_nothing():
    assert api._ollama_opts({}) == {}


def test_ollama_opts_partial_input_only_forwards_present_keys():
    out = api._ollama_opts({"stop": ["END"]})
    assert out == {"stop": ["END"]}


def test_tgi_params_forwards_stop_seed_and_repetition_penalty():
    out = api._tgi_params({
        "temperature": 0.5, "top_p": 0.8, "max_new_tokens": 64,
        "stop": ["END"], "seed": 7, "repetition_penalty": 1.2,
    })
    assert out["temperature"] == 0.5
    assert out["top_p"] == 0.8
    assert out["max_tokens"] == 64
    assert out["stop"] == ["END"]
    assert out["seed"] == 7
    assert out["repeat_penalty"] == 1.2


def test_tgi_params_empty_input_forwards_nothing():
    assert api._tgi_params({}) == {}


def test_ollama_durations_shape_and_values(monkeypatch):
    times = iter([100.0, 100.2, 101.0, 102.0])  # t_start, t_first, t_last, "now"(t_end)
    monkeypatch.setattr(api.time, "monotonic", lambda: next(times))
    t_start = api.time.monotonic()   # 100.0
    t_first = api.time.monotonic()   # 100.2
    t_last = api.time.monotonic()    # 101.0
    d = api._ollama_durations(t_start, t_first, t_last, ctok=42, ptok=10)
    assert d["eval_count"] == 42
    assert d["prompt_eval_count"] == 10
    assert d["load_duration"] == 0
    # total_duration measured against a 4th monotonic() call (t_end=102.0)
    assert d["total_duration"] == pytest.approx(2_000_000_000, rel=0.01)
    assert d["prompt_eval_duration"] == pytest.approx(200_000_000, rel=0.01)
    assert d["eval_duration"] == pytest.approx(800_000_000, rel=0.01)


def test_ollama_durations_handles_no_tokens_produced():
    """A request that upstream-errored before any token arrived — t_first/
    t_last stay None. Must not raise or produce negative durations."""
    d = api._ollama_durations(100.0, None, None, ctok=0, ptok=5)
    assert d["eval_count"] == 0
    assert d["prompt_eval_count"] == 5
    assert d["prompt_eval_duration"] == 0
    assert d["eval_duration"] == 0
    assert d["total_duration"] >= 0
