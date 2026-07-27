#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_quant_calib.calibrate_with_prompts() (P8, plan
bring-gb-synapse-gb-quant-and-async-nygaard.md) , the real-activation
calibration adapter over the vendored AutoRound AutoScheme search
(third_party/auto_round). AutoRound's own AutoScheme/GenScheme are mocked
throughout: a real run needs a real tokenizer + model + forward pass,
which this CPU-only test suite (like the rest of tests/test_gb_quant*.py)
deliberately does not require. These tests instead verify the adapter's
OWN logic: prompt resolution precedence, preset-name mapping, the
fp8/int8/int4 <-> gb_quant bits conversion, the separate delta_loss cache
namespace, and error handling , the actual scoring correctness is a
documented live-verification gap (see calibrate_with_prompts' docstring).

gb_quant_calib had ZERO tests before this file.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

import gb_quant_calib as gqc


class _TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 64, bias=False)
        self.b = nn.Linear(64, 64, bias=False)


@pytest.fixture(autouse=True)
def _isolated_cache_dir(monkeypatch, tmp_path):
    """Every test below exercises the real cache read/write path , isolate
    it from the user's real ~/.cache/greenboost AND from other tests in
    this file (two tests using the same default model_id/options/prompts
    would otherwise collide on the exact same cache_hash and silently read
    each other's stale result)."""
    monkeypatch.setattr(gqc, "_CACHE_DIR", str(tmp_path))


# ── _calib_prompts(): precedence ──────────────────────────────────────────

def test_calib_prompts_explicit_arg_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("GB_QUANT_CALIB_PROMPTS", str(tmp_path / "unused.txt"))
    result = gqc._calib_prompts(["explicit one", "explicit two"])
    assert result == ["explicit one", "explicit two"]


def test_calib_prompts_env_file_used_when_no_explicit_arg(monkeypatch, tmp_path):
    p = tmp_path / "prompts.txt"
    p.write_text("line one\nline two\n\nline three\n")
    monkeypatch.setenv("GB_QUANT_CALIB_PROMPTS", str(p))
    result = gqc._calib_prompts(None)
    assert result == ["line one", "line two", "line three"]


def test_calib_prompts_falls_back_to_default_corpus(monkeypatch):
    monkeypatch.delenv("GB_QUANT_CALIB_PROMPTS", raising=False)
    result = gqc._calib_prompts(None)
    assert result == list(gqc._DEFAULT_CALIB_PROMPTS)
    assert len(result) > 0


def test_calib_prompts_missing_env_file_falls_back(monkeypatch):
    monkeypatch.setenv("GB_QUANT_CALIB_PROMPTS", "/nonexistent/path/prompts.txt")
    result = gqc._calib_prompts(None)
    assert result == list(gqc._DEFAULT_CALIB_PROMPTS)


# ── calibrate_with_prompts(): option validation ───────────────────────────

def _fake_preset_name_to_scheme(name):
    class _FakeScheme:
        def __init__(self, bits):
            self.bits = bits
    return _FakeScheme(bits={"FP8": 8, "INT8": 8, "INT4": 4}[name])


def test_rejects_option_with_no_autoround_preset(monkeypatch):
    monkeypatch.setattr(gqc, "_require_auto_round",
                        lambda: (MagicMock(), MagicMock(), _fake_preset_name_to_scheme))
    with pytest.raises(ValueError, match="tq3"):
        gqc.calibrate_with_prompts(_TwoLayer(), tokenizer=MagicMock(),
                                   options=("fp8", "tq3"), verbose=False)


def test_empty_module_returns_empty_dict_without_calling_autoround(monkeypatch):
    """A module with no quantizable Linear layers must short-circuit before
    ever touching AutoScheme/GenScheme (nothing to calibrate)."""
    autoscheme_cls = MagicMock(side_effect=AssertionError("should not be constructed"))
    monkeypatch.setattr(gqc, "_require_auto_round",
                        lambda: (autoscheme_cls, MagicMock(), _fake_preset_name_to_scheme))

    class _Empty(nn.Module):
        pass

    result = gqc.calibrate_with_prompts(_Empty(), tokenizer=MagicMock(), verbose=False)
    assert result == {}


# ── calibrate_with_prompts(): adapter plumbing (AutoRound mocked) ────────

def _mock_autoround(monkeypatch, layer_config):
    """Patch _require_auto_round so calibrate_with_prompts runs entirely
    against fakes , returns the captured AutoScheme(...)/GenScheme(...)
    kwargs for assertions."""
    captured = {}

    class _FakeAutoScheme:
        def __init__(self, **kw):
            captured["autoscheme_kwargs"] = kw

    class _FakeGenScheme:
        def __init__(self, auto_scheme, model, quant_layer_names, fixed_layer_scheme, **kw):
            captured["quant_layer_names"] = quant_layer_names
            captured["fixed_layer_scheme"] = fixed_layer_scheme
            captured["gen_kwargs"] = kw

        def get_layer_config(self):
            return layer_config

    monkeypatch.setattr(gqc, "_require_auto_round",
                        lambda: (_FakeAutoScheme, _FakeGenScheme, _fake_preset_name_to_scheme))
    return captured


def test_layer_config_bits_converted_to_gb_quant_vocabulary(monkeypatch):
    layer_config = {
        "a": {"bits": 8, "data_type": "int"},
        "b": {"bits": 8, "data_type": "fp8"},
    }
    _mock_autoround(monkeypatch, layer_config)

    result = gqc.calibrate_with_prompts(_TwoLayer(), tokenizer=MagicMock(),
                                        prompts=["x"], verbose=False)

    assert result == {"a": 8, "b": "fp8"}


def test_int4_layer_config_converts_correctly(monkeypatch):
    layer_config = {"a": {"bits": 4, "data_type": "int"}}
    _mock_autoround(monkeypatch, layer_config)

    result = gqc.calibrate_with_prompts(_TwoLayer(), tokenizer=MagicMock(),
                                        prompts=["x"], verbose=False)

    assert result["a"] == 4


def test_quant_layer_names_passed_to_gen_scheme(monkeypatch):
    layer_config = {"a": {"bits": 8, "data_type": "int"},
                    "b": {"bits": 8, "data_type": "int"}}
    captured = _mock_autoround(monkeypatch, layer_config)

    gqc.calibrate_with_prompts(_TwoLayer(), tokenizer=MagicMock(),
                               prompts=["x"], verbose=False)

    assert set(captured["quant_layer_names"]) == {"a", "b"}
    assert captured["fixed_layer_scheme"] == {}


def test_avg_bits_defaults_to_midpoint_of_options(monkeypatch):
    layer_config = {"a": {"bits": 8, "data_type": "int"}}
    captured = _mock_autoround(monkeypatch, layer_config)

    gqc.calibrate_with_prompts(_TwoLayer(), tokenizer=MagicMock(),
                               options=("fp8", 4), prompts=["x"], verbose=False)

    # fp8 -> preset FP8 -> fake bits=8; int4 -> preset INT4 -> fake bits=4.
    # Midpoint of [8, 4] = 6.0.
    assert captured["autoscheme_kwargs"]["avg_bits"] == 6.0


def test_explicit_avg_bits_overrides_midpoint(monkeypatch):
    layer_config = {"a": {"bits": 8, "data_type": "int"}}
    captured = _mock_autoround(monkeypatch, layer_config)

    gqc.calibrate_with_prompts(_TwoLayer(), tokenizer=MagicMock(), avg_bits=5.5,
                               prompts=["x"], verbose=False)

    assert captured["autoscheme_kwargs"]["avg_bits"] == 5.5


# ── calibrate_with_prompts(): separate cache namespace ────────────────────

def test_cache_is_written_and_reloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(gqc, "_CACHE_DIR", str(tmp_path))
    layer_config = {"a": {"bits": 8, "data_type": "int"}}
    _mock_autoround(monkeypatch, layer_config)

    first = gqc.calibrate_with_prompts(_TwoLayer(), tokenizer=MagicMock(),
                                       prompts=["x"], model_id="m", verbose=False)
    assert first == {"a": 8}

    # Second call: patch _require_auto_round to blow up if actually invoked
    # again -- proves the cache hit short-circuits AutoRound entirely.
    monkeypatch.setattr(gqc, "_require_auto_round",
                        lambda: (_ for _ in ()).throw(AssertionError("should hit cache")))
    second = gqc.calibrate_with_prompts(_TwoLayer(), tokenizer=MagicMock(),
                                        prompts=["x"], model_id="m", verbose=False)
    assert second == {"a": 8}


def test_cache_namespace_is_separate_from_calibrate_sensitivity(monkeypatch, tmp_path):
    """Same model_id, same module -- calibrate_sensitivity's own cache file
    must not collide with calibrate_with_prompts' delta_loss cache file."""
    monkeypatch.setattr(gqc, "_CACHE_DIR", str(tmp_path))
    module = _TwoLayer()

    gqc.calibrate_sensitivity(module, precisions=("fp8",), model_id="m", verbose=False)
    sensitivity_files = set(Path(tmp_path).glob("sensitivity_m_*.json"))
    assert len(sensitivity_files) == 1

    _mock_autoround(monkeypatch, {"a": {"bits": 8, "data_type": "int"}})
    gqc.calibrate_with_prompts(module, tokenizer=MagicMock(), prompts=["x"],
                               model_id="m", verbose=False)
    delta_loss_files = set(Path(tmp_path).glob("sensitivity_m.delta_loss_*.json"))
    assert len(delta_loss_files) == 1
    # The two cache files are distinct paths (different namespace suffix).
    assert not sensitivity_files & delta_loss_files


def test_force_recompute_bypasses_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(gqc, "_CACHE_DIR", str(tmp_path))
    calls = {"n": 0}

    def _mock_with_count():
        calls["n"] += 1
        return _mock_autoround(monkeypatch, {"a": {"bits": 8, "data_type": "int"}})

    _mock_with_count()
    gqc.calibrate_with_prompts(_TwoLayer(), tokenizer=MagicMock(), prompts=["x"],
                               model_id="m", verbose=False)
    _mock_with_count()
    gqc.calibrate_with_prompts(_TwoLayer(), tokenizer=MagicMock(), prompts=["x"],
                               model_id="m", verbose=False, force_recompute=True)
    assert calls["n"] == 2


# ── _require_auto_round(): actionable error when the vendor tree is absent ─

def test_require_auto_round_raises_actionable_error_when_missing(monkeypatch):
    monkeypatch.setattr(gqc, "_ensure_auto_round_path", lambda: None)
    with patch.dict(sys.modules, {"auto_round": None,
                                  "auto_round.auto_scheme.gen_auto_scheme": None}):
        with pytest.raises(RuntimeError, match="install-synapse-engine"):
            gqc._require_auto_round()
