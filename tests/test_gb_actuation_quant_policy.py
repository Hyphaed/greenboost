#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_actuation.set_quant_policy() (P3 fix, plan
bring-gb-synapse-gb-quant-and-async-nygaard.md). Previously ANY `quality`
string was written to GB_QUALITY, which gb_quant.maybe_quantize_from_env
only recognizes as a TIER name (near_lossless/balanced/compact) , an
MCP-set precision token like "nvfp4"/"int8"/"tq3" landed in GB_QUALITY,
failed that tier check, and (with no GB_QUANT_BUDGET_GB set either) had
ZERO effect on the next pipeline run. Fixed by routing tier names to
GB_QUALITY and precision tokens to GB_QUANT_BITS via the shared
gb_quant.normalize_bits_token(), the same normalizer quant_cmds._normalize_
bits now delegates to.

Zero tests existed for this verb before this file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_actuation as ga


@pytest.fixture(autouse=True)
def _gated(monkeypatch):
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")


@pytest.fixture
def _env_file(tmp_path, monkeypatch):
    path = tmp_path / "inference.env"
    monkeypatch.setattr(ga, "INFERENCE_ENV", path)
    return path


def test_tier_name_writes_gb_quality(_env_file):
    result = ga.set_quant_policy(quality="near_lossless", confirm=True)
    assert result["updates"] == {"GB_QUALITY": "near_lossless"}
    assert "fp8_floor_tradeoff" not in result
    assert "GB_QUALITY=near_lossless" in _env_file.read_text()
    assert "GB_QUANT_BITS" not in _env_file.read_text()


@pytest.mark.parametrize("tier", ["near_lossless", "balanced", "compact"])
def test_all_tier_names_route_to_gb_quality(tier, _env_file):
    result = ga.set_quant_policy(quality=tier, confirm=True)
    assert result["updates"] == {"GB_QUALITY": tier}


def test_precision_token_writes_gb_quant_bits_not_gb_quality(_env_file):
    """The exact bug: a below-fp8 precision token must land in
    GB_QUANT_BITS (which maybe_quantize_from_env's legacy path actually
    reads), never in GB_QUALITY (where it was previously silently inert)."""
    result = ga.set_quant_policy(quality="nvfp4", confirm=True)
    assert result["updates"] == {"GB_QUANT_BITS": "nvfp4"}
    assert "GB_QUALITY" not in result["updates"]
    assert "fp8_floor_tradeoff" in result


@pytest.mark.parametrize("token,normalized", [
    ("int8", "8"), ("8", "8"), ("int4", "4"), ("4", "4"),
    ("tq3", "tq3"), ("tq2", "tq2"),
])
def test_precision_tokens_are_normalized_before_writing(token, normalized, _env_file):
    result = ga.set_quant_policy(quality=token, confirm=True)
    assert result["updates"] == {"GB_QUANT_BITS": normalized}
    assert "fp8_floor_tradeoff" in result


def test_fp8_itself_is_not_flagged_as_below_floor(_env_file):
    result = ga.set_quant_policy(quality="fp8", confirm=True)
    assert result["updates"] == {"GB_QUANT_BITS": "fp8"}
    assert "fp8_floor_tradeoff" not in result


def test_bf16_normalizes_to_the_int_sentinel(_env_file):
    result = ga.set_quant_policy(quality="bf16", confirm=True)
    assert result["updates"] == {"GB_QUANT_BITS": "16"}
    assert "fp8_floor_tradeoff" not in result


def test_auto_is_not_flagged_as_below_floor(_env_file):
    result = ga.set_quant_policy(quality="auto", confirm=True)
    assert result["updates"] == {"GB_QUANT_BITS": "auto"}
    assert "fp8_floor_tradeoff" not in result


def test_budget_gb_still_writes_independently_of_quality(_env_file):
    result = ga.set_quant_policy(budget_gb=8.5, quality="near_lossless", confirm=True)
    assert result["updates"] == {"GB_QUANT_BUDGET_GB": "8.5", "GB_QUALITY": "near_lossless"}


def test_dry_run_without_confirm_writes_nothing(_env_file, monkeypatch):
    monkeypatch.delenv("GB_ORCH_ACTUATE", raising=False)
    result = ga.set_quant_policy(quality="nvfp4", confirm=False)
    assert result["gate"]["allowed"] is False
    assert "dry_run" in result
    assert not _env_file.exists()
