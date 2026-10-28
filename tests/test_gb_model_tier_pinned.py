#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Pinned-host demotion in ModelTierManager (GB_TIER_PIN_HOST).

Everything except the last test is CPU-only. The last one needs a usable CUDA
device and is skipped, never failed, while a served model holds the card.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import torch

import gb_model_tier as mt
from _gpu_gate import gpu_skip_reason

GIB = 2**30


def _meminfo(total_gib, avail_gib):
    return lambda field: {"MemTotal": total_gib * GIB // 1024,
                          "MemAvailable": avail_gib * GIB // 1024}.get(field, 0)


@pytest.fixture
def make_tm(tmp_path):
    def _make(**kw):
        with patch("gb_model_tier._gb_session"):
            return mt.ModelTierManager(hbm_headroom_mb=512, t3_dir=str(tmp_path / "pages"), **kw)
    return _make


def test_budget_keeps_the_safety_floor_free(monkeypatch):
    monkeypatch.setattr(mt, "_meminfo_kib", _meminfo(64, 20))     # floor = 10% of 64 GiB = 6.4 GiB
    monkeypatch.delenv("GREENBOOST_HOST_RAM_SAFETY_MB", raising=False)
    assert mt._pinned_budget_ok(10 * GIB)
    assert not mt._pinned_budget_ok(14 * GIB)                    # leaves 6.0 GiB, under the floor


def test_budget_floor_can_be_overridden_in_mib(monkeypatch):
    monkeypatch.setattr(mt, "_meminfo_kib", _meminfo(64, 20))
    monkeypatch.setenv("GREENBOOST_HOST_RAM_SAFETY_MB", "1024")
    assert mt._pinned_budget_ok(18 * GIB)


def test_unknown_memory_figures_never_allow_pinning(monkeypatch):
    monkeypatch.setattr(mt, "_meminfo_kib", lambda field: 0)
    assert not mt._pinned_budget_ok(1)


def test_non_cuda_tensors_take_the_ordinary_path():
    t = torch.arange(6, dtype=torch.float32)
    out = mt._demote_to_pinned(t)
    assert not out.is_cuda and torch.equal(out, t)


def test_pinning_is_off_unless_asked_for(make_tm, monkeypatch):
    monkeypatch.delenv("GB_TIER_PIN_HOST", raising=False)
    assert make_tm().pin_host is False
    monkeypatch.setenv("GB_TIER_PIN_HOST", "1")
    assert make_tm().pin_host is True
    assert make_tm(pin_host=False).pin_host is False             # explicit argument wins


def test_default_path_is_the_plain_module_to_cpu(make_tm):
    tm, module = make_tm(pin_host=False), MagicMock()
    tm._module_to_host(module, non_blocking=True)
    module.to.assert_called_once_with("cpu", non_blocking=True)
    module._apply.assert_not_called()


def test_pinned_path_when_enabled_and_within_budget(make_tm, monkeypatch):
    monkeypatch.setattr(mt, "_pinned_budget_ok", lambda n: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: MagicMock())
    tm, module = make_tm(pin_host=True), MagicMock()
    tm._module_to_host(module, non_blocking=False)
    module._apply.assert_called_once_with(mt._demote_to_pinned)
    module.to.assert_not_called()


def test_falls_back_when_the_budget_is_exceeded(make_tm, monkeypatch):
    monkeypatch.setattr(mt, "_pinned_budget_ok", lambda n: False)
    tm, module = make_tm(pin_host=True), MagicMock()
    tm._module_to_host(module, non_blocking=True)
    module.to.assert_called_once_with("cpu", non_blocking=True)
    module._apply.assert_not_called()


@pytest.mark.skipif(gpu_skip_reason() is not None, reason=str(gpu_skip_reason()))
def test_cuda_tensor_lands_in_pinned_memory_with_the_same_values():
    t = torch.arange(1024, dtype=torch.float32, device="cuda")
    out = mt._demote_to_pinned(t)
    torch.cuda.synchronize()
    assert out.is_pinned() and torch.equal(out, t.cpu())
