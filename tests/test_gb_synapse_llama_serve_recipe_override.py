#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for LlamaCppBackend.serve()'s recipe-placement overrides
(NemoClaw audit, Phase 5/7 follow-up): n_gpu_layers_override/
kv_type_override/n_cpu_moe_override skip the heuristic -ngl/KV-type/
--n-cpu-moe decisions entirely, EXCEPT cpu_quirk (a confirmed hardware
crash) which always wins, and a dense (non-MoE) override asking for fewer
than all layers still goes through _gate_cpu_offload() — the same T2-spill
safety rule a live heuristic decision would have to clear.

Reuses test_gb_synapse_llama_serve.py's fixture harness (no real
subprocess/GPU: subprocess.Popen, NVML, the feeder probe, and
gb_shim_probe are all monkeypatched).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_synapse_backends as gsb
from tests.test_gb_synapse_llama_serve import (
    _FakeNvml, _no_shim, _shim_active, _stub_gb_cluster, _stub_gb_synapse,
    _stub_popen, backend, entry,
)


def test_n_gpu_layers_override_all_sets_ngl_999(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435, n_gpu_layers_override="all")

    cmd = _stub_popen["cmd"]
    assert cmd[cmd.index("-ngl") + 1] == "999"


def test_n_gpu_layers_override_int_skips_heuristic_entirely(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    """Even though the shim is active and this box would otherwise fit
    everything on GPU (fits_vram would be True), an explicit override
    still wins outright — no heuristic re-derivation."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))
    monkeypatch.setenv("GB_SYNAPSE_ALLOW_CPU_OFFLOAD", "1")  # this IS a dense partial offload

    backend.serve(entry, port=11435, n_gpu_layers_override=30)

    cmd = _stub_popen["cmd"]
    assert cmd[cmd.index("-ngl") + 1] == "30"


def test_kv_type_override_wins_over_heuristic_and_model_default(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435, kv_type_override="q4_0")

    cmd = _stub_popen["cmd"]
    assert cmd[cmd.index("--cache-type-k") + 1] == "q4_0"
    assert cmd[cmd.index("--cache-type-v") + 1] == "q4_0"


def test_cpu_quirk_wins_over_n_gpu_layers_override(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _no_shim, monkeypatch):
    """A confirmed hardware/driver crash for this arch must never be
    re-enabled by a recipe override — cpu_quirk is checked first,
    unconditionally, regardless of what the override asks for. Shim OFF
    (_no_shim) so fits_vram is genuinely False (with the shim active,
    fits_vram short-circuits True via "LD_PRELOAD" in env regardless of
    free VRAM, which would mask cpu_quirk — not what this test checks)."""
    import gb_synapse as gs
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(1.0))  # forces not fits_vram
    monkeypatch.setattr(gs, "ARCH_CPU_SPLIT_BROKEN", {"quirky-arch"})
    monkeypatch.delenv("GB_SYNAPSE_FORCE_SPLIT", raising=False)
    entry.arch = "quirky-arch"

    backend.serve(entry, port=11435, n_gpu_layers_override="all")

    cmd = _stub_popen["cmd"]
    assert cmd[cmd.index("-ngl") + 1] == "0"
    assert "--no-op-offload" in cmd


def test_dense_override_below_all_layers_gated_by_cpu_offload_rule(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _no_shim, monkeypatch):
    """The exact safety concern this override design has to close: a
    recipe asking for fewer than all layers on a DENSE model is still a
    capacity decision the T2-spill rule forbids by default — must raise,
    not silently serve, when the shim is off and no override env is set."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))
    monkeypatch.delenv("GB_SYNAPSE_ALLOW_CPU_OFFLOAD", raising=False)

    with pytest.raises(RuntimeError, match="CPU offload"):
        backend.serve(entry, port=11435, n_gpu_layers_override=30)

    assert "cmd" not in _stub_popen


def test_dense_override_below_all_layers_serves_under_explicit_env_override(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _no_shim, monkeypatch):
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))
    monkeypatch.setenv("GB_SYNAPSE_ALLOW_CPU_OFFLOAD", "1")

    backend.serve(entry, port=11435, n_gpu_layers_override=30)

    cmd = _stub_popen["cmd"]
    assert cmd[cmd.index("-ngl") + 1] == "30"


def test_moe_dense_override_check_does_not_apply_to_moe_models(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _no_shim, monkeypatch):
    """A MoE model's n_gpu_layers_override='all' + n_cpu_moe_override is the
    already-exempt --n-cpu-moe placement, not a dense partial offload — must
    NOT hit the CPU-offload gate even with the shim off and no env override."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))
    monkeypatch.delenv("GB_SYNAPSE_ALLOW_CPU_OFFLOAD", raising=False)
    entry.is_moe = True
    entry.n_experts = 256
    entry.n_experts_used = 8

    backend.serve(entry, port=11435, n_gpu_layers_override="all", n_cpu_moe_override=20)

    cmd = _stub_popen["cmd"]
    assert cmd[cmd.index("-ngl") + 1] == "999"
    assert cmd[cmd.index("--n-cpu-moe") + 1] == "20"


def test_no_override_preserves_existing_heuristic_behavior(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    """Byte-for-byte regression guard: calling serve() with none of the
    three new override params set must behave exactly as before this
    change — same shape as test_gb_synapse_llama_serve.py's own
    test_shim_active_extends_ctx_past_vram_only_floor."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435)

    cmd = _stub_popen["cmd"]
    assert cmd[cmd.index("-ngl") + 1] == "999"  # fits_vram path, unchanged
