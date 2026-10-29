#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_mcp.py's orchestrator-server-specific tools (a2a_gateway,
run_under_greenboost MCP wrapper). No CUDA, no daemon; subprocess/systemctl
calls are monkeypatched.

Note: gb_mcp.py's mirrored tools (greenboost_status, synapse_status, etc.)
are covered by tests/test_mcp_shapes.py, not here.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_mcp


# ── gb_plan sizes KV per layer kind (casebook C-07) ──────────────────────────

class _Entry:
    name = "hybrid-test"
    path = "/nonexistent/hybrid.gguf"
    n_bytes = 17_022_685_184
    quant = "IQ4_XS"


#: The real geometry of Qwen3.8-27B-Cold-Fusion-MTP-IQ4_XS, read from its own
#: GGUF metadata on 2026-08-22. 65 layers, of which only 17 are attention.
_HYBRID_SUMMARY = {
    "n_layers": 65, "n_kv_layers": 17, "n_recurrent_layers": 48,
    "n_kv_heads": 4, "head_dim": 256, "is_recurrent_only": False,
    "ssm_d_conv": 4, "ssm_conv_width": 10240, "ssm_state_elems": 786432,
    "dense_bytes": 0, "expert_bytes": 0, "is_moe": False,
}


def _plan_with(monkeypatch, summary):
    """Run gb_plan against a fixed summary, capturing the kv_gb it computes."""
    import gb_synapse
    import gb_placement
    monkeypatch.setattr(gb_synapse, "list_models", lambda: [_Entry()])
    monkeypatch.setattr(gb_synapse, "gguf_summary", lambda p: summary)
    seen = {}

    def _capture(dense_gb, expert_gb, kv_gb, vram_fill_pct):
        seen["kv_gb"] = kv_gb
        return {}

    monkeypatch.setattr(gb_placement, "plan_experts_and_emit", _capture)
    gb_mcp.gb_plan("hybrid-test")
    return seen["kv_gb"]


def test_gb_plan_charges_only_the_attention_layers(monkeypatch):
    """C-07's mechanism, found still live on this path 2026-08-22.

    gb_plan passed `n_layers` where every other caller passes `n_kv_layers`
    (gb_synapse_backends.py:876, gb_bench_kv.py:88). On this real model that
    charged all 65 layers the attention rate: 8.125 GiB against 2.271 GiB
    real, a 3.58x overestimate handed to the expert planner on an 11.3 GiB
    card. gguf_summary() had carried n_kv_layers all along.
    """
    import gb_synapse
    kv_gb = _plan_with(monkeypatch, _HYBRID_SUMMARY)

    all_layers = gb_synapse.estimate_kv_gb(
        65536, _Entry.n_bytes, _Entry.quant, n_layers=65,
        n_kv_heads=4, head_dim=256)
    assert kv_gb < all_layers / 3, (
        f"gb_plan charged {kv_gb:.3f} GiB, close to the all-layers "
        f"{all_layers:.3f} GiB it must no longer use")

    expected = gb_synapse.estimate_kv_gb(
        65536, _Entry.n_bytes, _Entry.quant, n_layers=17,
        n_kv_heads=4, head_dim=256)
    expected += gb_synapse.estimate_ssm_state_gb(48, 4, 10240, 786432)
    assert kv_gb == pytest.approx(expected, rel=1e-6)


def test_gb_plan_counts_the_recurrent_state_it_excluded_from_kv(monkeypatch):
    """Dropping 48 layers from KV without counting the state they DO hold
    would swing the estimate from over- to under-counting, and that state is
    VRAM-resident under Rule #1."""
    import gb_synapse
    with_ssm = _plan_with(monkeypatch, _HYBRID_SUMMARY)
    kv_only = gb_synapse.estimate_kv_gb(
        65536, _Entry.n_bytes, _Entry.quant, n_layers=17,
        n_kv_heads=4, head_dim=256)
    assert with_ssm > kv_only


def test_gb_plan_on_a_plain_transformer_is_unchanged(monkeypatch):
    """No recurrent layers means n_kv_layers == n_layers and no SSM term, so
    this fix must be a no-op for every non-hybrid model."""
    import gb_synapse
    plain = dict(_HYBRID_SUMMARY, n_layers=32, n_kv_layers=32,
                 n_recurrent_layers=0, ssm_state_elems=0)
    kv_gb = _plan_with(monkeypatch, plain)
    assert kv_gb == pytest.approx(gb_synapse.estimate_kv_gb(
        65536, _Entry.n_bytes, _Entry.quant, n_layers=32,
        n_kv_heads=4, head_dim=256), rel=1e-6)


def test_gb_plan_on_a_recurrent_only_model_charges_no_kv(monkeypatch):
    import gb_synapse
    rec = dict(_HYBRID_SUMMARY, n_kv_layers=0, n_recurrent_layers=65,
               is_recurrent_only=True)
    kv_gb = _plan_with(monkeypatch, rec)
    assert kv_gb == pytest.approx(
        gb_synapse.estimate_ssm_state_gb(65, 4, 10240, 786432), rel=1e-6)


def test_a2a_gateway_status_reports_active_and_enabled(monkeypatch):
    def _fake_run(cmd, **kw):
        out = "active\n" if cmd[1] == "is-active" else "enabled\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = gb_mcp.a2a_gateway(action="status")
    assert result["active"] is True
    assert result["enabled"] is True
    assert result["unit"] == "greenboost-a2a.service"


def test_a2a_gateway_status_handles_inactive_unit(monkeypatch):
    def _fake_run(cmd, **kw):
        out = "inactive\n" if cmd[1] == "is-active" else "disabled\n"
        return subprocess.CompletedProcess(cmd, 3, stdout=out, stderr="")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = gb_mcp.a2a_gateway(action="status")
    assert result["active"] is False
    assert result["enabled"] is False


def test_a2a_gateway_restart_dry_run_by_default(monkeypatch):
    monkeypatch.delenv("GB_ORCH_ACTUATE", raising=False)
    result = gb_mcp.a2a_gateway(action="restart", confirm=False)
    assert result["gate"]["allowed"] is False
    assert "dry_run" in result


def test_a2a_gateway_restart_dry_run_when_gate_not_double_confirmed(monkeypatch):
    monkeypatch.delenv("GB_ORCH_ACTUATE", raising=False)
    result = gb_mcp.a2a_gateway(action="restart", confirm=True)
    assert result["gate"]["allowed"] is False


def test_a2a_gateway_restart_executes_when_double_gated(monkeypatch, tmp_path):
    import os
    import gb_actuation
    tf = tmp_path / "a2a.token"
    tf.write_text("k" * 32 + "\n")
    os.chmod(tf, 0o600)
    monkeypatch.setenv("GB_A2A_TOKEN_FILE", str(tf))
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")
    calls = []
    def _fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    with gb_actuation.actuation_auth("k" * 32) as authed:
        assert authed
        result = gb_mcp.a2a_gateway(action="restart", confirm=True)
    assert result["ok"] is True
    # confirm=True + env alone (no authorization context) is a dry run
    assert gb_mcp.a2a_gateway(action="restart", confirm=True)["gate"]["allowed"] is False
    assert calls == [["systemctl", "restart", "greenboost-a2a.service"]]


def test_a2a_gateway_unknown_action_errors():
    result = gb_mcp.a2a_gateway(action="bogus")
    assert "error" in result


def test_run_under_greenboost_mcp_wrapper_dry_run():
    result = gb_mcp.run_under_greenboost(["true"], confirm=False)
    assert result["gate"]["allowed"] is False


def test_optimize_inference_snapshot_uses_real_gbsnapshot_keys(monkeypatch):
    """Regression: optimize_inference's snapshot projection used to ask for
    "phase"/"pressure", keys that have never existed on GbSnapshot.as_dict()
    (the real keys are "shim_phase"/"swap_pressure"/"t2_pressure") , the
    `if k in snap` guard silently dropped both every time instead of raising,
    so result["snapshot"] was permanently missing phase/pressure. Verified
    live via the greenboost-orchestrator MCP server before this fix."""
    import gb_monitor

    class _FakeSnap:
        def as_dict(self):
            return {
                "loaded": True, "vram_physical_mb": 11264, "t2_pool_mb": 43008,
                "t2_allocated_mb": 0, "t3_used_mb": 0,
                "shim_phase": "SERVING", "swap_pressure": 2, "t2_pressure": 1,
            }

    monkeypatch.setattr(gb_monitor, "snapshot", lambda: _FakeSnap())
    result = gb_mcp.optimize_inference()
    snap = result["snapshot"]
    assert snap["shim_phase"] == "SERVING"
    assert snap["swap_pressure"] == 2
    assert snap["t2_pressure"] == 1
    assert "phase" not in snap
    assert "pressure" not in snap
