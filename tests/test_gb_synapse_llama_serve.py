#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for LlamaCppBackend.serve()'s CPU-offload placement decision — the
owner rule (CLAUDE.md, 2026-08-01): weights that exceed VRAM must spill
through the shim to T2, never fall back to CPU offload.

Covers the two changes that make that rule real:
  1. `_gate_cpu_offload()` — raises by default whenever serve() is about to
     reduce -ngl as a capacity decision; only proceeds under the explicit
     GB_SYNAPSE_ALLOW_CPU_OFFLOAD=1 debugging override.
  2. The T2 KV/weights budget extension — ctx must no longer collapse to
     _clamp_ctx_to_budget's floor the instant the shim flips `fits_vram`
     true via "LD_PRELOAD" in env (the exact 2026-08-01 incident: shim
     active, ctx_kv_budget_gb stayed VRAM-only, ctx clamped to ~7680).

No real subprocess/GPU: subprocess.Popen, NVML, the feeder probe, and
gb_shim_probe are all monkeypatched, matching
test_gb_synapse_backends_embeddings.py's pattern for serve_embedding().
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_synapse_backends as gsb


class _FakeProc:
    pid = 4321

    def poll(self):
        return None


class _FakeNvml:
    def __init__(self, free_mb):
        self._free_mb = free_mb

    def mem(self):
        return (0, self._free_mb, self._free_mb * 1.1, 0.0)


@pytest.fixture
def backend():
    return gsb.LlamaCppBackend()


@pytest.fixture
def entry():
    """Scaled to the reference-workload incident: ~14.09 GB weights, 65
    dense (non-MoE, non-recurrent) layers — the exact shape that hit the
    2026-08-01 ctx-collapse bug. arch="" bypasses the engine-support and
    ARCH_CPU_SPLIT_BROKEN checks (neither is what this test exercises)."""
    import gb_synapse as gs
    return gs.ModelEntry(
        name="fable-fusion-test", path="/models/fable-fusion.gguf",
        arch="", quant="Q4_K_M", ctx_length=65536,
        n_bytes=int(14.09 * 1024 ** 3), n_layers=65,
        n_kv_heads=8, head_dim=128, is_moe=False,
    )


@pytest.fixture
def _stub_gb_synapse(monkeypatch, tmp_path):
    import gb_synapse as gs
    monkeypatch.setattr(gs, "engine_installed", lambda: True)
    monkeypatch.setattr(gs, "ENGINE_DIR", tmp_path / "engine")
    monkeypatch.setattr(gs, "SLOT_DIR", tmp_path / "slots")
    monkeypatch.setattr(gs, "_pcore_threads", lambda: 4)
    monkeypatch.setattr(gs, "_run_log_path", lambda label: tmp_path / f"{label}.log")
    monkeypatch.setattr(gs, "_read_ram_available_mb", lambda: 20000)
    monkeypatch.setattr(gs, "_find_mmproj", lambda entry: None)
    monkeypatch.setattr(gs, "_has_mtp", lambda entry: False)
    monkeypatch.setattr(gs, "_engine_supported_archs", lambda: set())
    captured = {}

    def _fake_launch(entry, upstream, port, internal_port, **kw):
        captured["launch_kwargs"] = kw
        return upstream, port

    monkeypatch.setattr(gs, "_launch_proxy_and_record", _fake_launch)
    return captured


@pytest.fixture
def _stub_gb_cluster(monkeypatch):
    import gb_cluster
    monkeypatch.setattr(gb_cluster, "feeders", lambda probe=True, **kw: [])


@pytest.fixture
def _stub_popen(monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env")
        return _FakeProc()

    monkeypatch.setattr(gsb.subprocess, "Popen", _fake_popen)
    return captured


def _shim_env_without_preload(**_kw):
    return {"GREENBOOST_ACTIVE": "1"}


def _shim_env_with_preload(**_kw):
    return {"GREENBOOST_ACTIVE": "1", "LD_PRELOAD": "/fake/libgreenboost_cuda.so"}


@pytest.fixture
def _no_shim(monkeypatch):
    """Shim unreachable — same shape as the 2026-08-01 incident before the
    193-vs-102 fix: probe reports failure, LD_PRELOAD never survives."""
    import gb_cluster
    import gb_shim_probe
    monkeypatch.setattr(gb_cluster, "shim_env",
                        lambda workload=None, enabled=None: _shim_env_without_preload())
    monkeypatch.setattr(gb_shim_probe, "shim_works_for_llama",
                        lambda engine_dir=None: (False, "reproduced known failure"))
    monkeypatch.setattr(gsb, "effective_vram_budget_mb",
                        lambda: (10500.0, 10500.0, {"t2_free_mb": 0.0, "t2_fraction": 0.5}))


@pytest.fixture
def _shim_active(monkeypatch):
    """Shim reachable and the probe passes — post-fix state."""
    import gb_cluster
    import gb_shim_probe
    monkeypatch.setattr(gb_cluster, "shim_env",
                        lambda workload=None, enabled=None: _shim_env_with_preload())
    monkeypatch.setattr(gb_shim_probe, "shim_works_for_llama",
                        lambda engine_dir=None: (True, "probe passed"))
    # 42 GB T2 pool free, 50% fraction — matches this box's real pool_brief
    # at the time of the incident (T2:0/42GB).
    monkeypatch.setattr(gsb, "effective_vram_budget_mb",
                        lambda: (10500.0, 31500.0, {"t2_free_mb": 43008.0, "t2_fraction": 0.5}))


# ── "never CPU offload" gate ───────────────────────────────────────────────

def test_dense_partial_offload_raises_by_default(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _no_shim, monkeypatch):
    """Weights (14.09 GB) exceed the VRAM-only budget (10.5 GB) and the shim
    is off — this is exactly the incident's shape. Must refuse, not silently
    degrade to PARTIAL CPU OFFLOAD."""
    monkeypatch.delenv("GB_SYNAPSE_ALLOW_CPU_OFFLOAD", raising=False)

    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    with pytest.raises(RuntimeError, match="CPU offload"):
        backend.serve(entry, port=11435)

    # Popen must never have been reached — the gate fires before spawning.
    assert "cmd" not in _stub_popen


def test_dense_partial_offload_serves_under_explicit_override(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _no_shim, monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_ALLOW_CPU_OFFLOAD", "1")

    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435)

    assert "cmd" in _stub_popen
    assert _stub_gb_synapse["launch_kwargs"]["placement"] == "PARTIAL CPU OFFLOAD (slow)"


def test_gate_cpu_offload_emits_dataflux_event_on_refusal(monkeypatch):
    import gb_synapse as gs
    entry = gs.ModelEntry(name="x", path="/x", n_bytes=1, n_layers=1)
    monkeypatch.delenv("GB_SYNAPSE_ALLOW_CPU_OFFLOAD", raising=False)
    monkeypatch.setattr(gsb, "effective_vram_budget_mb",
                        lambda: (0.0, 0.0, {"t2_free_mb": 100.0, "t2_fraction": 0.5}))
    emitted = {}
    monkeypatch.setitem(sys.modules, "gb_dataflux", type(sys)("gb_dataflux"))
    sys.modules["gb_dataflux"].emit = lambda event: emitted.update(event)

    with pytest.raises(RuntimeError):
        gsb._gate_cpu_offload(entry=entry, engine="llama.cpp", event_status="dense_partial_offload",
                              shim_active=False, shim_reason="off (test)", extra={"n_layers": 1})

    assert emitted["kind"] == "cpu_spillover"
    assert emitted["status"] == "dense_partial_offload"
    assert emitted["allowed_override"] is False


def test_gate_cpu_offload_override_skips_raise(monkeypatch):
    import gb_synapse as gs
    entry = gs.ModelEntry(name="x", path="/x", n_bytes=1, n_layers=1)
    monkeypatch.setenv("GB_SYNAPSE_ALLOW_CPU_OFFLOAD", "1")
    monkeypatch.setattr(gsb, "effective_vram_budget_mb",
                        lambda: (0.0, 0.0, {"t2_free_mb": 100.0, "t2_fraction": 0.5}))

    gsb._gate_cpu_offload(entry=entry, engine="llama.cpp", event_status="dense_partial_offload",
                          shim_active=False, shim_reason="off (test)", extra={})
    # No exception — reaching here is the assertion.


# ── T2 KV/weights budget extension ─────────────────────────────────────────

def test_shim_active_extends_ctx_past_vram_only_floor(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    """The exact 2026-08-01 regression: weights (14.09 GB) exceed physical
    VRAM (10.5 GB), the shim is active (LD_PRELOAD survives the probe), so
    fits_vram flips true — but ctx_kv_budget_gb used to stay VRAM-only,
    collapsing ctx to _clamp_ctx_to_budget's 0.25 GiB floor (~7680 tokens,
    the number reported live). With the T2 share folded in, ctx must clear
    that floor by a wide margin."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435)

    assert "cmd" in _stub_popen
    kwargs = _stub_gb_synapse["launch_kwargs"]
    assert kwargs["placement"] == "all-GPU"
    assert kwargs["ctx"] > 7680 * 2, (
        f"ctx={kwargs['ctx']} did not clear the pre-fix floor — T2 budget "
        f"extension did not reach _clamp_ctx_to_budget")


def test_shim_inactive_dense_offload_gate_cites_real_shim_reason(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _no_shim, monkeypatch):
    """The refusal message must carry the ACTUAL probe reason, not a generic
    string — this is what makes 'fix the shim' actionable instead of a dead
    end (Observability Must-Rule: a refusal must be as followable as a serve)."""
    monkeypatch.delenv("GB_SYNAPSE_ALLOW_CPU_OFFLOAD", raising=False)
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    with pytest.raises(RuntimeError, match="reproduced known failure"):
        backend.serve(entry, port=11435)


# ── MTP / slot-reuse flags actually reach llama-server (2026-08-10) ────────
# gb_synapse.serve() has passed mtp_draft_n/spec_draft_p_min/
# slot_prompt_similarity down to backend.serve() since 2026-08-05, but until
# this fix LlamaCppBackend never turned them into cmd-line flags — the
# constants (gs.MTP_DRAFT_N, gs.MTP_P_MIN, gs.GB_SLOT_PROMPT_SIMILARITY)
# existed and were documented as tuned, but nothing read them.

def _cmd_value_after(cmd: list, flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_slot_prompt_similarity_always_emitted(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    """Unconditional — governs slot reuse for every serve, not just MTP
    ones, so it must appear even with no MTP head."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435)

    cmd = _stub_popen["cmd"]
    assert "--slot-prompt-similarity" in cmd
    assert _cmd_value_after(cmd, "--slot-prompt-similarity") == "0.5"  # GB_SLOT_PROMPT_SIMILARITY default


def test_slot_prompt_similarity_per_call_override_wins(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435, slot_prompt_similarity=0.9)

    assert _cmd_value_after(_stub_popen["cmd"], "--slot-prompt-similarity") == "0.9"


def test_mtp_flags_emitted_when_active(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    """mtp_active requires _has_mtp(entry) True and fits_vram/is_moe —
    _shim_active gives the all-GPU placement (fits_vram=True)."""
    import gb_synapse as gs
    monkeypatch.setattr(gs, "_has_mtp", lambda entry: True)
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435)

    cmd = _stub_popen["cmd"]
    assert _cmd_value_after(cmd, "--spec-draft-n-max") == "4"   # MTP_DRAFT_N default
    assert _cmd_value_after(cmd, "--spec-draft-p-min") == "0.3"  # MTP_P_MIN default


def test_mtp_flags_per_call_override_wins(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    import gb_synapse as gs
    monkeypatch.setattr(gs, "_has_mtp", lambda entry: True)
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435, mtp_draft_n=6, spec_draft_p_min=0.1)

    cmd = _stub_popen["cmd"]
    assert _cmd_value_after(cmd, "--spec-draft-n-max") == "6"
    assert _cmd_value_after(cmd, "--spec-draft-p-min") == "0.1"


def test_mtp_flags_absent_when_no_mtp_head(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    """_stub_gb_synapse's default _has_mtp stub is False — the common case
    (no MTP head) must not pass --spec-type/--spec-draft-* at all."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435)

    cmd = _stub_popen["cmd"]
    assert "--spec-type" not in cmd
    assert "--spec-draft-n-max" not in cmd


# ── KV type override + extra_args precedence (2026-08-10 M1 fix) ───────────

def test_kv_type_override_reaches_cache_type_flags(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    """kv_type_override wins over _pick_kv_type()'s own budget-driven choice."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435, kv_type_override="q4_0")

    cmd = _stub_popen["cmd"]
    assert _cmd_value_after(cmd, "--cache-type-k") == "q4_0"
    assert _cmd_value_after(cmd, "--cache-type-v") == "q4_0"


def test_extra_args_cache_type_override_wins_over_default(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    """Regression for the 2026-08-03 M1 measurement-pass bug: an explicit
    --cache-type-k/-v in extra_args used to be silently ignored because
    LlamaCppBackend.serve() already emitted its own --cache-type-k/-v
    earlier in cmd, and llama.cpp's arg parser keeps the FIRST occurrence
    of a duplicate flag. _dedupe_extra_args_overrides() must strip the
    built-in pair before extra_args is appended, so the caller's value
    appears exactly once."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435, extra_args="--cache-type-k f16 --cache-type-v f16")

    cmd = _stub_popen["cmd"]
    assert cmd.count("--cache-type-k") == 1
    assert cmd.count("--cache-type-v") == 1
    assert _cmd_value_after(cmd, "--cache-type-k") == "f16"
    assert _cmd_value_after(cmd, "--cache-type-v") == "f16"


def test_extra_args_leaves_unrelated_flags_untouched(
        backend, entry, _stub_gb_synapse, _stub_gb_cluster, _stub_popen, _shim_active, monkeypatch):
    """_dedupe_extra_args_overrides() must only remove flags extra_args
    itself supplies — every other built-in flag (boolean or valued) stays
    exactly as emitted."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda *_a, **_k: _FakeNvml(10500.0))

    backend.serve(entry, port=11435, extra_args="--cache-type-k q8_0")

    cmd = _stub_popen["cmd"]
    assert cmd.count("--cache-type-k") == 1
    assert _cmd_value_after(cmd, "--cache-type-k") == "q8_0"
    # --cache-type-v (not named in extra_args) keeps its own built-in value
    assert cmd.count("--cache-type-v") == 1
    assert "--jinja" in cmd
    assert "--no-webui" in cmd
