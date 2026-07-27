#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_api.py (P7) — the public facade for ai-forge and other
external consumers. Every function is a thin delegate to the module that
actually owns the data/decision, so these tests verify the delegation
wiring (right module, right function, right params), not re-test the
underlying logic (already covered by test_gb_api_helpers.py,
test_gb_actuation_run_capped.py, test_gb_dataflux_stage.py,
test_gb_synapse_api_facade_helpers.py).
"""
import ast
from pathlib import Path

import pytest

import gb_api


def test_module_has_no_torch_import_at_top_level():
    """Ground rule: importing the backend must never touch the GPU. Every
    gb_api function must lazily `import gb_*` inside its own body."""
    tree = ast.parse(Path(gb_api.__file__).read_text())
    top_level_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = set()
    for node in top_level_imports:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        else:
            if node.module:
                names.add(node.module.split(".")[0])
    assert "torch" not in names
    assert "gb_quant" not in names
    assert "gb_synapse" not in names


def test_workload_env_delegates_to_shim_env(monkeypatch):
    import gb_cluster
    captured = {}

    def _fake_shim_env(workload, enabled=True, cudart_path=None, base_env=None):
        captured.update(workload=workload, enabled=enabled, cudart_path=cudart_path)
        return {"GREENBOOST_ACTIVE": "1"}
    monkeypatch.setattr(gb_cluster, "shim_env", _fake_shim_env)
    monkeypatch.setattr(gb_api, "cudart_for", lambda python: "/fake/libcudart.so.12")

    env = gb_api.workload_env("diffusion", python="/venv/bin/python")

    assert env == {"GREENBOOST_ACTIVE": "1"}
    assert captured["workload"] == "diffusion"
    assert captured["cudart_path"] == "/fake/libcudart.so.12"


def test_workload_env_no_cudart_lookup_without_python(monkeypatch):
    import gb_cluster
    captured = {}
    monkeypatch.setattr(gb_cluster, "shim_env",
                        lambda workload, enabled=True, cudart_path=None, base_env=None:
                        captured.update(cudart_path=cudart_path) or {})
    gb_api.workload_env("diffusion")
    assert captured["cudart_path"] is None


def test_workload_env_extra_merges_last(monkeypatch):
    import gb_cluster
    monkeypatch.setattr(gb_cluster, "shim_env",
                        lambda workload, enabled=True, cudart_path=None, base_env=None:
                        {"A": "1"})
    env = gb_api.workload_env("diffusion", extra={"A": "override", "B": "2"})
    assert env == {"A": "override", "B": "2"}


def test_cudart_for_delegates(monkeypatch):
    import gb_cluster
    monkeypatch.setattr(gb_cluster, "cudart_for", lambda py: f"resolved:{py}")
    assert gb_api.cudart_for("/venv/bin/python") == "resolved:/venv/bin/python"


def test_t2_pool_delegates(monkeypatch):
    import gb_tiering
    monkeypatch.setattr(gb_tiering, "t2_pool", lambda: {"total_mb": 1})
    assert gb_api.t2_pool() == {"total_mb": 1}


def test_gpu_state_delegates(monkeypatch):
    import gb_monitor
    monkeypatch.setattr(gb_monitor, "gpu_state", lambda: {"free_mb": 5000})
    assert gb_api.gpu_state() == {"free_mb": 5000}


def test_wait_for_vram_delegates(monkeypatch):
    import gb_monitor
    captured = {}
    monkeypatch.setattr(gb_monitor, "wait_for_vram",
                        lambda free_mb, timeout_s=60.0, poll_s=1.0:
                        captured.update(free_mb=free_mb, timeout_s=timeout_s) or True)
    assert gb_api.wait_for_vram(2000, timeout_s=30) is True
    assert captured == {"free_mb": 2000, "timeout_s": 30}


def test_wait_ready_delegates(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "wait_ready",
                        lambda url, path="/health", timeout_s=120.0, attempts=None: True)
    assert gb_api.wait_ready("http://x") is True


def test_pull_model_delegates(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "pull_model", lambda name, progress=None: f"entry:{name}")
    assert gb_api.pull_model("org/repo") == "entry:org/repo"


def test_run_capped_delegates(monkeypatch):
    import gb_actuation
    captured = {}
    monkeypatch.setattr(gb_actuation, "run_capped",
                        lambda argv, **kw: captured.update(argv=argv, **kw) or {"returncode": 0})
    result = gb_api.run_capped(["true"], mem_max_mb=1000)
    assert result == {"returncode": 0}
    assert captured["argv"] == ["true"]
    assert captured["mem_max_mb"] == 1000


def test_stage_delegates_and_is_usable_as_context_manager(monkeypatch, tmp_path):
    import gb_dataflux
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(tmp_path / "dataflux.jsonl"))
    gb_dataflux._READ_EVENTS_MEMO.clear()

    with gb_api.stage("test:stage", label="x"):
        pass

    events = [e for e in gb_dataflux.read_events() if e.get("kind") == "stage_profile"]
    assert len(events) == 1
    assert events[0]["stage"] == "test:stage"


def test_serve_gguf_delegates(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "serve_gguf",
                        lambda path, port, mmproj=None, ctx=0: {"pid": 1, "port": port})
    assert gb_api.serve_gguf("/x.gguf", 8081) == {"pid": 1, "port": 8081}


def test_endpoints_delegates(monkeypatch):
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "endpoints", lambda: {"FORGE_OLLAMA_URL": "http://x"})
    assert gb_api.endpoints() == {"FORGE_OLLAMA_URL": "http://x"}


def test_cluster_returns_the_real_module():
    import gb_cluster
    assert gb_api.cluster() is gb_cluster


def test_ssh_opts_delegates(monkeypatch):
    import gb_cluster
    monkeypatch.setattr(gb_cluster, "_ssh_opts",
                        lambda connect_timeout=10, compress=False: ["-o", f"ConnectTimeout={connect_timeout}"])
    assert gb_api.ssh_opts(connect_timeout=5) == ["-o", "ConnectTimeout=5"]
