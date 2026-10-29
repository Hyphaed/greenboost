# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""shim_env / gb_api.cpu_affinity: P-core pinning derived from the node's own topology (hybrid CPUs only).

Measured 2026-10-28 on an i9-14900KF (P-cores 0-15, E-cores 16-31): the model-load / quantise phase of a diffusion batch is
single-thread bound; letting the scheduler park it on an E-core costs minutes. The list comes from topology, never a literal.
"""
from types import SimpleNamespace

import pytest

import gb_cluster as gc
import gb_topology


def _topo(monkeypatch, p, e):
    monkeypatch.setattr(gb_topology, "get_topology", lambda: SimpleNamespace(p_core_cpus=p, e_core_cpus=e))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("GB_CPU_PIN", raising=False)


def test_hybrid_cpu_gets_the_p_cores_and_a_matching_thread_count(monkeypatch):
    _topo(monkeypatch, list(range(0, 16)), list(range(16, 32)))
    assert gc.cpu_affinity("diffusion") == {"cpus": list(range(16)), "cpulist": "0-15", "omp_num_threads": 16}


def test_a_non_hybrid_cpu_has_nothing_to_pin(monkeypatch):
    _topo(monkeypatch, list(range(8)), [])
    assert gc.cpu_affinity("diffusion") == {}


def test_unknown_topology_pins_nothing(monkeypatch):
    _topo(monkeypatch, [], [16, 17])
    assert gc.cpu_affinity("diffusion") == {}
    monkeypatch.setattr(gb_topology, "get_topology", lambda: (_ for _ in ()).throw(RuntimeError("no profile")))
    assert gc.cpu_affinity("diffusion") == {}


def test_llm_engines_size_their_own_threads(monkeypatch):
    _topo(monkeypatch, list(range(16)), list(range(16, 32)))
    assert gc.cpu_affinity("llm") == {} and gc.cpu_affinity("ggml") == {}


def test_the_owner_can_opt_out(monkeypatch):
    _topo(monkeypatch, list(range(16)), list(range(16, 32)))
    monkeypatch.setenv("GB_CPU_PIN", "all")
    assert gc.cpu_affinity("diffusion") == {}


def test_cpulist_format_matches_taskset():
    assert gc._cpulist([0, 1, 2, 3, 8, 9, 12]) == "0-3,8-9,12"
    assert gc._cpulist([5]) == "5" and gc._cpulist([]) == "" and gc._cpulist([3, 1, 2, 2]) == "1-3"


def test_shim_env_carries_the_decision_even_without_the_shim(monkeypatch):
    _topo(monkeypatch, list(range(16)), list(range(16, 32)))
    env = gc.shim_env("diffusion", enabled=False, base_env={})
    assert env["GB_CPU_AFFINITY"] == "0-15" and env["OMP_NUM_THREADS"] == "16"


def test_a_callers_own_thread_count_wins(monkeypatch):
    _topo(monkeypatch, list(range(16)), list(range(16, 32)))
    env = gc.shim_env("diffusion", enabled=False, base_env={"OMP_NUM_THREADS": "4"})
    assert env["OMP_NUM_THREADS"] == "4" and env["GB_CPU_AFFINITY"] == "0-15"


def test_llm_env_has_no_pinning_keys(monkeypatch):
    _topo(monkeypatch, list(range(16)), list(range(16, 32)))
    env = gc.shim_env("llm", enabled=False, base_env={})
    assert "GB_CPU_AFFINITY" not in env and "OMP_NUM_THREADS" not in env


def test_the_facade_delegates(monkeypatch):
    import gb_api
    _topo(monkeypatch, list(range(16)), list(range(16, 32)))
    assert gb_api.cpu_affinity("diffusion")["cpulist"] == "0-15"
