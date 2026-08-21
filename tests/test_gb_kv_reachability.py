"""Does lookahead KV prefetch have anything to prefetch?

The gate on that whole branch. Measured on this box 2026-08-20 while serving
the reference model at ctx 24576: KV sat entirely in T1 (427 MB tracked) and
the 6687 MB in T2 was all weights , a prefetch mechanism there would move zero
bytes. These tests pin the two things that make the answer trustworthy: a
measured per-token figure is preferred over the formula (which is known ~2.9x
high on this hybrid architecture), and a stale shim stats file is never read as
current state.
"""
from __future__ import annotations

import os
import time

import gb_synapse


def test_a_prior_measurement_is_preferred_over_the_formula(monkeypatch):
    monkeypatch.setattr(gb_synapse, "_read_run_state", lambda m: None)
    monkeypatch.setattr(gb_synapse, "_load_kv_measurement",
                        lambda m, c, k: 427.0)
    out = gb_synapse.kv_spill_reachability("m", 24576, "q4_0")
    assert out["source"].startswith("measured")
    assert abs(out["kv_bytes_per_token"] - (427 * 1024 * 1024) / 24576) < 1


def test_the_formula_is_labelled_as_an_estimate(monkeypatch):
    monkeypatch.setattr(gb_synapse, "_read_run_state", lambda m: None)
    monkeypatch.setattr(gb_synapse, "_load_kv_measurement", lambda m, c, k: None)

    class _E:
        name = "m"
        quant = "IQ4_XS"
        n_bytes = 17 * 1024 ** 3
        n_layers = 64
        n_kv_layers = 16
        n_kv_heads = 8
        head_dim = 128

    monkeypatch.setattr(gb_synapse, "list_models", lambda *a, **kw: [_E()])
    out = gb_synapse.kv_spill_reachability("m", 65536, "q4_0")
    assert out["source"].startswith("estimated")
    assert out["kv_bytes_per_token"] > 0


def test_it_says_so_when_it_cannot_size_the_model(monkeypatch):
    """Silence or a zero here would read as 'KV is tiny', which is the
    opposite of 'I could not tell'."""
    monkeypatch.setattr(gb_synapse, "_read_run_state", lambda m: None)
    monkeypatch.setattr(gb_synapse, "_load_kv_measurement", lambda m, c, k: None)
    monkeypatch.setattr(gb_synapse, "list_models", lambda *a, **kw: [])
    out = gb_synapse.kv_spill_reachability("nope", 4096, "f16")
    assert out["kv_bytes_per_token"] is None
    assert "no manifest entry" in out["note"]


def test_a_stale_stats_file_is_not_read_as_the_current_reserve(monkeypatch, tmp_path):
    """The file outlives the process that wrote it , reading it anyway sizes
    this serve against the previous one's reserve."""
    monkeypatch.setattr(gb_synapse, "_read_run_state", lambda m: None)
    monkeypatch.setattr(gb_synapse, "_load_kv_measurement", lambda m, c, k: 427.0)
    real_getmtime = os.path.getmtime
    monkeypatch.setattr(os.path, "getmtime",
                        lambda p: (time.time() - 3600
                                   if str(p).endswith("shim_stats")
                                   else real_getmtime(p)))
    out = gb_synapse.kv_spill_reachability("m", 24576, "q4_0")
    assert out["reserve_mb"] is None
    assert out["spill_ctx"] is None
    assert "stale" in out["note"]
