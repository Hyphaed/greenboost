"""Tests for missing_features.md item (k): _measured_tok_s()/record_measured_
tok_s() must key samples by (model, quant, ctx, kv_type), not by model name
alone, and must reject physically-impossible samples before they enter the
rolling average.

Real incident this closes: measured_tok_s.json held both a legitimate
Q4_K_M-era history AND a single 18355.3 tok/s outlier for a 27B dense model
on a 12 GB card under one flat per-model key — recommend() blended both into
one meaningless average. CPU-only, no GGUF, no CUDA, no network.
"""
import json

import gb_synapse as gs


def _fake_entry(name="qwen35", quant="Q4_K_M", n_bytes=18 * 1024 ** 3, n_layers=65):
    return gs.ModelEntry(name=name, path="", quant=quant, n_bytes=n_bytes, n_layers=n_layers)


def test_tok_s_key_format():
    assert gs._tok_s_key("Q4_K_M", 65536, "q8_0") == "Q4_K_M::65536::q8_0"
    # missing quant/kv_type still produce a stable, non-empty key
    assert gs._tok_s_key("", 0, "") == "unknown-quant::0::"


def test_load_tok_s_samples_migrates_legacy_flat_list(tmp_path, monkeypatch):
    legacy_file = tmp_path / "measured_tok_s.json"
    legacy_file.write_text(json.dumps({"qwen35": [3.2, 3.4, 6.1]}))
    monkeypatch.setattr(gs, "MEASURED_TOK_S_FILE", legacy_file)

    samples = gs._load_tok_s_samples()

    assert samples == {"qwen35": {gs._TOK_S_LEGACY_KEY: [3.2, 3.4, 6.1]}}


def test_record_measured_tok_s_separates_quant_swaps(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "MEASURED_TOK_S_FILE", tmp_path / "measured_tok_s.json")
    monkeypatch.setattr(gs, "_tok_s_sanity_ceiling", lambda model: None)
    monkeypatch.setattr(gs, "_df_emit_tok_s", lambda *a, **kw: None)

    gs.record_measured_tok_s("qwen35", 5.0, quant="Q4_K_M", ctx=65536, kv_type="q8_0")
    gs.record_measured_tok_s("qwen35", 5.5, quant="Q4_K_M", ctx=65536, kv_type="q8_0")
    gs.record_measured_tok_s("qwen35", 12.0, quant="IQ4_XS", ctx=32768, kv_type="f16")

    q4_avg = gs._measured_tok_s("qwen35", quant="Q4_K_M", ctx=65536)
    iq4_avg = gs._measured_tok_s("qwen35", quant="IQ4_XS", ctx=32768)

    assert q4_avg == 5.2  # round(5.25, 1) banker-rounds to 5.2
    assert iq4_avg == 12.0
    # the two configurations' histories must never blend into one average
    assert q4_avg != iq4_avg


def test_record_measured_tok_s_falls_back_to_run_state_when_config_omitted(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "MEASURED_TOK_S_FILE", tmp_path / "measured_tok_s.json")
    monkeypatch.setattr(gs, "_tok_s_sanity_ceiling", lambda model: None)
    monkeypatch.setattr(gs, "_df_emit_tok_s", lambda *a, **kw: None)
    monkeypatch.setattr(
        gs, "_read_run_state",
        lambda model: gs.ServerState(model=model, llama_pid=1, proxy_pid=1, port=0,
                                     internal_port=0, tensor_split="", quant="Q6_K",
                                     ctx=16384, kv_type="f16"))

    gs.record_measured_tok_s("qwen35", 7.0, source="proxy")  # no quant/ctx/kv_type passed

    assert gs._measured_tok_s("qwen35", quant="Q6_K", ctx=16384) == 7.0


def test_record_measured_tok_s_drops_sample_without_run_state(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "MEASURED_TOK_S_FILE", tmp_path / "measured_tok_s.json")
    monkeypatch.setattr(gs, "_read_run_state", lambda model: None)

    gs.record_measured_tok_s("qwen35", 7.0, source="proxy")  # no config, no run-state to find it

    assert gs._measured_tok_s("qwen35") is None


def test_record_measured_tok_s_rejects_sample_above_sanity_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "MEASURED_TOK_S_FILE", tmp_path / "measured_tok_s.json")
    monkeypatch.setattr(gs, "_tok_s_sanity_ceiling", lambda model: 500.0)
    emitted = []
    monkeypatch.setattr(gs, "_df_emit_tok_s", lambda *a, **kw: emitted.append((a, kw)))

    # the real incident: 18355.3 tok/s for a 27B dense model on a 12 GB card
    gs.record_measured_tok_s("qwen35", 18355.3, quant="IQ2_M", ctx=32768, kv_type="f16")

    assert gs._measured_tok_s("qwen35", quant="IQ2_M", ctx=32768) is None
    assert emitted == []  # the impossible sample never reaches the dataflux tok_s stream either


def test_tok_s_sanity_ceiling_derives_from_node_bandwidth_and_model_layer_size(monkeypatch):
    monkeypatch.setattr(gs, "_load_manifest", lambda: {"qwen35": _fake_entry()})
    monkeypatch.setattr("gb_topology._detect_vram_bw_gb_s", lambda: 672.0)

    ceiling = gs._tok_s_sanity_ceiling("qwen35")

    # bytes_per_layer = 18 GiB / 65 ≈ 296.9 MB; ceiling = 672e9 / 296.9e6 ≈ 2264 tok/s
    assert ceiling is not None
    assert 2000 < ceiling < 2600


def test_tok_s_sanity_ceiling_none_when_bandwidth_undetectable(monkeypatch):
    monkeypatch.setattr(gs, "_load_manifest", lambda: {"qwen35": _fake_entry()})
    monkeypatch.setattr("gb_topology._detect_vram_bw_gb_s", lambda: 0.0)

    assert gs._tok_s_sanity_ceiling("qwen35") is None


def test_measured_tok_s_coarse_rollup_when_no_config_given(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "MEASURED_TOK_S_FILE", tmp_path / "measured_tok_s.json")
    monkeypatch.setattr(gs, "_tok_s_sanity_ceiling", lambda model: None)
    monkeypatch.setattr(gs, "_df_emit_tok_s", lambda *a, **kw: None)

    gs.record_measured_tok_s("qwen35", 4.0, quant="Q4_K_M", ctx=65536, kv_type="q8_0")
    gs.record_measured_tok_s("qwen35", 8.0, quant="IQ4_XS", ctx=32768, kv_type="f16")

    # display-only rollup: blends across configs on purpose, unlike the keyed query above
    assert gs._measured_tok_s("qwen35") == 6.0
