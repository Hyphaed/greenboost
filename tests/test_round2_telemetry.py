"""Round-2 dataflux events (2026-10-30): sdpa_backend, compile_decision, engine_runtime_check, quant_conv_result."""
import pytest

import gb_dataflux


@pytest.fixture
def events(monkeypatch):
    got = []
    monkeypatch.setattr(gb_dataflux, "emit", lambda ev: got.append(ev))
    return got


def test_compile_decision_off_when_batch_small(events, monkeypatch):
    import gb_compile
    monkeypatch.delenv("GB_COMPILE", raising=False)
    assert gb_compile.apply(object(), n_images=2) is False
    e = events[-1]
    assert e["kind"] == "compile_decision" and e["compiled"] is False and e["n_images"] == 2 and e["status"] == "ok"


def test_sdpa_backend_event_when_off(events, monkeypatch):
    import gb_sdpa
    monkeypatch.setattr(gb_sdpa, "_applied", None)
    monkeypatch.setenv("GB_ATTN_BACKEND", "off")
    gb_sdpa.apply()
    e = [x for x in events if x["kind"] == "sdpa_backend"][-1]
    assert e["setting"] == "off" and e["order"] == ""


def test_engine_runtime_check_event(events):
    import gb_synapse
    gb_synapse._emit_engine_check("/x/bin", "13", "12", True)
    gb_synapse._emit_engine_check("/x/bin", "13", "13", True)
    assert [e["status"] for e in events] == ["warn", "ok"]
    assert gb_dataflux._is_incident(events[0]) and not gb_dataflux._is_incident(events[1])
    assert any("GB_SYNAPSE_NVCC" in h for h in gb_dataflux._diagnosis_hints(events[0], None, None, {}, []))


def test_quant_conv_result_event(events, monkeypatch):
    torch = pytest.importorskip("torch")
    import gb_quant_conv
    monkeypatch.setenv("GB_QUANT_CONV_MIN_TOTAL_MB", "0")
    monkeypatch.setenv("GB_QUANT_CONV_MIN_ELEMS", "16")
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.Conv2d(8, 16, 3), torch.nn.Conv2d(16, 16, 3), torch.nn.Conv2d(16, 4, 3))
    rep = gb_quant_conv.maybe_quantize_convs(m)
    e = [x for x in events if x["kind"] == "quant_conv_result"][-1]
    assert e["quantized"] == rep.quantized and e["bytes_after"] < e["bytes_before"]
