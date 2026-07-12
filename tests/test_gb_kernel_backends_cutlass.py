"""CPU-only routing tests for the Stage-A2 cutlass backend wiring.
gb_cutlass.available() is monkeypatched (the real extension is GPU + build +
GB_CUTLASS_ENABLE gated); these lock the resolve/supports/fallback contract.
"""
import gb_kernel_backends as kb


SM120 = (12, 0)
SM89 = (8, 9)


def test_cutlass_not_selected_when_unavailable(monkeypatch):
    monkeypatch.setattr(kb, "_cutlass_available", lambda: False)
    # explicit request for cutlass on nvfp4 falls back to gemlite when unbuilt
    assert kb.resolve_backend("nvfp4", 4096, 4096, SM120, env="cutlass") == "gemlite"


def test_cutlass_selected_when_available_on_sm120(monkeypatch):
    monkeypatch.setattr(kb, "_cutlass_available", lambda: True)
    assert kb.resolve_backend("nvfp4", 4096, 4096, SM120, env="cutlass") == "cutlass"


def test_cutlass_rejected_off_blackwell(monkeypatch):
    monkeypatch.setattr(kb, "_cutlass_available", lambda: True)
    # cc < 12 → supports() False → fall back
    assert kb.resolve_backend("nvfp4", 4096, 4096, SM89, env="cutlass") == "gemlite"


def test_cutlass_only_serves_nvfp4(monkeypatch):
    monkeypatch.setattr(kb, "_cutlass_available", lambda: True)
    # cutlass doesn't serve fp8/int4 — falls back
    assert kb.resolve_backend("fp8", 4096, 4096, SM120, env="cutlass") != "cutlass"
    assert kb.resolve_backend(4, 4096, 4096, SM120, env="cutlass") != "cutlass"


def test_supports_matrix(monkeypatch):
    monkeypatch.setattr(kb, "_cutlass_available", lambda: True)
    assert kb.supports("cutlass", "nvfp4", 4096, 4096, SM120) is True
    assert kb.supports("cutlass", "nvfp4", 4096, 4096, SM89) is False
    assert kb.supports("cutlass", "fp8", 4096, 4096, SM120) is False
    assert kb.supports("cutlass", "nvfp4", 4096, 4096, None) is False


def test_cutlass_processor_builder_stub_returns_none():
    # bench-gated: still a stub, gb_quant falls back to gemlite nvfp4 on None
    assert kb.build_cutlass_nvfp4_processor("nvfp4", "cuda", None) is None


def test_auto_never_picks_cutlass(monkeypatch):
    monkeypatch.setattr(kb, "_cutlass_available", lambda: True)
    # AUTO ships == gemlite; cutlass is opt-in via GB_KERNEL_BACKEND=cutlass
    assert kb.resolve_backend("nvfp4", 4096, 4096, SM120, env="auto") == "gemlite"
