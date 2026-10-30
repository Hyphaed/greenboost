"""gb_sdpa: the priority-order policy (CPU only)."""
import gb_sdpa as sd


def test_auto_prefers_cudnn_only_on_consumer_blackwell():
    assert sd.order("auto", (12, 0)) == ["cudnn", "flash", "efficient", "math"]
    assert sd.order("auto", (9, 0)) is None and sd.order("auto", (8, 9)) is None and sd.order("auto", None) is None


def test_explicit_choices_and_off():
    assert sd.order("flash", (8, 9))[0] == "flash" and sd.order("cudnn", None)[0] == "cudnn"
    assert sd.order("efficient", None) == ["efficient", "flash", "cudnn", "math"]
    assert sd.order("off", (12, 0)) is None and sd.order("garbage", (12, 0)) is None


def test_apply_without_a_gpu_changes_nothing(monkeypatch):
    monkeypatch.setattr(sd, "_applied", None)
    monkeypatch.setenv("GB_ATTN_BACKEND", "off")
    assert sd.apply() is None
