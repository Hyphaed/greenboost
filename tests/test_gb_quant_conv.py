"""gb_quant_conv: fp8 weight-only conv compression. CPU only (torch CPU has float8_e4m3fn); no CUDA, no model download."""
import pytest
import torch
import torch.nn as nn

import gb_quant_conv as qc


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)          # per test: the outcome must not depend on test order (fp8 error accumulates a few percent over layers, so the bounds below have headroom)



def _rel(a, b):
    return float((a - b).norm() / b.norm())


def test_weight_roundtrip_is_channelwise_and_close():
    w = torch.randn(32, 16, 3, 3) * torch.linspace(0.1, 3.0, 32).view(-1, 1, 1, 1)   # channels with very different ranges
    w8, s = qc.quantize_weight(w)
    assert w8.dtype == torch.float8_e4m3fn and s.shape == (32, 1, 1, 1)
    assert _rel(qc.dequantize_weight(w8, s, torch.float32), w) < 0.045
    assert qc.rel_error(w, w8, s) < 0.045


def test_transposed_weight_scales_per_output_channel():
    w = torch.randn(16, 8, 4, 4)
    w8, s = qc.quantize_weight(w, transposed=True)
    assert s.shape == (1, 8, 1, 1)


def _net():
    return nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1, dilation=1),
                         nn.ReLU(), nn.Conv2d(64, 3, 3, padding=1))


def test_quantized_net_matches_original_and_keeps_first_last():
    net = _net().eval()
    x = torch.randn(1, 3, 24, 24)
    with torch.no_grad():
        ref = net(x)
    ids = [id(m) for m in net]
    rep = qc.quantize_convs(net)
    assert [id(m) for m in net] == ids                      # same objects, patched in place
    assert rep.quantized == 2 and set(rep.kept) == {"0", "6"} and "first/last" in rep.kept["0"]
    assert hasattr(net[2], "weight_fp8") and not hasattr(net[0], "weight_fp8")
    assert rep.bytes_after < rep.bytes_before / 3.5          # fp32 -> fp8 (+ scales)
    with torch.no_grad():
        out = net(x)
    assert _rel(out, ref) < 0.08 and torch.isfinite(out).all()
    assert rep.max_rel_err < 0.045


@pytest.mark.parametrize("make,shape", [
    (lambda: nn.Conv1d(64, 64, 5, padding=2), (2, 64, 40)),
    (lambda: nn.Conv3d(16, 32, 3, padding=1), (1, 16, 5, 12, 12)),
    (lambda: nn.Conv2d(64, 128, 3, stride=2, padding=2, dilation=2, groups=2), (1, 64, 20, 20)),
    (lambda: nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), (1, 64, 16, 16)),
    (lambda: nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), (1, 64, 10, 10)),
    (lambda: nn.ConvTranspose3d(16, 16, 3, stride=2, padding=1, output_padding=1), (1, 16, 4, 6, 6)),
])
def test_every_conv_kind_matches(make, shape):
    m = make().eval()
    wrap = nn.Sequential(nn.Conv2d(1, 1, 1), m, nn.Conv2d(1, 1, 1)) if False else m
    x = torch.randn(*shape)
    with torch.no_grad():
        ref = m(x)
    rep = qc.quantize_convs(nn.ModuleList([nn.Identity(), m, nn.Identity()]), keep_first_last=False, min_elems=1)
    assert rep.quantized == 1, rep.kept
    with torch.no_grad():
        out = m(x)
    assert out.shape == ref.shape and _rel(out, ref) < 0.08


def test_transposed_output_size_argument():
    m = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1).eval()
    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        ref = m(x, output_size=(16, 16))
    qc.quantize_convs(nn.ModuleList([m]), keep_first_last=False, min_elems=1)
    with torch.no_grad():
        assert m(x, output_size=(16, 16)).shape == ref.shape


def test_guards():
    net = nn.Sequential(nn.Conv2d(3, 64, 3), nn.Conv2d(64, 64, 3), nn.Conv2d(64, 4, 1), nn.Conv2d(4, 4, 3), nn.Conv2d(4, 4, 3))
    rep = qc.quantize_convs(net, min_elems=20000)
    assert rep.quantized == 1 and "small" in rep.kept["2"]
    assert qc.quantize_convs(net, min_elems=20000).kept["1"] == "already fp8"            # idempotent
    tight = nn.Sequential(nn.Conv2d(3, 64, 3), nn.Conv2d(64, 64, 3), nn.Conv2d(64, 3, 3))
    r2 = qc.quantize_convs(tight, max_rel_err=1e-4, min_elems=1)
    assert r2.quantized == 0 and "over the" in r2.kept["1"] and hasattr(tight[1], "weight")      # layer over the error limit stays as it was
    named = nn.ModuleDict({"a": nn.Conv2d(3, 64, 3), "pos_embed": nn.Conv2d(64, 64, 3), "mid_norm_conv": nn.Conv2d(64, 64, 3), "z": nn.Conv2d(64, 3, 3)})
    r3 = qc.quantize_convs(named, min_elems=1)
    assert r3.kept["pos_embed"] == "skip list" and r3.kept["mid_norm_conv"] == "skip list"


def test_half_precision_input_uses_input_dtype():
    m = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.Conv2d(64, 64, 3, padding=1), nn.Conv2d(64, 3, 3, padding=1)).eval()
    qc.quantize_convs(m, min_elems=1)
    out = m[1](torch.randn(1, 64, 8, 8, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()


def test_dtype_casts_cannot_undo_the_compression():
    m = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.Conv2d(64, 64, 3, padding=1), nn.Conv2d(64, 3, 3, padding=1)).eval()
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        ref = m(x)
    qc.quantize_convs(m, min_elems=1)
    scale_before = m[1].weight_scale.clone()
    for cast in (m.half, m.bfloat16, m.float):
        cast()
        assert m[1].weight_fp8.dtype == torch.uint8 and m[1].weight_fp8.numel() == 64 * 64 * 9
        assert torch.equal(m[1].weight_scale, scale_before) and m[1].weight_scale.dtype == torch.int32     # scale bits untouched, not rounded to 16 bit
    with torch.no_grad():
        assert _rel(m(x), ref) < 0.08


def test_env_switch(monkeypatch):
    monkeypatch.setenv("GB_QUANT_CONV", "0")
    assert not qc.enabled()
    monkeypatch.delenv("GB_QUANT_CONV")
    assert qc.enabled()


def test_hook_is_gated_by_total_size(monkeypatch, capsys):
    small = nn.Sequential(nn.Conv2d(3, 64, 3), nn.Conv2d(64, 64, 3), nn.Conv2d(64, 3, 3))
    assert qc.maybe_quantize_convs(small) is None and not hasattr(small[1], "weight_fp8")        # a few hundred KB: nothing to win
    monkeypatch.setenv("GB_QUANT_CONV_MIN_TOTAL_MB", "0.001")
    rep = qc.maybe_quantize_convs(small)
    assert rep is not None and rep.quantized == 1 and "gb_quant_conv" in capsys.readouterr().out
    monkeypatch.setenv("GB_QUANT_CONV", "0")
    other = nn.Sequential(nn.Conv2d(3, 64, 3), nn.Conv2d(64, 64, 3), nn.Conv2d(64, 3, 3))
    assert qc.maybe_quantize_convs(other) is None


def test_quantize_module_fp8_calls_the_conv_hook(monkeypatch):
    import gb_quant
    calls = []
    monkeypatch.setattr(qc, "maybe_quantize_convs", lambda m, device=None: calls.append(device))
    gb_quant._maybe_conv_fp8(nn.Conv2d(1, 1, 1), "fp8", "cuda")
    gb_quant._maybe_conv_fp8(nn.Conv2d(1, 1, 1), 4, "cuda")             # int4: not our business
    assert calls == ["cuda"]


def test_recommendations_follow_the_gpu():
    blackwell = {r["path"]: r for r in qc.recommendations(cc=(12, 0), vram_gb=12)}
    assert blackwell["linear_fp8_w8a8_scaled_mm"]["applies"] and blackwell["conv_fp8_weight_only"]["default"] and "budget" in blackwell
    old = {r["path"]: r for r in qc.recommendations(cc=(7, 5), vram_gb=24)}
    assert not old["linear_fp8_w8a8_scaled_mm"]["applies"] and "budget" not in old
    assert all({"path", "how", "gain"} <= set(r) for r in blackwell.values())
