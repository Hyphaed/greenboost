# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb_vision: projector sizing arithmetic and image-input normalisation.

Evidence these tests encode (live, 2026-10-28, RTX 5070 12 GB, Qwen3.8-35B-A3B + mmproj):
the recipe's nCpuMoe=28 (measured without the projector) left ~0.8 GB free and the first image
killed the engine; 32 worked. A 16:9 frame at 896 px (~575 tokens) made the model deny features
that were there; >= 1024 tokens fixed it.
"""
import base64
import io
import threading

import pytest

import gb_vision as gv

QWEN_META = {
    "clip.projector_type": "qwen3vl_merger",
    "clip.vision.patch_size": 14,
    "clip.vision.spatial_merge_size": 2,
    "clip.vision.embedding_length": 1152,
    "clip.vision.attention.head_count": 16,
    "clip.vision.feed_forward_length": 4304,
    "clip.vision.block_count": 27,
}
LLAVA_META = {"clip.projector_type": "mlp", "clip.vision.patch_size": 14, "clip.vision.image_size": 336,
              "clip.vision.embedding_length": 1024, "clip.vision.attention.head_count": 16}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("GB_SYNAPSE_IMAGE_MIN_TOKENS", "GB_SYNAPSE_IMAGE_MAX_TOKENS", "GB_SYNAPSE_VISION_MARGIN", "GB_SYNAPSE_IMAGE_MAX_BYTES"):
        monkeypatch.delenv(k, raising=False)
    gv.clear_fit()


def test_qwen_profile_has_the_engines_token_floor_and_a_target_above_it():
    p = gv.profile_from_meta(QWEN_META, int(0.6 * gv.GIB))
    assert p.dynamic and p.min_tokens == 1024
    assert p.target_tokens == 1280 and p.max_tokens >= p.target_tokens
    assert p.px_per_token == 784 and p.source == "gguf"
    assert gv.cmd_flags(p) == ["--image-min-tokens", "1024", "--image-max-tokens", str(p.max_tokens)]


def test_fixed_size_projector_has_no_flags_and_no_budget():
    p = gv.profile_from_meta(LLAVA_META, int(0.3 * gv.GIB))
    assert not p.dynamic and gv.cmd_flags(p) == [] and gv.budget_from_profile(p) is None
    assert p.max_tokens == (336 // 14) ** 2


def test_unreadable_projector_is_sized_for_the_dynamic_class():
    p = gv.profile_from_meta({}, int(0.6 * gv.GIB), "/x/mmproj.gguf")
    assert p.source == "default" and p.dynamic and p.min_tokens == 1024


def test_env_overrides_the_floor_and_cap(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_IMAGE_MIN_TOKENS", "512")
    monkeypatch.setenv("GB_SYNAPSE_IMAGE_MAX_TOKENS", "2048")
    p = gv.profile_from_meta(QWEN_META)
    assert (p.min_tokens, p.max_tokens) == (512, 2048) and p.target_tokens == 640


def test_projector_costs_about_the_four_layers_the_incident_needed():
    """weights + encoder buffers ~1.8 GiB; with ~0.45 GiB of experts per layer that is 4 more layers: 28 -> 32."""
    p = gv.profile_from_meta(QWEN_META, int(0.6 * gv.GIB))
    extra = gv.extra_vram_gb(p)
    assert 1.4 < extra < 2.4
    layers = gv.extra_cpu_moe_layers(extra, 0.465)
    assert layers in (4, 5) and 28 + layers >= 32


def test_encoder_buffer_grows_with_the_square_of_the_tokens():
    p = gv.profile_from_meta(QWEN_META)
    assert gv.encoder_compute_gb(p, 2048) > 2.4 * gv.encoder_compute_gb(p, 1280)
    assert gv.encoder_compute_gb(p, 0) == 0.0 or gv.encoder_compute_gb(p, 0) > 0


def test_margin_env_scales_the_estimate(monkeypatch):
    p = gv.profile_from_meta(QWEN_META)
    base = gv.encoder_compute_gb(p)
    monkeypatch.setenv("GB_SYNAPSE_VISION_MARGIN", "2.5")
    assert gv.encoder_compute_gb(p) == pytest.approx(base * 2.0, rel=1e-6)


def test_no_experts_no_layers():
    assert gv.extra_cpu_moe_layers(1.8, 0.0) == 0 and gv.extra_cpu_moe_layers(0.0, 0.4) == 0


# ---- image normalisation


def _budget():
    return gv.budget_from_profile(gv.profile_from_meta(QWEN_META))


def test_a_frame_is_scaled_to_about_a_megapixel_and_snapped_to_the_merge_unit():
    b = _budget()
    w, h = gv.target_size(1280, 720, b)
    assert w % 28 == 0 and h % 28 == 0
    assert gv.tokens_for(w, h, b) >= b.min_tokens            # 896 px side (575 tokens) is what broke the critic
    assert abs((w / h) - (1280 / 720)) < 0.08


def test_a_small_crop_is_upscaled_to_the_token_floor():
    b = _budget()
    w, h = gv.target_size(160, 380, b)
    assert w * h > 160 * 380 * 4 and gv.tokens_for(w, h, b) >= b.min_tokens


def test_a_huge_sheet_is_downscaled_and_capped():
    b = _budget()
    w, h = gv.target_size(4000, 3000, b)
    assert max(w, h) <= b.max_side and gv.tokens_for(w, h, b) <= b.max_tokens * 1.1


def test_degenerate_sizes_do_not_crash():
    assert gv.target_size(0, 10, _budget()) == (0, 10)


def _png(w, h, rgba=False):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA" if rgba else "RGB", (w, h), (200, 30, 30, 120) if rgba else (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_normalize_bytes_outputs_a_jpeg_at_the_target_size():
    from PIL import Image
    b = _budget()
    out, info = gv.normalize_image_bytes(_png(320, 760), b)
    im = Image.open(io.BytesIO(out))
    assert im.format == "JPEG" and im.size == tuple(info["to"]) and info["tokens"] >= b.min_tokens
    assert len(out) <= b.max_bytes


def test_transparent_png_is_flattened_on_white():
    from PIL import Image
    out, _ = gv.normalize_image_bytes(_png(200, 200, rgba=True), _budget())
    r, g, bl = Image.open(io.BytesIO(out)).convert("RGB").getpixel((50, 50))
    assert r > 200 and g > 100                               # blended toward white, not black


def test_body_limit_is_respected(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_IMAGE_MAX_BYTES", "20000")
    b = _budget()
    out, _ = gv.normalize_image_bytes(_png(1600, 1200), b)
    assert len(out) <= 20000 or len(out) < len(_png(1600, 1200))


def test_corrupt_image_passes_through_for_the_engines_own_error():
    out, info = gv.normalize_image_bytes(b"not an image", _budget())
    assert out == b"not an image" and "skipped" in info


def test_data_url_roundtrip_and_remote_urls_untouched():
    b = _budget()
    url = "data:image/png;base64," + base64.b64encode(_png(64, 64)).decode()
    new, info = gv.normalize_data_url(url, b)
    assert new.startswith("data:image/jpeg;base64,") and info["tokens"] >= b.min_tokens
    assert gv.normalize_data_url("https://example.com/a.png", b)[0] == "https://example.com/a.png"   # local-only: never fetched
    assert gv.normalize_data_url("data:text/plain,hello", b)[1].get("skipped")


def test_openai_messages_only_touch_image_parts():
    b = _budget()
    url = "data:image/png;base64," + base64.b64encode(_png(90, 90)).decode()
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": url}}]}]
    out, infos = gv.normalize_openai_messages(msgs, b)
    assert out[0] == msgs[0] and out[1]["content"][0] == {"type": "text", "text": "look"}
    assert out[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg") and len(infos) == 1
    assert msgs[1]["content"][1]["image_url"]["url"] == url            # input not mutated
    assert gv.normalize_openai_messages(msgs, None) == (msgs, [])        # no budget (fixed projector): unchanged


def test_fit_context_is_thread_local():
    gv.set_fit("/a/mmproj.gguf", None, 1.5)
    seen = {}
    t = threading.Thread(target=lambda: seen.update(mm=gv.VISION_FIT.mmproj, gb=gv.VISION_FIT.extra_gb))
    t.start(); t.join()
    assert seen == {"mm": "", "gb": 0.0} and gv.VISION_FIT.extra_gb == 1.5
    gv.clear_fit()
    assert gv.VISION_FIT.mmproj == ""


def test_budget_roundtrips_through_a_dict():
    b = _budget()
    assert gv.budget_from_dict(b.as_dict()) == b and gv.budget_from_dict(None) is None and gv.budget_from_dict({}) is None
