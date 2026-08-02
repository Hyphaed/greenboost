#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_longlive_server.py (missing_features.md item (f) , persistent
video-serving engine). CPU-only: the diffusion pipeline, PIL, and ffmpeg are
all mocked/stubbed , this module has no live GPU + video model available
to test against (see the module's own docstring for the documented
ai-forge/GreenBoost cross-process VRAM incident that makes this an honest
limitation, not an oversight). These tests exercise gb_longlive_server.py's
OWN logic: image resolution, shot iteration, request parsing, dataflux
emission, and the HTTP handlers' request/response shape.
"""
import asyncio
import base64
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_longlive_server as gls


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _run(coro):
    return asyncio.run(coro)


# ── _load_image ──────────────────────────────────────────────────────────

def test_load_image_none_returns_none():
    assert gls._load_image(None) is None


def test_load_image_local_path(tmp_path):
    from PIL import Image
    p = tmp_path / "anchor.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(p)
    img = gls._load_image(str(p))
    assert img.size == (8, 8)
    assert img.mode == "RGB"


def test_load_image_base64():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(9, 9, 9)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    img = gls._load_image(b64)
    assert img.size == (4, 4)


# ── _encode_video ────────────────────────────────────────────────────────

def test_encode_video_writes_and_reads_temp_file_then_cleans_up():
    written_paths = []

    def _fake_export(frames, path, fps):
        written_paths.append(path)
        with open(path, "wb") as f:
            f.write(b"FAKEMP4BYTES")

    fake_module = MagicMock()
    fake_module.export_to_video = _fake_export
    with patch.dict(sys.modules, {"diffusers.utils": fake_module}):
        result = gls._encode_video(["frame0", "frame1"], fps=24)

    assert result == b"FAKEMP4BYTES"
    assert len(written_paths) == 1
    import os
    assert not os.path.exists(written_paths[0])   # cleaned up


# ── _generate_shot ───────────────────────────────────────────────────────

class _FakeOrch:
    def denoise_phase(self):
        return self


    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeResultFrames:
    def __init__(self, frames):
        self.frames = [frames]


def test_generate_shot_passes_prompt_and_frame_count(monkeypatch):
    captured_kwargs = {}

    def _fake_pipe(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResultFrames(["f0", "f1"])

    monkeypatch.setattr(gls, "PIPE", _fake_pipe)
    monkeypatch.setattr(gls, "ORCH", _FakeOrch())
    monkeypatch.setattr(gls, "_encode_video", lambda frames, fps: b"VIDEO")

    out = gls._generate_shot("a cat", None, num_frames=17, steps=5, fps=16,
                             seed=None, extra_kwargs={})
    assert out == b"VIDEO"
    assert captured_kwargs["prompt"] == "a cat"
    assert captured_kwargs["num_frames"] == 17
    assert captured_kwargs["num_inference_steps"] == 5
    assert "image" not in captured_kwargs
    assert "generator" not in captured_kwargs


def test_generate_shot_includes_image_when_provided(monkeypatch):
    captured_kwargs = {}

    def _fake_pipe(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResultFrames(["f0"])

    monkeypatch.setattr(gls, "PIPE", _fake_pipe)
    monkeypatch.setattr(gls, "ORCH", _FakeOrch())
    monkeypatch.setattr(gls, "_encode_video", lambda frames, fps: b"VIDEO")

    sentinel_image = object()
    gls._generate_shot("a cat", sentinel_image, num_frames=8, steps=4, fps=16,
                       seed=None, extra_kwargs={})
    assert captured_kwargs["image"] is sentinel_image


def test_generate_shot_seed_builds_generator(monkeypatch):
    captured_kwargs = {}

    def _fake_pipe(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResultFrames(["f0"])

    monkeypatch.setattr(gls, "PIPE", _fake_pipe)
    monkeypatch.setattr(gls, "ORCH", _FakeOrch())
    monkeypatch.setattr(gls, "_encode_video", lambda frames, fps: b"VIDEO")

    gls._generate_shot("a cat", None, num_frames=8, steps=4, fps=16,
                       seed=42, extra_kwargs={})
    import torch
    assert isinstance(captured_kwargs["generator"], torch.Generator)


def test_generate_shot_falls_back_to_videos_attr(monkeypatch):
    class _FakeResultVideos:
        def __init__(self):
            self.videos = [["v0"]]

    def _fake_pipe(**kwargs):
        return _FakeResultVideos()

    monkeypatch.setattr(gls, "PIPE", _fake_pipe)
    monkeypatch.setattr(gls, "ORCH", _FakeOrch())
    captured = {}
    monkeypatch.setattr(gls, "_encode_video",
                        lambda frames, fps: captured.setdefault("frames", frames) or b"V")
    gls._generate_shot("x", None, 8, 4, 16, None, {})
    assert captured["frames"] == ["v0"]


# ── _generate: multi-shot sequencing + concat ───────────────────────────────

def test_generate_single_shot_returns_clip_directly(monkeypatch, tmp_path):
    def _fake_shot(prompt, image, num_frames, steps, fps, seed, extra):
        return b"SINGLE_CLIP"
    monkeypatch.setattr(gls, "_generate_shot", _fake_shot)
    monkeypatch.setattr(gls, "_load_image", lambda spec: None)
    concat_called = {"n": 0}
    monkeypatch.setattr(gls, "_concat_clips", lambda paths: concat_called.__setitem__("n", concat_called["n"] + 1) or b"SHOULD_NOT_HAPPEN")

    out = gls._generate([{"prompt": "only shot"}], None, fps=16, seed=None)
    assert out == b"SINGLE_CLIP"
    assert concat_called["n"] == 0


def test_generate_multi_shot_concatenates(monkeypatch):
    calls = []

    def _fake_shot(prompt, image, num_frames, steps, fps, seed, extra):
        calls.append((prompt, image))
        return f"CLIP:{prompt}".encode()

    monkeypatch.setattr(gls, "_generate_shot", _fake_shot)
    sentinel_image = object()
    monkeypatch.setattr(gls, "_load_image", lambda spec: sentinel_image if spec else None)

    concat_paths_seen = []

    def _fake_concat(paths):
        concat_paths_seen.append(list(paths))
        return b"CONCATENATED"

    monkeypatch.setattr(gls, "_concat_clips", _fake_concat)

    out = gls._generate(
        [{"prompt": "shot one"}, {"prompt": "shot two"}], "some-image-spec",
        fps=16, seed=7)
    assert out == b"CONCATENATED"
    assert len(concat_paths_seen[0]) == 2
    # Only shot 0 gets the i2v anchor image.
    assert calls[0] == ("shot one", sentinel_image)
    assert calls[1] == ("shot two", None)


def test_generate_cleans_up_temp_clip_files(monkeypatch):
    import os
    written = []

    def _fake_shot(prompt, image, num_frames, steps, fps, seed, extra):
        return b"X"

    monkeypatch.setattr(gls, "_generate_shot", _fake_shot)
    monkeypatch.setattr(gls, "_load_image", lambda spec: None)

    seen_paths = []

    def _fake_concat(paths):
        seen_paths.extend(paths)
        return b"OUT"

    monkeypatch.setattr(gls, "_concat_clips", _fake_concat)
    gls._generate([{"prompt": "a"}, {"prompt": "b"}], None, fps=16, seed=None)
    for p in seen_paths:
        assert not os.path.exists(p)


# ── _emit_video_gen ──────────────────────────────────────────────────────

def test_emit_video_gen_ok():
    fake_dataflux = MagicMock()
    with patch.dict(sys.modules, {"gb_dataflux": fake_dataflux}):
        gls._emit_video_gen([{"prompt": "a cat"}], 1.23)
    event = fake_dataflux.emit.call_args[0][0]
    assert event["kind"] == "video_render"
    assert event["status"] == "ok"
    assert event["n_shots"] == 1
    assert "error" not in event


def test_emit_video_gen_error():
    fake_dataflux = MagicMock()
    with patch.dict(sys.modules, {"gb_dataflux": fake_dataflux}):
        gls._emit_video_gen([{"prompt": "a cat"}], 0.5, error="boom")
    event = fake_dataflux.emit.call_args[0][0]
    assert event["status"] == "error"
    assert event["error"] == "boom"


def test_emit_video_gen_never_raises_when_dataflux_missing():
    with patch.dict(sys.modules, {"gb_dataflux": None}):
        gls._emit_video_gen([{"prompt": "x"}], 1.0)   # must not raise


# ── HTTP handlers ────────────────────────────────────────────────────────

def test_video_generations_rejects_invalid_json():
    resp = _run(gls.video_generations(_FakeRequest(None)))
    assert resp.status == 400


def test_video_generations_rejects_empty_body():
    resp = _run(gls.video_generations(_FakeRequest({})))
    assert resp.status == 400


def test_video_generations_bare_prompt_builds_one_shot(monkeypatch):
    captured = {}

    def _fake_generate(shots, image_spec, fps, seed):
        captured["shots"] = shots
        captured["image_spec"] = image_spec
        return b"VIDEO"

    monkeypatch.setattr(gls, "_generate", _fake_generate)
    monkeypatch.setattr(gls, "_emit_video_gen", lambda *a, **k: None)

    resp = _run(gls.video_generations(_FakeRequest({"prompt": "a dog running"})))
    assert resp.status == 200
    assert len(captured["shots"]) == 1
    assert captured["shots"][0]["prompt"] == "a dog running"


def test_video_generations_shots_list_passed_through(monkeypatch):
    captured = {}

    def _fake_generate(shots, image_spec, fps, seed):
        captured["shots"] = shots
        return b"VIDEO"

    monkeypatch.setattr(gls, "_generate", _fake_generate)
    monkeypatch.setattr(gls, "_emit_video_gen", lambda *a, **k: None)

    body = {"shots": [{"prompt": "one"}, {"prompt": "two"}], "image": "/tmp/x.png", "seed": 5}
    resp = _run(gls.video_generations(_FakeRequest(body)))
    assert resp.status == 200
    assert len(captured["shots"]) == 2


def test_video_generations_returns_b64_video_on_success(monkeypatch):
    monkeypatch.setattr(gls, "_generate", lambda *a, **k: b"HELLOVIDEO")
    monkeypatch.setattr(gls, "_emit_video_gen", lambda *a, **k: None)
    resp = _run(gls.video_generations(_FakeRequest({"prompt": "x"})))
    assert resp.status == 200
    import json
    payload = json.loads(resp.body)
    decoded = base64.b64decode(payload["data"][0]["b64_video"])
    assert decoded == b"HELLOVIDEO"


def test_video_generations_500_on_generate_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("pipeline exploded")
    monkeypatch.setattr(gls, "_generate", _boom)
    emitted = {}
    monkeypatch.setattr(gls, "_emit_video_gen",
                        lambda shots, dur, error=None: emitted.update(error=error))
    resp = _run(gls.video_generations(_FakeRequest({"prompt": "x"})))
    assert resp.status == 500
    assert "pipeline exploded" in emitted["error"]


def test_health_reports_loading_when_pipe_none(monkeypatch):
    monkeypatch.setattr(gls, "PIPE", None)
    resp = _run(gls.health(_FakeRequest(None)))
    import json
    assert json.loads(resp.body)["status"] == "loading"


def test_health_reports_ok_when_pipe_set(monkeypatch):
    monkeypatch.setattr(gls, "PIPE", object())
    resp = _run(gls.health(_FakeRequest(None)))
    import json
    assert json.loads(resp.body)["status"] == "ok"


def test_models_reports_served_model_name(monkeypatch):
    monkeypatch.setattr(gls, "MODEL_NAME", "longlive-2.0-5b")
    resp = _run(gls.models(_FakeRequest(None)))
    import json
    body = json.loads(resp.body)
    assert body["data"][0]["id"] == "longlive-2.0-5b"


def test_build_app_registers_routes():
    app = gls.build_app()
    paths = {r.resource.canonical for r in app.router.routes()}
    assert "/health" in paths
    assert "/v1/models" in paths
    assert "/v1/video/generations" in paths
