#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_longlive_server.py — persistent video-serving engine (missing_features.md
item (f)). Same "load once, serve many" shape as gb_diffusion_server.py
(single global pipeline, aiohttp, /health + /v1/models), generalized from
that server's image-only `AutoPipelineForText2Image` to a video-capable
`diffusers.DiffusionPipeline` (text-to-video or image-to-video), so a
multi-shot rollout doesn't pay the full model-load + gb-quant cost on every
render the way a fresh subprocess per call does.

Scope note, read before pointing this at a real model
-------------------------------------------------------
This is a GENERIC engine for diffusers-compatible video pipelines (WAN2.x
and similar checkpoints installable via `diffusers.DiffusionPipeline.
from_pretrained`), NOT a loader for LongLive's actual bespoke checkpoint
format. LongLive (the reference workload named in missing_features.md item
(f)) ships a custom FourOverSix/NVFP4 fork with its own inference code
(`~/Dev/ai-forge/forge/runners/longlive.py`, a separate `.venv-longlive`
environment) that this server does not replicate.

More importantly: ai-forge's OWN prior attempt at a warm persistent server
for LongLive specifically (`forge/runners/longlive.py`'s "serve" mode) is
documented there as KNOWN BROKEN under GreenBoost, for a real, diagnosed
root cause, not just "untested" — quoting that file's own comment (2026-07-
26/27 incident): GreenBoost's shim never releases VRAM cross-process
(`gb_quant._gb_cache_release()`'s `torch.cuda.empty_cache()` no-ops while
GreenBoost is active), so a second process (a separate VAE-decode
subprocess) reliably hit clean CUDA OOM against a warm server's own VRAM
reservation. In-process decode inside the warm server was ALSO tried
(generator+KV-cache offload, gc.collect(), every combination) and never
reliably avoided CUDA managed-UVM page-fault thrashing (100% GPU util at
~45W of 250W — memory-stalled, not compute-bound; 5-15+ minutes for a
decode that takes 12s on a clean GPU). ai-forge's LongLive pipeline
therefore defaults to "oneshot" (a fresh subprocess per render, full VRAM
release on exit) specifically to sidestep this, and its own comment says
not to use "serve" until the underlying GreenBoost bug is fixed upstream.

**Do not point this server at LongLive until that GreenBoost cross-process
VRAM release bug is fixed.** It may still be useful for other video
pipelines whose VAE decode has smaller memory pressure or that don't spill
into GreenBoost's T2/T3 tiers as aggressively — that has not been measured
either, since no live GPU + video model was available to test against this
session. Treat this module as structurally complete and unit-tested with a
stubbed pipeline, NOT end-to-end verified.

Usage:
    python3 gb_longlive_server.py --model org/repo --served-model-name name \
        --quant fp8 --host 127.0.0.1 --port 12435
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import os
import tempfile
import time

from aiohttp import web

PIPE = None
ORCH = None
MODEL_NAME = ""

# gb_quant precision-ladder tokens, same mapping gb_diffusion_server.py uses.
_QUANT_TO_BITS = {"FP8": "fp8", "INT8": 8, "INT4": 4}

# Default video generation params, chosen to be a cheap smoke-test shape
# (short clip, few steps) rather than a production-quality default — a
# caller with real quality requirements should pass these explicitly.
_DEFAULT_NUM_FRAMES = 33
_DEFAULT_STEPS = 20
_DEFAULT_FPS = 16


def _load(model: str, quant: str) -> None:
    global PIPE, ORCH
    import torch
    from diffusers import DiffusionPipeline
    from gb_diffusion_orch import DiffusionOrchestrator
    import gb_quant

    print(f"[gb_longlive_server] loading {model} (gb-quant bits="
          f"{_QUANT_TO_BITS.get(quant.upper(), 'fp8')!r}) ...", flush=True)
    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    orch = DiffusionOrchestrator(pipe)
    bits = _QUANT_TO_BITS.get(quant.upper(), "fp8")
    # quality=None , same reasoning as gb_diffusion_server.py: quality=
    # defaults to "near_lossless" and silently overrides --quant otherwise.
    gb_quant.quantize_encoders(pipe, bits=bits, quality=None)
    gb_quant.quantize_denoiser(pipe, bits=bits, quality=None)
    PIPE, ORCH = pipe, orch
    print(f"[gb_longlive_server] ready, serving '{model}'", flush=True)


def _load_image(spec: "str | None"):
    """Resolve an i2v anchor image: an absolute local path, or a base64
    data string. None -> None (text-to-video call)."""
    if not spec:
        return None
    from PIL import Image
    if os.path.isabs(spec) and os.path.exists(spec):
        return Image.open(spec).convert("RGB")
    raw = base64.b64decode(spec)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_video(frames, fps: int) -> bytes:
    """PIL frame list -> MP4 bytes via diffusers.utils.export_to_video (the
    same ffmpeg-backed helper the diffusers ecosystem already standardizes
    on for exactly this), written to a temp file and read back since that
    helper's only public contract is a filesystem path."""
    from diffusers.utils import export_to_video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_path = tf.name
    try:
        export_to_video(frames, tmp_path, fps=fps)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _generate_shot(prompt: str, image, num_frames: int, steps: int,
                   fps: int, seed: "int | None", extra_kwargs: dict) -> bytes:
    """Blocking single-shot generate , run via run_in_executor so /health
    stays responsive while a render is in flight (mirrors
    gb_diffusion_server.py's own _generate)."""
    import torch
    kwargs = dict(prompt=prompt, num_frames=num_frames,
                 num_inference_steps=steps, **extra_kwargs)
    if image is not None:
        kwargs["image"] = image
    if seed is not None:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    with ORCH.denoise_phase():
        result = PIPE(**kwargs)
    frames = result.frames[0] if hasattr(result, "frames") else result.videos[0]
    return _encode_video(frames, fps)


def _generate(shots: "list[dict]", image_spec: "str | None", fps: int,
             seed: "int | None") -> bytes:
    """Render every shot sequentially and concatenate the resulting clips
    via ffmpeg's concat demuxer. This is a deliberately SIMPLE interpretation
    of "multi-shot" for a generic pipeline , it does NOT replicate LongLive's
    actual chunked causal-rollout streaming (continuous KV-cache state
    across chunks via model_kwargs.num_frame_per_block), which only
    LongLive's own bespoke inference code implements. Each shot here is an
    independent generation; only shot 0 gets the i2v anchor image, matching
    the convention ai-forge's own gen_longlive.py CLI already uses."""
    image = _load_image(image_spec)
    clip_paths = []
    try:
        for i, shot in enumerate(shots):
            prompt = shot.get("prompt", "")
            num_frames = int(shot.get("num_frames") or _DEFAULT_NUM_FRAMES)
            steps = int(shot.get("num_inference_steps") or _DEFAULT_STEPS)
            extra = {k: v for k, v in shot.items()
                    if k not in ("prompt", "num_frames", "num_inference_steps")}
            clip = _generate_shot(prompt, image if i == 0 else None,
                                  num_frames, steps, fps, seed, extra)
            fd, path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            with open(path, "wb") as f:
                f.write(clip)
            clip_paths.append(path)

        if len(clip_paths) == 1:
            with open(clip_paths[0], "rb") as f:
                return f.read()
        return _concat_clips(clip_paths)
    finally:
        for p in clip_paths:
            if os.path.exists(p):
                os.remove(p)


def _concat_clips(paths: "list[str]") -> bytes:
    import subprocess
    fd, list_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        with open(list_path, "w") as f:
            for p in paths:
                f.write(f"file '{p}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
             "-c", "copy", out_path],
            check=True, capture_output=True)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (list_path, out_path):
            if os.path.exists(p):
                os.remove(p)


def _emit_video_gen(shots: "list[dict]", duration_s: float,
                    error: "str | None" = None) -> None:
    try:
        import gb_dataflux
        prompts_blob = "|".join(s.get("prompt", "") for s in shots)
        gb_dataflux.emit({
            "kind": "video_render", "status": "error" if error else "ok",
            "model": MODEL_NAME, "engine": "video",
            "prompt_hash": hashlib.sha256(prompts_blob.encode()).hexdigest()[:12],
            "n_shots": len(shots), "duration_s": round(duration_s, 2),
            **({"error": error} if error else {}),
        })
    except Exception:
        pass


async def video_generations(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except (ValueError, TypeError):
        return web.json_response({"error": {"message": "invalid JSON body"}}, status=400)

    shots = body.get("shots")
    if not shots:
        prompt = body.get("prompt", "")
        if not prompt:
            return web.json_response(
                {"error": {"message": "body must carry 'shots' (list of "
                                      "{'prompt': str, ...}) or a bare 'prompt'"}},
                status=400)
        shots = [{"prompt": prompt, "num_frames": body.get("num_frames"),
                 "num_inference_steps": body.get("num_inference_steps")}]
    image_spec = body.get("image")
    fps = int(body.get("fps") or _DEFAULT_FPS)
    seed = body.get("seed")
    seed = int(seed) if seed is not None else None

    t0 = time.time()
    loop = asyncio.get_event_loop()
    try:
        video_bytes = await loop.run_in_executor(
            None, _generate, shots, image_spec, fps, seed)
    except Exception as exc:
        _emit_video_gen(shots, time.time() - t0, error=str(exc))
        return web.json_response({"error": {"message": str(exc)}}, status=500)
    _emit_video_gen(shots, time.time() - t0)

    return web.json_response({
        "created": int(time.time()),
        "data": [{"b64_video": base64.b64encode(video_bytes).decode()}],
    })


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok" if PIPE is not None else "loading",
                              "model": MODEL_NAME})


async def models(request: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/video/generations", video_generations)
    return app


def main() -> None:
    global MODEL_NAME
    ap = argparse.ArgumentParser(description="gb-synapse persistent video-gen server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument("--quant", default="fp8")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    MODEL_NAME = args.served_model_name
    _load(args.model, args.quant)

    web.run_app(build_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
