#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_diffusion_server.py — minimal OpenAI-images-compatible server around
DiffusionOrchestrator (gb_diffusion_orch.py) + gb_quant's diffusion
quantizers. Used by gb_synapse's DiffusersBackend for HF diffusers
pipelines (FLUX, SDXL, LTX, ...) served through the same :11434 proxy as
every other gb-synapse engine — the proxy's /v1/{path:.*} passthrough needs
zero changes for /v1/images/generations.

Single global pipeline, single-request-at-a-time — no batching queue, that's
future work. Exposes just enough of the OpenAI images surface for
gb_synapse_api.py's raw passthrough to work unchanged: /health, /v1/models,
/v1/images/generations.

Usage:
    python3 gb_diffusion_server.py --model org/repo --served-model-name name \
        --quant fp8 --host 127.0.0.1 --port 12434
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import time

from aiohttp import web

PIPE = None
ORCH = None
MODEL_NAME = ""

# gb_quant precision-ladder tokens, same mapping gb_llm_server.py uses for
# the transformers text backend.
_QUANT_TO_BITS = {"FP8": "fp8", "INT8": 8, "INT4": 4}


def _load(model: str, quant: str) -> None:
    global PIPE, ORCH
    import torch
    from diffusers import AutoPipelineForText2Image
    from gb_diffusion_orch import DiffusionOrchestrator
    import gb_quant

    print(f"[gb_diffusion_server] loading {model} (gb-quant bits="
          f"{_QUANT_TO_BITS.get(quant.upper(), 'fp8')!r}) ...", flush=True)
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.bfloat16)
    orch = DiffusionOrchestrator(pipe)
    bits = _QUANT_TO_BITS.get(quant.upper(), "fp8")
    gb_quant.quantize_encoders(pipe, bits=bits)
    gb_quant.quantize_denoiser(pipe, bits=bits)
    PIPE, ORCH = pipe, orch
    print(f"[gb_diffusion_server] ready, serving '{model}'", flush=True)


def _generate(prompt: str, n: int, size: str, steps: int) -> list[bytes]:
    """Blocking single-shot generate — run via run_in_executor so it doesn't
    stall the event loop (e.g. /health) while it runs."""
    if "x" in size.lower():
        w, h = (int(x) for x in size.lower().split("x", 1))
    else:
        w, h = 1024, 1024
    with ORCH.denoise_phase():
        images = PIPE(prompt=prompt, num_images_per_prompt=n,
                      width=w, height=h, num_inference_steps=steps).images
    out = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _emit_image_gen(prompt: str, n: int, size: str, steps: int,
                    duration_s: float, error: "str | None" = None) -> None:
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "image_gen", "status": "error" if error else "ok",
            "model": MODEL_NAME, "engine": "diffusers",
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:12],
            "n": n, "size": size, "steps": steps, "duration_s": round(duration_s, 2),
            **({"error": error} if error else {}),
        })
    except Exception:
        pass


async def images_generations(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except (ValueError, TypeError):
        return web.json_response({"error": {"message": "invalid JSON body"}}, status=400)
    prompt = body.get("prompt", "")
    n = int(body.get("n") or 1)
    size = body.get("size") or "1024x1024"
    steps = int(body.get("steps") or 28)

    t0 = time.time()
    loop = asyncio.get_event_loop()
    try:
        pngs = await loop.run_in_executor(None, _generate, prompt, n, size, steps)
    except Exception as exc:
        _emit_image_gen(prompt, n, size, steps, time.time() - t0, error=str(exc))
        return web.json_response({"error": {"message": str(exc)}}, status=500)
    _emit_image_gen(prompt, n, size, steps, time.time() - t0)

    data = [{"b64_json": base64.b64encode(png).decode()} for png in pngs]
    return web.json_response({"created": int(time.time()), "data": data})


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok" if PIPE is not None else "loading",
                              "model": MODEL_NAME})


async def models(request: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/images/generations", images_generations)
    return app


def main() -> None:
    global MODEL_NAME
    ap = argparse.ArgumentParser(description="gb-synapse diffusers image-gen server")
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
