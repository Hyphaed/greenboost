"""FLUX/SD diffusion pipeline for UI asset generation."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from greenboost_cli.environment.settings import GB_HOME
from greenboost_cli.diffusion.models import (
    MODELS, UI_PROMPT_TEMPLATES, UI_NEGATIVE, auto_select_model,
)

_CACHE_DIR     = GB_HOME / "design_assets" / ".cache"
_CACHE_INDEX   = _CACHE_DIR / "index.json"
_VITALS_FILE   = Path("/run/greenboost/diffuser_vitals.json")
_VITALS_FALLBACK = GB_HOME / "diffuser_vitals.json"

# Lazy-loaded pipeline cache
_loaded_pipe: dict[str, Any] = {}


def _write_vitals(data: dict) -> None:
    """Write diffuser pipeline state to /run/greenboost/diffuser_vitals.json (or GB_HOME fallback)."""
    payload = {"ts": time.time(), **data}
    blob = json.dumps(payload)
    for dest in (_VITALS_FILE, _VITALS_FALLBACK):
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(blob)
            return
        except OSError:
            continue


def _cuda_memory_mb() -> tuple[int, int, int]:
    """Return (allocated_mb, reserved_mb, peak_mb) from torch.cuda if available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0, 0, 0
        stats = torch.cuda.memory_stats()
        alloc = stats.get("allocated_bytes.all.current", 0) // (1024 * 1024)
        rsv   = stats.get("reserved_bytes.all.current",  0) // (1024 * 1024)
        peak  = stats.get("allocated_bytes.all.peak",    0) // (1024 * 1024)
        return alloc, rsv, peak
    except Exception:
        return 0, 0, 0


def _cache_key(prompt: str, model_key: str, w: int, h: int, seed: int | None, steps: int) -> str:
    sig = f"{prompt}|{model_key}|{w}|{h}|{seed}|{steps}"
    return hashlib.sha256(sig.encode()).hexdigest()[:16]


def _cache_lookup(key: str) -> Path | None:
    if not _CACHE_INDEX.exists():
        return None
    try:
        index = json.loads(_CACHE_INDEX.read_text())
        entry = index.get(key)
        if entry and Path(entry).exists():
            return Path(entry)
    except Exception:
        pass
    return None


def _cache_store(key: str, path: Path) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index: dict = {}
    if _CACHE_INDEX.exists():
        try:
            index = json.loads(_CACHE_INDEX.read_text())
        except Exception:
            pass
    index[key] = str(path)
    _CACHE_INDEX.write_text(json.dumps(index, indent=2))


def _build_prompt(
    asset_type: str,
    style: str,
    colors: str,
    mood: str,
    product_type: str,
    lora_trigger: str,
    prompt_prefix: str,
) -> str:
    template = UI_PROMPT_TEMPLATES.get(asset_type, UI_PROMPT_TEMPLATES["hero"])
    prompt   = template.format(style=style, colors=colors, mood=mood, product_type=product_type)
    if lora_trigger:
        prompt = f"{lora_trigger}, {prompt}"
    if prompt_prefix and not prompt.startswith(prompt_prefix):
        prompt = f"{prompt_prefix}, {prompt}"
    return prompt


def load_pipeline(model_key: str, use_lora: str | None = None) -> Any:
    """Load (and cache in-process) a diffusion pipeline."""
    # Apply GreenBoost env before any torch.cuda call.
    # diffusion=True zeros GREENBOOST_KV_RESERVE_MB — no KV cache in diffusion.
    try:
        from greenboost_cli.greenboost.gb_torch import apply_gb_torch_env
        apply_gb_torch_env(diffusion=True)
    except ImportError:
        pass

    cache_key = f"{model_key}:{use_lora}"
    if cache_key in _loaded_pipe:
        return _loaded_pipe[cache_key]

    try:
        import torch
    except ImportError:
        raise ImportError("torch not installed. Run: pip install torch")

    _write_vitals({"state": "loading", "model": model_key, "pipeline": "FLUX" if "flux" in model_key.lower() else "SD", "pid": os.getpid()})

    try:
        from diffusers import (
            FluxPipeline, FluxTransformer2DModel,
            StableDiffusionPipeline,
        )
    except ImportError:
        raise ImportError(
            "diffusers not installed. Run: pip install diffusers transformers accelerate"
        )

    cfg    = MODELS[model_key]
    repo   = cfg["repo"]
    family = cfg.get("family", "flux")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CPU spillover not permitted — GreenBoost must provide CUDA. "
            "Check that the kernel module is loaded and torch is the cu130-tagged build."
        )
    device = "cuda"
    dtype  = torch.float16

    try:
        if family == "sd":
            pipe = StableDiffusionPipeline.from_pretrained(
                repo, torch_dtype=dtype, safety_checker=None
            ).to(device)
            pipe.enable_attention_slicing()

        elif cfg.get("quantization") == "fp8":
            try:
                from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
                transformer = FluxTransformer2DModel.from_pretrained(
                    repo, subfolder="transformer", torch_dtype=torch.bfloat16
                )
                quantize_(transformer, float8_dynamic_activation_float8_weight())
            except (ImportError, Exception):
                transformer = FluxTransformer2DModel.from_pretrained(
                    repo, subfolder="transformer", torch_dtype=torch.bfloat16
                )
            pipe = FluxPipeline.from_pretrained(repo, transformer=transformer,
                                                 torch_dtype=torch.bfloat16).to(device)

        elif cfg.get("quantization") == "nf4":
            try:
                from transformers import BitsAndBytesConfig
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                transformer = FluxTransformer2DModel.from_pretrained(
                    repo, subfolder="transformer",
                    quantization_config=nf4_config, torch_dtype=torch.bfloat16
                )
            except ImportError:
                transformer = FluxTransformer2DModel.from_pretrained(
                    repo, subfolder="transformer", torch_dtype=torch.bfloat16
                )
            pipe = FluxPipeline.from_pretrained(repo, transformer=transformer,
                                                 torch_dtype=torch.bfloat16).to(device)
        else:
            pipe = FluxPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to(device)

    except torch.cuda.OutOfMemoryError:
        _loaded_pipe.clear()
        gc.collect()
        torch.cuda.empty_cache()
        raise

    # Load LoRA weights for FLUX models
    if family == "flux" and use_lora:
        loras = cfg.get("loras", {})
        if use_lora in loras:
            lora_cfg = loras[use_lora]
            try:
                pipe.load_lora_weights(
                    lora_cfg["repo"],
                    weight_name=lora_cfg.get("weight", ""),
                    adapter_name=use_lora,
                )
                pipe.set_adapters([use_lora], adapter_weights=[lora_cfg.get("scale", 0.8)])
            except Exception as e:
                print(f"  ⚠  LoRA load failed ({use_lora}): {e}", file=sys.stderr)

    _loaded_pipe[cache_key] = pipe
    alloc, rsv, peak = _cuda_memory_mb()
    _write_vitals({
        "state": "ready", "model": model_key,
        "pipeline": "FLUX" if "flux" in model_key.lower() else "SD",
        "pid": os.getpid(), "vram_alloc_mb": alloc,
        "vram_reserved_mb": rsv, "vram_peak_mb": peak,
    })
    return pipe


def generate_ui_asset(
    asset_type: str,
    output_path: Path,
    style: str = "glassmorphism",
    colors: str = "deep blue and violet gradient with white",
    mood: str = "professional and modern",
    product_type: str = "SaaS application",
    model_key: str | None = None,
    use_lora: str | None = None,
    use_turboquant: bool = True,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    custom_prompt: str | None = None,
    use_cache: bool = True,
) -> Path:
    """Generate a UI design asset. Returns path to PNG."""
    import torch

    if model_key is None:
        model_key = auto_select_model()

    cfg    = MODELS.get(model_key, MODELS["klein-fp8"])
    family = cfg.get("family", "flux")

    # Fall back from T2-requiring model if no GreenBoost
    from greenboost_cli.diffusion.models import GREENBOOST_AVAILABLE
    if cfg.get("requires_greenboost") and not GREENBOOST_AVAILABLE:
        print(f"  ⚠  {model_key} requires GreenBoost. Falling back to klein-fp8.")
        model_key = "klein-fp8"
        cfg    = MODELS["klein-fp8"]
        family = "flux"

    default_w, default_h = cfg.get("default_size", (512, 512))
    width  = width  or default_w
    height = height or default_h
    steps  = cfg["steps"]

    lora_trigger  = ""
    if use_lora and use_lora in cfg.get("loras", {}):
        lora_trigger = cfg["loras"][use_lora].get("trigger", "")

    prompt = custom_prompt or _build_prompt(
        asset_type, style, colors, mood, product_type,
        lora_trigger, cfg.get("prompt_prefix", ""),
    )

    if use_cache:
        key    = _cache_key(prompt, model_key, width, height, seed, steps)
        cached = _cache_lookup(key)
        if cached:
            print(f"  ✓  Cache hit → {cached}")
            return cached

    # Warn on T3 spillover — generation becomes ~100× slower on NVMe swap.
    try:
        from greenboost_cli.greenboost.monitor import get_tier_stats
        _stats = get_tier_stats()
        if _stats and _stats.get("t3_swap_used_mb", 0) > 0:
            t3_gb = round(_stats["t3_swap_used_mb"] / 1024, 1)
            print(f"  ⚠  T3 NVMe spillover active ({t3_gb} GB) — generation will be ~100× slower!",
                  file=sys.stderr)
    except Exception:
        pass

    pipe      = load_pipeline(model_key, use_lora if family == "flux" else None)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CPU spillover not permitted — GreenBoost must provide CUDA. "
            "Check that the kernel module is loaded and torch is the cu130-tagged build."
        )
    device    = "cuda"
    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
    neg       = UI_NEGATIVE if family in ("sd", "controlnet") else None

    kwargs: dict = dict(
        prompt=prompt, height=height, width=width,
        num_inference_steps=steps, generator=generator,
    )
    if neg is not None:
        kwargs["negative_prompt"] = neg

    # Apply TurboQuant+ K/V compression when requested and gb_attn is available.
    # k_bits=4/v_bits=3 gives ~4× DMA bandwidth reduction with near-zero quality loss.
    _tq_ctx = None
    if use_turboquant and device == "cuda":
        try:
            from greenboost_cli.greenboost.gb_torch import load_gb_attn
            gb_attn = load_gb_attn()
            if gb_attn is not None:
                _tq_ctx = gb_attn.turboquant_attention(k_bits=4, v_bits=3, layer_adaptive=True)
        except Exception:
            pass

    t0 = time.time()
    alloc0, _, _ = _cuda_memory_mb()
    _write_vitals({
        "state": "generating", "model": model_key,
        "pipeline": "FLUX" if family == "flux" else "SD",
        "pid": os.getpid(), "vram_alloc_mb": alloc0,
        "gen_step": 0, "gen_total_steps": steps,
        "last_prompt": prompt[:120],
    })
    from contextlib import nullcontext
    with (_tq_ctx if _tq_ctx is not None else nullcontext()):
        with torch.inference_mode():
            result = pipe(**kwargs)
    elapsed = time.time() - t0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(str(output_path))

    alloc1, rsv1, peak1 = _cuda_memory_mb()
    _write_vitals({
        "state": "ready", "model": model_key,
        "pipeline": "FLUX" if family == "flux" else "SD",
        "pid": os.getpid(), "vram_alloc_mb": alloc1,
        "vram_reserved_mb": rsv1, "vram_peak_mb": peak1,
        "gen_step": steps, "gen_total_steps": steps,
        "last_gen_s": round(elapsed, 2), "last_prompt": prompt[:120],
        "last_image": str(output_path),
    })

    print(f"  ✓  Generated {asset_type} [{model_key}] in {elapsed:.1f}s → {output_path}")

    if use_cache and seed is not None:
        key = _cache_key(prompt, model_key, width, height, seed, steps)
        _cache_store(key, output_path)

    return output_path
