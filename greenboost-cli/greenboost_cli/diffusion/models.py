"""FLUX and SD model registry for diffusion pipeline."""
from __future__ import annotations

from pathlib import Path

def _greenboost_available() -> bool:
    """True if the GreenBoost kernel module is loaded (sysfs check)."""
    return Path("/sys/class/greenboost/greenboost/status").exists()

GREENBOOST_AVAILABLE = _greenboost_available()

MODELS: dict[str, dict] = {
    "klein-fp8": {
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "family": "flux",
        "quantization": "fp8",
        "steps": 4,
        "vram_gb": 9,
        "speed": "4–8s/image",
        "default_size": (1280, 720),
        "text_encoder": "Qwen3",
        "loras": {
            "arcane": {
                "repo": "DeverStyle/Flux.2-Klein-Loras",
                "weight": "klein-arcane.safetensors",
                "trigger": "arcane_visual_style",
                "scale": 0.80,
            },
            "sts2": {
                "repo": "DeverStyle/Flux.2-Klein-Loras",
                "weight": "klein-sts2.safetensors",
                "trigger": "sts2_style",
                "scale": 0.75,
            },
        },
    },
    "flux1-nf4": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "family": "flux",
        "quantization": "nf4",
        "steps": 28,
        "vram_gb": 8,
        "speed": "~3min/image",
        "default_size": (1024, 1024),
        "text_encoder": "T5",
    },
    "flux2-nf4": {
        "repo": "black-forest-labs/FLUX.2-dev",
        "family": "flux",
        "quantization": "nf4",
        "steps": 28,
        "vram_gb": 16,
        "speed": "~4–8min/image",
        "default_size": (1024, 1024),
        "text_encoder": "Mistral3",
        "requires_greenboost": True,
    },
    "sd15": {
        "repo": "runwayml/stable-diffusion-v1-5",
        "family": "sd",
        "steps": 25,
        "vram_gb": 2,
        "speed": "~5s/image",
        "default_size": (512, 512),
        "description": "General-purpose illustration & UI mockups",
    },
    "openjourney": {
        "repo": "prompthero/openjourney",
        "family": "sd",
        "steps": 25,
        "vram_gb": 2,
        "speed": "~6s/image",
        "default_size": (512, 512),
        "description": "Artistic/stylized Midjourney-like output",
        "prompt_prefix": "mdjrny-v4 style",
    },
    "realistic": {
        "repo": "SG161222/Realistic_Vision_V2.0",
        "family": "sd",
        "steps": 20,
        "vram_gb": 2,
        "speed": "~5s/image",
        "default_size": (512, 512),
        "description": "Photorealistic product/scene rendering",
    },
}

UI_PROMPT_TEMPLATES = {
    "hero": (
        "ultra-detailed digital illustration, modern {style} website hero image, "
        "{mood} atmosphere, {colors}, professional UI/UX design aesthetic, "
        "clean composition, high quality, cinematic lighting, "
        "concept art for a {product_type} product, no text, no UI elements"
    ),
    "background": (
        "abstract digital background texture, {style} aesthetic, "
        "{colors}, subtle geometric patterns, web design background, "
        "seamless, premium quality, no text, clean"
    ),
    "mood": (
        "design mood board, {style} visual style, {colors}, "
        "{mood} feeling, modern digital product aesthetic, "
        "concept art, visual language reference, no text, no labels"
    ),
    "illustration": (
        "modern digital illustration, flat design with depth, "
        "{style} aesthetic, {colors}, icon concept art for {product_type}, "
        "clean vector-art style, professional, minimal"
    ),
    "brand": (
        "brand identity visual, {style} corporate design language, "
        "{colors}, premium brand aesthetic, "
        "abstract logo concept, geometric, clean, professional"
    ),
}

UI_NEGATIVE = (
    "text, watermark, signature, UI chrome, browser, screenshot, "
    "ugly, blurry, low quality, distorted, extra elements, noise, artifacts"
)


def vram_introspect() -> float:
    """Return total VRAM in GB. Raises RuntimeError if CUDA is not available."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CPU spillover not permitted — GreenBoost must provide CUDA. "
            "Check that the kernel module is loaded and torch is the cu130-tagged build."
        )
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def auto_select_model() -> str:
    """Pick the best model based on available VRAM and live GreenBoost tier stats."""
    vram_gb = vram_introspect()
    if vram_gb < 10:
        return "klein-fp8"

    # Re-check at call time — module-level flag is frozen at import
    gb_loaded = _greenboost_available()
    if not gb_loaded:
        return "flux1-nf4" if vram_gb < 18 else "flux2-nf4"

    # With GreenBoost, use live T2 available MB to decide whether flux2-nf4 fits.
    # flux2-nf4 needs ~16 GB VRAM effective; T2 provides the headroom.
    try:
        from greenboost_cli.greenboost.monitor import get_tier_stats
        stats = get_tier_stats()
        if stats:
            t2_avail_gb = stats.get("t2_available_mb", 0) / 1024
            t2_pressure = stats.get("t2_pressure", 0)
            oom_active  = stats.get("oom_active", False)
            # Only use flux2 when T2 has ≥8 GB free and is not under pressure
            if t2_avail_gb >= 8.0 and t2_pressure == 0 and not oom_active:
                return "flux2-nf4"
            return "flux1-nf4"
    except Exception:
        pass

    return "flux2-nf4" if vram_gb >= 18 else "flux1-nf4"
