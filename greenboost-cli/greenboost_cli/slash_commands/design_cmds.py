"""Design slash commands: /design /design-gen /design-intel /design-models."""
from __future__ import annotations

import sys
from pathlib import Path

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.environment.settings import GB_HOME

_ASSETS_DIR = GB_HOME / "design_assets"


def _design(args: str, session, settings: dict) -> None:
    """Run the full design pipeline: intelligence + optional image generation."""
    query = args.strip().strip('"').strip("'")
    if not query:
        print("  Usage: /design \"product or feature description\"")
        return

    try:
        from greenboost_cli.design.intelligence import (
            generate_design_system, format_design_system, is_available
        )
        if not is_available(settings):
            print("  ⚠  ui-ux-pro-max-skill data not found — design intelligence unavailable.")
        else:
            ds = generate_design_system(query, settings=settings)
            print(format_design_system(ds))
    except ImportError:
        print("  ⚠  design intelligence module unavailable.")


def _design_gen(args: str, session, settings: dict) -> None:
    """Generate a UI asset with FLUX/SD diffusion."""
    from greenboost_cli.diffusion.models import MODELS, auto_select_model

    parts = args.strip().split()
    if not parts:
        print("  Usage: /design-gen <asset_type> [--model klein-fp8] [--lora arcane]"
              " [--style ...] [--colors ...] [--seed N] [--size WxH]")
        print("  Asset types: hero background mood illustration brand")
        return

    asset_type = parts[0]
    model_key  = settings.get("diffusion_model") or auto_select_model()
    use_lora   = None
    style      = "glassmorphism"
    colors     = "deep blue and violet gradient"
    mood       = "professional and modern"
    product    = "SaaS application"
    seed       = None
    width = height = None

    i = 1
    while i < len(parts):
        if parts[i] == "--model"  and i + 1 < len(parts): model_key = parts[i+1]; i += 2
        elif parts[i] == "--lora" and i + 1 < len(parts): use_lora  = parts[i+1]; i += 2
        elif parts[i] == "--style" and i + 1 < len(parts): style    = parts[i+1]; i += 2
        elif parts[i] == "--colors" and i + 1 < len(parts): colors  = parts[i+1]; i += 2
        elif parts[i] == "--seed"  and i + 1 < len(parts):
            try: seed = int(parts[i+1])
            except ValueError: pass
            i += 2
        elif parts[i] == "--size" and i + 1 < len(parts):
            try:
                w_str, h_str = parts[i+1].lower().split("x")
                width, height = int(w_str), int(h_str)
            except Exception: pass
            i += 2
        else:
            i += 1

    out_dir = Path(settings.get("diffusion_output_dir") or _ASSETS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix   = f"_{seed}" if seed is not None else ""
    out_path = out_dir / f"{asset_type}_{model_key}{suffix}.png"

    print(f"  Generating {asset_type} with {model_key} …")
    try:
        from greenboost_cli.diffusion.pipeline import generate_ui_asset
        path = generate_ui_asset(
            asset_type=asset_type,
            output_path=out_path,
            style=style, colors=colors, mood=mood, product_type=product,
            model_key=model_key, use_lora=use_lora,
            use_turboquant=settings.get("greenboost_turboquant", False),
            width=width, height=height, seed=seed,
        )
        # Optional OpenCV post-process
        try:
            from greenboost_cli.diffusion.postprocess import postprocess
            outputs = postprocess(path, sharpen=True, webp=True)
            if outputs and outputs[0] != path:
                print(f"  ✓  Post-processed → {outputs[0]}")
        except ImportError:
            pass
    except ImportError as e:
        print(f"  ✗  diffusers/torch not installed: {e}")
        print("     Run: pip install greenboost-cli[diffusion]")


def _design_intel(args: str, session, settings: dict) -> None:
    """Query the design intelligence database."""
    query = args.strip().strip('"').strip("'")
    if not query:
        print("  Usage: /design-intel \"<query>\"")
        return

    try:
        from greenboost_cli.design.intelligence import search, is_available
        if not is_available(settings):
            print("  ⚠  ui-ux-pro-max-skill data not found.")
            return
        results = search(query, top_k=3, settings=settings)
        for domain, rows in results.items():
            if not rows:
                continue
            print(f"\n── {domain} ──")
            for row in rows:
                print("  " + " | ".join(f"{k}: {v}" for k, v in list(row.items())[:3] if v))
    except ImportError:
        print("  ⚠  design intelligence unavailable.")


def _design_models(args: str, session, settings: dict) -> None:
    """List available diffusion models."""
    from greenboost_cli.diffusion.models import MODELS, GREENBOOST_AVAILABLE, vram_introspect

    vram = vram_introspect()
    print(f"\n  Available diffusion models  (VRAM: {vram:.1f} GB)")
    print()
    for key, cfg in MODELS.items():
        gb_req   = "  (requires GreenBoost)" if cfg.get("requires_greenboost") else ""
        loras    = list(cfg.get("loras", {}).keys())
        lora_str = f"  LoRAs: {', '.join(loras)}" if loras else ""
        desc     = cfg.get("description", "")
        marker   = "✓" if (not cfg.get("requires_greenboost") or GREENBOOST_AVAILABLE) else "✗"
        print(f"  {marker}  {key:<16}  {cfg['speed']:<22}  ~{cfg['vram_gb']}GB{gb_req}{lora_str}")
        if desc:
            print(f"     {'':16}  {desc}")
    print()
    if GREENBOOST_AVAILABLE:
        print("  ✓  GreenBoost detected (T2 RAM overflow available)")


register_command("design",        _design,       "Run full design pipeline  (/design \"description\")")
register_command("design-gen",    _design_gen,   "Generate UI asset with FLUX/SD  (/design-gen hero)")
register_command("design-intel",  _design_intel, "Query design intelligence DB  (/design-intel \"query\")")
register_command("design-models", _design_models,"List available diffusion models")
