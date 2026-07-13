"""LoRA loading and fusing helpers for FLUX pipelines."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def load_lora(pipe: Any, repo: str, weight_name: str = "", adapter_name: str = "default",
              scale: float = 0.8) -> bool:
    """Load a LoRA adapter into a pipeline. Returns True on success."""
    try:
        pipe.load_lora_weights(repo, weight_name=weight_name or None, adapter_name=adapter_name)
        pipe.set_adapters([adapter_name], adapter_weights=[scale])
        return True
    except Exception as e:
        print(f"  ⚠  LoRA load failed ({adapter_name}): {e}", file=sys.stderr)
        return False


def unload_lora(pipe: Any) -> None:
    """Remove all LoRA adapters from pipeline."""
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass


def fuse_lora(pipe: Any, lora_scale: float = 1.0) -> bool:
    """Permanently fuse LoRA weights into the base model (faster inference, no hot-swap)."""
    try:
        pipe.fuse_lora(lora_scale=lora_scale)
        return True
    except Exception as e:
        print(f"  ⚠  LoRA fuse failed: {e}", file=sys.stderr)
        return False


def list_loras_for_model(model_key: str) -> dict[str, dict]:
    """Return LoRA configs available for the given model key."""
    from greenboost_cli.diffusion.models import MODELS
    cfg = MODELS.get(model_key, {})
    return cfg.get("loras", {})
