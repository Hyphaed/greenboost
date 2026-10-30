"""gb_cnn_tune: the torch flags that make convolutional workers (Real-ESRGAN, SwinIR-class upscalers, SAM2/vision CNN parts, UNets, video VAEs) faster on a tensor-core GPU.

Measured 2026-10-29 on the RTX 5070 (Real-ESRGAN x4plus, fp16, 280x280 tiles, 40 timed runs after warm-up):
  eager NCHW (before)        108.1 ms/tile   peak 685 MiB
  channels_last                81.5 ms/tile   peak 379 MiB     <- 1.33x faster, 45% less peak VRAM
  cudnn.benchmark only       108.8 ms/tile   peak 991 MiB     <- no gain on this net, +45% peak: OFF by default
  channels_last + benchmark   81.4 ms/tile   peak 685 MiB
So only channels_last is on by default. Env: GB_CNN_CHANNELS_LAST=0 disables it, GB_CNN_CUDNN_BENCHMARK=1 enables the autotuner (worth trying on a net whose shapes repeat many times).
Importing needs only torch; nothing here touches the GPU.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "off", "false", "no")


def channels_last_enabled() -> bool:
    return _flag("GB_CNN_CHANNELS_LAST", True)


def apply(model: nn.Module, *, channels_last: Optional[bool] = None, benchmark: Optional[bool] = None) -> nn.Module:
    """Convert `model` (in place) to channels_last when it holds 4-D convolutions, and set the cudnn autotuner. Returns the model. Pair with `to_input` for the inputs."""
    cl = channels_last_enabled() if channels_last is None else channels_last
    torch.backends.cudnn.benchmark = _flag("GB_CNN_CUDNN_BENCHMARK", False) if benchmark is None else benchmark
    if cl and any(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) for m in model.modules()):
        model.to(memory_format=torch.channels_last)
    return model


def to_input(x: torch.Tensor, channels_last: Optional[bool] = None) -> torch.Tensor:
    """Give a 4-D input the memory format the tuned model expects (no copy when it already has it)."""
    cl = channels_last_enabled() if channels_last is None else channels_last
    return x.contiguous(memory_format=torch.channels_last) if cl and x.dim() == 4 else x
