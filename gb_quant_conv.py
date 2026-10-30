"""gb_quant_conv: FP8 weight-only compression for convolution layers (the conv half of gb-quant's fp8 default).

gb_quant covers nn.Linear (GemLite FP8 W8A16, torch._scaled_mm W8A8). Convolutions are most of the weights of UNets, upscalers (Real-ESRGAN, SwinIR-class CNNs), video
VAEs and vision towers, and until now they stayed 16/32-bit. This module stores every eligible Conv1d/2d/3d and ConvTranspose1d/2d/3d weight as float8_e4m3fn with one fp32 scale
per output channel (the same channel-wise e4m3 scheme as gb_quant's fp8 Linear), and dequantizes it to the input dtype for each forward (a transient copy of one layer's
weight, cuDNN does the convolution). It saves capacity (1 B/param vs 2 B fp16, 4 B fp32); it does not speed the convolution up.

Why weight-only and not fp8 x fp8 (W8A8) for convs: the 5070 does 238 TFLOP/s fp8 vs 128 fp16 (workflow/fp8-prefill-blackwell-scope.md), but a W8A8 conv needs
an implicit-GEMM kernel or an im2col that expands activations ~9x for 3x3 on a 544 GB/s card; the 1x1 convs are Linear in disguise and are served by gb_quant's Linear path.
Recorded as a not-built option in workflow/gb-quant-conv.md.

Recipe (from the literature, see workflow/gb-quant-conv.md):
  * per-output-channel scales, e4m3 (Intel, "Efficient Post-training Quantization with FP8 Formats", arXiv 2309.14592: E4M3 for language, E3M4 marginally better for vision;
    torch has no e3m4, e4m3fn with per-channel scales is the closest);
  * the first and the last convolution stay in high precision (same paper: quantizing the first/last operators costs the most accuracy on CNNs);
  * a per-layer guard: a layer whose fp8 round-trip error exceeds `max_rel_err` (default 6%; fp8 normally gives ~2.6%) stays unquantized;
  * tiny layers (< `min_elems` weights) stay unquantized: nothing to gain.

Env: GB_QUANT_CONV=0 disables the hook in gb_quant.quantize_module; GB_QUANT_CONV_MIN_ELEMS, GB_QUANT_CONV_MAX_REL_ERR, GB_QUANT_CONV_KEEP_FIRST_LAST=0 override the defaults.
Importing needs only torch; nothing touches the GPU until a module is quantized.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

E4M3_MAX = 448.0
_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_TRANSPOSED_TYPES = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
_CONV_FN = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
_TCONV_FN = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}
# Name fragments left alone (mirrors gb_quant._DEFAULT_SKIP_MODULES for the parts that make sense on convs).
DEFAULT_SKIP = ("embed", "norm", "lm_head", "proj_out")


def enabled() -> bool:
    return os.environ.get("GB_QUANT_CONV", "1").strip().lower() not in ("0", "off", "false", "no")


def available() -> bool:
    return hasattr(torch, "float8_e4m3fn")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _channel_dims(weight: torch.Tensor, transposed: bool) -> tuple:
    """Dims reduced to get one scale per OUTPUT channel (Conv: dim 0; ConvTranspose: dim 1)."""
    return tuple(d for d in range(weight.dim()) if d != (1 if transposed else 0))


def quantize_weight(weight: torch.Tensor, transposed: bool = False):
    """(float8_e4m3fn weight, fp32 per-output-channel scale shaped to broadcast against the weight)."""
    w = weight.detach().float()
    amax = w.abs().amax(dim=_channel_dims(w, transposed), keepdim=True).clamp_min(1e-12)
    scale = amax / E4M3_MAX
    return (w / scale).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


def dequantize_weight(w8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (w8.to(torch.float32) * scale).to(dtype)


# Stored form on the module: the fp8 weight as a uint8 view and the fp32 scale as an int32 view. Both are NON-floating dtypes on purpose: module.half(), .to(torch.bfloat16),
# .float() and friends only cast floating tensors, so they cannot silently re-expand the weights to 16 bit or round the scales to fp16 (measured 2026-10-29: a float8 buffer IS
# cast by .to(float16)). The views cost nothing.
def pack(w8: torch.Tensor, scale: torch.Tensor):
    return w8.contiguous().view(torch.uint8), scale.to(torch.float32).contiguous().view(torch.int32)


def unpack(m: nn.Module):
    return m.weight_fp8.view(torch.float8_e4m3fn), m.weight_scale.view(torch.float32)


def rel_error(weight: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor) -> float:
    w = weight.detach().float()
    d = dequantize_weight(w8, scale, torch.float32)
    return float((d - w).norm() / w.norm().clamp_min(1e-12))


def _nsp(m: nn.Module) -> int:
    return len(m.kernel_size) if isinstance(m.kernel_size, (tuple, list)) else 1


def _fp8_conv_forward(self: nn.Module, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    dt = input.dtype if input.is_floating_point() else torch.float32
    w = dequantize_weight(*unpack(self), dt)
    bias = self.bias.to(dt) if self.bias is not None else None
    n = _nsp(self)
    if isinstance(self, _TRANSPOSED_TYPES):
        output_size = args[0] if args else kwargs.get("output_size")
        out_pad = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, n, self.dilation)
        return _TCONV_FN[n](input, w, bias, self.stride, self.padding, out_pad, self.groups, self.dilation)
    if self.padding_mode != "zeros":
        return _CONV_FN[n](F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), w, bias, self.stride, (0,) * n, self.dilation, self.groups)
    return _CONV_FN[n](input, w, bias, self.stride, self.padding, self.dilation, self.groups)


def conv_weight_bytes(module: nn.Module) -> int:
    """Bytes of the conv weights that are still unquantized 16/32-bit tensors."""
    return sum(m.weight.numel() * m.weight.element_size() for m in module.modules()
               if isinstance(m, _CONV_TYPES + _TRANSPOSED_TYPES) and getattr(m, "weight", None) is not None and m.weight.dtype.is_floating_point and not m.weight.is_meta)


@dataclass
class ConvQuantReport:
    quantized: int = 0
    kept: Dict[str, str] = field(default_factory=dict)       # layer name -> why it stayed as it was
    bytes_before: int = 0
    bytes_after: int = 0
    max_rel_err: float = 0.0
    mean_rel_err: float = 0.0
    layers: List[str] = field(default_factory=list)

    def summary(self) -> str:
        mb = lambda b: f"{b / 1048576:.1f} MiB"  # noqa: E731
        return (f"gb_quant_conv: {self.quantized} conv layer(s) fp8 ({mb(self.bytes_before)} -> {mb(self.bytes_after)}), {len(self.kept)} kept, "
                f"rel err mean {self.mean_rel_err * 100:.2f}% max {self.max_rel_err * 100:.2f}%")


def quantize_convs(module: nn.Module, *, device: Optional[str] = None, min_elems: Optional[int] = None, max_rel_err: Optional[float] = None,
                   keep_first_last: Optional[bool] = None, skip: Iterable[str] = DEFAULT_SKIP) -> ConvQuantReport:
    """Replace the weight of every eligible conv in `module` with fp8 + per-channel scale, IN PLACE (the module objects stay alive and keep their identity, like gb_quant's
    Linear patch). Moves one layer at a time when `device` is given. Returns a report; never raises for a layer it cannot handle (it stays as it was and the report says why)."""
    rep = ConvQuantReport()
    if not available():
        rep.kept["*"] = "torch has no float8_e4m3fn"
        return rep
    min_elems = int(_env_float("GB_QUANT_CONV_MIN_ELEMS", 16384)) if min_elems is None else min_elems
    max_rel_err = _env_float("GB_QUANT_CONV_MAX_REL_ERR", 0.06) if max_rel_err is None else max_rel_err
    if keep_first_last is None:
        keep_first_last = os.environ.get("GB_QUANT_CONV_KEEP_FIRST_LAST", "1").strip().lower() not in ("0", "off", "false", "no")
    convs = [(n, m) for n, m in module.named_modules() if isinstance(m, _CONV_TYPES + _TRANSPOSED_TYPES)]
    skip = tuple(skip)
    errs: List[float] = []
    for i, (name, m) in enumerate(convs):
        lname = name or type(m).__name__
        if hasattr(m, "weight_fp8"):
            rep.kept[lname] = "already fp8"
            continue
        if m.weight is None or m.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or m.weight.is_meta:
            rep.kept[lname] = "no real weight"
            continue
        if keep_first_last and len(convs) > 2 and i in (0, len(convs) - 1):
            rep.kept[lname] = "first/last conv stays high precision"
            continue
        if any(s in lname.lower() for s in skip):
            rep.kept[lname] = "skip list"
            continue
        if m.weight.numel() < min_elems:
            rep.kept[lname] = f"small ({m.weight.numel()} weights)"
            continue
        try:
            w = m.weight.data if device is None else m.weight.data.to(device)
            transposed = isinstance(m, _TRANSPOSED_TYPES)
            w8, scale = quantize_weight(w, transposed)
            err = rel_error(w, w8, scale)
            if not (err == err) or err > max_rel_err:
                rep.kept[lname] = f"fp8 error {err * 100:.1f}% over the {max_rel_err * 100:.0f}% limit"
                continue
            rep.bytes_before += m.weight.numel() * m.weight.element_size()
            p8, ps = pack(w8, scale)
            m.register_buffer("weight_fp8", p8)
            m.register_buffer("weight_scale", ps)
            del m._parameters["weight"]
            m.weight = None                      # type: ignore[assignment]
            m.forward = _fp8_conv_forward.__get__(m, type(m))
            if device is not None and m.bias is not None:
                m.bias.data = m.bias.data.to(device)
            rep.bytes_after += p8.numel() + ps.numel() * 4
            errs.append(err)
            rep.layers.append(lname)
        except Exception as e:  # noqa: BLE001  - a layer we cannot handle stays as it was
            rep.kept[lname] = f"{type(e).__name__}: {e}"
    rep.quantized = len(errs)
    if errs:
        rep.max_rel_err, rep.mean_rel_err = max(errs), sum(errs) / len(errs)
    return rep


def maybe_quantize_convs(module: nn.Module, device: Optional[str] = None) -> Optional[ConvQuantReport]:
    """The gb_quant.quantize_module hook: fp8 the convs of `module` only when it is worth it, else return None and touch nothing.

    Worth it = enabled (GB_QUANT_CONV != 0), torch has e4m3, and the unquantized conv weights total at least GB_QUANT_CONV_MIN_TOTAL_MB (default 256 MiB). Measured 2026-10-29 on the
    5070: Real-ESRGAN x4 (31.8 MiB of fp16 weights) saves 15.8 MiB with fp8 and loses 13.6 dB of fidelity (43.7 dB against the fp32 result; fp16 itself gives 57.3 dB) at the same speed,
    so small CNNs stay as they are; a UNet or video VAE with hundreds of MiB of conv weights is where the capacity matters. Quality-first, same reasoning as the fp8 floor."""
    if not (enabled() and available()):
        return None
    if conv_weight_bytes(module) < _env_float("GB_QUANT_CONV_MIN_TOTAL_MB", 256.0) * 1048576:
        return None
    rep = quantize_convs(module, device=device)
    print(rep.summary(), flush=True)
    try:
        import gb_dataflux
        gb_dataflux.emit({"node": "host", "label": "gb_quant_conv", "kind": "quant_conv_result", "n_items": rep.quantized, "items": [type(module).__name__], "duration_s": 0.0, "status": "ok",
                          "quantized": rep.quantized, "kept": len(rep.kept), "bytes_before": rep.bytes_before, "bytes_after": rep.bytes_after,
                          "mean_rel_err": round(rep.mean_rel_err, 5), "max_rel_err": round(rep.max_rel_err, 5)})
    except Exception:  # noqa: BLE001  - telemetry never stops a quantize
        pass
    return rep


def recommendations(cc: Optional[tuple] = None, vram_gb: Optional[float] = None) -> List[dict]:
    """Which fp8-class compression path to use for which kind of model, for THIS GPU (cc = (major, minor), vram_gb = total VRAM; both detected when None).
    Shown by the quant_advisor MCP tool as `fp8_paths`, so an agent working on a GPU like the RTX 5070 (sm_120, 12 GB) gets the same answer every time. Numbers are the measured
    ones from workflow/gb-quant-conv.md; `applies` says whether the path exists on this GPU."""
    if cc is None or vram_gb is None:
        try:
            import torch
            if torch.cuda.is_available():
                cc = cc or tuple(torch.cuda.get_device_capability(0))
                vram_gb = vram_gb or torch.cuda.get_device_properties(0).total_memory / 2**30
        except Exception:  # noqa: BLE001
            pass
    fp8_hw = bool(cc) and tuple(cc) >= (8, 9)
    small = bool(vram_gb) and vram_gb <= 16
    out = [
        {"path": "linear_fp8_w8a16", "model_kinds": "transformers, DiT, text encoders (every nn.Linear)", "applies": True, "default": True,
         "how": "gb_quant.quantize_module(bits='fp8'): GemLite e4m3, channel-wise scales, bf16 compute", "gain": "1 B/param: half of bf16, about 2.6% weight error (the quality floor)"},
        {"path": "linear_fp8_w8a8_scaled_mm", "model_kinds": "compute-bound Linear work: prefill, diffusion denoise", "applies": fp8_hw, "default": False,
         "how": "GB_KERNEL_BACKEND=scaled_mm (torch._scaled_mm, cuBLASLt; both dims 16-aligned)",
         "gain": "fp8 MMA 238 TFLOP/s vs 128 fp16 on the 5070 (measured, workflow/fp8-prefill-blackwell-scope.md); decode is bandwidth bound and does not gain"},
        {"path": "conv_fp8_weight_only", "model_kinds": "UNets, CNN upscalers, video VAEs, vision towers (Conv1d/2d/3d, ConvTranspose)", "applies": available(), "default": True,
         "how": "gb_quant.quantize_module(bits='fp8') calls gb_quant_conv when the conv weights total >= GB_QUANT_CONV_MIN_TOTAL_MB (256); first and last conv stay high precision; GB_QUANT_CONV=0 disables",
         "gain": "capacity only (no speedup): 4x smaller than fp32, 2x than fp16; FLUX.2 VAE 160 -> 82 MiB with 46.9 dB between fp8 and bf16 decodes; Real-ESRGAN 31.8 -> 16.0 MiB but 43.7 dB vs the fp32 result (fp16: 57.3), so below the gate it stays fp16"},
        {"path": "vae", "model_kinds": "image/video VAEs", "applies": True, "default": False,
         "how": "keep bf16 (gb_quant skips 'vae' components); decode OOM is activation bound: tile or slice the decode", "gain": "fp8 conv weights would save under 100 MiB of a VAE; peak VRAM is activations (2.6 GB at 1024 px)"},
        {"path": "kv_cache", "model_kinds": "LLM / VLM serving", "applies": True, "default": False,
         "how": "llama.cpp: kv=f16, q8_0 when the context must grow; torch engines: gb_attn TurboQuant K/V", "gain": "KV fp8 raised batched throughput 56% in TensorRT-LLM's guide; a single stream on 12 GB gains capacity, not speed"},
        {"path": "gguf_llm", "model_kinds": "GGUF (Qwen3.8-35B-A3B APEX)", "applies": True, "default": True,
         "how": "Q4-class GGUF with -fa on, MoE experts on the CPU as needed, MTP, ubatch sized to the free VRAM", "gain": "llama.cpp has no fp8 compute path for GGUF, so fp8 does not apply to it"},
        {"path": "below_fp8", "model_kinds": "any", "applies": True, "default": False,
         "how": "nvfp4/int4 only with an explicit owner decision", "gain": "nvfp4 measured ~9% slower and lossier than fp8 on this card"},
    ]
    if small:
        out.append({"path": "budget", "model_kinds": "any", "applies": True, "default": True,
                    "how": f"{vram_gb:.0f} GB card: keep the peak under ~10 GB (the desktop holds 1 to 7 GB), one heavy model at a time, text encoders quantized then freed",
                    "gain": "avoids the OOM class of failure"})
    out.append({"path": "known_issue_half_conv_under_shim", "model_kinds": "torch CNNs run under the GreenBoost shim", "applies": True, "default": False,
                "how": "fp16 and bf16 convolutions returned NaN under the shim in the ai-forge mesh env (fp32 was fine; not cuDNN, not T2, not expandable segments); run such workers shimless or in fp32 until fixed",
                "gain": "correctness"})
    return out
