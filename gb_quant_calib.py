"""
gb_quant_calib.py , per-layer sensitivity calibration for the gb_quant quality planner.

Computes the relative quantization error for every nn.Linear in a model at each
candidate precision.  These estimates drive plan_quality()'s per-layer bit-allocation
decisions, turning quality-blind uniform compression into "spend bits where they matter".

Algorithm , relative Frobenius error proxy:

    rel_err(W, bits) = ‖W − Wq‖_F / ‖W‖_F

where Wq is the dequantized weight obtained by quantizing W to `bits` and
reconstructing in float32.  This equals the expected output error

    E[‖(W − Wq)·x‖] / E[‖W·x‖]   for x ~ N(0, I)

(by isotropy of Gaussian activations), which means the Frobenius ratio is the
right sensitivity proxy for randomly distributed inputs.  The dominant driver of
quantization sensitivity is the weight distribution (dynamic range, outlier fraction),
not the specific activation pattern , so random-activation calibration captures the
key variance at zero forward-pass cost and no model-specific setup.

A future `calibrate_with_prompts()` extension can refine with real activations for
the sub-0.1% regime; the API and cache schema are designed to slot it in cleanly.

Caching:
    Results are written to  ~/.cache/greenboost/sensitivity_<model_id>_<hash>.json.
    The hash covers model architecture (param shapes) + precision set + n_tokens.
    Warm loads are O(µs); cold runs are O(seconds, CPU only , no GPU needed).

Public API:
    from gb_quant_calib import calibrate_sensitivity

    sensitivity = calibrate_sensitivity(
        pipe.transformer,                   # nn.Module or diffusers pipeline
        precisions=("fp8", 4, "tq3"),       # precisions to evaluate
        model_id="flux2-klein-9b",          # for cache filename
    )
    # → {"single_blocks.0.linear1": {"fp8": 0.003, 4: 0.021, "tq3": 0.038}, …}
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

# Precision values this module knows how to emulate.
_KNOWN_PRECISIONS: frozenset = frozenset({16, "fp8", "e4m3", 8, 4, "nvfp4", "tq3", "tq2"})

# Minimum in_features to bother calibrating (mirrors _delegate_patch's MIN_SIZE=32).
_MIN_IN_FEATURES = 32

# Default layer-name substrings that are always kept at BF16 (mirrors gb_quant defaults).
_DEFAULT_SKIP_MODULES: Tuple[str, ...] = (
    "lm_head", "vision", "visual", "embed", "norm", "proj_out"
)

_CACHE_DIR = os.path.expanduser(
    os.environ.get("GB_QUANT_CACHE_DIR", "~/.cache/greenboost")
)


# ---------------------------------------------------------------------------
# Weight dequantization proxies , CPU float32 only, no GemLite / Triton needed
# ---------------------------------------------------------------------------

def _dequant_fp8(w: torch.Tensor, **_kw) -> torch.Tensor:
    """FP8 e4m3fn weight-only: per-channel absmax scale + cast."""
    # Use native fp8 dtype when available (torch ≥ 2.1 + CUDA 11.8+).
    if hasattr(torch, "float8_e4m3fn"):
        scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
        wq = (w / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)
        return wq * scale
    # Fallback: clamp to e4m3 dynamic range [−448, 448] + round to 256 levels.
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    return (w / scale).clamp(-448.0, 448.0).div(1.0 / 256).round().mul(1.0 / 256) * scale


def _dequant_int8(w: torch.Tensor, **_kw) -> torch.Tensor:
    """INT8 per-row absmax quantization and dequantization."""
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    return (w / scale).round().clamp(-127.0, 127.0) * scale


def _dequant_int4(w: torch.Tensor, group_size: int = 64, **_kw) -> torch.Tensor:
    """INT4 HQQ-style group-wise min-max quantization and dequantization."""
    out_f, in_f = w.shape
    pad = (-in_f) % group_size
    wp = torch.nn.functional.pad(w, (0, pad)) if pad else w
    g = wp.reshape(out_f, -1, group_size)
    mn = g.amin(dim=-1, keepdim=True)
    mx = g.amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp_min(1e-12) / 15.0  # 4-bit: 16 levels
    wq = ((g - mn) / scale).round().clamp(0.0, 15.0) * scale + mn
    return wq.reshape(out_f, -1)[:, :in_f]


def _dequant_nvfp4(w: torch.Tensor, **_kw) -> torch.Tensor:
    """NVFP4 emulation: 4-bit FP per group-16, FP8-scaled.
    Approximated as block absmax with 4-bit FP mantissa (e2m1 range ≈ 6.0)."""
    out_f, in_f = w.shape
    gs = 16
    pad = (-in_f) % gs
    wp = torch.nn.functional.pad(w, (0, pad)) if pad else w
    g = wp.reshape(out_f, -1, gs)
    # FP4 e2m1 max magnitude = 6.0; scale with FP8 absmax per group.
    scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 6.0
    wq = (g / scale).round().clamp(-6.0, 6.0) * scale
    return wq.reshape(out_f, -1)[:, :in_f]


def _dequant_tq3(w: torch.Tensor, **_kw) -> torch.Tensor:
    """TurboQuant tq3 proxy: INT3 per-row absmax (symmetric, 7 levels)."""
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 3.5
    return (w / scale).round().clamp(-3.0, 3.0) * scale


def _dequant_tq2(w: torch.Tensor, **_kw) -> torch.Tensor:
    """TurboQuant tq2 proxy: INT2 per-row absmax (symmetric, 3 levels)."""
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 1.5
    return (w / scale).round().clamp(-1.0, 1.0) * scale


_DEQUANT_FN = {
    "fp8": _dequant_fp8,
    "e4m3": _dequant_fp8,   # alias
    8: _dequant_int8,
    4: _dequant_int4,
    "nvfp4": _dequant_nvfp4,
    "tq3": _dequant_tq3,
    "tq2": _dequant_tq2,
}


def relative_quant_error(weight: torch.Tensor, bits,
                          group_size: int = 64) -> float:
    """
    Relative Frobenius error from quantizing `weight` to `bits`:

        ‖W − Wq‖_F / ‖W‖_F

    Returns 0.0 for bits=16 (BF16, no quantization) or for weights too small
    to quantize.  Returns float('nan') for unsupported `bits`.

    All computation is on CPU in float32.  The weight tensor can be on any device.
    """
    if bits == 16:
        return 0.0
    dequant_fn = _DEQUANT_FN.get(bits)
    if dequant_fn is None:
        return float("nan")

    w = weight.detach().float().cpu()
    if w.dim() != 2:
        return float("nan")
    out_f, in_f = w.shape
    if in_f < _MIN_IN_FEATURES:
        return 0.0  # too small → stays BF16 anyway

    w_norm = w.norm(p="fro")
    if w_norm < 1e-12:
        return 0.0

    wq = dequant_fn(w, group_size=group_size)
    err = (w - wq).norm(p="fro") / w_norm
    return float(err)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _model_hash(root: nn.Module, precisions: tuple, n_tokens: int) -> str:
    """Stable cache key from model parameter shapes + precision set."""
    parts = [f"{n}:{p.shape}" for n, p in root.named_parameters()]
    blob = (
        "\n".join(parts[:256])
        + f"|prec={sorted(str(p) for p in precisions)}"
        + f"|tok={n_tokens}"
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_path(model_id: str, cache_hash: str) -> str:
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_id)
    return os.path.join(_CACHE_DIR, f"sensitivity_{safe_id}_{cache_hash}.json")


def _load_cache(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)  # atomic write


def _fix_key_types(d: dict) -> dict:
    """Restore int keys (JSON serialises all keys as strings)."""
    out: dict = {}
    for layer_name, errs in d.items():
        fixed: dict = {}
        for k, v in errs.items():
            try:
                fixed[int(k)] = float(v)
            except (ValueError, TypeError):
                fixed[k] = float(v)
        out[layer_name] = fixed
    return out


def _serializable(d: dict) -> dict:
    """Convert all values to JSON-safe types."""
    out: dict = {}
    for layer_name, errs in d.items():
        out[layer_name] = {str(k): float(v) for k, v in errs.items()}
    return out


# ---------------------------------------------------------------------------
# Module extraction helper
# ---------------------------------------------------------------------------

def _iter_quantizable_linears(
    module: nn.Module,
    skip_modules: Tuple[str, ...],
) -> Iterable[Tuple[str, nn.Linear]]:
    """Yield (name, layer) for every quantizable nn.Linear in `module`."""
    for name, layer in module.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if any(s in name for s in skip_modules):
            continue
        if layer.in_features < _MIN_IN_FEATURES:
            continue
        if layer.weight is None:
            continue
        yield name, layer


# ---------------------------------------------------------------------------
# Main calibration entry point
# ---------------------------------------------------------------------------

def calibrate_sensitivity(
    module: nn.Module,
    precisions: Union[Tuple, list] = ("fp8", 4, "tq3"),
    model_id: str = "model",
    group_size: int = 64,
    n_tokens: int = 128,
    skip_modules: Tuple[str, ...] = _DEFAULT_SKIP_MODULES,
    verbose: bool = True,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    """
    Compute per-layer relative quantization error for each precision in `precisions`.

    Returns a dict mapping layer names to per-precision relative errors:
        {layer_name: {bits: rel_err, …}, …}
    where rel_err = ‖W − Wq‖_F / ‖W‖_F ∈ [0, 1].

    Results are cached to disk , second call for the same model + precision set
    loads from disk in milliseconds.  Use `force_recompute=True` to skip cache.

    Args:
        module:          nn.Module to calibrate (on any device, including CPU).
        precisions:      precision values to evaluate, e.g. ("fp8", 4, "nvfp4", "tq3").
        model_id:        human-readable string used in the cache filename.
        group_size:      INT4 quantization group size (matches runtime config).
        n_tokens:        reserved for real-activation calibration (currently unused;
                         affects cache key so changing it invalidates the cache).
        skip_modules:    layer name substrings to exclude (kept at BF16 unconditionally).
        verbose:         print progress to stdout.
        force_recompute: ignore disk cache, always recompute.
    """
    precisions = tuple(precisions)
    cache_hash = _model_hash(module, precisions, n_tokens)
    cpath = _cache_path(model_id, cache_hash)

    if not force_recompute:
        cached = _load_cache(cpath)
        if cached is not None:
            if verbose:
                print(f"[gb_quant_calib] sensitivity cache loaded: {cpath}",
                      flush=True)
            return _fix_key_types(cached)

    if verbose:
        print(f"[gb_quant_calib] computing per-layer sensitivity "
              f"({len(precisions)} precisions) for {model_id!r} …", flush=True)

    results: Dict[str, Dict] = {}
    layers = list(_iter_quantizable_linears(module, skip_modules))
    n = len(layers)

    for i, (name, layer) in enumerate(layers):
        layer_errs: Dict = {}
        for bits in precisions:
            err = relative_quant_error(layer.weight, bits, group_size=group_size)
            layer_errs[bits] = err
        results[name] = layer_errs
        if verbose and (i % 100 == 0 or i == n - 1):
            print(f"[gb_quant_calib]   {i + 1}/{n} layers …", flush=True)

    gc.collect()
    _save_cache(cpath, _serializable(results))
    if verbose:
        print(f"[gb_quant_calib] done , {len(results)} layers calibrated. "
              f"Cache saved: {cpath}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Convenience: calibrate_pipeline (one call per component)
# ---------------------------------------------------------------------------

def calibrate_pipeline_components(
    pipe,
    components: Tuple[str, ...] = ("transformer", "text_encoder"),
    precisions: Union[Tuple, list] = ("fp8", 4, "tq3"),
    model_id: str = "pipeline",
    group_size: int = 64,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict]]:
    """
    Calibrate multiple pipeline components separately.

    Returns:
        {component_name: {layer_name: {bits: rel_err}}, …}

    Each component is cached independently (cache key includes the component name
    in model_id) so re-running after changing only the transformer doesn't
    invalidate the encoder cache.
    """
    out: Dict[str, Dict[str, Dict]] = {}
    for comp_name in components:
        mod = getattr(pipe, comp_name, None)
        if mod is None or not isinstance(mod, nn.Module):
            if verbose:
                print(f"[gb_quant_calib] skip {comp_name!r} (not found or not Module)",
                      flush=True)
            continue
        comp_id = f"{model_id}.{comp_name}"
        out[comp_name] = calibrate_sensitivity(
            mod, precisions=precisions, model_id=comp_id,
            group_size=group_size, verbose=verbose,
        )
    return out
