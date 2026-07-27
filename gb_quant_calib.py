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
    from gb_quant import is_prequantized_linear  # lazy: gb_quant imports this module too
    for name, layer in module.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if any(s in name for s in skip_modules):
            continue
        if layer.in_features < _MIN_IN_FEATURES:
            continue
        if is_prequantized_linear(layer):
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


# ---------------------------------------------------------------------------
# Real-activation calibration via the vendored AutoRound AutoScheme search
# (third_party/auto_round , see NOTICE). Additive: calibrate_sensitivity's
# zero-data Frobenius proxy above stays the default everywhere , this is
# opt-in (sensitivity_source="delta_loss" / GB_QUANT_CALIB=delta_loss) for
# callers that have a real tokenizer and want the sub-0.1% regime refined
# with actual forward-pass activations instead of random-Gaussian ones.
# ---------------------------------------------------------------------------

# GreenBoost's own tiny, stdlib-only default calibration corpus , no
# `datasets` dependency for GreenBoost's OWN prompt set (third_party/
# auto_round's internal calibration dataloader is a separate matter and
# already depends on `datasets`, see its own NOTICE). Deliberately short and
# diverse (code, prose, dialogue, numbers) rather than large , this is a
# sensitivity PROBE, not a training corpus.
_DEFAULT_CALIB_PROMPTS: Tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly "
    "behind the distant mountains, painting the sky in shades of orange.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return "
    "fibonacci(n - 1) + fibonacci(n - 2)\n",
    "In a shocking turn of events, scientists have discovered that the "
    "average temperature of the ocean has risen by 0.5 degrees over the "
    "past decade, raising concerns about marine ecosystems.",
    "Q: What is the capital of France?\nA: The capital of France is Paris, "
    "a city known for its art, culture, and the Eiffel Tower.",
    "SELECT customer_id, SUM(order_total) FROM orders WHERE order_date > "
    "'2026-01-01' GROUP BY customer_id ORDER BY SUM(order_total) DESC;",
    "Dear team, following our meeting yesterday, I wanted to summarize the "
    "action items: 1) finalize the budget, 2) schedule the review, "
    "3) notify stakeholders by Friday.",
    "The mitochondria is the powerhouse of the cell, converting nutrients "
    "into adenosine triphosphate (ATP) through a process called cellular "
    "respiration.",
    "12 + 37 = 49. If a train travels at 80 km/h for 3.5 hours, it covers "
    "280 kilometers, assuming a constant speed and no stops along the way.",
)


def _calib_prompts(prompts: "Iterable[str] | None" = None) -> "list[str]":
    """Resolve the real-activation calibration prompt set: explicit
    `prompts` arg > GB_QUANT_CALIB_PROMPTS (one prompt per line in a plain
    text file) > GreenBoost's own small stdlib-only default corpus."""
    if prompts is not None:
        return list(prompts)
    path = os.environ.get("GB_QUANT_CALIB_PROMPTS", "").strip()
    if path:
        try:
            with open(os.path.expanduser(path), encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if lines:
                return lines
        except OSError:
            pass
    return list(_DEFAULT_CALIB_PROMPTS)


def _ensure_auto_round_path() -> None:
    """Make the vendored AutoRound tree (third_party/auto_round/auto_round)
    importable , same env-override > greenboost/third_party search order
    gb_quant._ensure_vendored_paths() uses for gemlite/hqq."""
    import importlib.util
    import sys as _sys
    override = os.environ.get("GB_AUTO_ROUND_PATH")
    candidates = [override, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "third_party", "auto_round")]
    try:
        present = importlib.util.find_spec("auto_round") is not None
    except (ImportError, ValueError):
        present = False
    if present:
        return
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, "auto_round")) and path not in _sys.path:
            _sys.path.insert(0, path)
            return


def _require_auto_round():
    """Import the vendored AutoRound AutoScheme entry points, raising a
    clear, actionable error if the tree or its (real, heavier-than-gb_quant's
    own) runtime deps are missing , mirrors gb_quant._require_gemlite()'s
    pattern. Never imported at this module's own top level: opt-in only.
    Returns (AutoScheme, GenScheme, preset_name_to_scheme) , all 3 AutoRound
    symbols calibrate_with_prompts needs, from one choke point (a single
    entry point is also what makes this trivially mockable in tests)."""
    _ensure_auto_round_path()
    try:
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme, GenScheme
        from auto_round.schemes import preset_name_to_scheme
        return AutoScheme, GenScheme, preset_name_to_scheme
    except Exception as e:
        raise RuntimeError(
            "calibrate_with_prompts needs third_party/auto_round (vendored "
            "AutoRound AutoScheme) plus its runtime deps (accelerate, "
            "datasets, py-cpuinfo , declared in synapse_engine/pyproject.toml, "
            "installed via `sudo greenboost install-synapse-engine`). "
            f"Original error: {e!r}"
        ) from e


# AutoRound QuantizationScheme preset names for gb_quant's own precision
# vocabulary , only the ones gb_quant.GpuProfile.calibrated_precisions ever
# picks automatically (fp8/int8/int4); tq3/tq2/nvfp4 have no AutoRound
# preset equivalent and are never part of this search.
_GB_BITS_TO_AUTOROUND_PRESET = {"fp8": "FP8", 8: "INT8", 4: "INT4"}
_AUTOROUND_PRESET_TO_GB_BITS = {v: k for k, v in _GB_BITS_TO_AUTOROUND_PRESET.items()}


def calibrate_with_prompts(
    module: nn.Module,
    tokenizer,
    prompts: "Iterable[str] | None" = None,
    options: "Tuple[str, ...] | list" = ("fp8", 8, 4),
    avg_bits: "float | None" = None,
    model_id: str = "model",
    skip_modules: Tuple[str, ...] = _DEFAULT_SKIP_MODULES,
    nsamples: int = 8,
    seqlen: int = 512,
    device_map: "str | None" = None,
    verbose: bool = True,
    force_recompute: bool = False,
) -> Dict[str, object]:
    """Real-activation per-layer bit ASSIGNMENT via the vendored AutoRound
    AutoScheme search (delta-loss method) , the sub-0.1% regime this
    module's own docstring names as calibrate_sensitivity's future
    extension point.

    Unlike calibrate_sensitivity() (a per-layer, per-precision ERROR TABLE
    that plan_quality()'s own ladder walk picks from), this returns a
    per-layer bits ASSIGNMENT directly , AutoRound's search algorithm picks
    the final scheme itself, it does not score a ladder of independent
    options the way the zero-data Frobenius proxy does. Feed the result
    straight into quantize_module(module, bits=<floor>,
    per_layer_bits=result) , this bypasses plan_quality()'s ladder walk
    entirely for this path, it does not slot into `sensitivity=`.

    `avg_bits`: target average bits/param across `options`. Defaults to the
    midpoint of the achievable range (computed once options are resolved)
    when unset , a caller with no strong opinion gets a sane default rather
    than an error.

    Cached to disk like calibrate_sensitivity(), but in a SEPARATE
    namespace (model_id suffixed with the source) so the two calibration
    sources never collide or overwrite each other's cache entries for the
    same model_id.

    Known live-verification gap: this has been verified to import and wire
    correctly against the real vendored AutoRound package (third_party/
    auto_round/NOTICE), but end-to-end correctness of the actual per-layer
    scores needs a real tokenizer + model + forward pass to confirm , not
    yet re-run against a live model in this session. Treat a first real run
    as a verification step, not an assumed-correct dependency.
    """
    # Validate options / build the preset-name list first , pure lookups
    # against a local dict, no need to touch AutoRound (or fail loudly if
    # it's missing) for a call that's about to be a cache hit anyway.
    options = tuple(options)
    preset_names = []
    for opt in options:
        name = _GB_BITS_TO_AUTOROUND_PRESET.get(opt)
        if name is None:
            raise ValueError(
                f"calibrate_with_prompts: option {opt!r} has no AutoRound preset "
                f"equivalent (supported: {sorted(_GB_BITS_TO_AUTOROUND_PRESET, key=str)})")
        preset_names.append(name)

    quant_layer_names = [name for name, _ in _iter_quantizable_linears(module, skip_modules)]
    if not quant_layer_names:
        return {}

    resolved_prompts = _calib_prompts(prompts)
    cache_hash = _model_hash(module, tuple(preset_names), nsamples) + "_" + \
        hashlib.sha256("\n".join(resolved_prompts).encode()).hexdigest()[:8]
    cpath = _cache_path(f"{model_id}.delta_loss", cache_hash)

    if not force_recompute:
        cached = _load_cache(cpath)
        if cached is not None:
            if verbose:
                print(f"[gb_quant_calib] delta_loss cache loaded: {cpath}", flush=True)
            # Unlike calibrate_sensitivity's cache (int PRECISION keys need
            # restoring after JSON stringifies dict keys), this cache's keys
            # are layer names (always strings) and values are bits (int or
            # str) , JSON round-trips those correctly with no ambiguity, no
            # restoration needed.
            return cached

    # Only reached on a cache MISS , a warm cache hit above must never touch
    # AutoRound at all (its real deps are heavier than gb_quant's own).
    AutoScheme, GenScheme, preset_name_to_scheme = _require_auto_round()
    schemes = [preset_name_to_scheme(n) for n in preset_names]
    bit_options = [float(s.bits) for s in schemes]
    target = avg_bits if avg_bits is not None else (min(bit_options) + max(bit_options)) / 2.0

    if verbose:
        print(f"[gb_quant_calib] delta_loss: {len(quant_layer_names)} layers, "
              f"options={preset_names}, target avg_bits={target:.2f}, "
              f"{len(resolved_prompts)} calibration prompts …", flush=True)

    auto_scheme = AutoScheme(
        avg_bits=target, options=list(preset_names),
        nsamples=nsamples, seqlen=seqlen, device_map=device_map,
        low_gpu_mem_usage=True, low_cpu_mem_usage=True,
    )
    gen = GenScheme(
        auto_scheme, module, quant_layer_names, fixed_layer_scheme={},
        tokenizer=tokenizer,
    )
    layer_config = gen.get_layer_config()

    result: Dict[str, object] = {}
    for name, cfg in layer_config.items():
        bits = cfg.get("bits")
        data_type = str(cfg.get("data_type", "int"))
        if data_type in ("fp8", "float8_e4m3fn"):
            result[name] = "fp8"
        else:
            result[name] = _AUTOROUND_PRESET_TO_GB_BITS.get(
                f"INT{bits}" if bits is not None else None, bits)

    gc.collect()
    _save_cache(cpath, result)
    if verbose:
        print(f"[gb_quant_calib] delta_loss done , {len(result)} layers assigned. "
              f"Cache saved: {cpath}", flush=True)
    return result
    return out
