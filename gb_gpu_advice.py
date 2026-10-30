"""gb_gpu_advice: the per-GPU serving recommendations learned on the RTX 5070 class (consumer Blackwell, compute capability 12.x, 12 GB), keyed on what NVML reports.

The MCP tool `gpu_advice` (greenboost-orchestrator) returns this list. Nothing here names a card: the GPU is described by its detected compute capability and total VRAM, each item says
whether it `applies` on that GPU, with a `confidence` of "measured" (a number on the 5070 in a workflow doc) or "derived" (follows from measured facts or from sizing arithmetic) and an
`evidence` list of the docs holding the numbers. Pure data plus `detect()`; no process is started and no GPU context is created (NVML only).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

W = "workflow/"
MEASURED_ON = "sm_120, 12 GB (RTX 5070 class), 2026-10"


def detect() -> dict:
    """{'name', 'cc': (major, minor) or None, 'vram_gb': float or None, 'free_gb'} from NVML (no CUDA context). Missing pieces are None."""
    out: dict = {"name": "", "cc": None, "vram_gb": None, "free_gb": None}
    try:
        from gb_nvml import get_nvml
        nv = get_nvml(0)
        out["name"] = nv.device_name() or ""
        _, free_mb, total_mb, _ = nv.mem()
        if total_mb:
            out["vram_gb"], out["free_gb"] = round(total_mb / 1024, 2), round(free_mb / 1024, 2)
    except Exception:  # noqa: BLE001
        pass
    try:
        import gb_topology
        cc = gb_topology._detect_compute_capability()
        if cc and "." in cc:
            out["cc"] = tuple(int(x) for x in cc.split(".")[:2])
    except Exception:  # noqa: BLE001
        pass
    return out


def advice(cc: Optional[Tuple[int, int]], vram_gb: Optional[float]) -> List[dict]:
    """The recommendation list for a GPU of compute capability `cc` and `vram_gb` total VRAM (pure)."""
    blackwell = bool(cc) and cc[0] == 12
    fp8_hw = bool(cc) and tuple(cc) >= (8, 9)
    small = bool(vram_gb) and vram_gb <= 16
    on = "measured on " + MEASURED_ON

    def item(id_, text, applies, conf, evidence, numbers=""):
        d = {"id": id_, "advice": text, "applies": bool(applies), "confidence": conf, "evidence": [W + e for e in evidence]}
        if numbers:
            d["numbers"] = numbers + ("" if applies and (blackwell or conf == "derived") else f" ({on}; re-measure on this GPU)")
        return d

    return [
        item("attention_backend", "Torch attention: cuDNN first, then Flash, Efficient, Math (gb_sdpa, GB_ATTN_BACKEND=auto does this on compute capability 12.x only).", blackwell, "measured",
             ["missing-5070-features.md", "overnight-improvements.md"], "head_dim 128: cuDNN 5-10% faster than FlashAttention-2, both 2.2x the memory-efficient kernel; head_dim 64 ties"),
        item("cnn_channels_last", "CNN workers (upscalers, UNets, video VAEs): channels_last memory format (gb_cnn_tune); cudnn.benchmark gave nothing.", True, "measured",
             ["missing-5070-features.md"], "Real-ESRGAN 1.33x faster and 45% less VRAM"),
        item("regional_compile", "Diffusion/video transformers: regional torch.compile (gb_compile) only for batches of 8 or more images; the compile cost is paid once, cache persists.", True, "measured",
             ["missing-5070-features.md"], "11-13% per image, on for GB_COMPILE_MIN_IMAGES >= 8"),
        item("expert_cache", "MoE GGUF with experts on the host: build and select the expert-cache engine, keep ALL experts on the host and give the free VRAM to --moe-cache-mib. "
             "cache = free VRAM - dense - KV - vision - 0.4 GB x ubatch/512 - MTP draft - 11% safety (min 0.5 GB); used from 3 GB; GB_MOE_CACHE=0|auto|<MiB>.", True, "measured",
             ["gb-moe-cache.md"], "Qwen3.8-35B-A3B: 54 tok/s (layer split) -> 74 / 93 / 95-101 tok/s at 3 / 4.5 / 5.5 GB cache; 6.5 GB (card full) unstable"),
        item("vision_reserve", "Serving a VLM: reserve ~1.73 GB for the projector plus vision buffers before sizing the cache (gb_vision.extra_vram_gb); vision=false frees it.", True, "derived",
             ["vision-serving.md", "gb-moe-cache.md"], "1.73 GB at the default image token budget of the Qwen3.8 projector; exact value computed per projector at serve time"),
        item("mtp_stacking", "MTP stacks with the expert cache at draft depth 2 (not the default 4): the draft context costs 1.5 GB (depth 2) to 2.0 GB (depth 4) of VRAM and deeper verify batches touch experts the cache lacks.", True, "measured",
             ["gb-moe-cache.md", "overnight-improvements.md"], "cache only 128 tok/s; cache + MTP depth 2 = 181; depth 3 = 160; depth 4 does not load"),
        item("power_limit", "Do not tune the power limit for decode: it never binds (decode is bandwidth bound and draws less than the lowest allowed limit).", True, "measured",
             ["missing-5070-features.md"], "decode 101-112 tok/s at every limit from 175 to 300 W; board draw avg 116-126 W"),
        item("vae_precision", "Keep VAEs in bf16 (gb_quant skips them); their decode OOM is activation bound, so tile or slice the decode instead of quantizing.", True, "measured",
             ["gb-quant-conv.md"], "fp8 conv weights save under 100 MiB of a VAE; decode peak 2.6 GB at 1024 px"),
        item("fp8_conv", "fp8 weight-only convs (gb_quant_conv) only when the module's conv weights total 256 MiB or more (GB_QUANT_CONV_MIN_TOTAL_MB); capacity only, no speedup; smaller CNNs stay fp16.", fp8_hw, "measured",
             ["gb-quant-conv.md"], "FLUX.2 VAE 160 -> 82 MiB at 46.9 dB; Real-ESRGAN below the gate loses 13.6 dB for 15.8 MiB"),
        item("fp8_linear_prefill", "Compute-bound Linear work (prefill, denoise): fp8 W8A8 via torch._scaled_mm (GB_KERNEL_BACKEND=scaled_mm); decode is bandwidth bound and gains nothing.", fp8_hw, "measured",
             ["fp8-prefill-blackwell-scope.md"], "fp8 MMA 238 vs 128 fp16 TFLOP/s"),
        item("vram_budget", f"{'%.0f GB card: ' % vram_gb if vram_gb else ''}keep the peak under ~10 GB (the desktop holds 1-7 GB), one heavy model at a time, text encoders freed after encoding.", small, "derived",
             ["gb-quant.md"]),
    ]


def gpu_advice(ids: Optional[List[str]] = None, only_applicable: bool = False) -> dict:
    """Detected GPU + the advice list (compact: one line per item, evidence as doc paths)."""
    g = detect()
    items = advice(g["cc"], g["vram_gb"])
    if ids:
        items = [i for i in items if i["id"] in ids]
    if only_applicable:
        items = [i for i in items if i["applies"]]
    return {"gpu": {"name": g["name"], "compute_capability": ".".join(map(str, g["cc"])) if g["cc"] else None, "vram_gb": g["vram_gb"], "free_gb": g["free_gb"]},
            "measured_on": MEASURED_ON, "advice": items,
            "note": "applies=false means the item does not exist or is untested on this GPU; confidence 'measured' = a number in the evidence doc, 'derived' = follows from those facts"}
