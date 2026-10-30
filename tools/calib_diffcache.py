#!/usr/bin/env python3
"""Calibrate gb_diffcache on FLUX.1-dev nf4: for fixed seeds, render a no-cache baseline and each cache setting with the SAME loaded pipeline, then score the cached images
against the baseline (PSNR, SSIM, mean absolute pixel error) and record wall time per image and the skip rate. LPIPS is not used: it needs a downloaded network (offline box).

  GB_SHIMLESS=1 bash ~/Dev/ai-forge/workers/image/_gen_gb.sh tools/calib_diffcache.py --out OUT_DIR [--size 768] [--steps 28] [--configs 0.002:1,0.005:1,0.01:1]

A config is threshold:max_skip. 8-bit T5 (about 4.7 GB) encodes the prompts first and is freed; the nf4 transformer (about 6.7 GB) then stays on the card, so the peak stays under 8 GB at 768 px.
Wrap the call in `flock /tmp/gb-gpu.lock` when another job shares the card. Writes OUT_DIR/results.json and the images."""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gb_diffcache  # noqa: E402

CACHE = Path(os.environ.get("HF_HOME_FLUX", Path.home() / "Dev" / "ComfyUI" / "hf_cache"))
MODEL = "black-forest-labs/FLUX.1-dev"
# Six different subjects with six fixed seeds: a face-less illustration, a texture-heavy scene, flat graphics with text-like shapes, architecture, a still life and a landscape.
CASES = [
    (101, "a flat vector illustration of a red fox sitting in a snowy forest, clean shapes, soft colours"),
    (202, "a cluttered wooden workshop with tools on the walls, warm light, detailed texture"),
    (303, "a minimalist poster of a lighthouse on a cliff at sunset, bold geometric shapes"),
    (404, "a narrow cobblestone street in an old european town, rainy evening, reflections"),
    (505, "a still life of fruit in a ceramic bowl on a linen cloth, soft window light"),
    (606, "a wide mountain valley with a river, autumn colours, morning mist"),
]


def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--configs", default="0.002:1,0.005:1,0.01:1,0.02:2")
    ap.add_argument("--cases", type=int, default=len(CASES))
    ap.add_argument("--encode-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sage", action="store_true", help="also render with SageAttention (diffusers attention backend 'sage'; needs `sageattention` importable) and score it like a cache config")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfgs = [(float(t), int(m)) for t, m in (c.split(":") for c in a.configs.split(",") if c)]
    cases = CASES[:a.cases]

    from diffusers import BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig as TfBitsAndBytesConfig
    from transformers import T5EncoderModel
    from PIL import Image
    from skimage.metrics import structural_similarity

    snaps = sorted((CACHE / "models--black-forest-labs--FLUX.1-dev" / "snapshots").glob("*"))
    snap = snaps[-1]
    emb_file = out / "embeddings.pt"
    if a.encode_only or not emb_file.exists():
        # 8-bit T5 (about 4.7 GB) in its own process: it cannot be freed reliably inside the one that then loads the transformer.
        if not a.encode_only:
            import subprocess
            subprocess.run([sys.executable, __file__, "--out", str(out), "--cases", str(a.cases), "--encode-only"], check=True)
        else:
            t5 = T5EncoderModel.from_pretrained(snap / "text_encoder_2", quantization_config=TfBitsAndBytesConfig(load_in_8bit=True), torch_dtype=torch.bfloat16, device_map={"": 0})
            pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, transformer=None, vae=None, text_encoder_2=t5, cache_dir=str(CACHE))
            pipe.text_encoder.to("cuda")
            emb = []
            with torch.no_grad():
                for _, prompt in cases:
                    pe, pooled, _ids = pipe.encode_prompt(prompt=prompt, prompt_2=None, device="cuda")
                    emb.append((pe.cpu(), pooled.cpu()))
            torch.save(emb, emb_file)
            return 0
    emb = torch.load(emb_file)

    tr = FluxTransformer2DModel.from_pretrained(snap / "transformer", quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
                                                torch_dtype=torch.bfloat16, device_map={"": 0})
    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, transformer=tr, text_encoder=None, text_encoder_2=None, cache_dir=str(CACHE))
    pipe.vae.to("cuda")
    pipe.vae.enable_tiling()
    print(f"[calib] ready, allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    def render(i, label):
        pe, pooled = emb[i]
        torch.cuda.synchronize()
        t0 = time.time()
        img = pipe(prompt_embeds=pe.to("cuda"), pooled_prompt_embeds=pooled.to("cuda"), height=a.size, width=a.size, guidance_scale=3.5, num_inference_steps=a.steps,
                   generator=torch.Generator(device="cpu").manual_seed(cases[i][0])).images[0]
        torch.cuda.synchronize()
        sec = time.time() - t0
        img.save(out / f"{label}_{cases[i][0]}.png")
        return np.asarray(img.convert("RGB")), sec

    render(0, "warmup")
    res = {"size": a.size, "steps": a.steps, "model": "FLUX.1-dev nf4", "cases": [s for s, _ in cases], "configs": {}}
    base = [render(i, "base") for i in range(len(cases))]
    res["configs"]["baseline"] = {"seconds": [round(s, 2) for _, s in base], "median_s": round(float(np.median([s for _, s in base])), 2)}
    variants = [(f"{thr}:{mskip}", thr, mskip) for thr, mskip in cfgs] + ([("sage", None, 0)] if a.sage else [])
    for key, thr, mskip in variants:
        rows, skips = [], []
        for i in range(len(cases)):
            if thr is None:
                pipe.transformer.set_attention_backend("sage")
                img, sec = render(i, "sage")
                pipe.transformer.set_attention_backend("native")
                st = {"skip_rate": 0.0}
            else:
                gb_diffcache.patch(pipe.transformer, threshold=thr, max_skip=mskip)
                gb_diffcache.reset(pipe.transformer)
                img, sec = render(i, f"c{key.replace(':', '_')}")
                st = gb_diffcache.status(pipe.transformer)
                gb_diffcache.unpatch(pipe.transformer)
            ref = base[i][0]
            rows.append({"seed": cases[i][0], "seconds": round(sec, 2), "psnr": round(psnr(img, ref), 2), "ssim": round(float(structural_similarity(img, ref, channel_axis=2)), 4),
                         "mae": round(float(np.abs(img.astype(np.float64) - ref).mean()), 2), "skip_rate": round(st["skip_rate"], 3)})
            skips.append(st["skip_rate"])
        med = lambda k: round(float(np.median([r[k] for r in rows])), 3)  # noqa: E731
        res["configs"][key] = {"rows": rows, "median_s": med("seconds"), "median_psnr": med("psnr"), "min_psnr": min(r["psnr"] for r in rows), "median_ssim": med("ssim"),
                               "min_ssim": min(r["ssim"] for r in rows), "median_mae": med("mae"), "skip_rate": round(float(np.mean(skips)), 3),
                               "speedup": round(res["configs"]["baseline"]["median_s"] / med("seconds"), 3)}
        print(f"[calib] {key}: skip {res['configs'][key]['skip_rate']}, speedup {res['configs'][key]['speedup']}, psnr {med('psnr')}, ssim {med('ssim')}", flush=True)
        (out / "results.json").write_text(json.dumps(res, indent=1))
    res["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "configs"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
