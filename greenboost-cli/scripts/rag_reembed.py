#!/usr/bin/env python3
"""Re-embed the greenboost-cli RAG index with the current embedder, on the GPU.

Why: the live index was built with all-MiniLM-L6-v2 (the silent fallback). The default
embedder is now Qwen/Qwen3-Embedding-0.6B (override: GB_EMBED_MODEL). An index only
matches the model that built it, so every chunk's vector has to be recomputed. The text
of every chunk is already in the index, so no source file is read.

Safe by construction:
  * the current index keeps answering searches until the very last step, which is one
    atomic swap to a new generation;
  * Ctrl+C or a crash changes nothing live; run the script again and it resumes from the
    4,096-chunk checkpoints under <rag dir>/reembed.tmp;
  * it refuses to start while a model server is loaded (the card is needed) or when the
    card has no room. The CPU is not an option: it needs days for this many chunks.

Usage (from anywhere; uses the greenboost-cli venv's torch/transformers):
    /usr/local/lib/greenboost/cli-venv/bin/python3 greenboost-cli/scripts/rag_reembed.py --measure 512
    /usr/local/lib/greenboost/cli-venv/bin/python3 greenboost-cli/scripts/rag_reembed.py
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

# Before torch is imported: expandable segments avoid the fragmentation that made a real run
# climb to 10.2 of 11.6 GiB. (An explicit setting from the caller wins.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))               # greenboost-cli/  (the repo's engine, not an older installed one)
sys.path.insert(0, str(_HERE.parent.parent))        # repo root       (gb_synapse, gb_dataflux)


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h {rem // 60:02d}m" if h else f"{rem // 60}m {rem % 60:02d}s"


def _running_servers() -> list[str]:
    try:
        import gb_synapse
        return [s.get("model", "?") for s in gb_synapse.ps()]
    except Exception:
        return []


def _vram() -> str:
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return f"{(total - free) / 2**30:.1f} of {total / 2**30:.1f} GiB used"
    except Exception:
        return "unknown"


def preflight(force: bool) -> tuple[dict, list]:
    from greenboost_cli.rag import engine as E
    emb, meta = E._load_store()
    built = E.read_manifest().get("embed_model")
    print(f"index:      {len(meta):,} chunks, built with {built}")
    print(f"embedder:   {E.EMBED_MODEL} (target)")
    print(f"GPU now:    {_vram()}")
    servers = _running_servers()
    if servers:
        print(f"model server running: {', '.join(servers)}")
        if not force:
            sys.exit("\nA model server is loaded and holds the card. Stop it first "
                     "(/exit unloads it, or: greenboost synapse stop), then run this again. "
                     "--force skips this check.")
    if built == E.EMBED_MODEL:
        sys.exit("\nThe index is already built with the target embedder; nothing to do.")
    return {"built": built}, meta


def measure(n: int, meta: list) -> None:
    """Embed a random sample on the GPU, write nothing, print speed, VRAM peak and the ETA."""
    import numpy as np
    import torch
    from greenboost_cli.rag import engine as E

    texts = [m.get("text", "") or " " for m in meta]
    rng = random.Random(7)
    idx = rng.sample(range(len(texts)), min(n, len(texts)))
    sample = sorted((texts[i] for i in idx), key=len)
    model = E._get_model("cuda")
    torch.cuda.reset_peak_memory_stats()
    model.encode(sample[:8], batch_size=8, show_progress_bar=False)       # warm-up, not timed
    t0 = time.time()
    vecs = model.encode(sample, batch_size=32, convert_to_tensor=True, show_progress_bar=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    rate = len(sample) / dt
    mean_all = sum(len(t) for t in texts) / len(texts)
    mean_sample = sum(len(t) for t in sample) / len(sample)
    eta = len(texts) / rate * (mean_all / max(mean_sample, 1))
    E.release_gpu_embedder()
    print(f"\nmeasured:   {len(sample)} chunks in {dt:.1f}s = {rate:.1f} chunks/s, "
          f"{vecs.shape[1]}-dim, peak GPU {peak:.2f} GiB")
    print(f"estimate:   all {len(texts):,} chunks in about {_fmt(eta)} "
          f"(mean chunk {mean_all:.0f} chars vs {mean_sample:.0f} in the sample)")
    print("Nothing was written. Run again without --measure to rebuild the index.")


def run_full(yes: bool) -> None:
    from greenboost_cli.rag import engine as E
    if not yes:
        ans = input("\nRe-embed the whole index now? The current index keeps working until "
                    "the final swap; Ctrl+C is safe. [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            sys.exit("Cancelled; nothing changed.")
    last = [0.0]
    t0 = time.time()

    def progress(done: int, total: int, eta_s: float) -> None:
        if time.time() - last[0] >= 30 or done >= total:
            last[0] = time.time()
            print(f"  {done:>9,}/{total:,} chunks  {100 * done / total:5.1f}%  "
                  f"~{_fmt(eta_s)} left  GPU {_vram()}", flush=True)

    try:
        r = E.reembed_store(progress=progress)
    except RuntimeError as e:
        sys.exit(f"\n{e}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. The current index is untouched; run this again to resume.")
    print(f"\nstatus: {r['status']}  chunks: {r.get('chunks'):,}  "
          f"dims: {r.get('dims')}  took {_fmt(time.time() - t0)}")
    if r["status"] != "done":
        return
    print(f"manifest embed_model now: {E.read_manifest().get('embed_model')}")
    print("\nsanity search on the new index:")
    for q in ("where is the embedding fallback model chosen", "how does the MCP client time out a server"):
        hits = E.search(q, top_k=3, min_score=0.0)
        print(f"  {q!r}")
        for h in hits[:3]:
            print(f"     {float(h.get('score', 0)):.3f}  {str(h.get('file'))[-70:]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", type=int, metavar="N", help="embed N sample chunks, write nothing, print speed and ETA")
    ap.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    ap.add_argument("--force", action="store_true", help="run even if a model server is loaded")
    args = ap.parse_args()
    os.environ.setdefault("GB_EMBED_DEVICE", "auto")
    from greenboost_cli.hf_token import install_warning_filter
    install_warning_filter(action="export HF_TOKEN=<your-token>  (then run this script again), "
                                  "or start gb and type /huggingface-token <your-token>")
    _, meta = preflight(args.force)
    if args.measure:
        measure(args.measure, meta)
    else:
        run_full(args.yes)


if __name__ == "__main__":
    main()
