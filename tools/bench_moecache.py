#!/usr/bin/env python3
"""One serve + measurement of the GPU expert cache on a MoE model, through gb_synapse.serve (the real sizing path). Reproduces the tables in workflow/gb-moe-cache.md and
workflow/overnight-improvements.md. Wrap it in `flock /tmp/gb-gpu.lock` when a game shares the card.

  GB_SYNAPSE_ENGINE_DIR=<engine> python3 tools/bench_moecache.py --model qwen38-35b-a3b-apex-miniplus --mtp 1 --ubatch 512 [--cache auto|0|<MiB>] [--vision 0|1] [--ctx 16384]

Prints one JSON line: planned cache, decode tok/s (median of 3 x 200 tokens, engine `timings`), prefill tok/s (about 2.5k-token prompt), peak and idle card VRAM, free VRAM after load."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def vram_mib() -> int:
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])


def chat(port: int, content: str, n: int) -> dict:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38-35b-a3b-apex-miniplus")
    ap.add_argument("--mtp", choices=("0", "1"), default="0")
    ap.add_argument("--cache", default="auto", help="GB_MOE_CACHE: auto, 0 (off) or MiB")
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--vision", choices=("0", "1"), default="0")
    ap.add_argument("--draft-n", type=int, default=None)
    a = ap.parse_args()
    os.environ["GB_MOE_CACHE"] = a.cache
    os.environ["GB_SYNAPSE_MTP"] = a.mtp
    os.environ["GB_MOE_CACHE_MTP"] = a.mtp
    import gb_synapse as gs
    state = str(gs._run_log_path(a.model)).replace(".log", ".json")
    try:
        pid = json.load(open(state)).get("llama_pid", 0)
        if pid and os.path.exists(f"/proc/{pid}") and open(f"/proc/{pid}/stat").read().split()[2] != "Z":
            sys.exit(f"a server for {a.model} is already running (pid {pid}): serve() would reuse it and this would measure its settings, not these. Stop it first.")
    except (OSError, ValueError):
        pass
    base = vram_mib()
    t0 = time.time()
    import contextlib
    import io
    import re
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))      # a timeout must still stop the server it started (an orphan keeps 11 GB and the next serve reuses it)
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap):
        st = gs.serve(a.model, explicit=True, trigger="bench_moecache", keep=True, ctx=a.ctx, ubatch_size=a.ubatch, vision=(a.vision == "1"), mtp_draft_n=a.draft_n)
    print(cap.getvalue(), file=sys.stderr)
    m = re.search(r"(\d+) MiB GPU expert cache \((.*?)\)", cap.getvalue())
    out = {"planned_cache_mib": int(m.group(1)) if m else None, "plan": m.group(2) if m else None, "mtp_kept": "MTP off while" not in cap.getvalue(), "mtp": a.mtp, "cache": a.cache, "ubatch": a.ubatch, "vision": a.vision, "ctx": a.ctx, "engine": st.engine, "load_s": round(time.time() - t0, 1)}
    try:
        port = st.internal_port
        out["after_load_used_mib"] = vram_mib()
        chat(port, "Say hi.", 8)  # warm up
        dec, peak = [], vram_mib()
        for q in ("Write a short story about a lighthouse keeper.", "Explain how a transistor works in detail.", "List twenty facts about the moon, one per line."):
            r = chat(port, q, 200)
            tm = r.get("timings") or {}
            dec.append(tm.get("predicted_per_second") or 0.0)
            out.setdefault("draft_accept", []).append(tm.get("draft_n_accepted") and round(tm["draft_n_accepted"] / max(1, tm.get("draft_n", 1)), 2))
            peak = max(peak, vram_mib())
        out["decode_tok_s"] = [round(x, 1) for x in dec]
        out["decode_median"] = round(statistics.median(dec), 1)
        long_prompt = " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog {i * 7}." for i in range(180))
        r = chat(port, "Summarise in one word: " + long_prompt, 4)
        tm = r.get("timings") or {}
        out["prefill_tokens"] = tm.get("prompt_n")
        out["prefill_tok_s"] = round(tm.get("prompt_per_second") or 0.0, 1)
        out["peak_used_mib"] = max(peak, vram_mib())
        out["baseline_mib"] = base
    finally:
        gs.stop(a.model)
        time.sleep(3)
        out["after_stop_used_mib"] = vram_mib()
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
