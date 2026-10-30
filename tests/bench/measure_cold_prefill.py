#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Measure COLD prefill throughput (prompt processing, pp tok/s) on a live
gb-synapse proxy.

Sends N unique long prompts (random content => zero slot/prefix cache reuse,
so this is real cold prefill, not recurrent replay), non-streaming, and reads
llama.cpp's own `timings` block from the response. Prefill tok/s =
timings.prompt_per_second (== prompt_n / (prompt_ms/1000)); same source
gb_session_audit.prefill_tok_s uses. Decode is reported too but with
max_tokens tiny it is noise — use run_gguf_decode.py for decode.

Serve first (shim defaults on now), then run this against :11369. Example used
to pin `ubatch: 2048` into serving/recipes/qwen38-35b-a3b-apex-miniplus.yaml
(see the ubatch sweep table in workflow/gb-synapse.md):

    python gb_synapse.py serve qwen38-35b-a3b-apex-miniplus --ubatch 2048
    python tests/bench/measure_cold_prefill.py --runs 3 --tokens 3000

Usage:
    measure_cold_prefill.py [--model NAME] [--base URL] [--runs N]
                            [--tokens APPROX] [--label TAG]
"""
import argparse
import json
import random
import statistics
import string
import sys
import time
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:11369/v1/chat/completions"
DEFAULT_MODEL = "qwen38-35b-a3b-apex-miniplus"


def _unique_prompt(ntok: int) -> str:
    """~ntok tokens of unique nonsense so no slot/prefix cache can hit."""
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 9)))
             for _ in range(int(ntok * 0.9))]
    salt = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return (f"Session {salt}. Read this corpus and reply with the single word OK.\n\n"
            + " ".join(words))


def _one(base: str, model: str, ntok: int) -> "tuple[dict, float]":
    body = json.dumps({
        "model": model, "stream": False, "max_tokens": 1, "temperature": 0,
        "messages": [{"role": "user", "content": _unique_prompt(ntok)}],
    }).encode()
    req = urllib.request.Request(base, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return (d.get("timings") or {}), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--label", default="prefill")
    a = ap.parse_args()

    pps_rows: "list[float]" = []
    for i in range(a.runs):
        tim, wall = _one(a.base, a.model, a.tokens)
        pn, pms = tim.get("prompt_n"), tim.get("prompt_ms")
        pps = tim.get("prompt_per_second")
        if pps is None and pn and pms:
            pps = pn / (pms / 1000.0)
        print(f"  [{a.label} {i+1}/{a.runs}] prompt_n={pn} prompt_ms={pms} "
              f"prefill_tok_s={pps:.1f} wall={wall:.1f}s", flush=True)
        if pps:
            pps_rows.append(pps)
    if pps_rows:
        print(f"== {a.label}: median prefill {statistics.median(pps_rows):.1f} tok/s "
              f"over {len(pps_rows)} runs", flush=True)
        return 0
    print("no timings returned — is a model serving on the proxy?", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
