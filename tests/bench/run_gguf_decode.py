#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""tests/bench/run_gguf_decode.py, Speed Program Phase 0: the real GGUF
end-to-end harness.

WHY THIS EXISTS: tests/bench/run_real_model.py only exercises the
`transformers` fallback path (gb_synapse_fallback.py), it cannot measure the
actual reference workload, a GGUF served through llama.cpp via
gb_synapse_backends.select_backend()/serve(). This harness drives that real
production path directly (same command line real traffic gets, same
placement decisions, same crash-avoidance quirks like ARCH_CPU_SPLIT_BROKEN,
see workflow/known-issues.md's qwen35moe entry and the plan's Finding 14).
It does NOT special-case any model, it calls the real serve() and measures
whatever command line that model actually gets, CPU-only, GPU-active, or
partial-split, because that's what production does too.

Reports prefill and decode tok/s SEPARATELY (they have different
bottlenecks): prefill is bandwidth-bound over a large batch, decode is
latency-bound at batch 1. TTFT (time to first token) approximates prefill
time; token-to-token gaps after that approximate decode time.

Usage:
    python3 tests/bench/run_gguf_decode.py --model "satgeze/qwen36-35b-uncensored-1m:latest" \\
        [--prompt-tokens 512,8000,46000] [--max-new-tokens 64] [--tag baseline] \\
        [--ctx 4096] [--port 18088]

Each (model, prompt length) pair gets its own server start/stop cycle, model
load time is excluded from the tok/s numbers but recorded separately.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gb_bench  # noqa: E402


def _make_prompt(n_tokens: int) -> str:
    """Cheap, deterministic filler to approximate a prompt of roughly
    n_tokens tokens (whitespace-word tokenization is a rough proxy across
    tokenizers; exact token count isn't the point, comparable PREFILL COST
    across runs at the same n_tokens value is)."""
    unit = "The quick brown fox jumps over the lazy dog near the river bank. "
    words_needed = max(1, int(n_tokens * 0.75))  # ~0.75 words/token, rough
    text = (unit * ((words_needed // len(unit.split())) + 1))
    return " ".join(text.split()[:words_needed])


def _stream_chat(port: int, model: str, prompt: str, max_new_tokens: int) -> dict:
    """POST a streaming chat completion, timing TTFT and inter-token gaps.
    Returns {"ttft_s", "decode_s", "n_tokens", "prefill_tok_s", "decode_tok_s"}."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"}, method="POST")

    t_start = time.monotonic()
    t_first = None
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            # Reasoning models (Qwen3-family thinking mode) stream thinking
            # tokens as "reasoning_content", not "content", both represent
            # real decode work and both must count toward decode tok/s, or a
            # thinking model silently measures as producing zero tokens.
            if delta.get("content") or delta.get("reasoning_content"):
                if t_first is None:
                    t_first = time.monotonic()
                n_tokens += 1
    t_end = time.monotonic()

    ttft_s = (t_first - t_start) if t_first else None
    decode_s = (t_end - t_first) if t_first and n_tokens > 1 else None
    return {
        "ttft_s": round(ttft_s, 3) if ttft_s else None,
        "decode_s": round(decode_s, 3) if decode_s else None,
        "n_tokens": n_tokens,
        "prefill_tok_s": None,  # prompt token count isn't echoed by the API; TTFT is the proxy
        "decode_tok_s": round((n_tokens - 1) / decode_s, 2) if decode_s and n_tokens > 1 else None,
    }


def _concurrent_chat(port: int, model: str, prompt: str, max_new_tokens: int,
                      n_clients: int) -> dict:
    """Fire n_clients concurrent streaming requests, all starting at
    approximately the same wall-clock time (Phase 3's gate, Finding 10):
    does aggregate tok/s at concurrency scale, or is the server already
    saturated at concurrency 1? Per-client results come from the same
    _stream_chat() used by the single-client path, so the two are directly
    comparable. Returns aggregate + per-client stats; a client that errors
    is excluded from the aggregate (its error is still reported)."""
    t_wall0 = time.monotonic()
    results = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_clients) as pool:
        futures = [pool.submit(_stream_chat, port, model, prompt, max_new_tokens)
                   for _ in range(n_clients)]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                errors.append(str(e))
    wall_s = time.monotonic() - t_wall0

    total_tokens = sum(r["n_tokens"] for r in results)
    per_client_tok_s = [r["decode_tok_s"] for r in results if r["decode_tok_s"]]
    return {
        "n_clients": n_clients,
        "n_ok": len(results),
        "n_errors": len(errors),
        "errors": errors[:3],  # cap, don't flood the log with N identical failures
        "wall_s": round(wall_s, 3),
        # Aggregate tok/s: total tokens produced by ALL clients over the
        # shared wall-clock window, this is the number Finding 10's gate
        # actually compares (does concurrency raise total throughput).
        "aggregate_tok_s": round(total_tokens / wall_s, 2) if wall_s > 0 else None,
        "mean_per_client_tok_s": round(sum(per_client_tok_s) / len(per_client_tok_s), 2)
                                  if per_client_tok_s else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model spec, e.g. satgeze/qwen36-35b-uncensored-1m:latest")
    ap.add_argument("--prompt-tokens", default="512,8000",
                     help="comma-separated approx prompt lengths to sweep (46000 is the "
                          "documented worst case from known-issues.md, add it explicitly "
                          "once shorter runs are validated, it's slow)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=18088)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--clients", default="1",
                     help="comma-separated concurrency levels to sweep, e.g. 1,2,4,8 "
                          "(Phase 3's gate: aggregate tok/s at 4 clients >= 1.5x at 1 client). "
                          "n_slots is left at serve()'s own default (-1 = llama.cpp auto, "
                          "n_parallel=4) regardless of this sweep, per Finding 10")
    args = ap.parse_args()

    import gb_synapse as gs
    import gb_synapse_backends as backends

    entry = gs._resolve_model(args.model)
    backend = backends.select_backend(entry)
    print(f"[run_gguf_decode] model={entry.name} arch={entry.arch} backend={type(backend).__name__}")

    t_load0 = time.monotonic()
    try:
        # n_slots=-1: serve()'s own default (Finding 10), llama.cpp's "auto"
        # (n_parallel=4, kv_unified=true), not hardcoded here either.
        state = backend.serve(entry, args.port, ctx=args.ctx, use_cluster=False, n_slots=-1)
    except Exception as exc:
        print(f"[run_gguf_decode] serve() raised: {exc}", file=sys.stderr)
        gb_bench.emit_bench_result(
            path="run_gguf_decode", config={"model": args.model, "ctx": args.ctx},
            model=args.model, tag=args.tag, error=str(exc))
        return 1
    port = state.port
    load_s = round(time.monotonic() - t_load0, 2)
    try:
        # serve() already waited for readiness internally (state.ready); a
        # large GGUF can still legitimately still be loading (SERVE_READY_GRACE_S
        # timeout), poll /health a bit longer before giving up, same
        # tolerance the CLI itself extends to a big model.
        if not state.ready:
            for _ in range(180):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("server never became ready within 180s past serve()'s own grace period")
        load_s = round(time.monotonic() - t_load0, 2)

        prompt_lens = [int(x) for x in args.prompt_tokens.split(",") if x.strip()]
        client_counts = [int(x) for x in args.clients.split(",") if x.strip()]

        # Warm-up: a single request against the FIRST prompt length, with the
        # existing 503-retry-on-still-loading logic. Once this succeeds the
        # server is confirmed warm, nothing later in the sweep needs the
        # retry loop (avoids N concurrent clients all separately racing the
        # same 503 window, which would just serialize on the retry sleeps).
        warm_prompt = _make_prompt(prompt_lens[0])
        result, exc = None, None
        for attempt in range(10):
            try:
                result = _stream_chat(port, entry.name, warm_prompt, args.max_new_tokens)
                exc = None
                break
            except Exception as e:
                exc = e
                if "503" in str(e) and attempt < 9:
                    print(f"[run_gguf_decode] still loading (503), retry {attempt + 1}/10...")
                    time.sleep(15)
                    continue
                break
        if exc is not None:
            print(f"[run_gguf_decode] warm-up FAILED: {exc}", file=sys.stderr)
            gb_bench.emit_bench_result(
                path="run_gguf_decode",
                config={"model": args.model, "ctx": args.ctx, "prompt_tokens": prompt_lens[0]},
                model=args.model, tag=args.tag, load_s=load_s, error=str(exc))
            return 1
        print(f"[run_gguf_decode] warm-up prompt_tokens={prompt_lens[0]}: "
              f"ttft={result['ttft_s']}s decode_tok_s={result['decode_tok_s']}")
        warm_config = {"model": args.model, "ctx": args.ctx, "prompt_tokens": prompt_lens[0],
                       "max_new_tokens": args.max_new_tokens, "n_clients": 1}
        gb_bench.emit_bench_result(
            path="run_gguf_decode", config=warm_config, model=args.model, tag=args.tag,
            decode_tok_s=result["decode_tok_s"], load_s=load_s)

        baseline_tok_s = result["decode_tok_s"]  # for the >=1.5x gate check below

        for n_tok in prompt_lens:
            prompt = _make_prompt(n_tok)
            for n_clients in client_counts:
                if n_tok == prompt_lens[0] and n_clients == 1:
                    continue  # already covered by the warm-up run above
                config = {"model": args.model, "ctx": args.ctx, "prompt_tokens": n_tok,
                          "max_new_tokens": args.max_new_tokens, "n_clients": n_clients}
                try:
                    if n_clients == 1:
                        r = _stream_chat(port, entry.name, prompt, args.max_new_tokens)
                        agg_tok_s, mean_tok_s = r["decode_tok_s"], r["decode_tok_s"]
                        print(f"[run_gguf_decode] prompt_tokens={n_tok} clients=1: "
                              f"ttft={r['ttft_s']}s decode_tok_s={r['decode_tok_s']}")
                    else:
                        r = _concurrent_chat(port, entry.name, prompt, args.max_new_tokens, n_clients)
                        agg_tok_s, mean_tok_s = r["aggregate_tok_s"], r["mean_per_client_tok_s"]
                        gate = (agg_tok_s / baseline_tok_s) if (agg_tok_s and baseline_tok_s) else None
                        print(f"[run_gguf_decode] prompt_tokens={n_tok} clients={n_clients}: "
                              f"n_ok={r['n_ok']}/{n_clients} aggregate_tok_s={agg_tok_s} "
                              f"mean_per_client={mean_tok_s} "
                              f"(vs clients=1 baseline {baseline_tok_s}: "
                              f"{f'{gate:.2f}x' if gate else 'n/a'}, "
                              f"Phase 3 gate is >=1.5x at 4 clients)")
                        if r["errors"]:
                            print(f"[run_gguf_decode]   sample errors: {r['errors']}", file=sys.stderr)
                except Exception as e:
                    print(f"[run_gguf_decode] prompt_tokens={n_tok} clients={n_clients} FAILED: {e}",
                          file=sys.stderr)
                    gb_bench.emit_bench_result(path="run_gguf_decode", config=config,
                                                model=args.model, tag=args.tag, load_s=load_s,
                                                error=str(e))
                    continue
                gb_bench.emit_bench_result(
                    path="run_gguf_decode", config=config, model=args.model, tag=args.tag,
                    decode_tok_s=agg_tok_s, load_s=load_s)
    finally:
        gs.stop(entry.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
