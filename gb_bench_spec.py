#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_bench_spec.py , GB-6: measure what speculative decoding is actually paying.

The served reference model carries an MTP (multi-token prediction) head in its
own weights, and `gb_synapse.serve()` already exposes the two knobs that drive
it (`mtp_draft_n` -> `--spec-draft-n-max`, `spec_draft_p_min` ->
`--spec-draft-p-min`). What has never existed is a measurement that says
whether a given depth is paying on THIS box, for THIS model, today , the
current defaults (4 and 0.3) come from a sweep run on the previous reference
model on 2026-08-05 and were carried forward on the assumption they transfer.

Why the depth is not obvious, and why "pick the highest" is wrong: decode here
is bandwidth-bound, so one forward pass costs the same wall time whether it
emits one token or five, which argues for depth. But a rejected draft token is
work thrown away AND it displaces the KV of the tokens that follow, so past
some depth the acceptance rate falls faster than the parallelism helps. The
2026-08-05 sweep found exactly that shape , 2:5.15, 3:5.58, 4:6.50, 6:4.40,
8:5.76 tok/s , non-monotonic, with the best value in the middle.

This script re-runs that sweep as a first-class tool instead of an ad-hoc
session: it re-serves the model at each depth, replays a fixed conversation,
and records `spec_decode` telemetry per depth so the choice is followable
afterwards rather than living in a session transcript.

Because each depth needs a re-serve (the flag is a launch parameter), a full
sweep costs a few minutes per point. `--depths` keeps that explicit.

Run:  python3 gb_bench_spec.py --depths 0,2,4,6      # full sweep, re-serves
      python3 gb_bench_spec.py --current             # measure what is served now
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

DEFAULT_DEPTHS = (0, 2, 4, 6)

# Prompts chosen to span what the MTP head is good and bad at. A draft head
# predicts confidently on formulaic continuations and poorly on genuinely
# novel text, so measuring only one shape reports the acceptance rate of that
# shape and calls it the model's.
PROMPTS = (
    "List the first 12 prime numbers, comma separated.",
    "Write a Python function that reverses a linked list. Code only.",
    "Explain in three sentences why PCIe bandwidth limits local LLM decode.",
)


def _measure_once(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    import gb_bench_turn as bt
    messages = [{"role": "system", "content": "You are a benchmark fixture."},
                {"role": "user", "content": prompt}]
    t0 = time.monotonic()
    d = bt._post(f"{base_url}/chat/completions", {
        "model": model, "messages": messages, "stream": False,
        "max_tokens": max_tokens, "temperature": 0.0})
    wall = time.monotonic() - t0
    usage = d.get("usage") or {}
    tim = d.get("timings") or {}
    out = {
        "wall_s": round(wall, 2),
        "completion_tokens": usage.get("completion_tokens", 0),
        "decode_tok_s": tim.get("predicted_per_second"),
        "prompt_ms": tim.get("prompt_ms"),
    }
    # llama-server reports draft acceptance when speculative decoding ran. Its
    # absence is meaningful (the depth was 0, or the model has no draft head)
    # and is recorded as such rather than defaulted to zero.
    for k in ("draft_n", "draft_n_accepted"):
        if k in tim:
            out[k] = tim[k]
    if out.get("draft_n"):
        out["accept_rate"] = round(out.get("draft_n_accepted", 0) / out["draft_n"], 3)
    return out


def measure_current(base_url: str, model: str = "", max_tokens: int = 96,
                    repeats: int = 2, emit: bool = True) -> dict:
    """Measure the depth that is being served right now, without re-serving."""
    import gb_synapse
    if not model:
        ps = gb_synapse.ps()
        if not ps:
            return {"error": "no serve session running"}
        model = ps[0]["model"]
    served = next((s for s in gb_synapse.ps() if s.get("model") == model), {})
    depth = served.get("mtp_draft_n")

    rows = []
    for prompt in PROMPTS:
        for _ in range(repeats):
            rows.append({"prompt": prompt[:40], **_measure_once(base_url, model, prompt, max_tokens)})
    rates = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
    # At depth 0 nothing is drafted, so nothing can be rejected and the engine
    # reports acceptance = 1.0. Passing that through would read as "drafting is
    # working perfectly" when the truth is "drafting is off" , the exact
    # confusion this module's accept_rate_source field exists to prevent.
    # Measured 2026-08-20: depth 0 really does come back as 1.0 from engine
    # timings, so this has to be suppressed here rather than trusted.
    accepts = ([] if depth in (0, None)
               else [r["accept_rate"] for r in rows if r.get("accept_rate") is not None])
    summary = {
        "model": model,
        "draft_n": depth,
        "median_tok_s": round(statistics.median(rates), 2) if rates else None,
        "mean_tok_s": round(statistics.fmean(rates), 2) if rates else None,
        "samples": len(rates),
        "median_accept_rate": round(statistics.median(accepts), 3) if accepts else None,
        "accept_rate_source": (
            "engine timings" if accepts else
            "not applicable at depth 0 (nothing drafted, so nothing rejected)"
            if depth in (0, None) else
            "engine reported none (no draft head)"),
    }
    if emit:
        try:
            import gb_dataflux
            gb_dataflux.emit({"node": "host", "label": "synapse",
                              "kind": "spec_decode", **summary})
        except Exception:
            pass
    return {"summary": summary, "runs": rows}


def sweep(base_url: str, model: str = "", depths=DEFAULT_DEPTHS,
          max_tokens: int = 96, repeats: int = 2, confirm: bool = False) -> dict:
    """Re-serve at each depth and measure. Requires --confirm: this stops and
    restarts the live server, which is exactly the kind of thing that should
    not happen because a benchmark was run by accident."""
    import gb_synapse
    if not model:
        ps = gb_synapse.ps()
        if not ps:
            return {"error": "no serve session running; pass --model"}
        model = ps[0]["model"]
    if not confirm:
        return {"error": "sweep re-serves the model at each depth; pass --confirm",
                "would_measure": list(depths), "model": model}

    results = []
    failed = None
    try:
      for d in depths:
        gb_synapse.stop(model)
        gb_synapse.serve(model, mtp_draft_n=d)
        # The serve call returns once /health is green, but the first request
        # still pays a cold prefill; the measurement below discards nothing, so
        # give the engine a beat rather than folding load noise into depth 0.
        time.sleep(2)
        r = measure_current(base_url, model, max_tokens, repeats, emit=True)
        r["summary"]["draft_n"] = d          # authoritative: what we just served
        results.append(r["summary"])
    except Exception as e:
        # Same hazard as gb_bench_kv.sweep: an exception used to skip the
        # restore entirely and leave the box with nothing serving, which is
        # worse than leaving it on the wrong depth.
        failed = f"{type(e).__name__}: {e}"
    best = max((r for r in results if r.get("median_tok_s")),
               key=lambda r: r["median_tok_s"], default=None)

    # Leave the box at the WINNER, not at whatever was measured last.
    #
    # Found the hard way on 2026-08-20: a sweep over 0,2,4,6 ends with the
    # server at 6, and a sweep that included 0 left it serving at 3.80 tok/s
    # against the 6.03 it had before the benchmark ran. Nothing warned; the
    # next session would simply have been 37% slower for no visible reason.
    # A tool that measures the machine must not degrade it.
    restored = None
    try:
        target = best["draft_n"] if best else (depths[0] if depths else None)
        if target is not None and (failed or not results
                                   or target != results[-1]["draft_n"]):
            gb_synapse.stop(model)
            gb_synapse.serve(model, mtp_draft_n=target)
            restored = target
    except Exception as e:
        restored = f"FAILED: {type(e).__name__}: {e}"

    return {"model": model, "depths": results, "best": best,
            "restored_depth": restored,
            **({"failed": failed} if failed else {}),
            "note": ("depth is non-monotonic on this class of model — the best "
                     "value is expected in the middle, not at the end")}


def main() -> None:
    p = argparse.ArgumentParser(prog="gb_bench_spec.py", description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:11369/v1")
    p.add_argument("--model", default="")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--current", action="store_true",
                   help="measure the running server only, no re-serve")
    p.add_argument("--depths", default="",
                   help="comma-separated draft depths to sweep (re-serves)")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    if a.current or not a.depths:
        res = measure_current(a.base_url, a.model, a.max_tokens, a.repeats)
    else:
        res = sweep(a.base_url, a.model, [int(x) for x in a.depths.split(",")],
                    a.max_tokens, a.repeats, a.confirm)
    if a.json:
        print(json.dumps(res, indent=1))
        return
    if "error" in res:
        print(f"error: {res['error']}")
        raise SystemExit(1)
    if "summary" in res:
        s = res["summary"]
        print(f"model {s['model']}   draft_n={s['draft_n']}")
        print(f"median decode  {s['median_tok_s']} tok/s   (mean {s['mean_tok_s']}, "
              f"n={s['samples']})")
        print(f"accept rate    {s['median_accept_rate']}   [{s['accept_rate_source']}]")
        return
    for r in res["depths"]:
        print(f"draft_n {r['draft_n']:>2}   {r['median_tok_s']:>6} tok/s   "
              f"accept {r['median_accept_rate']}")
    if res.get("best"):
        print(f"\nbest: draft_n={res['best']['draft_n']} at {res['best']['median_tok_s']} tok/s")
    print(res.get("note", ""))


if __name__ == "__main__":
    main()
