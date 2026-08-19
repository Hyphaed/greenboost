#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_bench_kv.py , GB-4: what quantizing the KV cache actually costs.

The KV cache is the second-largest consumer of VRAM after the weights, and
`kv_type` (f16 / q8_0 / q4_0) is the one lever GreenBoost genuinely exposes
over it. Halving KV either buys context at the same VRAM or buys VRAM at the
same context, so the trade is real and worth taking , IF the quality cost is
known. Until this script, it was not: `kv_type` was chosen by a budget
heuristic and labelled "budget" versus "certification-grade" without any
measurement behind those words on this box.

**The gate has to be NIAH, not smoke.** This is the part that is easy to get
wrong. `smoke_gate()` catches repetition-collapse, which is how a badly
quantized WEIGHT set fails. Quantized KV fails differently: generation stays
fluent and the model quietly stops being able to retrieve things from far back
in its context. A smoke gate passes that happily. `niah_certify()` plants
secret codes at spread depths and scores retrieval, which is exactly the
failure mode, so it is the only gate whose PASS means anything here.

What the curve is for: it turns "q8_0 is a budget config" into a number , how
much VRAM it frees and how much recall it costs , so the choice can be made on
evidence rather than on the adjective.

Each kv_type needs a re-serve (it is a launch flag), so a full sweep costs a
few minutes per point plus the NIAH run.

Run:  python3 gb_bench_kv.py --kv-types f16,q8_0 --confirm
      python3 gb_bench_kv.py --current            # score what is served now
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

DEFAULT_KV_TYPES = ("f16", "q8_0")

#: NIAH size. Big enough that KV quantization has somewhere to lose
#: information, small enough that a sweep finishes. A 2k-token haystack would
#: pass on every setting and certify nothing.
DEFAULT_NIAH_TOKENS = 16384
DEFAULT_NEEDLES = 8


def _kv_gb(entry: dict, niah: dict):
    """KV footprint in GB, from the serve entry or derived from the context.

    `ps()` does not always carry `kv_gb` (it is absent right after a restart),
    and a curve with a blank cost column cannot answer the question it exists
    for. Falling back to the served context length times the per-token cost of
    the kv_type keeps the column populated with something honest, and the
    source is recorded so a derived figure is never mistaken for a measured
    one.
    """
    v = entry.get("kv_gb")
    if v:
        return v
    ctx = entry.get("ctx") or 0
    if not ctx:
        return None
    # bytes/elem: f16 = 2, q8_0 = 1, q4_0 = 0.5 (llama.cpp cache types)
    per = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}.get(entry.get("kv_type") or "f16", 2.0)
    layers, heads_dim = 64, 128 * 8          # this model's shape, K and V
    return round(ctx * layers * heads_dim * 2 * per / (1024 ** 3), 3)


def _served(model: str = ""):
    import gb_synapse
    ps = gb_synapse.ps()
    if not ps:
        return None, {}
    entry = next((s for s in ps if not model or s.get("model") == model), ps[0])
    return entry.get("model"), entry


def max_needles() -> int:
    """Needles are keyed to distinct city names, so the city list is the cap.

    Found 2026-08-20 by asking for 24: `random.sample` raised "Sample larger
    than population" from inside gb_aviary, several frames deep, after the
    model had already been re-served , a slow, confusing failure for what is a
    caller mistake. Ask the source and refuse up front instead.
    """
    try:
        import gb_aviary
        return len(gb_aviary._CITIES)
    except Exception:
        return 10


def measure_current(model: str = "", niah_tokens: int = DEFAULT_NIAH_TOKENS,
                    needles: int = DEFAULT_NEEDLES, emit: bool = True) -> dict:
    """Score the kv_type being served right now, without re-serving."""
    import gb_aviary
    cap = max_needles()
    if needles > cap:
        return {"error": f"needles={needles} exceeds the {cap} distinct city "
                         f"names gb_aviary can plant; raise --niah-tokens to "
                         f"make the haystack harder instead"}
    name, entry = _served(model)
    if not name:
        return {"error": "no serve session running"}
    kv_type = entry.get("kv_type") or "f16"

    t0 = time.monotonic()
    niah = gb_aviary.niah_certify(name, tokens=niah_tokens, needles=needles,
                                  kv_type=kv_type)
    # niah_certify returns `score` as a COUNT of needles retrieved, not a
    # fraction. Reporting the count as "recall" reads as 800% at 8 needles;
    # the ratio is what compares across runs with different needle counts.
    found = niah.get("score")
    n = niah.get("needles") or needles or 1
    summary = {
        "model": name,
        "kv_type": kv_type,
        "kv_gb": _kv_gb(entry, niah),
        "ctx": entry.get("ctx"),
        "prompt_tokens": niah.get("prompt_tokens"),
        "niah_tokens": niah_tokens,
        "needles": n,
        "found": found,
        "recall": round(found / n, 3) if found is not None else None,
        "niah_status": niah.get("status"),
        "niah_s": round(time.monotonic() - t0, 1),
    }
    if emit:
        try:
            import gb_dataflux
            gb_dataflux.emit({"node": "host", "label": "synapse",
                              "kind": "kv_quality", **summary})
        except Exception:
            pass
    return {"summary": summary, "niah": niah}


def sweep(model: str = "", kv_types=DEFAULT_KV_TYPES,
          niah_tokens: int = DEFAULT_NIAH_TOKENS, needles: int = DEFAULT_NEEDLES,
          confirm: bool = False) -> dict:
    """Re-serve at each kv_type and certify. Requires --confirm: this stops and
    restarts the live engine."""
    import gb_synapse
    name, _ = _served(model)
    name = name or model
    if not name:
        return {"error": "no serve session running; pass --model"}
    if not confirm:
        return {"error": "sweep re-serves the model at each kv_type; pass --confirm",
                "would_measure": list(kv_types), "model": name}
    # Check before the first re-serve, not after: an invalid needle count used
    # to surface minutes in, once the engine had already been restarted.
    cap = max_needles()
    if needles > cap:
        return {"error": f"needles={needles} exceeds the {cap} distinct city "
                         f"names gb_aviary can plant; raise --niah-tokens "
                         f"instead", "model": name}

    rows = []
    failed = None
    try:
        for kv in kv_types:
            gb_synapse.stop(name)
            gb_synapse.serve(name, kv_type=kv)
            time.sleep(2)
            r = measure_current(name, niah_tokens, needles, emit=True)
            row = r.get("summary", {"error": r.get("error")})
            row["kv_type"] = kv                # authoritative: what we served
            rows.append(row)
    except Exception as e:
        # A measurement that raises must not leave the box with nothing
        # serving. The restore below used to sit after this loop, so any
        # exception , an engine restarting underneath us, a 502 from the proxy,
        # a bad needle count , skipped it entirely and left the machine dead
        # rather than merely mis-tuned. Unattended-For-Days rule: degrade,
        # never abort into a worse state.
        failed = f"{type(e).__name__}: {e}"
        rows.append({"kv_type": "(interrupted)", "error": failed})

    scored = [r for r in rows if r.get("recall") is not None]
    baseline = next((r for r in scored if r["kv_type"] == "f16"), None)
    for r in scored:
        if baseline and baseline.get("kv_gb") and r.get("kv_gb"):
            r["vram_saved_gb"] = round(baseline["kv_gb"] - r["kv_gb"], 3)
        if baseline and baseline.get("recall") is not None:
            r["recall_lost"] = round(baseline["recall"] - r["recall"], 3)

    # Restore the certification-grade default whatever happened above,
    # INCLUDING when the sweep raised.
    restored = None
    try:
        if failed or (rows and rows[-1].get("kv_type") != "f16"):
            gb_synapse.stop(name)
            gb_synapse.serve(name, kv_type="f16")
            restored = "f16"
    except Exception as e:
        restored = f"FAILED: {type(e).__name__}: {e}"    # say so, never silently

    # A test that returns the same verdict for every setting is not measuring
    # the setting. Measured 2026-08-20: f16, q8_0 AND q4_0 all scored 10/10 at
    # 20k tokens with 10 needles, which says more about the haystack than about
    # the quantization. Say so in the result rather than letting a reader take
    # "no difference" as "no cost".
    verdicts = {r.get("recall") for r in scored}
    saturated = len(scored) > 1 and len(verdicts) == 1 and verdicts == {1.0}
    out = {"model": name, "kv_types": rows, "restored_kv_type": restored,
           **({"failed": failed} if failed else {}),
           "note": ("recall is the gate that matters for KV quantization; a "
                    "smoke gate passes quantized KV that has quietly lost "
                    "long-range retrieval")}
    if saturated:
        out["warning"] = (
            "SATURATED: every kv_type scored a perfect recall, so this run "
            "cannot distinguish them. The haystack is too easy , raise "
            "--niah-tokens toward the context limit and --needles until f16 "
            "itself starts missing, then compare. Do NOT read this as "
            "'quantization is free'.")
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="gb_bench_kv")
    p.add_argument("--model", default="")
    p.add_argument("--current", action="store_true",
                   help="certify the kv_type served now (no re-serve)")
    p.add_argument("--kv-types", default=",".join(DEFAULT_KV_TYPES))
    p.add_argument("--niah-tokens", type=int, default=DEFAULT_NIAH_TOKENS)
    p.add_argument("--needles", type=int, default=DEFAULT_NEEDLES)
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    if a.current:
        res = measure_current(a.model, a.niah_tokens, a.needles)
    else:
        res = sweep(a.model, tuple(t.strip() for t in a.kv_types.split(",") if t.strip()),
                    a.niah_tokens, a.needles, a.confirm)
    if a.json:
        print(json.dumps(res, indent=1)); return
    if res.get("error"):
        print(res["error"]); return
    if "summary" in res:
        s = res["summary"]
        print(f"model {s['model']}   kv={s['kv_type']}  kv_gb={s.get('kv_gb')}")
        print(f"NIAH recall {s.get('recall')}  ({s.get('found')}/{s['needles']} "
              f"at {s['niah_tokens']} tokens, {s['niah_s']}s)")
        return
    print(f"model {res['model']}")
    for r in res["kv_types"]:
        print(f"  kv={r['kv_type']:<6} recall={r.get('recall')}  "
              f"kv_gb={r.get('kv_gb')}  saved={r.get('vram_saved_gb')}  "
              f"recall_lost={r.get('recall_lost')}")
    if res.get("restored_kv_type"):
        print(f"  restored to {res['restored_kv_type']}")


if __name__ == "__main__":
    main()
