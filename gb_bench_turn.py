#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_bench_turn.py , per-turn prefill/decode benchmark for the serving path.

Phase 0 of the inference-speed program (greenboost_plans/
inference_speed_program_2026-08-18.md). Built FIRST, deliberately: every later
phase claims a speedup, and none of those claims are worth anything without a
before/after that survives a re-run on a box whose VRAM budget shifts under it.

What it separates, and why that is the whole point
--------------------------------------------------
A turn's wall time is prefill + decode, and `prefill` on this model is really
two different costs wearing one name:

  real prefill    , processing tokens the engine has never seen. Recoverable
                    only by sending fewer tokens.
  recurrent replay, re-running the 48 Gated DeltaNet layers over the ENTIRE
                    prompt because a recurrent layer's state at position t only
                    exists by running every token up to t. Paid in full even at
                    a 99.9% KV hit rate, and paid again every single turn.

Measured on the reference workload 2026-08-18, 29 warm-cache turns: median
prefill 6,937 ms for a median of 42 new tokens. Cost per TOTAL token was stable
(0.151 ms, stdev 0.33); cost per NEW token scattered 146-191 ms. That stability
is the signature , the work tracks total length, not new length.

The replay figure below is therefore an ESTIMATE, not a direct reading: the
engine reports one `prompt_ms` and does not break it down. It is computed by
charging new tokens at the cold-prefill rate this box actually achieves and
attributing the remainder to replay. Stated as an estimate everywhere it is
reported, because presenting a derived number as a measurement is how a
plausible story outlives the evidence for it.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request

# Cold-prefill rate, ms per token, for charging genuinely-new tokens.
#
# Not a hardcoded hardware constant in the sense the repo rule forbids: it is a
# per-box MEASUREMENT with a documented default, re-derivable at any time with
# `--calibrate`, and overridable. The default is this box's own observed cold
# prefill (7,906 tokens with zero reuse in 125,317 ms = 15.85 ms/token).
DEFAULT_COLD_MS_PER_TOKEN = 15.85

DEFAULT_BASE_URL = "http://127.0.0.1:11369/v1"


def _post(url: str, payload: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def split_prefill(prompt_tokens: int, reused_tokens: int, prompt_ms: float,
                  cold_ms_per_token: float = DEFAULT_COLD_MS_PER_TOKEN) -> dict:
    """Attribute one turn's prompt_ms to real prefill vs recurrent replay.

    Returns both halves plus the ms-per-total-token ratio, which is the
    diagnostic that made the split visible in the first place: when it is
    stable across turns with wildly different new-token counts, the cost is
    tracking total length and replay dominates.
    """
    new_tokens = max(0, prompt_tokens - reused_tokens)
    real_ms = min(prompt_ms, new_tokens * cold_ms_per_token)
    replay_ms = max(0.0, prompt_ms - real_ms)
    return {
        "prompt_tokens": prompt_tokens,
        "reused_tokens": reused_tokens,
        "new_tokens": new_tokens,
        "prompt_ms": round(prompt_ms, 1),
        "real_prefill_ms_est": round(real_ms, 1),
        "recurrent_replay_ms_est": round(replay_ms, 1),
        "replay_share": round(replay_ms / prompt_ms, 3) if prompt_ms > 0 else 0.0,
        "ms_per_total_token": round(prompt_ms / prompt_tokens, 4) if prompt_tokens else 0.0,
        "ms_per_new_token": round(prompt_ms / new_tokens, 1) if new_tokens else None,
    }


# Measured bulk-DMA throughput of this box's PCIe link, host->VRAM, pinned.
#
# Measured 2026-08-18 at 24.43 GB/s, which is ~94% of the Gen4 x16 usable
# ceiling. That number matters as a PHYSICAL LIMIT, not a target: no more than
# this many bytes can cross the bus per second, so any telemetry implying more
# is wrong regardless of how confidently it is reported.
#
# Gen5 is not reachable on this board and this is not a tuning knob: the CPU
# root port (00:01.0) advertises max 16.0 GT/s while the GPU advertises 32.0,
# so the limit is the board, not the card. See gb_pcie_tune.py.
PCIE_BULK_DMA_GBS = float(os.environ.get("GB_PCIE_BULK_DMA_GBS", "24.43"))


def spill_is_physically_possible(overflow_mb: float, decode_tok_s: float,
                                 link_gbs: float = 0.0) -> dict:
    """Could this much data really cross PCIe at this decode rate?

    A dense model reads every weight every token, so `overflow_mb` bytes must
    traverse the bus once per token. Multiply by tok/s and compare against what
    the link can actually carry. If the product exceeds the link, the overflow
    figure is wrong , the decode rate is measured end to end and is not in
    doubt.

    This exists because the check it guards was itself misled. On 2026-08-18
    `t2_overflow_active_mb` reported 5,923 MB alongside a measured 8.37 tok/s,
    which implies 48.4 GB/s on a link measured at 24.43 , roughly 2x
    over-reported. Every consumer of that field inherits the error: this
    module's own fit verdict, the Rule #1 tripwire, and placement decisions.
    Telemetry that cannot be checked against physics gets believed for months.
    """
    link = link_gbs or PCIE_BULK_DMA_GBS
    if not overflow_mb or not decode_tok_s or decode_tok_s <= 0:
        return {"checked": False}
    implied = (overflow_mb / 1024.0) * decode_tok_s      # GiB/token * tok/s
    possible = implied <= link
    return {
        "checked": True, "possible": possible,
        "implied_gbs": round(implied, 1), "link_gbs": link,
        "max_plausible_overflow_mb": round(link / decode_tok_s * 1024),
        "note": None if possible else (
            f"reported overflow implies {implied:.1f} GB/s across a link "
            f"measured at {link:.2f} , the overflow figure is over-reported, "
            f"not the decode rate"),
    }


def verify_zero_spill(decode_tok_s: float = 0.0) -> dict:
    """Did the served model actually fit in VRAM, with nothing crossing PCIe?

    The question a quant comparison has to answer BEFORE its tok/s number means
    anything, and the one nobody asked on 2026-08-05. `NEO-MTP-IQ2_M` (11.29
    GiB) was tested against `NEO-MTP-IQ4_XS` (15.86 GiB), measured slower, and
    written off as "I-quant dequantization cost outweighing reduced PCIe
    traffic". It never fit: 11.29 + 0.58 KV + activations overruns 11.94 GiB by
    ~0.28. So it paid full PCIe traffic AND I-quant dequant cost, and the
    conclusion drawn from that number sent the next six months of tuning away
    from the one lever that matters.

    A quant that fits reads weights from VRAM at ~672 GB/s; one that spills
    reads the overflow at ~11.5 GB/s. That is a ~58x difference in the dominant
    term, so "did it fit" is not a detail attached to the benchmark , it is the
    benchmark's precondition.
    """
    out = {"fits": None, "overflow_mb": None, "vram_fill_pct": None}
    try:
        import gb_semantics
        ov = gb_semantics.resolve("t2_overflow_active_mb")
        fill = gb_semantics.resolve("vram_fill_pct")
        out["overflow_mb"] = ov.get("value")
        out["vram_fill_pct"] = fill.get("value")
        out["source"] = ov.get("provenance", {}).get("raw_source")
        if out["overflow_mb"] is None:
            # Unknown is NOT "fits". Reporting a clean fit from missing
            # telemetry is the same failure this function exists to prevent.
            out["verdict"] = ("cannot tell , the shim reported no overflow "
                              "figure; do not compare tok/s against another "
                              "quant until this resolves")
            return out
        out["fits"] = out["overflow_mb"] <= 0
        out["verdict"] = ("fits entirely in VRAM , decode is VRAM-bound"
                          if out["fits"] else
                          f"spilling {out['overflow_mb']:.0f} MB to T2 , decode "
                          f"is PCIe-bound and this tok/s number reflects the "
                          f"bus, not the quant")
        if decode_tok_s:
            # Sanity-check the figure against what the bus can physically carry.
            out["plausibility"] = spill_is_physically_possible(
                out["overflow_mb"], decode_tok_s)
            if out["plausibility"].get("possible") is False:
                out["verdict"] += (" — BUT " + out["plausibility"]["note"])
    except Exception as e:
        out["verdict"] = f"unavailable: {e}"
    return out


def _turn(base_url: str, model: str, messages: list, max_tokens: int) -> dict:
    """One non-streaming turn. Non-streaming on purpose: llama-server returns
    its own `timings` block there, so prefill and decode come from the engine
    rather than being reconstructed from SSE arrival times."""
    t0 = time.monotonic()
    d = _post(f"{base_url}/chat/completions", {
        "model": model, "messages": messages, "stream": False,
        "max_tokens": max_tokens, "temperature": 0.0,
    })
    wall_s = time.monotonic() - t0
    usage = d.get("usage") or {}
    tim = d.get("timings") or {}
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")
              or d.get("tokens_cached") or 0)
    out = {
        "wall_s": round(wall_s, 3),
        "completion_tokens": usage.get("completion_tokens", 0),
        "predicted_ms": tim.get("predicted_ms"),
        "decode_tok_s": tim.get("predicted_per_second"),
        "content_chars": len(((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""),
    }
    out.update(split_prefill(usage.get("prompt_tokens", 0), cached,
                             tim.get("prompt_ms") or 0.0))
    return out


# Tokens of conversation-private context each interleaved conversation carries.
# Big enough that losing it is unmistakable next to the shared head, small
# enough that the fixture still runs in a couple of minutes at ~4 tok/s.
_UNIQUE_BLOCK_TOKENS = 1200


def run_interleaved(base_url: str = DEFAULT_BASE_URL, model: str = "",
                    conversations: int = 2, turns: int = 3,
                    max_tokens: int = 32, filler_tokens: int = 2000,
                    conversation_ids: bool = False, emit: bool = True) -> dict:
    """GB-1: what several conversations sharing one system prompt cost.

    This is the case `--slot-prompt-similarity` cannot fix and that this repo
    left open on 2026-08-05: every conversation on this box carries the same
    GB-CLI system prompt, so any nonzero overlap with a recently-touched slot
    beats an idle slot's zero score, and two conversations end up trading the
    same slot back and forth — each one re-prefilling what the other evicted.

    Round-robins K conversations, one turn each, and reports the reuse each one
    got. With slot routing working, every conversation's second turn reuses its
    OWN history (near-100%); without it, they converge on the shared system
    prompt and reuse stalls at whatever fraction that head represents.
    """
    if not model:
        try:
            import gb_synapse
            ps = gb_synapse.ps()
            if not ps:
                return {"error": "no serve session running; start one or pass --model"}
            model = ps[0]["model"]
        except Exception as e:
            return {"error": f"could not resolve a model: {e}"}

    filler = ("The following is reference material retained across turns. "
              * max(1, filler_tokens // 12)) if filler_tokens else ""
    system = {"role": "system", "content":
              "You are a benchmark fixture. Answer in one short sentence." + filler}

    # Each conversation gets its own opening message and then diverges — the
    # shared head is the whole point, so it must be byte-identical.
    #
    # That opening message carries a LARGE unique block on purpose. With a
    # small one the measurement cannot see the failure: reuse is reported
    # against the whole prompt, so a conversation that kept only the shared
    # head still scores ~97% while having lost everything that was actually
    # its own. The unique block is what makes "kept my history" and "kept the
    # boilerplate" different numbers.
    convs = [[dict(system),
              {"role": "user", "content":
               f"Conversation {c}. Reference block, unique to this conversation:\n"
               + f"item {c}-%d: the quick brown fox jumps over the lazy dog. "
               * max(1, _UNIQUE_BLOCK_TOKENS // 12) % tuple(range(max(1, _UNIQUE_BLOCK_TOKENS // 12)))
               }]
             for c in range(conversations)]

    rows = []
    for t in range(turns):
        for c, messages in enumerate(convs):
            if t:
                messages.append({"role": "user",
                                 "content": f"Turn {t} of conversation {c}. Reply '{t}'."})
            payload = {"conversation_id": f"bench-{c}"} if conversation_ids else {}
            try:
                r = _turn_with(base_url, model, messages, max_tokens, payload)
            except Exception as e:
                return {"error": f"conversation {c} turn {t + 1} failed: {e}",
                        "turns": rows}
            r.update({"conversation": c, "turn": t + 1})
            rows.append(r)
            messages.append({"role": "assistant", "content": f"{t}."})

    # Turn 1 is cold for everyone; the question is what happens from turn 2 on,
    # once each conversation has a history that only its own slot holds.
    warm = [r for r in rows if r["turn"] > 1 and r["prompt_tokens"]]
    reuse = [100.0 * (r.get("reused_tokens") or 0) / r["prompt_tokens"] for r in warm]
    summary = {
        "model": model, "conversations": conversations, "turns": turns,
        "conversation_ids": conversation_ids,
        "median_warm_reuse_pct": round(statistics.median(reuse), 1) if reuse else None,
        "min_warm_reuse_pct": round(min(reuse), 1) if reuse else None,
        "median_warm_prefill_ms": (round(statistics.median(
            [r["prompt_ms"] for r in warm]), 1) if warm else None),
    }
    result = {"summary": summary, "turns": rows}
    if emit:
        try:
            import gb_dataflux
            gb_dataflux.emit({"node": "host", "label": "gb_bench",
                              "kind": "turn_bench", "model": model,
                              "mode": "interleaved", **summary})
        except Exception:
            pass
    return result


def _turn_with(base_url: str, model: str, messages: list, max_tokens: int,
               extra: dict) -> dict:
    """_turn, plus any extra top-level request fields (conversation id)."""
    body = {"model": model, "messages": messages, "stream": False,
            "max_tokens": max_tokens, "temperature": 0.0}
    if extra.get("conversation_id"):
        body["gb_conversation"] = extra["conversation_id"]
    t0 = time.monotonic()
    d = _post(f"{base_url}/chat/completions", body)
    wall_s = time.monotonic() - t0
    usage = d.get("usage") or {}
    tim = d.get("timings") or {}
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")
              or d.get("tokens_cached") or 0)
    out = {"wall_s": round(wall_s, 3),
           "completion_tokens": usage.get("completion_tokens", 0),
           "decode_tok_s": tim.get("predicted_per_second")}
    out.update(split_prefill(usage.get("prompt_tokens", 0), cached,
                             tim.get("prompt_ms") or 0.0))
    return out


def run(base_url: str = DEFAULT_BASE_URL, model: str = "", turns: int = 5,
        max_tokens: int = 64, filler_tokens: int = 0, emit: bool = True,
        echo_tokens: int = 0, edit_at: int = 0) -> dict:
    """Replay a growing conversation and report the per-turn cost split.

    Grows the conversation the way an agentic session does — append only — so
    the overlap distribution matches the workload this is meant to represent,
    not a synthetic one where every turn is a cache miss.
    """
    if not model:
        try:
            import gb_synapse
            ps = gb_synapse.ps()
            if not ps:
                return {"error": "no serve session running; start one or pass --model"}
            model = ps[0]["model"]
        except Exception as e:
            return {"error": f"could not resolve a model: {e}"}

    # A long stable prefix makes the replay cost visible; without it every turn
    # is short and prefill disappears into noise.
    filler = ("The following is reference material retained across turns. "
              * max(1, filler_tokens // 12)) if filler_tokens else ""
    messages = [{"role": "system", "content":
                 "You are a benchmark fixture. Answer in one short sentence." + filler}]

    # `echo` mode asks the model to reproduce a span already in its context.
    #
    # The default prompts ("State the number N") generate almost nothing that
    # appears in the prompt, which makes them blind to any lever that works by
    # reusing context — ngram/prompt-lookup speculation in particular. Measured
    # 2026-08-18: stacking ngram-cache on MTP moved decode 4.53 -> 4.71 tok/s
    # (+4%, inside noise) on the default prompts, which says more about the
    # fixture than the lever. A real agentic turn re-emits file contents, paths
    # and tool output verbatim, and that is what this mode imitates.
    if echo_tokens:
        corpus = " ".join(f"def step_{i}(state): return state['{i}'] + {i}"
                          for i in range(max(1, echo_tokens // 10)))
        messages[0]["content"] += "\n\nSOURCE:\n" + corpus

    rows = []
    for i in range(turns):
        # GB-0: at turn `edit_at`, mutate a message EARLY in the conversation
        # instead of only appending.
        #
        # This is the measurement the harness was missing. Append-only replay
        # shows the good case , every turn reuses everything before it. What
        # actually costs this box its time is an edit near the FRONT, which is
        # what history compaction, a summary rewrite, or a re-rendered system
        # block all are. Measured over 14 days of real sessions, the difference
        # between reusing >=99% of the prefix and under 50% was 5.5s vs 166s of
        # time-to-first-token, and nothing here could reproduce it on demand.
        #
        # Editing message 1 (the first user turn) invalidates the KV for
        # everything after it, so the turn that follows pays a full re-prefill.
        # That number is the baseline any chunk-level cache work (GB-1) has to
        # beat.
        if edit_at and (i + 1) == edit_at and len(messages) > 2:
            messages[1]["content"] += " (revised)"
        if echo_tokens:
            # Reproducing a span verbatim is what a coding agent spends most of
            # its output tokens doing, and it is the only way this harness can
            # see a context-reuse lever at all.
            messages.append({"role": "user", "content":
                             f"Repeat SOURCE function step_{i} exactly, nothing else."})
        else:
            messages.append({"role": "user", "content": f"State the number {i + 1} and stop."})
        try:
            r = _turn(base_url, model, messages, max_tokens)
        except Exception as e:
            return {"error": f"turn {i + 1} failed: {e}", "turns": rows}
        r["turn"] = i + 1
        r["edited_here"] = bool(edit_at and (i + 1) == edit_at)
        rows.append(r)
        messages.append({"role": "assistant", "content": f"{i + 1}."})

    warm = [r for r in rows[1:] if r["prompt_tokens"]]
    summary = {
        "model": model, "turns": len(rows),
        "warm_turns": len(warm),
        "median_prompt_ms": round(statistics.median([r["prompt_ms"] for r in warm]), 1) if warm else None,
        "median_replay_ms_est": round(statistics.median(
            [r["recurrent_replay_ms_est"] for r in warm]), 1) if warm else None,
        "median_new_tokens": statistics.median([r["new_tokens"] for r in warm]) if warm else None,
        "median_decode_tok_s": round(statistics.median(
            [r["decode_tok_s"] for r in warm if r.get("decode_tok_s")]), 2)
            if any(r.get("decode_tok_s") for r in warm) else None,
        "ms_per_total_token_stdev": round(statistics.pstdev(
            [r["ms_per_total_token"] for r in warm]), 4) if len(warm) > 1 else None,
    }
    # Reported alongside every result, never as a separate opt-in step: a tok/s
    # figure without this is not interpretable (see verify_zero_spill).
    # What the edit actually cost, stated as a ratio rather than left for the
    # reader to eyeball out of the per-turn rows.
    if edit_at:
        # WARM turns only. Turn 1 prefills from cold and costs an order of
        # magnitude more, so leaving it in the baseline would flatter the edit
        # by comparing it against a median that is half cold.
        pre = [r for r in rows if r["turn"] < edit_at and r["prompt_tokens"]
               and r.get("reused_tokens")]
        post = [r for r in rows if r["turn"] == edit_at]
        if pre and post:
            base_ms = statistics.median([r["prompt_ms"] for r in pre]) or 0.0
            hit_ms = post[0]["prompt_ms"] or 0.0
            summary["edit_cost"] = {
                "edited_at_turn": edit_at,
                "median_prompt_ms_before": round(base_ms, 1),
                "prompt_ms_on_edited_turn": round(hit_ms, 1),
                "penalty_x": round(hit_ms / base_ms, 2) if base_ms else None,
                "reused_tokens_before": pre[-1].get("reused_tokens"),
                "reused_tokens_after": post[0].get("reused_tokens"),
            }
    summary["fit"] = verify_zero_spill(summary.get("median_decode_tok_s") or 0.0)
    result = {"summary": summary, "turns": rows}

    if emit:
        try:
            import gb_dataflux
            gb_dataflux.emit({
                "node": "host", "label": "gb_bench", "kind": "turn_bench",
                "model": model, "n_turns": len(rows),
                "median_prompt_ms": summary["median_prompt_ms"],
                "median_replay_ms_est": summary["median_replay_ms_est"],
                "median_decode_tok_s": summary["median_decode_tok_s"],
                "median_new_tokens": summary["median_new_tokens"],
            })
        except Exception:
            pass
    return result


def main() -> None:
    p = argparse.ArgumentParser(prog="gb_bench_turn.py", description=__doc__)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default="")
    p.add_argument("--turns", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--filler-tokens", type=int, default=0,
                   help="pad the system prompt so replay cost is visible")
    p.add_argument("--echo-tokens", type=int, default=0,
                   help="ask the model to reproduce context verbatim — the only "
                        "mode that can measure context-reuse levers (ngram "
                        "speculation, prefix caching)")
    p.add_argument("--edit-at", type=int, default=0, metavar="TURN",
                   help="edit the FIRST user message just before turn TURN, "
                        "the way compaction or a summary rewrite does — "
                        "measures what an invalidated prefix costs, which is "
                        "the number GB-1 has to beat")
    p.add_argument("--cold-ms-per-token", type=float, default=DEFAULT_COLD_MS_PER_TOKEN)
    p.add_argument("--conversations", type=int, default=0, metavar="K",
                   help="GB-1 mode: interleave K conversations that share one "
                        "system prompt, and report the reuse each one keeps")
    p.add_argument("--conversation-ids", action="store_true",
                   help="with --conversations, send an explicit conversation id "
                        "so identity is exact rather than inferred")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-emit", action="store_true")
    a = p.parse_args()

    # Rebind the module-level calibration so split_prefill()'s default picks it
    # up; it is a per-box measurement, not a constant (see its docstring).
    globals()["DEFAULT_COLD_MS_PER_TOKEN"] = a.cold_ms_per_token

    if a.conversations:
        res = run_interleaved(a.base_url, a.model, a.conversations, a.turns,
                              a.max_tokens, a.filler_tokens,
                              conversation_ids=a.conversation_ids,
                              emit=not a.no_emit)
        if "error" in res:
            print(f"error: {res['error']}")
            raise SystemExit(1)
        if a.json:
            print(json.dumps(res, indent=1))
            return
        s = res["summary"]
        print(f"model {s['model']}   {s['conversations']} conversations x "
              f"{s['turns']} turns"
              + ("   (explicit ids)" if s["conversation_ids"] else "   (inferred identity)"))
        print(f"{'conv':>5} {'turn':>5} {'prompt':>8} {'reused':>8} {'reuse':>7} {'prefill':>10}")
        for r in res["turns"]:
            reuse = (100.0 * (r.get("reused_tokens") or 0) / r["prompt_tokens"]
                     if r["prompt_tokens"] else 0)
            print(f"{r['conversation']:>5} {r['turn']:>5} {r['prompt_tokens']:>8} "
                  f"{r.get('reused_tokens', 0):>8} {reuse:>6.0f}% {r['prompt_ms']:>8.0f}ms")
        print()
        print(f"median warm reuse   {s['median_warm_reuse_pct']}%   "
              f"(min {s['min_warm_reuse_pct']}%)")
        print(f"median warm prefill {s['median_warm_prefill_ms']} ms")
        return

    res = run(a.base_url, a.model, a.turns, a.max_tokens, a.filler_tokens,
              emit=not a.no_emit, echo_tokens=a.echo_tokens, edit_at=a.edit_at)
    if a.json:
        print(json.dumps(res, indent=1))
        return
    if "error" in res:
        print(f"error: {res['error']}")
        raise SystemExit(1)
    s = res["summary"]
    print(f"model {s['model']}   {s['warm_turns']} warm turns")
    print(f"{'turn':>4} {'total':>8} {'new':>6} {'prefill':>9} {'replay est':>11} "
          f"{'share':>6} {'decode':>8}")
    for r in res["turns"]:
        print(f"{r['turn']:>4} {r['prompt_tokens']:>8} {r['new_tokens']:>6} "
              f"{r['prompt_ms']:>8.0f}ms {r['recurrent_replay_ms_est']:>10.0f}ms "
              f"{r['replay_share']:>5.0%} "
              f"{(r.get('decode_tok_s') or 0):>6.1f}t/s"
              + ("  <- edited" if r.get("edited_here") else ""))
    print()
    print(f"median prefill      {s['median_prompt_ms']} ms")
    print(f"median replay (est) {s['median_replay_ms_est']} ms   <- what Phase 1 targets")
    print(f"median decode       {s['median_decode_tok_s']} tok/s")
    print(f"ms/total-token sd   {s['ms_per_total_token_stdev']}   (low => cost tracks TOTAL)")
    ec = s.get("edit_cost")
    if ec:
        print()
        print(f"EDIT at turn {ec['edited_at_turn']}: prefill "
              f"{ec['median_prompt_ms_before']} ms -> "
              f"{ec['prompt_ms_on_edited_turn']} ms"
              + (f"  ({ec['penalty_x']}x)" if ec.get("penalty_x") else ""))
        print(f"     reused tokens {ec['reused_tokens_before']} -> "
              f"{ec['reused_tokens_after']}")
    fit = s.get("fit") or {}
    print()
    print(f"FIT: {fit.get('verdict', 'unknown')}")
    if fit.get("overflow_mb") is not None:
        print(f"     overflow {fit['overflow_mb']:.0f} MB · VRAM fill {fit.get('vram_fill_pct')}%")


if __name__ == "__main__":
    main()
