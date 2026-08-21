"""Tests for proxy-owned tok/s measurement in gb_synapse_api.

Exercises the measurement helpers directly (no live aiohttp server), plus a
simulation of the SSE-chunk accumulation loop that ollama_chat/ollama_generate
run, asserting the correct decode rate is handed to gb_synapse.record_measured_
tok_s.
"""
import json

import pytest

pytest.importorskip("aiohttp")

import gb_synapse_api as api


def test_usage_counts_extracts_and_keeps_last():
    assert api._usage_counts({}, (0, 0)) == (0, 0)
    assert api._usage_counts({"usage": {"completion_tokens": 50, "prompt_tokens": 12}},
                              (0, 0)) == (50, 12)
    # a chunk without usage keeps the previous values
    assert api._usage_counts({"choices": [{}]}, (50, 12)) == (50, 12)
    # null usage keeps previous
    assert api._usage_counts({"usage": {"completion_tokens": None, "prompt_tokens": None}},
                              (50, 12)) == (50, 12)
    # partial usage (only one field present) updates just that field
    assert api._usage_counts({"usage": {"completion_tokens": 60}}, (50, 12)) == (60, 12)


def test_record_tok_s_computes_decode_rate(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "gb_synapse.record_measured_tok_s",
        lambda model, tok_s, source="", **kw: captured.update(
            model=model, tok_s=tok_s, source=source, **kw))
    # 101 tokens, first→last span 1.0s → 100 tok after the first / 1.0s = 100 tok/s
    api._record_tok_s("qwen3", t_first=10.0, t_last=11.0, completion_tokens=101)
    assert captured["model"] == "qwen3"
    assert abs(captured["tok_s"] - 100.0) < 1e-6
    assert captured["source"] == "proxy"


def test_record_tok_s_skips_incomplete(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "gb_synapse.record_measured_tok_s",
        lambda model, tok_s, source="", **kw: calls.append((model, tok_s, source)))
    api._record_tok_s("m", None, 5.0, 10)          # no first-token time
    api._record_tok_s("m", 1.0, 1.0, 10)           # zero interval
    api._record_tok_s("m", 1.0, 2.0, 1)            # single token
    api._record_tok_s("m", 1.0, 2.0, 0)            # no usage
    assert calls == []


def test_record_tok_s_never_raises(monkeypatch):
    def _boom(model, tok_s, source=""):
        raise RuntimeError("store down")
    monkeypatch.setattr("gb_synapse.record_measured_tok_s", _boom)
    api._record_tok_s("m", 1.0, 2.0, 100)          # must swallow the error


def test_stream_loop_simulation(monkeypatch):
    """Mirror the ollama_chat accumulation: content chunks then a final usage
    chunk, and assert the recorded rate matches the observed span."""
    captured = {}
    monkeypatch.setattr(
        "gb_synapse.record_measured_tok_s",
        lambda model, tok_s, source="", **kw: captured.update(
            model=model, tok_s=tok_s, source=source, **kw))

    # completion_tokens is the UPSTREAM's own count and is independent of how
    # many SSE pieces carried it (one delta can hold several tokens), so the
    # three content chunks below still exercise the accumulation loop.
    # It is set above _MIN_TOK_S_SAMPLE_TOKENS deliberately: since 2026-08-17
    # _record_tok_s() drops completions too short to be a decode-rate
    # measurement, and the original 3-token fixture now falls under that floor,
    # which would make this test assert the very behaviour that guard removes.
    chunks = [
        {"choices": [{"delta": {"content": "a"}}]},
        {"choices": [{"delta": {"content": "b"}}]},
        {"choices": [{"delta": {"content": "c"}}]},
        {"choices": [], "usage": {"completion_tokens": 101}},
    ]
    times = iter([100.0, 100.5, 101.0])   # monotonic() for the 3 content pieces
    monkeypatch.setattr(api.time, "monotonic", lambda: next(times))

    t_first = t_last = None
    ctok = ptok = 0
    for chunk in chunks:
        piece = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content", "")
        if piece:
            now = api.time.monotonic()
            if t_first is None:
                t_first = now
            t_last = now
        ctok, ptok = api._usage_counts(chunk, (ctok, ptok))
    api._record_tok_s("qwen3", t_first, t_last, ctok)

    # (101-1) tokens / (101.0 - 100.0)s = 100 tok/s
    assert captured["tok_s"] == pytest.approx(100.0)


def test_record_non_stream_telemetry_reads_native_timings(monkeypatch):
    """Regression for the 2026-08-05 gap: a stream:false request through
    openai_passthrough recorded nothing (only the SSE branch called
    _record_tok_s), even though llama-server's own non-streaming response
    already carries a `timings` block with predicted_per_second. GB-CLI's
    BACKEND_REGISTRY talks this exact route for its real traffic."""
    tok_s_captured = {}
    cache_captured = {}
    monkeypatch.setattr(
        "gb_synapse.record_measured_tok_s",
        lambda model, tok_s, source="", **kw: tok_s_captured.update(
            model=model, tok_s=tok_s, source=source, **kw))
    monkeypatch.setattr(
        "gb_synapse.record_prompt_cache_sample",
        # **kw deliberately: a fixed-signature stub turns an ADDITIVE kwarg on
        # the real function into a TypeError that the caller's `except
        # Exception: pass` swallows, leaving `captured` empty and the test
        # failing with a confusing KeyError instead of naming the cause. Same
        # over-specified-stub trap the completion_tokens work hit on 2026-08-17.
        lambda model, ttft_ms, hit_pct, reused, **kw: cache_captured.update(
            model=model, ttft_ms=ttft_ms, hit_pct=hit_pct, reused=reused, **kw))

    raw = json.dumps({
        "usage": {"completion_tokens": 150, "prompt_tokens": 29,
                   "prompt_tokens_details": {"cached_tokens": 10}},
        "timings": {"predicted_n": 150, "predicted_ms": 16975.754,
                    "predicted_per_second": 8.836, "prompt_ms": 770.622},
    }).encode()
    api._record_non_stream_telemetry("qwen3", raw)

    assert tok_s_captured["model"] == "qwen3"
    assert tok_s_captured["tok_s"] == pytest.approx(8.836)
    assert tok_s_captured["source"] == "proxy"
    assert cache_captured["ttft_ms"] == pytest.approx(770.622)
    assert cache_captured["reused"] == 10
    assert cache_captured["hit_pct"] == pytest.approx(100.0 * 10 / 29)


def test_record_non_stream_telemetry_derives_rate_without_predicted_per_second(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "gb_synapse.record_measured_tok_s",
        lambda model, tok_s, source="", **kw: captured.update(tok_s=tok_s, **kw))
    monkeypatch.setattr("gb_synapse.record_prompt_cache_sample", lambda *a, **k: None)
    raw = json.dumps({
        "usage": {"completion_tokens": 100, "prompt_tokens": 10},
        "timings": {"predicted_n": 100, "predicted_ms": 10000.0},
    }).encode()
    api._record_non_stream_telemetry("qwen3", raw)
    assert captured["tok_s"] == pytest.approx(10.0)


def test_non_stream_applies_the_same_sample_floor_as_streaming(monkeypatch):
    """A decode rate measured over a handful of tokens is noise, not a sample.

    The streaming path has always enforced _MIN_TOK_S_SAMPLE_TOKENS; this
    branch was taught to record telemetry later (2026-08-05) and never
    inherited the floor, so 2-token replies landed as full-weight samples
    scoring hundreds of tok/s. Measured consequence on the reference workload
    (2026-08-17): samples of 254.9 and 283.8 tok/s pulled the mean to 33.2
    against a median of 13.9, which is what produced a false regression alert.
    """
    calls = []
    monkeypatch.setattr("gb_synapse.record_measured_tok_s",
                        lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr("gb_synapse.record_prompt_cache_sample", lambda *a, **k: None)
    short = api._MIN_TOK_S_SAMPLE_TOKENS - 1
    raw = json.dumps({
        "usage": {"completion_tokens": short, "prompt_tokens": 10},
        "timings": {"predicted_n": short, "predicted_ms": 12.0},
    }).encode()
    api._record_non_stream_telemetry("qwen3", raw)
    assert calls == [], f"a {short}-token generation must not become a tok/s sample"


def test_non_stream_records_completion_tokens_for_weighting(monkeypatch):
    """Samples must carry their generation length, or every consumer has to
    treat a short reply and a long one as equal evidence — the duration-blindness
    that made the outliers above indistinguishable from real measurements."""
    captured = {}
    monkeypatch.setattr(
        "gb_synapse.record_measured_tok_s",
        lambda model, tok_s, source="", **kw: captured.update(tok_s=tok_s, **kw))
    monkeypatch.setattr("gb_synapse.record_prompt_cache_sample", lambda *a, **k: None)
    raw = json.dumps({
        "usage": {"completion_tokens": 150, "prompt_tokens": 10},
        "timings": {"predicted_n": 150, "predicted_ms": 10000.0},
    }).encode()
    api._record_non_stream_telemetry("qwen3", raw)
    assert captured["completion_tokens"] == 150


def test_samples_carry_the_kv_depth_they_were_decoded_at(monkeypatch):
    """`ctx` is the configured window; `prompt_tokens` is how much of it was
    occupied. Decode rate falls as attention walks a longer cache, so without
    this field a slow sample cannot be attributed to the window setting rather
    than the conversation length — exactly the ambiguity that left
    "3.7 tok/s at ctx=65536 vs median 13.9 at ctx=32768" unresolvable."""
    captured = {}
    monkeypatch.setattr(
        "gb_synapse.record_measured_tok_s",
        lambda model, tok_s, source="", **kw: captured.update(tok_s=tok_s, **kw))
    monkeypatch.setattr("gb_synapse.record_prompt_cache_sample", lambda *a, **k: None)

    # Streaming path
    api._record_tok_s("qwen3", t_first=10.0, t_last=20.0,
                      completion_tokens=101, prompt_tokens=18374)
    assert captured["prompt_tokens"] == 18374

    # Non-streaming path — same field, same meaning
    captured.clear()
    raw = json.dumps({
        "usage": {"completion_tokens": 150, "prompt_tokens": 4950},
        "timings": {"predicted_per_second": 13.9},
    }).encode()
    api._record_non_stream_telemetry("qwen3", raw)
    assert captured["prompt_tokens"] == 4950


def test_kv_depth_absent_stays_distinguishable_from_zero(monkeypatch):
    """0 means "caller didn't know". The emitter drops the field entirely in
    that case rather than publishing a KV depth of zero, which would read as a
    genuine empty-cache measurement."""
    import gb_synapse
    events = []
    monkeypatch.setattr("gb_dataflux.emit", lambda ev: events.append(ev))

    gb_synapse._df_emit_tok_s("qwen3", 5.0, "proxy", quant="q", ctx=65536,
                              kv_type="f16", completion_tokens=100)
    assert "prompt_tokens" not in events[-1]

    gb_synapse._df_emit_tok_s("qwen3", 5.0, "proxy", quant="q", ctx=65536,
                              kv_type="f16", completion_tokens=100,
                              prompt_tokens=18374)
    assert events[-1]["prompt_tokens"] == 18374


def test_record_non_stream_telemetry_skips_incomplete(monkeypatch):
    calls = []
    monkeypatch.setattr("gb_synapse.record_measured_tok_s",
                         lambda *a, **k: calls.append(a))
    # single/zero completion tokens -> skip, same guard as _record_tok_s
    api._record_non_stream_telemetry("m", json.dumps({"usage": {"completion_tokens": 1}}).encode())
    api._record_non_stream_telemetry("m", json.dumps({"usage": {"completion_tokens": 0}}).encode())
    # no usage at all -> skip
    api._record_non_stream_telemetry("m", json.dumps({}).encode())
    # malformed JSON -> never raises
    api._record_non_stream_telemetry("m", b"not json")
    assert calls == []


# ── llama.cpp native routes must reach the engine (found 2026-08-20) ────────

def test_the_proxy_forwards_tokenize():
    """`gb_aviary.niah_certify()` sizes its haystack with /tokenize against
    gb-synapse's own port. The proxy forwarded /v1/* and /slots only, so the
    call 404'd and NIAH certification could not run at all , which is why
    CLAUDE.md records it as never having been run against this checkpoint.
    """
    import gb_synapse_api as api
    assert "/tokenize" in api._LLAMA_NATIVE_ROUTES


def test_the_native_routes_are_read_only():
    """Only side-effect-free engine routes may be forwarded blind. Anything
    that mutates state belongs behind an explicit handler, not a wildcard."""
    import gb_synapse_api as api
    mutating = {"/completion", "/infill", "/slots", "/lora-adapters", "/apply-template"}
    assert not (set(api._LLAMA_NATIVE_ROUTES) & mutating)


def test_every_native_route_is_registered_on_the_app():
    import gb_synapse_api as api
    app = api.build_app() if hasattr(api, "build_app") else None
    if app is None:
        pytest.skip("no build_app() on this build")
    paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
    for route in api._LLAMA_NATIVE_ROUTES:
        assert route in paths, f"{route} not registered"


# ---------------------------------------------------------------------------
# Inter-token latency percentiles (2026-08-20)
# ---------------------------------------------------------------------------
# tok_s is a mean, and with MTP speculative decode the real gap distribution is
# bimodal: near-zero inside an accepted draft batch, a whole forward pass at the
# boundary. A mean describes no token that actually happened, which is why the
# p95 exists at all. These tests pin the shape, not a number.

def _content_frame(txt: str = "x") -> bytes:
    import json as _json
    return ("data: " + _json.dumps({"choices": [{"delta": {"content": txt}}]})
            + "\n").encode()


def _bimodal_telemetry(batches: int = 12):
    """A stream shaped like real speculative decode: one slow forward pass,
    then three near-instant accepted draft tokens."""
    import gb_synapse_api as api
    t = api._SSETelemetry()
    ts = 100.0
    for _ in range(batches):
        t.feed(ts, _content_frame())
        ts += 0.190                      # forward-pass boundary
        for _ in range(3):
            t.feed(ts, _content_frame())
            ts += 0.002                  # inside the accepted draft
    return t


def test_p95_exposes_the_tail_a_mean_hides():
    stats = _bimodal_telemetry().latency_stats()
    assert stats is not None
    # Median sits inside the draft batch; p95 exposes the pass boundary.
    assert stats["p50_ms"] < 5.0
    assert stats["p95_ms"] > 100.0
    # Roughly one gap in four is a boundary.
    assert 0.20 < stats["slow_token_ratio"] < 0.30
    assert stats["max_ms"] >= stats["p95_ms"] >= stats["p50_ms"]


def test_too_few_gaps_returns_none_not_a_confident_number():
    """A p95 over three samples is arithmetic, not evidence , the same n=3
    population problem the 2026-08-18 audit found in tok_s_drop advisories."""
    import gb_synapse_api as api
    t = api._SSETelemetry()
    for i in range(4):
        t.feed(100.0 + i * 0.1, _content_frame())
    assert t.latency_stats() is None


def test_latency_ring_is_bounded():
    """This proxy is long-lived and an unbounded per-request buffer already
    cost 37.5 GB once (2026-08-18). The ring must not grow."""
    import gb_synapse_api as api
    t = api._SSETelemetry()
    for i in range(api._SSETelemetry.LAT_RING + 500):
        t.feed(100.0 + i * 0.01, _content_frame())
    stats = t.latency_stats()
    assert stats["gap_samples"] == api._SSETelemetry.LAT_RING
    assert len(t._gaps) == api._SSETelemetry.LAT_RING


def test_first_gap_excludes_prefill():
    """The interval before the first token carries the whole prompt-eval and
    would skew every percentile if counted as an inter-token gap."""
    import gb_synapse_api as api
    t = api._SSETelemetry()
    t.feed(100.0, _content_frame())        # first token, after a long prefill
    for i in range(1, 40):
        t.feed(100.0 + i * 0.010, _content_frame())
    stats = t.latency_stats()
    assert stats["gap_samples"] == 39
    assert stats["max_ms"] < 20.0          # no 100-second prefill gap in here


def test_latency_block_rides_on_the_tok_s_event(monkeypatch):
    """The percentiles must land on the SAME event as tok_s , they describe
    one turn, and splitting them would let a consumer read one without the
    other."""
    import gb_synapse
    seen = {}
    monkeypatch.setattr(gb_synapse, "_df_emit_tok_s",
                        lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(gb_synapse, "_load_tok_s_samples", lambda: {})
    monkeypatch.setattr(gb_synapse, "_save_tok_s_samples", lambda s: None)
    monkeypatch.setattr(gb_synapse, "_read_run_state", lambda m: None)
    gb_synapse.record_measured_tok_s(
        "m", 5.0, source="proxy", quant="q", ctx=1, kv_type="f16",
        completion_tokens=100, prompt_tokens=10,
        latency={"p50_ms": 2.0, "p95_ms": 190.0, "max_ms": 200.0,
                 "slow_token_ratio": 0.25, "gap_samples": 99})
    assert seen.get("latency", {}).get("p95_ms") == 190.0


# ── speculative-decode accounting (2026-08-20) ─────────────────────────────
#
# The engine has always reported draft_n / draft_n_accepted on every response
# that drafted anything (llama.cpp server-task.cpp). The proxy dropped both,
# so acceptance was only ever measurable through gb_bench_spec.py's synthetic
# prompts , never on the traffic that actually runs. Acceptance is the state
# the 2026-08-05 depth sweep was missing when it found the tok/s-vs-depth
# curve non-monotonic.

def _timings_frame(**timings) -> bytes:
    import json as _json
    return ("data: " + _json.dumps({"choices": [{"delta": {"content": "x"}}],
                                    "timings": timings}) + "\n").encode()


def test_streaming_captures_the_engines_draft_counters():
    import gb_synapse_api as api
    t = api._SSETelemetry()
    t.feed(100.0, _content_frame())
    t.feed(100.2, _timings_frame(prompt_ms=50.0, draft_n=40, draft_n_accepted=26))
    assert t.spec_stats() == {"draft_n": 40, "draft_n_accepted": 26}


def test_a_stream_that_drafted_nothing_reports_none_not_zero():
    """Depth 0 makes the engine report 100% acceptance, because nothing was
    drafted and so nothing could be rejected. None keeps 'drafting is off'
    distinguishable from 'drafting is working perfectly'."""
    import gb_synapse_api as api
    t = api._SSETelemetry()
    t.feed(100.0, _content_frame())
    t.feed(100.2, _timings_frame(prompt_ms=50.0))
    assert t.spec_stats() is None


def test_spec_block_rides_on_the_same_tok_s_event(monkeypatch):
    import gb_synapse
    seen = {}
    monkeypatch.setattr(gb_synapse, "_df_emit_tok_s",
                        lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(gb_synapse, "_load_tok_s_samples", lambda: {})
    monkeypatch.setattr(gb_synapse, "_save_tok_s_samples", lambda s: None)
    monkeypatch.setattr(gb_synapse, "_read_run_state", lambda m: None)
    gb_synapse.record_measured_tok_s(
        "m", 5.0, source="proxy", quant="q", ctx=1, kv_type="f16",
        completion_tokens=100, prompt_tokens=10,
        spec={"draft_n": 40, "draft_n_accepted": 26})
    assert seen.get("spec") == {"draft_n": 40, "draft_n_accepted": 26}


def test_non_streaming_reads_the_counters_out_of_its_timings_block(monkeypatch):
    """A stream:false request carries the whole timings block in one piece ,
    the same coverage gap that once left tok/s itself unrecorded here."""
    import json as _json
    import gb_synapse
    import gb_synapse_api as api
    seen = {}
    monkeypatch.setattr(gb_synapse, "record_measured_tok_s",
                        lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(gb_synapse, "record_prompt_cache_sample",
                        lambda *a, **kw: None)
    body = _json.dumps({
        "usage": {"completion_tokens": 120, "prompt_tokens": 30},
        "timings": {"predicted_per_second": 6.1, "predicted_ms": 19000.0,
                    "prompt_ms": 400.0, "draft_n": 48, "draft_n_accepted": 31},
    }).encode()
    api._record_non_stream_telemetry("m", body)
    assert seen.get("spec") == {"draft_n": 48, "draft_n_accepted": 31}


# ── adaptive draft depth (2026-08-20) ──────────────────────────────────────

def test_depth_is_not_stamped_unless_the_feature_is_on(monkeypatch):
    """Default off: the pinned (4, 0.3) constant is a real measured result on
    this box, and a controller that cannot beat it should not be running."""
    import gb_synapse_api as api
    monkeypatch.setattr(api, "_ADAPTIVE_DRAFT", False)
    c = api._SpecDepth()
    c.depth = 2
    body = {"messages": []}
    assert c.stamp(body) is False
    assert "speculative.n_max" not in body


def test_a_client_that_set_the_field_itself_wins(monkeypatch):
    import gb_synapse_api as api
    monkeypatch.setattr(api, "_ADAPTIVE_DRAFT", True)
    c = api._SpecDepth()
    c.depth = 2
    body = {"messages": [], "speculative.n_max": 6}
    assert c.stamp(body) is False
    assert body["speculative.n_max"] == 6


def test_depth_is_stamped_once_chosen(monkeypatch):
    import gb_synapse_api as api
    monkeypatch.setattr(api, "_ADAPTIVE_DRAFT", True)
    c = api._SpecDepth()
    c.depth = 3
    body = {"messages": []}
    assert c.stamp(body) is True
    assert body["speculative.n_max"] == 3


def test_a_single_bad_turn_does_not_move_the_depth(monkeypatch):
    """Depth is non-monotonic in tok/s, so reacting to one sample is the same
    n=3 mistake the tuner exists to prevent."""
    import gb_synapse_api as api
    monkeypatch.setattr(api, "_ADAPTIVE_DRAFT", True)
    monkeypatch.setattr(api, "_served_draft_depth", lambda: 4)
    c = api._SpecDepth()
    c.observe({"draft_n": 20, "draft_n_accepted": 1})
    assert c.depth is None


def test_sustained_low_acceptance_lowers_the_depth(monkeypatch):
    import gb_synapse_api as api
    monkeypatch.setattr(api, "_ADAPTIVE_DRAFT", True)
    monkeypatch.setattr(api, "_served_draft_depth", lambda: 4)
    monkeypatch.setattr(api, "_emit_spec_depth", lambda *a, **kw: None)
    c = api._SpecDepth()
    for _ in range(api._SPEC_WINDOW):
        c.observe({"draft_n": 20, "draft_n_accepted": 2})
    assert c.depth == 3


def test_stale_shim_stats_are_not_read_as_current(monkeypatch, tmp_path):
    """The stats file outlives the process that wrote it; a stale one reports
    the previous serve's overflow as if it were now."""
    import os
    import time as _time
    import gb_synapse_api as api
    api._t2_cache.update(ts=0.0, mb=None)
    f = tmp_path / "shim_stats"
    f.write_text("t2_overflow_total_mb=6687\n")
    old = _time.time() - 3600
    os.utime(f, (old, old))
    monkeypatch.setattr(api, "_T2_STALE_S", 60.0)
    monkeypatch.setattr("builtins.open", open)
    real_getmtime = os.path.getmtime
    monkeypatch.setattr(os.path, "getmtime",
                        lambda p: old if p == "/run/greenboost/shim_stats"
                        else real_getmtime(p))
    assert api._t2_overflow_mb() is None
