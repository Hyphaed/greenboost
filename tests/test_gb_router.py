"""Route each turn to the cheapest LOCAL model that can serve it.

Measured spread this routes across, same box, same day: MoE + expert offload at
45.72 tok/s against dense Qwen3.8-27B at 2.7-3.5. ~15x, and it is an
architecture difference, not a quality ranking — the dense model IS stronger
(SWE-bench Pro 61.7 vs 49.5), and no Qwen3.8 MoE exists to have both.

Bias is deliberately toward FAST: a wrong escalation costs ~15x throughput,
while a FAST turn that needed DEEP is visible immediately and can be re-asked.
"""
from __future__ import annotations

import pytest

import gb_router as r


def _u(text: str) -> list:
    return [{"role": "user", "content": text}]


POOL = r.Pool(fast="moe-fast", deep="dense-deep",
              endpoints={"moe-fast": "http://127.0.0.1:11369/v1",
                         "dense-deep": "http://127.0.0.1:11369/v1"})


# ── classification ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "run the tests", "list the files in src/", "cat pyproject.toml",
    "grep for TODO", "commit this with a short message", "show git status",
])
def test_mechanical_turns_go_fast(text):
    assert r.classify(_u(text)) == r.FAST


@pytest.mark.parametrize("text", [
    "why is decode slow on this box?",
    "design a caching layer for the prefix store",
    "compare the trade-offs between q8_0 and f16 KV",
    "what is the root cause of the manifest wipe?",
    "should we refactor the backend selection?",
    "explain how the shim spills to T2",
])
def test_reasoning_turns_go_deep(text):
    assert r.classify(_u(text)) == r.DEEP


def test_tool_output_continuation_stays_fast():
    """The bulk of an agentic session: the model is reading a file or command
    result, not reasoning from scratch. This is why routing pays at all."""
    assert r.classify(_u("continue"), tool_result_chars=50_000) == r.FAST


def test_reasoning_request_beats_tool_output_volume():
    """A big tool result does not make a judgement question mechanical."""
    assert r.classify(_u("why did that fail?"), tool_result_chars=50_000) == r.DEEP


def test_long_prompt_without_a_reasoning_cue_escalates():
    assert r.classify(_u("x" * 1500)) == r.DEEP


def test_long_mechanical_request_is_not_escalated_by_length_alone():
    """Length is a poor proxy for difficulty and is consulted last."""
    assert r.classify(_u("run " + "a" * 100)) == r.FAST


def test_empty_conversation_defaults_to_fast():
    assert r.classify([]) == r.FAST


def test_structured_content_blocks_are_read():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "why is this slow?"}]}]
    assert r.classify(msgs) == r.DEEP


# ── local-first enforcement ──────────────────────────────────────────────────

def test_remote_endpoint_in_pool_is_refused():
    """Refuse loudly rather than silently serving from a cloud endpoint."""
    bad = r.Pool(fast="m", deep="m",
                 endpoints={"m": "https://integrate.api.nvidia.com/v1"})
    assert r.check_pool_is_local(bad)
    with pytest.raises(ValueError, match="Local-First"):
        r.route(_u("run the tests"), pool=bad)


def test_lan_feeder_is_accepted_as_local():
    """Flagging the owner's own feeder would make the rule unusable."""
    ok = r.Pool(fast="m", deep="m", endpoints={"m": "http://192.168.0.12:11369/v1"})
    assert r.check_pool_is_local(ok) == []


def test_unverifiable_endpoint_fails_closed(monkeypatch):
    """A wrong 'local' silently switches the rule off, so ambiguity is remote."""
    monkeypatch.setattr(r, "_is_local_url", lambda u: False)
    assert r.check_pool_is_local(
        r.Pool(fast="m", endpoints={"m": "not-a-url"}))


# ── routing ──────────────────────────────────────────────────────────────────

def test_route_picks_the_right_model_and_explains_itself():
    fast = r.route(_u("run the tests"), pool=POOL)
    assert fast["model"] == "moe-fast" and fast["turn_class"] == r.FAST
    assert fast["reason"]

    deep = r.route(_u("why is decode slow?"), pool=POOL)
    assert deep["model"] == "dense-deep" and deep["turn_class"] == r.DEEP


def test_single_model_pool_routes_everything_to_it():
    """A pool with one model is valid, not an error."""
    only = r.Pool(fast="solo", endpoints={"solo": "http://127.0.0.1:11369/v1"})
    assert r.route(_u("why is this slow?"), pool=only)["model"] == "solo"
    assert r.route(_u("run tests"), pool=only)["model"] == "solo"
