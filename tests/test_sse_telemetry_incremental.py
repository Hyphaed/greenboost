"""_SSETelemetry must match the batch parser exactly, without hoarding bytes.

The streaming path used to append every raw chunk of a response to a list and
parse it afterwards. That was the only unbounded per-request buffer in the hot
path, and this proxy is long-lived: on 2026-08-18 `gb_synapse_api.py` was
measured holding 37.5 GB of anonymous memory (7.6 GB swapped) after twenty
minutes of ordinary agentic traffic on a 61 GB box, and the OOM killer took the
user's terminal twice that day.

Equivalence with the old parser is the whole safety argument for the change, so
it is tested directly rather than assumed.
"""
import json

import pytest

import gb_synapse_api as A


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _content(text):
    return {"choices": [{"delta": {"content": text}}]}


def _feed_all(chunks):
    t = A._SSETelemetry()
    for i, raw in enumerate(chunks):
        t.feed(float(i), raw)
    return t.result()


def _batch(chunks):
    return A._parse_sse_telemetry_timed([(float(i), c) for i, c in enumerate(chunks)])


STREAMS = {
    "plain prose": [
        _sse({"choices": [{"delta": {"role": "assistant"}}]}),
        _sse(_content("Hello")),
        _sse(_content(" world")),
        _sse({"usage": {"completion_tokens": 2, "prompt_tokens": 41},
              "timings": {"prompt_ms": 123.5}}),
        b"data: [DONE]\n\n",
    ],
    "tool-call only": [
        _sse({"choices": [{"delta": {"role": "assistant"}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]}),
        _sse({"usage": {"completion_tokens": 7, "prompt_tokens": 900}}),
    ],
    "reasoning then content": [
        _sse({"choices": [{"delta": {"reasoning_content": "hmm"}}]}),
        _sse(_content("answer")),
    ],
    "no content at all": [
        _sse({"choices": [{"delta": {"role": "assistant"}}]}),
        b"data: [DONE]\n\n",
    ],
    "malformed json mixed in": [
        b"data: {not json at all\n\n",
        _sse(_content("ok")),
        b"garbage line without a data prefix\n",
    ],
}


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_incremental_matches_the_batch_parser(name):
    chunks = STREAMS[name]
    assert _feed_all(chunks) == _batch(chunks), name


def test_a_line_split_across_chunks_still_parses():
    """iter_any() splits on TCP boundaries, not SSE ones."""
    whole = _sse(_content("split me"))
    half = len(whole) // 2
    t = A._SSETelemetry()
    t.feed(1.0, whole[:half])
    t.feed(2.0, whole[half:])
    _, _, _, t_first, t_last, _ = t.result()
    assert t_first == 2.0 and t_last == 2.0, "content latched on the wrong chunk"


def test_first_content_latches_after_a_role_only_opening_delta():
    """The 2026-08-01 incident: role-only frames are not decode starting."""
    t = A._SSETelemetry()
    t.feed(1.0, _sse({"choices": [{"delta": {"role": "assistant"}}]}))
    t.feed(9.0, _sse(_content("first real token")))
    t.feed(11.0, _sse(_content(" more")))
    _, _, _, t_first, t_last, _ = t.result()
    assert t_first == 9.0
    assert t_last == 11.0


def test_the_carry_over_buffer_is_bounded():
    """A stream that never sends a newline must not reintroduce the leak."""
    t = A._SSETelemetry()
    blob = b"x" * (1 << 16)
    for i in range(64):                     # 4 MiB with no delimiter
        t.feed(float(i), blob)
    assert len(t._partial) <= A._SSETelemetry.MAX_PARTIAL


def test_feed_never_raises_on_junk():
    t = A._SSETelemetry()
    for raw in (b"\xff\xfe\x00binary", b"", b"data:\n", b"data: null\n",
                b'data: {"choices": []}\n'):
        t.feed(1.0, raw)                    # must not raise
    assert t.result()[3] is None


def test_memory_does_not_scale_with_response_length():
    """The property the fix exists for."""
    import sys

    t = A._SSETelemetry()
    for i in range(20000):
        t.feed(float(i), _sse(_content("token")))
    held = sys.getsizeof(t._partial)
    assert held < 4096, f"still holding {held} bytes of stream"
    assert t.result()[3] == 0.0             # and it still measured correctly
