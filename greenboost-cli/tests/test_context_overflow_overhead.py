"""Context overflow caused by FIXED overhead must be explained, not retried.

Compaction shrinks history. It cannot touch the system prompt or the tool
schemas of every connected MCP server, which are re-sent in full on every
request. Live 2026-08-18: ten MCP servers, 238 tools, `n_prompt_tokens: 29507`
against `n_ctx: 16384` for a thirty-word message and an essentially empty
history , the CLI compacted, freed a few hundred tokens, retried, failed
identically, and surfaced a raw 400.
"""
import pytest

from greenboost_cli.inference.adapters import ContextOverflowError, _overflow_numbers

REAL_400 = (
    "Error code: 400 - {'error': {'code': 400, 'message': 'request (29507 tokens) "
    "exceeds the available context size (16384 tokens), try increasing it', "
    "'type': 'exceed_context_size_error', 'n_prompt_tokens': 29507, 'n_ctx': 16384}}"
)


def test_the_servers_own_numbers_are_recovered():
    assert _overflow_numbers(REAL_400) == (29507, 16384)


def test_prose_only_body_still_yields_numbers():
    """Older llama-server builds report it only in the message text."""
    prose = "request (8324 tokens) exceeds the available context size (7680 tokens)"
    assert _overflow_numbers(prose) == (8324, 7680)


def test_an_unrelated_error_yields_no_numbers():
    assert _overflow_numbers("Connection refused") == (0, 0)


def test_the_exception_carries_the_numbers():
    e = ContextOverflowError("boom", prompt_tokens=29507, n_ctx=16384)
    assert e.prompt_tokens == 29507 and e.n_ctx == 16384


def test_the_exception_still_works_without_numbers():
    """Every existing raise site passed only a message."""
    e = ContextOverflowError("plain")
    assert e.prompt_tokens == 0 and e.n_ctx == 0
    assert isinstance(e, RuntimeError)          # callers catch RuntimeError


@pytest.mark.parametrize("prompt,hist_tokens,n_ctx,overhead_dominated", [
    (29507,   200, 16384, True),    # the real incident: history is noise
    (29507, 20000, 16384, False),   # history IS the bulk , compaction can help
    (8324,    100,  7680, True),
    (20000, 19000, 16384, False),
])
def test_overhead_domination_is_decided_by_arithmetic(prompt, hist_tokens, n_ctx,
                                                      overhead_dominated):
    """Dropping the ENTIRE history and still not fitting means compaction is futile."""
    assert ((prompt - hist_tokens) >= n_ctx) is overhead_dominated
