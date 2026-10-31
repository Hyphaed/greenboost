"""A fallback to the weaker embedder is announced, once, with the reason."""
from __future__ import annotations

from greenboost_cli.rag import engine as E


def test_fallback_is_announced_once_with_reason_and_fix(monkeypatch):
    seen = []
    import greenboost_cli.terminal.theme as T
    monkeypatch.setattr(T, "emit_warn", lambda m: seen.append(m))
    monkeypatch.setattr(E, "_fallback_warned", False)
    err = ImportError("cannot import name 'find_pruneable_heads_and_indices'")
    E._warn_embed_fallback(err)
    E._warn_embed_fallback(err)
    assert len(seen) == 1
    assert "find_pruneable_heads_and_indices" in seen[0]
    assert E.EMBED_FALLBACK in seen[0] and "GB_EMBED_MODEL" in seen[0]
