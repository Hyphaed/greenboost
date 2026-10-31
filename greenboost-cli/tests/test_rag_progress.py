"""Long RAG commands say what they're doing and how long is left."""
from __future__ import annotations

import numpy as np

from greenboost_cli.rag import engine as E
from greenboost_cli.slash_commands import rag_cmds as C


class _Vec:
    def __init__(self, a): self.a = np.asarray(a, dtype="float32")
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


class _Model:
    prompts = {}
    def encode(self, texts, **kw):
        return _Vec([[float(len(t)), 1.0] for t in texts])


def test_progress_reports_every_slice_and_vectors_land_in_the_right_rows(monkeypatch):
    monkeypatch.setattr(E, "_PROGRESS_SLICE", 4)
    texts = ["x" * n for n in (30, 5, 12, 50, 7, 20, 3, 41, 9, 16)]          # unsorted on purpose
    seen = []
    out = E._embed_with_progress(_Model(), texts, {}, "cuda", lambda d, t, eta, dev: seen.append((d, t, eta, dev)))
    assert [int(v) for v in out[:, 0]] == [len(t) for t in texts]             # each row kept its own text
    assert seen[0] == (0, 10, None, "cuda")                                    # first: how much work, on what
    assert [s[0] for s in seen] == [0, 4, 8, 10] and seen[-1][2] == 0          # ends at the total with 0 left
    assert all(s[2] is not None for s in seen[1:])


def test_eta_is_weighted_by_characters_not_by_count(monkeypatch):
    """Length-sorted slices: the long chunks come last. A count-based countdown would claim
    most of the work was done when most of the characters are still to come."""
    monkeypatch.setattr(E, "_PROGRESS_SLICE", 2)
    texts = ["a" * 10] * 6 + ["b" * 1000] * 2
    etas = []
    E._embed_with_progress(_Model(), texts, {}, "cpu", lambda d, t, eta, dev: etas.append((d, eta)))
    d6 = next(e for d, e in etas if d == 6)                                     # 6 of 8 done by count
    assert d6 > 0                                                               # but the heavy part is still ahead


def test_embed_only_slices_when_asked_and_when_big_enough(monkeypatch):
    calls = []
    monkeypatch.setattr(E, "_pick_device", lambda n, b: "cpu")
    monkeypatch.setattr(E, "_get_model", lambda device=None, name=None: _Model())
    E._embed(["a"] * 10, progress=lambda *a: calls.append(a))                   # below the slice size: no callbacks
    assert calls == []
    monkeypatch.setattr(E, "_PROGRESS_SLICE", 3)
    E._embed(["a"] * 10, progress=lambda *a: calls.append(a))
    assert calls and calls[0][:2] == (0, 10)


def test_printer_announces_the_work_then_throttles_and_finishes(capsys, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    cb = C._progress_printer("embedding")
    cb(0, 10136, None, "cuda")
    cb(256, 10136, 200.0, "cuda")                    # 0 s later: throttled, no line
    clock[0] += 9
    cb(1024, 10136, 200.0, "cuda")
    clock[0] += 1
    cb(10136, 10136, 0.0, "cuda")
    lines = [l.strip() for l in capsys.readouterr().out.splitlines()]
    assert lines[0] == "embedding: 10,136 chunks on the GPU…"
    assert len(lines) == 3 and "1,024/10,136" in lines[1] and "about 3m 20s left" in lines[1]
    assert lines[2].endswith("done")


def test_printer_says_when_it_is_the_slow_cpu(capsys):
    C._progress_printer("embedding")(0, 300, None, "cpu")
    assert "on the CPU (slower)" in capsys.readouterr().out


def test_eta_formatting():
    assert C._fmt_eta(None) == "estimating…" and C._fmt_eta(45) == "45s"
    assert C._fmt_eta(200) == "3m 20s" and C._fmt_eta(5400) == "1h 30m"
