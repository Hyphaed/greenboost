"""gb_compile: policy and wiring (CPU only, torch.compile stubbed)."""
import torch
import torch.nn as nn

import gb_compile as gc


def test_policy(monkeypatch):
    monkeypatch.delenv("GB_COMPILE", raising=False)
    assert not gc.should_compile(None) and not gc.should_compile(1) and not gc.should_compile(7) and gc.should_compile(8) and gc.should_compile(40)
    monkeypatch.setenv("GB_COMPILE", "1")
    assert gc.should_compile(1)
    monkeypatch.setenv("GB_COMPILE", "0")
    assert not gc.should_compile(100)
    monkeypatch.setenv("GB_COMPILE", "auto"); monkeypatch.setenv("GB_COMPILE_MIN_IMAGES", "3")
    assert gc.should_compile(3) and not gc.should_compile(2)


class _T(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])


def test_apply_wraps_blocks_or_stays_eager(monkeypatch):
    seen = []
    monkeypatch.setattr(torch, "compile", lambda m, **kw: (seen.append(kw), m)[1])
    monkeypatch.setenv("GB_COMPILE", "1")
    assert gc.apply(_T()) is True and len(seen) == 3 and all(k["fullgraph"] is False for k in seen)
    monkeypatch.setenv("GB_COMPILE", "auto")
    assert gc.apply(_T(), n_images=2) is False and len(seen) == 3
    monkeypatch.setenv("GB_COMPILE", "1")
    assert gc.apply(nn.Linear(2, 2)) is False                      # no repeated blocks: stays eager


def test_failure_leaves_model_eager(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no inductor")
    monkeypatch.setattr(torch, "compile", boom)
    monkeypatch.setenv("GB_COMPILE", "1")
    assert gc.apply(_T()) is False
