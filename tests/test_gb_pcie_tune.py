"""PCIe link inspection must not overclaim, and must refuse unsafe writes.

GreenBoost's T2 tier is the GPU reading host memory over PCIe, so when a model
does not fit VRAM that link IS the decode bottleneck — 390 ms of bus against
17 ms of compute, measured 2026-08-18. There is a real gap to explain (the link
trains Gen4 x16, ~26 GB/s usable, while the shim measures ~11.5), and
MaxReadReq is the classic cause.

These tests pin the honesty properties, because a tuning tool that guesses is
worse than none: it must never invent a MaxReadReq it could not read, never
claim Gen5 is reachable when the root port caps the link, and never poke PCIe
config space without root.
"""
from __future__ import annotations

import pytest

import gb_pcie_tune as t


def test_mrrs_encoding_matches_the_pcie_spec():
    """DevCtl MaxReadReq is log2(bytes) - 7."""
    assert t.MRRS_BYTES == {0: 128, 1: 256, 2: 512, 3: 1024, 4: 2048, 5: 4096}


def test_unreadable_mrrs_is_reported_as_unreadable(monkeypatch):
    """Never fabricate a value the tool could not actually read."""
    monkeypatch.setattr(t, "gpu_bdf", lambda: "0000:01:00.0")
    monkeypatch.setattr(t, "_lspci_caps", lambda b: {})
    monkeypatch.setattr(t, "_sysfs", lambda b, n: "16.0 GT/s PCIe" if "speed" in n else "16")
    notes = " ".join(t.report()["notes"])
    assert "unreadable" in notes and "root" in notes
    assert "MaxReadReq is" not in notes, "claimed a value it never read"


def test_link_below_device_max_is_attributed_to_the_board(monkeypatch):
    """The GPU advertises Gen5 while the board trains Gen4 — say which is the
    limit, rather than implying software can fix it."""
    monkeypatch.setattr(t, "gpu_bdf", lambda: "0000:01:00.0")
    monkeypatch.setattr(t, "_lspci_caps", lambda b: {"max_read_req_bytes": 512})
    monkeypatch.setattr(t, "_sysfs", lambda b, n:
                        {"current_link_speed": "16.0 GT/s PCIe",
                         "max_link_speed": "32.0 GT/s PCIe"}.get(n, "16"))
    notes = " ".join(t.report()["notes"])
    assert "root port is the limit" in notes
    assert "Not fixable in software" in notes


def test_already_maxed_mrrs_suggests_nothing(monkeypatch):
    monkeypatch.setattr(t, "gpu_bdf", lambda: "0000:01:00.0")
    monkeypatch.setattr(t, "_lspci_caps", lambda b: {"max_read_req_bytes": 4096})
    monkeypatch.setattr(t, "_sysfs", lambda b, n: "16.0 GT/s PCIe")
    notes = " ".join(t.report()["notes"])
    assert "nothing to raise" in notes


def test_low_mrrs_is_flagged_with_the_mechanism(monkeypatch):
    monkeypatch.setattr(t, "gpu_bdf", lambda: "0000:01:00.0")
    monkeypatch.setattr(t, "_lspci_caps", lambda b: {"max_read_req_bytes": 512})
    monkeypatch.setattr(t, "_sysfs", lambda b, n: "16.0 GT/s PCIe")
    notes = " ".join(t.report()["notes"])
    assert "512 bytes" in notes and "round trip" in notes


def test_apply_refuses_without_root(monkeypatch):
    monkeypatch.setattr(t.os if hasattr(t, "os") else __import__("os"),
                        "geteuid", lambda: 1000)
    r = t.apply_mrrs("0000:01:00.0")
    assert r["ok"] is False and "root" in r["error"]


def test_apply_rejects_a_target_the_spec_cannot_encode(monkeypatch):
    import os
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    r = t.apply_mrrs("0000:01:00.0", target_bytes=3000)
    assert r["ok"] is False and "must be one of" in r["error"]


def test_no_gpu_is_an_error_not_a_guess(monkeypatch):
    monkeypatch.setattr(t, "gpu_bdf", lambda: "")
    assert "error" in t.report()
