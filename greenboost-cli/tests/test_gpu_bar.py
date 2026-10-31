"""The bottom bar names the card, and only real other jobs: not this CLI, not "python3"."""
from __future__ import annotations

import os
import subprocess
import sys

from greenboost_cli.terminal import cc_features as F
from greenboost_cli.terminal.dock import DockState, status1_frags


def _text(frags) -> str:
    return "".join(t for _, t in frags)


def test_title_is_the_card_and_its_size_from_the_driver():
    snap = {"name": "NVIDIA GeForce RTX 5070", "total": 12227}
    assert F.gpu_title(snap) == "NVIDIA RTX 5070 12G"
    assert F.gpu_title({"name": "NVIDIA RTX A4000", "total": 16376}) == "NVIDIA RTX A4000 16G"   # not tied to this box
    assert F.gpu_title(None) == "" and F.gpu_title({"total": 1}) == ""


def test_holders_exclude_this_cli_interpreters_and_the_model_server():
    apps = [(1, "python3", 3900),                              # this CLI's own embedder
            (2, "FLUX generation", 8100),
            (3, "gb-synapse model (llama-server)", 9000),
            (4, "ComfyUI", 500),                               # too small to matter
            (5, "python3.14", 2000),                           # a bare interpreter says nothing
            (6, "pid 99", 2000),
            (7, "ComfyUI", 2500)]
    shown = F.gpu_holders_to_show(apps, is_own=lambda pid: pid == 1)
    assert [(l, m) for _, l, m in shown] == [("ComfyUI", 2500), ("FLUX generation", 8100)][::-1]


def test_own_process_detection_covers_children_but_not_strangers():
    assert F._is_own_process(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert F._is_own_process(child.pid)
    finally:
        child.kill()
    assert not F._is_own_process(1)


def _bar(**kw):
    return _text(status1_frags(DockState(**kw), 200))


def test_the_bar_shows_the_card_then_another_job_when_there_is_one():
    assert "GPU: NVIDIA RTX 5070 12G" in _bar(gpu_name="NVIDIA RTX 5070 12G")
    both = _bar(gpu_name="NVIDIA RTX 5070 12G", gpu_owner="FLUX generation 8.1G")
    assert "GPU: NVIDIA RTX 5070 12G · FLUX generation 8.1G" in both
    assert "python3" not in _bar(gpu_name="NVIDIA RTX 5070 12G")
    assert "GPU:" not in _bar()                                  # no driver, no job: nothing to say


def test_without_a_driver_name_the_old_owner_text_still_shows():
    assert "GPU: FLUX generation 8.1G" in _bar(gpu_owner="FLUX generation 8.1G")


def test_model_is_named_right_of_the_gpu():
    both = _bar(gpu_name="NVIDIA RTX 5070 12G", model="qwen38-35b-a3b-apex-miniplus")
    assert "GPU: NVIDIA RTX 5070 12G  ·  qwen38-35b-a3b-apex-miniplus" in both
    assert "qwen" not in _bar(gpu_name="NVIDIA RTX 5070 12G")
