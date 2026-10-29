#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""qwen35moe GGUF handling (2026-10-02, Qwen3.8-35B-A3B Distill first serve).

Two real defects, pinned with synthetic GGUFs built by the vendored writer:
  * `attention.recurrent_layers` is a per-layer BOOL mask for qwen35moe;
    len() of it is n_layers, so n_kv_layers came out 0.
  * the MTP head ships as a second file (`blk.<n_layers>.*` only), which
    _has_mtp() could not see, so the model served without speculation.

CPU-only, no hardware, no network."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse as gs

_VENDORED_GGUF_PY = str(Path(__file__).parent.parent / "greenboost-sources"
                         / "llama.cpp" / "gguf-py")
if _VENDORED_GGUF_PY not in sys.path:
    sys.path.insert(0, _VENDORED_GGUF_PY)
from gguf.gguf_writer import GGUFWriter  # noqa: E402


def _write(path: Path, arch: str, tensors: list[str], kv=None) -> Path:
    w = GGUFWriter(str(path), arch=arch)
    w.add_context_length(4096)
    w.add_block_count(8)
    for fn in (kv or []):
        fn(w)
    for name in tensors:
        w.add_tensor(name, np.zeros((4, 4), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_bool_mask_counts_true_entries(tmp_path):
    mask = [True, True, True, False] * 2          # 6 recurrent, 2 attention
    p = _write(tmp_path / "m.gguf", "qwen35moe", ["blk.0.attn.weight"],
               kv=[lambda w: w.add_array("qwen35moe.attention.recurrent_layers", mask)])
    s = gs.gguf_summary(str(p))
    assert (s["n_recurrent_layers"], s["n_kv_layers"]) == (6, 2)


def test_index_list_keeps_length_semantics(tmp_path):
    p = _write(tmp_path / "m.gguf", "falcon-h1", ["blk.0.attn.weight"],
               kv=[lambda w: w.add_array("falcon-h1.attention.recurrent_layers", [0, 2, 4])])
    s = gs.gguf_summary(str(p))
    assert (s["n_recurrent_layers"], s["n_kv_layers"]) == (3, 5)


def test_mtp_head_found_beside_main_and_gated_by_layer_index(tmp_path):
    main = _write(tmp_path / "main.gguf", "llama",
                  [f"blk.{i}.attn.weight" for i in range(8)])
    # pad the main file so the head is under the size fraction
    with open(main, "ab") as f:
        f.write(b"\0" * 4096)
    head = _write(tmp_path / "mtp-head.gguf", "llama",
                  ["token_embd.weight", "blk.8.attn_q.weight", "blk.8.attn_k.weight"])
    other = _write(tmp_path / "other.gguf", "llama", ["blk.0.attn.weight"])
    entry = SimpleNamespace(path=str(main), n_layers=8)
    assert gs._mtp_head_path(entry) == str(head)
    assert gs._has_mtp(entry) is True
    # negative control: a model whose own blocks reach index 8 has no separate head
    assert gs._mtp_head_path(SimpleNamespace(path=str(main), n_layers=9)) is None
    assert other.exists()                          # never selected: holds blk.0


def test_mtp_head_disabled_by_env(tmp_path, monkeypatch):
    main = _write(tmp_path / "main.gguf", "llama", [f"blk.{i}.attn.weight" for i in range(8)])
    with open(main, "ab") as f:
        f.write(b"\0" * 8192)
    _write(tmp_path / "mtp-head.gguf", "llama", ["blk.8.attn_q.weight"])
    monkeypatch.setenv("GB_SYNAPSE_MTP", "0")
    assert gs._mtp_head_path(SimpleNamespace(path=str(main), n_layers=8)) is None
