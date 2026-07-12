# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
Test that gb_attn.patch_sdpa() records a dataflux "turboquant_activate" event
(gb_dataflux.py) , works standalone on a single host, no cluster/feeder
required. patch_sdpa() only swaps F.scaled_dot_product_attention (a closure
build, no actual tensor math), so this needs no real CUDA device.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_attn


def test_patch_sdpa_emits_turboquant_activate_event():
    gb_attn.unpatch_sdpa()  # ensure clean state regardless of test order
    try:
        with patch("gb_dataflux.emit") as mock_emit:
            gb_attn.patch_sdpa(k_bits=4, v_bits=3, device="cpu")
        mock_emit.assert_called_once()
        ev = mock_emit.call_args[0][0]
        assert ev["kind"] == "turboquant_activate"
        assert ev["k_bits"] == 4
        assert ev["v_bits"] == 3
        assert ev["node"] == "host"
    finally:
        gb_attn.unpatch_sdpa()


def test_patch_sdpa_records_mode():
    gb_attn.unpatch_sdpa()
    try:
        with patch("gb_dataflux.emit") as mock_emit:
            gb_attn.patch_sdpa(k_bits=4, v_bits=3, device="cpu", mode="turboquant")
        ev = mock_emit.call_args[0][0]
        assert ev["mode"] == "turboquant"
    finally:
        gb_attn.unpatch_sdpa()
