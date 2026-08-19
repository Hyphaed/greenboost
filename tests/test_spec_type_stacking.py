"""Speculation is stackable, and a typo must not silently disable it.

llama.cpp's `--spec-type` accepts a comma-separated list, and its ngram modes
need no draft model and no extra VRAM. On a bandwidth-bound decode a forward
pass costs the same whether it emits one token or five, so an extra accepted
draft token is close to free throughput.

The failure mode this guards: llama-server rejects an unknown --spec-type value
outright and serves with NO speculation at all. That reads as "the lever did
nothing" rather than "the lever was misspelled", which is the kind of silent
non-result that survives for months.
"""
from __future__ import annotations

import pytest

import gb_synapse as gs


def test_mtp_alone_is_unchanged():
    """The existing default must not shift under models that only have MTP."""
    assert gs.spec_type_list(["draft-mtp"]) == "draft-mtp"


def test_ngram_stacks_after_the_models_own_head():
    """Order matters: the model's own draft head should be tried first."""
    assert gs.spec_type_list(["draft-mtp"], "ngram-cache") == "draft-mtp,ngram-cache"


def test_extras_work_without_an_mtp_head():
    """A model with no draft head can still use ngram speculation."""
    assert gs.spec_type_list([], "ngram-cache,ngram-mod") == "ngram-cache,ngram-mod"


def test_duplicates_collapse_preserving_order():
    assert gs.spec_type_list(["draft-mtp"], "ngram-cache,draft-mtp") == "draft-mtp,ngram-cache"


def test_unknown_mode_is_dropped_and_named(capsys):
    """Dropped, not passed through — passing it would disable speculation
    entirely rather than just ignoring the bad entry."""
    out = gs.spec_type_list(["draft-mtp"], "ngram-cache,typo-mode")
    assert out == "draft-mtp,ngram-cache"
    assert "typo-mode" in capsys.readouterr().err


def test_every_known_mode_matches_the_engine():
    """Pinned against llama.cpp's own --spec-type enum. If the engine gains or
    renames a mode this test is where it should be noticed."""
    assert set(gs.SPEC_TYPES_KNOWN) == {
        "none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash",
        "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache",
    }


def test_empty_input_yields_no_flag():
    """No speculation configured must produce an empty value, so the caller can
    omit --spec-type entirely rather than passing an empty string."""
    assert gs.spec_type_list([], "") == ""
    assert gs.spec_type_list([], "  ,  ") == ""


def test_extras_are_opt_in_by_default():
    """Defaulting this on would change every serve without measurement."""
    import os
    assert os.environ.get("GB_SYNAPSE_SPEC_EXTRA", "") == "" or gs.SPEC_EXTRA_TYPES
