#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Source-level guard over DI-14 Phase 2: the shim's memory sizing must stay
DERIVED, never re-hardcoded to host-shaped GB literals.

The five values below were flat constants sized for the owner's own box
(8 GB safety reserve, 8 GB pre-registration cap, 2048 MB workstation-reserve
ceiling, 512 MB headroom ceiling, 3072 MB KV fallback). On a different node
every one of them is wrong in a way that does not announce itself: an 8 GB
feeder handed a "3072 MB KV reserve" has lost 37% of its card before a single
weight lands, and nothing errors , it just quietly fits less and spills more.

They were all replaced with percentage-of-capacity forms during the 2026-07-13
dynamic-intelligence pass. Nothing pinned them afterwards, which is the gap
this file closes: the derivations are several screens apart in a 7,000-line
file, and a plausible-looking literal is exactly the kind of edit that gets
made while chasing something else.

The sanctioned form throughout is `max(small_floor, pct * capacity)` , a
percentage that scales with the node, and a small floor so a tiny node still
gets something usable. A floor is fine. A CEILING expressed as an absolute
size is the thing that breaks on hardware bigger than the author's.

No compiler needed: a grep over the C source, same technique as
test_shim_vmm_attribute_constant.py.
"""
import re
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_SHIM_C = _REPO_DIR / "greenboost_cuda_shim.c"


def _src() -> str:
    return _SHIM_C.read_text(errors="replace")


def test_no_kmod_safety_reserve_mirrors_the_kmod_autosize_rule():
    """Without the kmod there is no sysfs value to read, and the fallback used
    to be a flat 8 GB. It must mirror greenboost.c's own rule instead:
    8% of THIS node's RAM, clamped to [6, 32] GB."""
    src = _src()
    # Anchor on the log string, which is unique to the implementation. The
    # phrase "no-kmod" alone also appears in a comment ~1200 lines earlier,
    # and anchoring there tests nothing.
    assert "no-kmod 8%-RAM autosize" in src, "the autosize branch is gone"
    end = src.index('"no-kmod 8%-RAM autosize"')
    block = src[end - 900:end]
    assert "gb_host_ram_bytes()" in block, "reserve no longer reads real RAM"
    assert re.search(r"\*\s*8ULL\s*/\s*100ULL", block), "8% derivation missing"
    assert "< 6ULL" in block and "> 32ULL" in block, "[6,32] GB clamp missing"


def test_pool_prereg_cap_is_a_percentage_with_a_floor():
    """Was a flat 8 GB cap; must be a percentage of MemTotal with a small
    floor, plus an env override so a node can be told otherwise."""
    src = _src()
    assert "GB_POOL_PREREG_PCT_DEFAULT" in src
    assert "GB_POOL_PREREG_FLOOR_BYTES" in src
    assert "GREENBOOST_POOL_PREREG_PCT" in src, "env override missing"
    m = re.search(r"#define\s+GB_POOL_PREREG_PCT_DEFAULT\s+(\d+)ULL", src)
    assert m and 1 <= int(m.group(1)) <= 90, "pre-reg percentage is not a percentage"


def test_workstation_reserve_ceiling_scales_with_vram():
    """The ceiling must be %-of-VRAM when VRAM is known. An absolute value is
    allowed ONLY as the fallback for when it is not."""
    src = _src()
    assert "GB_WS_RESERVE_MAX_PCT" in src, "ceiling is no longer a percentage"
    assert "GB_WS_RESERVE_MIN_BYTES" in src, "small floor removed"
    i = src.index("size_t max_reserve = real_total")
    block = src[i:i + 400]
    # Assert the COMPUTATION, not just that a symbol is spelled somewhere.
    # A guard that only checks for a name passes happily after someone
    # redefines that name to an absolute size , which is precisely the
    # regression being guarded against.
    assert re.search(r"real_total\s*\*\s*GB_WS_RESERVE_MAX_PCT\s*/\s*100ULL", block), \
        "the ceiling no longer scales with real VRAM"
    assert "GB_WS_RESERVE_MAX_FALLBACK_BYTES" in block, "fallback not on the unknown path"
    # The absolute value must be reachable only when real_total is falsy.
    assert block.index("GB_WS_RESERVE_MAX_PCT") < block.index("GB_WS_RESERVE_MAX_FALLBACK_BYTES")
    m = re.search(r"#define\s+GB_WS_RESERVE_MAX_PCT\s+(\d+)ULL", src)
    assert m and 1 <= int(m.group(1)) <= 90, "the ceiling percentage is not a percentage"


def test_vram_headroom_has_no_absolute_ceiling():
    """The old 512 MB ceiling over-capped large cards to a flat value. Only a
    floor may remain."""
    src = _src()
    i = src.index("size_t pct2 = gb_physical_vram_bytes")
    block = src[i:i + 400]
    assert "floor_bytes" in block, "headroom floor missing"
    assert not re.search(r"if\s*\(\s*pct2\s*>\s*\d", block), \
        "an absolute ceiling on headroom is back"


def test_kv_reserve_fallback_scales_instead_of_using_3072mb():
    """3072 MB is 37% of an 8 GB feeder. When VRAM is not yet known the shim
    must ask cuMemGetInfo directly rather than assume the owner's card."""
    src = _src()
    # Anchor on the auto-derivation itself; `g_kv_reserve_from_env` is tested
    # in several places and the first hit is not this one.
    i = src.index("size_t auto_kv_mb;")
    block = src[i:i + 1400]
    # Assert on USE, not mention: the block deliberately still names 3072 in
    # the comment explaining why it went away, and that comment is worth
    # keeping , it carries the reason (37% of an 8 GB feeder).
    assert not re.search(r"auto_kv_mb\s*=\s*3072", block), \
        "the flat 3072 MB KV fallback is back"
    assert "real_cuMemGetInfo" in block, "no longer queries the real card"
    assert re.search(r"\*\s*25\s*/\s*100", block), "25%-of-VRAM derivation missing"


def test_the_owner_rule_is_still_recorded_at_each_site():
    """These derivations look arbitrary without the rule that produced them.
    Losing the comment is how the next reader 'simplifies' one back."""
    assert _src().count("2026-07-13") >= 3


# ── netd wire-parser truncation (found 2026-08-19 by tests/c/fuzz_netd_protocol.c)

_NETD_C = _REPO_DIR / "greenboost_netd.c"


def test_h2d_copy_size_is_computed_at_one_width():
    """min(data_len, req_size) must not compare at 32 bits and copy at 64.

    The original form compared `data_len < (uint32_t)req_size` and then copied
    `(size_t)req_size`. A req_size with small low bits and set high bits lost
    the comparison and won the copy, so copy_size exceeded the bytes actually
    received , an over-read of the receive buffer, caught by ASan as a 30 GB
    memcpy out of an 8 KB payload.
    """
    src = _NETD_C.read_text(errors="replace")
    assert "(data_len < (uint32_t)req_size)" not in src, \
        "the 32-bit truncation in the H2D copy-size clamp is back"
    i = src.index("size_t   copy_size")
    block = src[i:i + 200]
    assert "(uint64_t)data_len" in block, \
        "the H2D clamp no longer promotes both sides to 64 bits"
