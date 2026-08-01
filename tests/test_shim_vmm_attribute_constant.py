#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Source-level guard pinning the exact regression that cost the 2026-08-01
session: greenboost_cuda_shim.c and greenboost_vmm_override.c both
hardcoded CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED as 193
instead of its real value from cuda.h, 102. 193 is outside
CU_DEVICE_ATTRIBUTE_MAX, so the real driver call always failed, the shim's
"report VMM=0 on Blackwell so ggml picks the T2-spillable legacy pool"
override never fired, and ggml aborted the process instead of falling back
(see CLAUDE.md's "T2 Spill Through the Shim, Never CPU Offload" rule and
gb_shim_probe.py's module docstring).

No compiler needed — a plain grep over the C sources. Cheap, and it is the
only thing standing between a future edit and silently reintroducing a
number that looks plausible (it's the correct BIT WIDTH coincidentally
associated with other CUDA constants nearby) but is wrong.
"""
import re
import sys
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_DIR))

_SHIM_C = _REPO_DIR / "greenboost_cuda_shim.c"
_VMM_OVERRIDE_C = _REPO_DIR / "greenboost_vmm_override.c"


def test_shim_defines_vmm_attribute_as_102():
    text = _SHIM_C.read_text()
    m = re.search(
        r"#define\s+CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED\s+(\d+)",
        text)
    assert m, "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED define not found"
    assert m.group(1) == "102", (
        f"greenboost_cuda_shim.c defines the VMM-supported attribute as "
        f"{m.group(1)}, not 102 (cuda.h's real value) — this is the exact "
        f"2026-08-01 regression: 193 is outside CU_DEVICE_ATTRIBUTE_MAX, so "
        f"the Blackwell VMM=0 override silently never fires.")


def test_vmm_override_defines_attribute_as_102():
    text = _VMM_OVERRIDE_C.read_text()
    m = re.search(r"#define\s+GB_ATTR_VMM_SUPPORTED\s+(\d+)", text)
    assert m, "GB_ATTR_VMM_SUPPORTED define not found"
    assert m.group(1) == "102"


def test_no_stray_193_vmm_attribute_code_reference():
    """Belt-and-suspenders: neither file should CODE-reference 193 as the
    VMM attribute anywhere (the dead runtime-API companion branch that used
    to match `attr == 193` was removed rather than fixed, since the CUDA
    runtime headers have no VMM-supported member at all). Only scans
    non-comment lines — the incident is deliberately documented BY NUMBER
    in nearby comments, which must not trip this guard."""
    for path in (_SHIM_C, _VMM_OVERRIDE_C):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("//", 1)[0].strip()
            if code.startswith("*") or code.startswith("/*"):
                continue  # block-comment body/opener
            if "193" in code and ("vmm" in code.lower() or "attr" in code.lower()):
                raise AssertionError(
                    f"{path.name}:{lineno} still CODE-references 193 in a "
                    f"VMM/attribute context: {line.strip()!r}")
