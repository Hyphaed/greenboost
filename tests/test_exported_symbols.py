# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
Symbol-export integration test.

Catches:
  - `greenboost_cuda.map` drift (a hook is defined in the .c but not
    exported because the map file wasn't updated - silent shim bypass)
  - Missing `__attribute__((alias))` aliases (PR-R `_ptds`/`_ptsz`
    variants)
  - Accidental removal of a hook the design relies on

Runs `nm -D libgreenboost_cuda.so` and verifies the expected hook set is
present.  Skipped if the .so hasn't been built yet - run `make shim`
first.
"""
import os
import subprocess
from pathlib import Path

import pytest


SHIM_PATH = Path(__file__).parent.parent / "libgreenboost_cuda.so"


# Hooks that MUST be exported.  This list is the authoritative spec.
# Adding a new hook? Add it here AND to greenboost_cuda.map AND to
# gb_get_hook in greenboost_cuda_shim.c.
EXPECTED_HOOKS = {
    # Allocation
    "cudaMalloc",
    "cudaMallocAsync",
    "cudaFree",
    "cudaFreeAsync",             # PR-Z fix - was missing from .map until caught
    "cuMemAlloc_v2",
    "cuMemAllocAsync",
    "cuMemAllocFromPoolAsync",   # PR-N - PyTorch 2.4+ expandable_segments
    "cuMemFree_v2",
    "cuMemFreeAsync",
    # Path A/B/C primitives
    "cuMemCreate",
    "cuMemRelease",
    "cuMemMap",
    "cuMemUnmap",
    "cuMemSetAccess",
    # Prefetch (load-bearing for Path C anchor)
    "cuMemPrefetchAsync",
    "cudaMemPrefetchAsync",
    # Memcpy
    "cudaMemcpy",
    "cudaMemcpyAsync",
    # Launch
    "cuLaunchKernel",
    "cuLaunchKernelEx",          # PR-O - CUDA 12+ graph mode
    "cuLaunchCooperativeKernel", # PR-Z fix - was missing from .map until caught
    "cudaLaunchKernel",
    # Device info - inflated VRAM reporting
    "cuDeviceTotalMem",
    "cuDeviceTotalMem_v2",
    "cuDeviceGetAttribute",
    "cuMemGetInfo",
    "cuMemGetInfo_v2",
    "cudaMemGetInfo",
    "cudaGetDeviceCount",
    "cudaSetDevice",
    "cudaGetDeviceProperties",
    "cudaGetDeviceProperties_v2",
    # Sync
    "cudaStreamSynchronize",
    # Register
    "__cudaRegisterFunction",
    # Driver-entry-point lookup (introduced in CUDA 11.3; both @.so.12 and @@.so.13
    # trampolines must be present - a bug in greenboost_cuda_v12.c's keepalive table
    # or the LTO compile flags silently drops the @libcudart.so.12 export).
    "cudaGetDriverEntryPointByVersion",
    # NOTE: cudaMallocManaged and cuStreamSynchronize were intentionally
    # removed from EXPECTED_HOOKS in PR-Z.  Neither has a function definition
    # in greenboost_cuda_shim.c, so listing them in EXPECTED_HOOKS would
    # forever-fail the test.  Apps calling cudaMallocManaged directly get
    # the unmodified driver behaviour (intentional - they're not asking for
    # GreenBoost overflow routing).  If a future PR adds the hook, add it
    # back here and to greenboost_cuda.map at the same time.
}

# PR-R: per-thread-default-stream aliases.  Each must be exported AND
# resolve to the same memory address as its base symbol (ELF alias
# verified via the address-equality check below).
EXPECTED_PTSZ_ALIASES = {
    "cudaMemcpyAsync_ptsz":          "cudaMemcpyAsync",
    "cudaMemPrefetchAsync_ptsz":     "cudaMemPrefetchAsync",
    "cudaLaunchKernel_ptsz":         "cudaLaunchKernel",  # PR-LL
}
EXPECTED_PTDS_ALIASES = {
    "cuMemAllocAsync_ptds":          "cuMemAllocAsync",
    "cuMemAllocFromPoolAsync_ptds":  "cuMemAllocFromPoolAsync",
    "cuMemFreeAsync_ptds":           "cuMemFreeAsync",
    "cuMemPrefetchAsync_ptds":       "cuMemPrefetchAsync",
    "cuLaunchKernel_ptds":           "cuLaunchKernel",
    "cuLaunchKernelEx_ptds":         "cuLaunchKernelEx",
    "cuLaunchCooperativeKernel_ptds": "cuLaunchCooperativeKernel",
}


@pytest.fixture(scope="module")
def shim_symbols():
    """Run `nm -D` once per module, parse {name -> {addresses}}.

    A symbol can have multiple addresses when it's defined under multiple
    ELF versions (e.g. cudaMemPrefetchAsync@@libcudart.so.12 default +
    cudaMemPrefetchAsync@libcudart.so.12 trampoline).  We track all of
    them and the alias-resolution test asserts that base and alias
    intersect on at least one address.
    """
    if not SHIM_PATH.exists():
        pytest.skip(f"{SHIM_PATH} not built - run `make shim` first")
    out = subprocess.check_output(
        ["nm", "-D", str(SHIM_PATH)], stderr=subprocess.DEVNULL
    ).decode()
    syms = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-2] == "T":
            addr = parts[0]
            # Strip ELF version suffix: "name@@GB_HOOKS" -> "name"
            name = parts[-1].split("@")[0]
            syms.setdefault(name, set()).add(addr)
    return syms


class TestExportedHooks:
    """The .so must export every hook in EXPECTED_HOOKS.  A missing one
    means either the map file lost an entry or the .c file dropped the
    function definition - both have caused silent shim bypass in the
    past."""

    @pytest.mark.parametrize("symbol", sorted(EXPECTED_HOOKS))
    def test_hook_is_exported(self, shim_symbols, symbol):
        assert symbol in shim_symbols, (
            f"Hook '{symbol}' missing from libgreenboost_cuda.so. "
            f"Check greenboost_cuda.map exports it and the .c file "
            f"defines it without `static`."
        )


class TestPTSZAliases:
    """PR-R: each _ptsz alias must resolve to the same address as its
    base symbol (GCC __attribute__((alias)) creates an ELF alias, so
    address equality is the integrity check)."""

    @pytest.mark.parametrize("alias,base", sorted(EXPECTED_PTSZ_ALIASES.items()))
    def test_ptsz_alias_resolves(self, shim_symbols, alias, base):
        assert alias in shim_symbols, f"_ptsz alias '{alias}' not exported"
        assert base in shim_symbols, f"Base symbol '{base}' not exported"
        # __attribute__((alias)) emits one symbol pointing at the same code as
        # the target.  When the base has multiple version entries, only ONE
        # of them (the default @@) needs to match the alias address.
        common = shim_symbols[alias] & shim_symbols[base]
        assert common, (
            f"'{alias}' at {sorted(shim_symbols[alias])} ∩ "
            f"'{base}' at {sorted(shim_symbols[base])} is empty "
            f"- the __attribute__((alias)) is broken or replaced with a thunk."
        )


class TestPTDSAliases:
    """PR-R: each _ptds alias must resolve to the same address as its
    base symbol - driver-API per-thread-default-stream variants."""

    @pytest.mark.parametrize("alias,base", sorted(EXPECTED_PTDS_ALIASES.items()))
    def test_ptds_alias_resolves(self, shim_symbols, alias, base):
        assert alias in shim_symbols, f"_ptds alias '{alias}' not exported"
        assert base in shim_symbols, f"Base symbol '{base}' not exported"
        common = shim_symbols[alias] & shim_symbols[base]
        assert common, (
            f"'{alias}' at {sorted(shim_symbols[alias])} ∩ "
            f"'{base}' at {sorted(shim_symbols[base])} is empty"
        )


class TestSo12Trampolines:
    """Positive guard: each symbol in this table MUST have a non-default
    (@libcudart.so.12) trampoline present in the .so.  These are built by
    greenboost_cuda_v12.c via .symver inline asm; they can silently disappear
    when that TU is accidentally compiled with -flto (strips .symver) or when
    its keepalive table loses an entry.

    The test reads `readelf --dyn-syms --wide` and checks for the `@` (not `@@`)
    form of each name with the libcudart.so.12 version tag.
    """

    REQUIRED_SO12_TRAMPOLINES = {
        "cudaGetDriverEntryPointByVersion",
        "cudaMalloc",
        "cudaFree",
        "cudaMemGetInfo",
    }

    @pytest.fixture(scope="class")
    def so12_syms(self):
        if not SHIM_PATH.exists():
            pytest.skip(f"{SHIM_PATH} not built - run `make shim` first")
        out = subprocess.check_output(
            ["readelf", "--dyn-syms", "--wide", str(SHIM_PATH)],
            stderr=subprocess.DEVNULL,
        ).decode()
        present = set()
        for line in out.splitlines():
            # Match `name@libcudart.so.12` (single @, non-default)
            # readelf format: "  N: addr size type bind vis ndx name@version"
            if "@libcudart.so.12" in line and "@@libcudart.so.12" not in line:
                parts = line.split()
                if parts:
                    name = parts[-1].split("@")[0]
                    present.add(name)
        return present

    @pytest.mark.parametrize("symbol", sorted(REQUIRED_SO12_TRAMPOLINES))
    def test_so12_trampoline_present(self, so12_syms, symbol):
        assert symbol in so12_syms, (
            f"@libcudart.so.12 trampoline for '{symbol}' is missing. "
            f"Check greenboost_cuda_v12.c keepalive table and that the TU "
            f"is compiled WITHOUT -flto (SHIM_CFLAGS_V12 in Makefile)."
        )


class TestNoSo12DefaultVersion:
    """Regression guard for the cu128 infinite-recursion hang (2026-06-12).

    greenboost_cuda_v12.c builds a `cudaX@libcudart.so.12` trampoline whose
    body literally calls the bare name `cudaX(...)`.  If a hook body's DEFAULT
    (@@) version is ALSO libcudart.so.12 (because the symbol was listed only in
    the .so.12 node of greenboost_cuda.map), that intra-object call binds to the
    trampoline's own @.so.12 alias instead of the real body → the function calls
    itself forever.  PyTorch cu128 hit this on the very first CUDA call
    (torch.cuda.get_device_properties) and spun at 100 % CPU with the GPU idle.

    Invariant: NO libcudart hook may default to @@libcudart.so.12.  Every
    .so.12 export must be a NON-default trampoline (@), with the real body
    defaulting to @@libcudart.so.13 , exactly how cudaMalloc has always worked.
    """

    @pytest.fixture(scope="class")
    def versioned_syms(self):
        if not SHIM_PATH.exists():
            pytest.skip(f"{SHIM_PATH} not built - run `make shim` first")
        out = subprocess.check_output(
            ["readelf", "--dyn-syms", "--wide", str(SHIM_PATH)],
            stderr=subprocess.DEVNULL,
        ).decode()
        defaults_on_so12 = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 8 or parts[3] != "FUNC":
                continue
            name = parts[7]
            # `name@@ver` is a DEFAULT versioned definition; `name@ver` is not.
            if "@@libcudart.so.12" in name:
                defaults_on_so12.append(name.split("@")[0])
        return defaults_on_so12

    def test_no_hook_defaults_to_so12(self, versioned_syms):
        assert versioned_syms == [], (
            "These hook bodies DEFAULT to @@libcudart.so.12, which makes the "
            "greenboost_cuda_v12.c trampolines recurse infinitely (100%% CPU "
            "hang under PyTorch cu128): "
            f"{sorted(set(versioned_syms))}.  Fix: list them in the "
            "libcudart.so.13 node of greenboost_cuda.map so the body defaults "
            "to @@.so.13 while the .so.12 trampoline stays a non-default alias."
        )


class TestHookCount:
    """Sanity check - the hook count should grow over time as new CUDA
    APIs are intercepted, never shrink.  If this fails, something was
    removed that probably shouldn't have been."""

    def test_at_least_33_hooks_exported(self, shim_symbols):
        # Count exported T symbols only; nm -D output already filtered.
        gb_hooks = [s for s in shim_symbols if not s.startswith("_")]
        # Floor of 33 was the count after PR-S; tracking will grow over time.
        assert len(gb_hooks) >= 33, (
            f"Only {len(gb_hooks)} hooks exported, expected ≥33.  "
            f"Did a recent change drop the symbol map entries?"
        )
