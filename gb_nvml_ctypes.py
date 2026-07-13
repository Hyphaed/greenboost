#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_nvml_ctypes.py , pynvml-compatible NVML bindings over ctypes.

Drop-in fallback for the `pynvml` package (nvidia-ml-py) used when it is not
installed in the running Python environment. Binds libnvidia-ml.so.1 directly,
which ships with every NVIDIA driver , so GreenBoost telemetry (gb_telemetry,
gb_nvml, gb_vitals_helper) gets real VRAM/util/temp/power/ECC/PCIe data in any
venv without a pip dependency.

Implements exactly the surface those modules call. Import pattern:

    try:
        import pynvml
    except ImportError:
        import gb_nvml_ctypes as pynvml
"""
from __future__ import annotations

import ctypes
from ctypes import (POINTER, byref, c_char, c_char_p, c_int, c_uint,
                    c_ulonglong, c_void_p)

# ── constants (values from nvml.h) ──────────────────────────────────────────
NVML_SUCCESS = 0

NVML_TEMPERATURE_GPU = 0

NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2

NVML_PCIE_UTIL_TX_BYTES = 0
NVML_PCIE_UTIL_RX_BYTES = 1

NVML_MEMORY_ERROR_TYPE_CORRECTED = 0
NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
NVML_VOLATILE_ECC = 0
NVML_AGGREGATE_ECC = 1

NVML_FEATURE_DISABLED = 0
NVML_FEATURE_ENABLED = 1
NVML_NVLINK_DEVICE_TYPE_GPU = 0

NVML_P2P_CAPS_INDEX_READ = 0
NVML_P2P_STATUS_OK = 0

_NVML_DEVICE_NAME_BUFFER_SIZE = 96
_NVML_PCI_BUS_ID_LEGACY_SIZE = 16
_NVML_PCI_BUS_ID_SIZE = 32


class NVMLError(Exception):
    """NVML call returned a non-success nvmlReturn_t code."""

    def __init__(self, code: int, func: str = ""):
        self.value = code
        super().__init__(f"NVML error {code}{f' from {func}' if func else ''}")


# ── structs ──────────────────────────────────────────────────────────────────

class c_nvmlMemory_t(ctypes.Structure):
    _fields_ = [("total", c_ulonglong), ("free", c_ulonglong), ("used", c_ulonglong)]


class c_nvmlUtilization_t(ctypes.Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class c_nvmlPciInfo_t(ctypes.Structure):
    # nvmlPciInfo_t layout since NVML 9 (busIdLegacy first, busId last)
    _fields_ = [
        ("busIdLegacy", c_char * _NVML_PCI_BUS_ID_LEGACY_SIZE),
        ("domain", c_uint),
        ("bus", c_uint),
        ("device", c_uint),
        ("pciDeviceId", c_uint),
        ("pciSubSystemId", c_uint),
        ("busId", c_char * _NVML_PCI_BUS_ID_SIZE),
    ]


class _ComputeCapability:
    """Result object with .major/.minor, iterable like pynvml's tuple form."""

    def __init__(self, major: int, minor: int):
        self.major = major
        self.minor = minor

    def __iter__(self):
        return iter((self.major, self.minor))


# ── library loading / call helper ────────────────────────────────────────────

_lib: "ctypes.CDLL | None" = None


def _nvml() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        try:
            _lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError:
            _lib = ctypes.CDLL("libnvidia-ml.so")  # dev symlink fallback
    return _lib


def _sym(name: str, *fallbacks: str):
    """Resolve an NVML symbol, preferring versioned names (e.g. _v3, _v2)."""
    lib = _nvml()
    for n in (name, *fallbacks):
        try:
            return getattr(lib, n)
        except AttributeError:
            continue
    raise AttributeError(f"NVML symbol not found: {name}")


def _call(fn, *args, func: str = "") -> None:
    rc = fn(*args)
    if rc != NVML_SUCCESS:
        raise NVMLError(rc, func)


# ── API ──────────────────────────────────────────────────────────────────────

def nvmlInit() -> None:
    _call(_sym("nvmlInit_v2", "nvmlInit"), func="nvmlInit")


def nvmlShutdown() -> None:
    _call(_sym("nvmlShutdown"), func="nvmlShutdown")


def nvmlDeviceGetCount() -> int:
    n = c_uint()
    _call(_sym("nvmlDeviceGetCount_v2", "nvmlDeviceGetCount"), byref(n),
          func="nvmlDeviceGetCount")
    return n.value


def nvmlDeviceGetHandleByIndex(index: int) -> c_void_p:
    h = c_void_p()
    _call(_sym("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex"),
          c_uint(index), byref(h), func="nvmlDeviceGetHandleByIndex")
    return h


def nvmlDeviceGetHandleByPciBusId(bus_id) -> c_void_p:
    if isinstance(bus_id, str):
        bus_id = bus_id.encode("ascii")
    h = c_void_p()
    _call(_sym("nvmlDeviceGetHandleByPciBusId_v2", "nvmlDeviceGetHandleByPciBusId"),
          c_char_p(bus_id), byref(h), func="nvmlDeviceGetHandleByPciBusId")
    return h


def nvmlDeviceGetIndex(handle) -> int:
    idx = c_uint()
    _call(_sym("nvmlDeviceGetIndex"), handle, byref(idx), func="nvmlDeviceGetIndex")
    return idx.value


def nvmlDeviceGetName(handle) -> str:
    buf = ctypes.create_string_buffer(_NVML_DEVICE_NAME_BUFFER_SIZE)
    _call(_sym("nvmlDeviceGetName"), handle, buf,
          c_uint(_NVML_DEVICE_NAME_BUFFER_SIZE), func="nvmlDeviceGetName")
    return buf.value.decode("utf-8", errors="replace")


def nvmlDeviceGetMemoryInfo(handle) -> c_nvmlMemory_t:
    mem = c_nvmlMemory_t()
    _call(_sym("nvmlDeviceGetMemoryInfo"), handle, byref(mem),
          func="nvmlDeviceGetMemoryInfo")
    return mem


def nvmlDeviceGetUtilizationRates(handle) -> c_nvmlUtilization_t:
    util = c_nvmlUtilization_t()
    _call(_sym("nvmlDeviceGetUtilizationRates"), handle, byref(util),
          func="nvmlDeviceGetUtilizationRates")
    return util


def nvmlDeviceGetTemperature(handle, sensor: int) -> int:
    t = c_uint()
    _call(_sym("nvmlDeviceGetTemperature"), handle, c_uint(sensor), byref(t),
          func="nvmlDeviceGetTemperature")
    return t.value


def nvmlDeviceGetPowerUsage(handle) -> int:
    mw = c_uint()
    _call(_sym("nvmlDeviceGetPowerUsage"), handle, byref(mw),
          func="nvmlDeviceGetPowerUsage")
    return mw.value


def nvmlDeviceGetPowerManagementLimit(handle) -> int:
    mw = c_uint()
    _call(_sym("nvmlDeviceGetPowerManagementLimit"), handle, byref(mw),
          func="nvmlDeviceGetPowerManagementLimit")
    return mw.value


def nvmlDeviceGetClockInfo(handle, clock_type: int) -> int:
    mhz = c_uint()
    _call(_sym("nvmlDeviceGetClockInfo"), handle, c_uint(clock_type), byref(mhz),
          func="nvmlDeviceGetClockInfo")
    return mhz.value


def nvmlDeviceGetMaxClockInfo(handle, clock_type: int) -> int:
    # Needed by gb_topology/gb_synapse hardware detection (rule: derive
    # bandwidths from the executing node, never reference-box literals).
    mhz = c_uint()
    _call(_sym("nvmlDeviceGetMaxClockInfo"), handle, c_uint(clock_type), byref(mhz),
          func="nvmlDeviceGetMaxClockInfo")
    return mhz.value


def nvmlDeviceGetMemoryBusWidth(handle) -> int:
    bits = c_uint()
    _call(_sym("nvmlDeviceGetMemoryBusWidth"), handle, byref(bits),
          func="nvmlDeviceGetMemoryBusWidth")
    return bits.value


def nvmlDeviceGetPcieThroughput(handle, counter: int) -> int:
    kbs = c_uint()
    _call(_sym("nvmlDeviceGetPcieThroughput"), handle, c_uint(counter), byref(kbs),
          func="nvmlDeviceGetPcieThroughput")
    return kbs.value


def nvmlDeviceGetTotalEccErrors(handle, error_type: int, counter_type: int) -> int:
    count = c_ulonglong()
    _call(_sym("nvmlDeviceGetTotalEccErrors"), handle, c_uint(error_type),
          c_uint(counter_type), byref(count), func="nvmlDeviceGetTotalEccErrors")
    return count.value


def nvmlDeviceGetCudaComputeCapability(handle) -> _ComputeCapability:
    major, minor = c_int(), c_int()
    _call(_sym("nvmlDeviceGetCudaComputeCapability"), handle, byref(major),
          byref(minor), func="nvmlDeviceGetCudaComputeCapability")
    return _ComputeCapability(major.value, minor.value)


def nvmlDeviceGetPciInfo(handle) -> c_nvmlPciInfo_t:
    pci = c_nvmlPciInfo_t()
    _call(_sym("nvmlDeviceGetPciInfo_v3", "nvmlDeviceGetPciInfo_v2",
               "nvmlDeviceGetPciInfo"), handle, byref(pci),
          func="nvmlDeviceGetPciInfo")
    return pci


nvmlDeviceGetPciInfo_v3 = nvmlDeviceGetPciInfo


def nvmlDeviceGetCurrPcieLinkGeneration(handle) -> int:
    v = c_uint()
    _call(_sym("nvmlDeviceGetCurrPcieLinkGeneration"), handle, byref(v),
          func="nvmlDeviceGetCurrPcieLinkGeneration")
    return v.value


def nvmlDeviceGetMaxPcieLinkGeneration(handle) -> int:
    v = c_uint()
    _call(_sym("nvmlDeviceGetMaxPcieLinkGeneration"), handle, byref(v),
          func="nvmlDeviceGetMaxPcieLinkGeneration")
    return v.value


def nvmlDeviceGetCurrPcieLinkWidth(handle) -> int:
    v = c_uint()
    _call(_sym("nvmlDeviceGetCurrPcieLinkWidth"), handle, byref(v),
          func="nvmlDeviceGetCurrPcieLinkWidth")
    return v.value


def nvmlDeviceGetMaxPcieLinkWidth(handle) -> int:
    v = c_uint()
    _call(_sym("nvmlDeviceGetMaxPcieLinkWidth"), handle, byref(v),
          func="nvmlDeviceGetMaxPcieLinkWidth")
    return v.value


def nvmlDeviceGetNvLinkState(handle, link: int) -> int:
    state = c_uint()
    _call(_sym("nvmlDeviceGetNvLinkState"), handle, c_uint(link), byref(state),
          func="nvmlDeviceGetNvLinkState")
    return state.value


def nvmlDeviceGetNvLinkRemoteDeviceType(handle, link: int) -> int:
    dtype = c_uint()
    _call(_sym("nvmlDeviceGetNvLinkRemoteDeviceType"), handle, c_uint(link),
          byref(dtype), func="nvmlDeviceGetNvLinkRemoteDeviceType")
    return dtype.value


def nvmlDeviceGetNvLinkRemotePciInfo_v2(handle, link: int) -> c_nvmlPciInfo_t:
    pci = c_nvmlPciInfo_t()
    _call(_sym("nvmlDeviceGetNvLinkRemotePciInfo_v2",
               "nvmlDeviceGetNvLinkRemotePciInfo"), handle, c_uint(link),
          byref(pci), func="nvmlDeviceGetNvLinkRemotePciInfo_v2")
    return pci


def nvmlDeviceGetP2PStatus(handle1, handle2, caps_index: int) -> int:
    status = c_uint()
    _call(_sym("nvmlDeviceGetP2PStatus"), handle1, handle2, c_uint(caps_index),
          byref(status), func="nvmlDeviceGetP2PStatus")
    return status.value
