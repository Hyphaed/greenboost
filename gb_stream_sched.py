#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_stream_sched.py — GreenBoost multi-stream CUDA scheduler.

Creates dedicated CUDA streams for each workload class so compute, memory
transfers, and quantization can overlap:

  gemm      — weight GEMMs and attention (main compute)
  transfer  — H2D / D2H model page moves (async CPU↔GPU)
  quant     — quantization passes (gb_quant calls)
  copy      — inter-buffer copies, VAE encode/decode staging

Usage:
    import gb_stream_sched as gs

    # Context-manager: run code on a named stream
    with gs.on("transfer"):
        tensor.to("cuda", non_blocking=True)

    # Wait for a stream to finish from another stream
    gs.wait_for("transfer", on="gemm")

    # Full barrier: all streams sync
    gs.barrier()

    # CUDA Graph capture on a named stream
    with gs.capture("gemm") as graph:
        model(inputs)
    graph.replay()

    # Get the raw torch.cuda.Stream object
    stream = gs.stream("gemm")
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Dict, Optional

_streams: Dict[str, "torch.cuda.Stream"] = {}
_STREAM_NAMES = ("gemm", "transfer", "quant", "copy")

# NVTX for Nsight Systems timeline ranges
_nvtx_annotate = None
def _init_nvtx():
    global _nvtx_annotate
    _nvtx_python = os.path.expanduser("~/Dev/nvidia_dcgm/NVTX/python/src")
    if _nvtx_python not in sys.path:
        sys.path.insert(0, _nvtx_python)
    try:
        import nvtx
        _nvtx_annotate = nvtx.annotate
    except Exception:
        pass
_init_nvtx()

@contextlib.contextmanager
def _nvtx(msg: str, color: str = "blue"):
    if _nvtx_annotate is not None:
        with _nvtx_annotate(message=msg, color=color, domain="GreenBoost"):
            yield
    else:
        yield


def _ensure_init():
    global _streams
    if _streams:
        return
    import torch
    for name in _STREAM_NAMES:
        _streams[name] = torch.cuda.Stream()


def stream(name: str) -> "torch.cuda.Stream":
    """Return the raw CUDA stream for a workload class."""
    _ensure_init()
    if name not in _streams:
        raise KeyError(f"Unknown stream '{name}'. Available: {list(_streams)}")
    return _streams[name]


@contextlib.contextmanager
def on(name: str):
    """Context manager: execute block on the named CUDA stream."""
    _ensure_init()
    import torch
    with _nvtx(f"gb:stream:{name}", color="blue"):
        with torch.cuda.stream(_streams[name]):
            yield _streams[name]


def wait_for(source: str, on: str = "gemm"):
    """
    Make the `on` stream wait for all work queued on the `source` stream.
    Inserts a CUDA event; no host synchronization.
    """
    _ensure_init()
    import torch
    ev = torch.cuda.Event()
    _streams[source].record_event(ev)
    _streams[on].wait_event(ev)


def barrier():
    """Synchronize all named streams to the host. Use sparingly."""
    _ensure_init()
    import torch
    torch.cuda.synchronize()


@contextlib.contextmanager
def capture(name: str = "gemm"):
    """
    CUDA Graph capture context on the named stream.
    Yields the graph object; caller calls graph.replay() to re-run.

    Example:
        with gs.capture("gemm") as g:
            out = model(x)
        g.replay()  # zero launch overhead
    """
    _ensure_init()
    import torch

    s = _streams[name]
    g = torch.cuda.CUDAGraph()

    # Warm-up: one real pass to populate caches before capture
    # (caller is responsible for having run at least one warmup already)
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            yield g


def prefetch_async(tensor: "torch.Tensor", name: str = "transfer") -> "torch.cuda.Event":
    """
    Move a CPU tensor to GPU asynchronously on the transfer stream.
    Returns a CUDA event the caller can wait on before consuming the tensor.

    Usage:
        ev = gs.prefetch_async(weights_cpu)
        with gs.on("gemm"):
            gs.wait_for("transfer", on="gemm")
            result = model_layer(input, weights_gpu)
    """
    _ensure_init()
    import torch

    ev = torch.cuda.Event()
    with on(name):
        tensor.data = tensor.data.to("cuda", non_blocking=True)
        _streams[name].record_event(ev)
    return ev


def offload_async(tensor: "torch.Tensor", name: str = "transfer") -> "torch.cuda.Event":
    """
    Move a GPU tensor to pinned CPU memory asynchronously on the transfer stream.
    Returns a CUDA event the caller can wait on before reading the CPU data.
    """
    _ensure_init()
    import torch

    ev = torch.cuda.Event()
    with on(name):
        cpu_pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
        cpu_pinned.copy_(tensor, non_blocking=True)
        tensor.data = cpu_pinned
        _streams[name].record_event(ev)
    return ev
