#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""gb_failures.py — typed serve-failure classification with confidence signals.

When gb_synapse.serve() or gb_synapse_backends fails (either by raising an
exception or returning a refusal), the failure surfaces through this module
as a FailureClassification carrying:
  - kind: a closed-set enum of failure types (FailureKind)
  - reason: human-readable explanation with confidence caveats baked in
  - next_step: concrete, actionable remedy
  - confidence: "high" when the signal is fresh/verified, "low" when
    stale/heuristic/inferred

This module's design is ported from NemoClaw's failure-classifier.ts pattern:
a verdict derived from unverified, stale, or heuristic evidence (rather than
freshly measured) is returned at confidence:"low" with an explicit caveat
embedded in the reason string. Never silently present a low-confidence verdict
as equally solid as a high-confidence one.

Port discipline: the FailureKind enum and classify_serve_failure() signature
are designed for THIS codebase's real failure modes (see RESEARCH section
below), not for NemoClaw's domain (gateway presets, OpenShell policy). The
confidence-and-caveat PATTERN is ported exactly; the failure taxonomy is
GreenBoost's own.

RESEARCH (2026-08-05):
  - serving/probe.py: 12 named pre-serve failure reasons (gguf_unreadable,
    gguf_malformed, load_timeout, load_crashed, health_unreachable,
    health_unhealthy, completion_timeout, completion_malformed,
    completion_empty, tool_call_unsupported, tool_call_malformed,
    probe_aborted). This module handles failures AFTER a probe passes — the
    layer "above" probe.py for failures the probe cannot pre-empt.
  - gb_synapse_backends._gate_cpu_offload(): raises RuntimeError when serve()
    would reduce -ngl below all layers as a capacity decision and the shim
    is not reliably active or T2 spill is unavailable. Signals: shim status
    from gb_shim_probe.py (ok/fails cached, vulnerable to stale cache), T2
    free from telemetry (best-effort, may be inferred). Example scenario:
    (low confidence) shim was probed 3 hours ago, cache says "works", but a
    reboot or driver update has happened since.
  - gb_shim_probe.shim_works_for_llama(): verdict for "does the CUDA shim
    work with this llama.cpp build?" Returns (ok, reason) tuple. Cached by
    (shim .so mtime, engine version, ggml cudart soname). Failure markers:
    "invalid device function", "cuda error", early process exit, timeout
    without /health. Signals: fresh if just probed, low confidence if cached.
  - gb_cluster.ensure_feeder_ready(): returns bool after (1) SSH connectivity
    check, (2) remote Python + deps check, (3) optional model rsync. Fails
    loudly with RuntimeError if any step fails. Signals: fresh (just checked).
  - gb_aviary.smoke_gate()/niah_certify(): quality checks on real model
    output. Return verdict dict with quality status. Signals: fresh (just
    ran inference).
  - Port detection: GB_SYNAPSE_PORT or engine port binding conflicts not
    extensively searched, but would surface as OSError/BindError on port
    listen. Signals: immediate.
  - Format compatibility: Already caught by probe.py's gguf_malformed, but
    could theoretically surface during serve if a format issue (e.g. missing
    rope.dimension_sections, Ollama blob incompatibility) only manifests
    under load. Signals: detected immediately, high confidence.

FailureKind closed set — mapped from real codebase failure modes:
  - SHIM_BROKEN: gb_shim_probe verdict failed or stale
  - CAPACITY_EXCEEDED: _gate_cpu_offload() refusal (T2 spill unavailable)
  - FEEDER_UNREACHABLE: gb_cluster.ensure_feeder_ready() failed
  - PORT_CONFLICT: GB_SYNAPSE_PORT or engine port already bound
  - QUALITY_GATE_FAILED: gb_aviary quality check failed
  - ENGINE_MISSING: gb_synapse engine not built/installed
  - UNKNOWN: anything else

Confidence downgrade signals (mapped to "low"):
  - Shim status from cache older than some threshold (stale cache)
  - T2 telemetry not freshly read (best-effort, inferred)
  - Feeder state not just-checked (stale assumption about connectivity)
  - Any heuristic that hasn't been directly verified this session
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class FailureKind(str, Enum):
    """Closed set of GreenBoost serve failure types."""

    SHIM_BROKEN = "shim_broken"
    """The CUDA shim (libgreenboost_cuda.so) is not reliably working with the
    current llama.cpp engine. Detected via gb_shim_probe or known from prior
    initialization failure."""

    CAPACITY_EXCEEDED = "capacity_exceeded"
    """Model placement would require CPU offload, but T2 spill is unavailable
    or shim is not active. Raised by _gate_cpu_offload()."""

    FEEDER_UNREACHABLE = "feeder_unreachable"
    """A feeder cluster node is not reachable or doesn't have required
    dependencies. From gb_cluster.ensure_feeder_ready()."""

    PORT_CONFLICT = "port_conflict"
    """GB_SYNAPSE_PORT or engine port is already in use."""

    QUALITY_GATE_FAILED = "quality_gate_failed"
    """Quality check (smoke_gate/niah_certify) detected a problem with model
    output (e.g., repetition collapse, too-low bit width)."""

    ENGINE_MISSING = "engine_missing"
    """gb_synapse engine (llama.cpp build) is not installed or not ready."""

    UNKNOWN = "unknown"
    """Failure did not match any of the above categories."""


@dataclass(frozen=True)
class FailureClassification:
    """A serve failure classified with confidence signal.

    confidence="high": the signal is fresh, verified, or directly measured
    this session. Treat as authoritative.

    confidence="low": the signal is cached, inferred, or not freshly verified.
    Embed a caveat in reason explaining why — agent must treat as advisory
    and may retry after confirming the underlying state.
    """

    kind: FailureKind
    reason: str
    next_step: str
    confidence: str  # "high" | "low"

    def __post_init__(self):
        if self.confidence not in ("high", "low"):
            raise ValueError(
                f"confidence must be 'high' or 'low', got {self.confidence!r}"
            )


def classify_serve_failure(
    exception: Exception | None = None,
    kind_hint: FailureKind | None = None,
    model_name: str | None = None,
    shim_status: tuple[bool, str] | None = None,
    shim_cache_age_s: float | None = None,
    t2_facts: dict[str, Any] | None = None,
    t2_freshly_read: bool = False,
    feeder_check_failed: bool = False,
    feeder_hostname: str | None = None,
    port_number: int | None = None,
    quality_check_result: dict[str, Any] | None = None,
    extra_context: dict[str, Any] | None = None,
) -> FailureClassification:
    """Classify a serve failure with confidence signal.

    Args:
        exception: The exception raised during serve, if any.
        kind_hint: Optional hint about the failure type (e.g., from a catch
            block that knows it came from _gate_cpu_offload).
        model_name: Name of the model being served.
        shim_status: Result from gb_shim_probe.shim_works_for_llama(),
            tuple of (ok: bool, reason: str).
        shim_cache_age_s: Age of the shim cache entry in seconds. When
            non-None and > some staleness threshold, confidence is downgraded.
        t2_facts: Dict with T2 telemetry (e.g., {"t2_free_mb": 2048, ...}).
            When absent or not freshly read, signals may be inferred.
        t2_freshly_read: Whether t2_facts came from a live read this turn
            (vs. cached/inferred). Used to set confidence.
        feeder_check_failed: Whether ensure_feeder_ready() failed.
        feeder_hostname: Hostname of the feeder if relevant.
        port_number: Port number if a port conflict is suspected.
        quality_check_result: Dict from gb_aviary.smoke_gate() or similar,
            with "verdict" key ("PASS"/"FAIL") and "reason" detail.
        extra_context: Additional context dict for future extensibility.

    Returns:
        FailureClassification with kind, reason (including confidence
        caveat if needed), next_step, and confidence signal.
    """
    extra_context = extra_context or {}

    # Helper to downgrade confidence when a signal is stale.
    def _stale_caveat(signal_name: str, age_s: float | None = None) -> str:
        """Return a caveat string for a stale/unverified signal."""
        if age_s is not None and age_s > 0:
            return (
                f" The {signal_name} verdict is {age_s:.0f} seconds old "
                f"(cached); a configuration change (driver update, reboot, "
                f"kernel module reload) may have occurred since. "
                f"Re-probe or re-check this signal before trusting it."
            )
        return (
            f" The {signal_name} signal is inferred or not freshly verified; "
            f"treat this verdict as advisory."
        )

    # Route by hint or exception message inspection.
    if kind_hint == FailureKind.QUALITY_GATE_FAILED or (
        quality_check_result and quality_check_result.get("verdict") == "FAIL"
    ):
        reason_detail = quality_check_result.get("reason", "unknown issue") if quality_check_result else "unknown issue"
        return FailureClassification(
            kind=FailureKind.QUALITY_GATE_FAILED,
            reason=(
                f"Model '{model_name}' failed quality check: {reason_detail}."
            ),
            next_step=(
                "Verify the model's actual quality (run gb_aviary.niah_certify() "
                "with a longer context if available; check repetition and "
                "token diversity). If quality is confirmed poor, try a different "
                "quantization level (higher bit width, different quant method) "
                "or model. If quality is actually good, the check may have a "
                "false positive — investigate the check's own sensitivity."
            ),
            confidence="high",
        )

    if kind_hint == FailureKind.FEEDER_UNREACHABLE or feeder_check_failed:
        return FailureClassification(
            kind=FailureKind.FEEDER_UNREACHABLE,
            reason=(
                f"Feeder '{feeder_hostname or 'unknown'}' is not reachable or "
                f"missing required dependencies (confirmed by ensure_feeder_ready)."
            ),
            next_step=(
                f"Verify SSH connectivity to the feeder (ssh {feeder_hostname or '<feeder>'} 'true'); "
                "check that greenboost and required Python packages are installed on the feeder; "
                "run `greenboost feeders diag all` for full diagnostics. "
                "If the feeder is down, bring it up before retrying serve."
            ),
            confidence="high",
        )

    if kind_hint == FailureKind.PORT_CONFLICT or port_number:
        return FailureClassification(
            kind=FailureKind.PORT_CONFLICT,
            reason=(
                f"Port {port_number or os.environ.get('GB_SYNAPSE_PORT', '11369')} "
                f"is already in use (bind failed)."
            ),
            next_step=(
                f"Kill any existing gb-synapse or llama-server on this port "
                f"(lsof -i :{port_number or '11369'}; pkill -f 'llama-server|synapse'), "
                f"or choose a different port (set GB_SYNAPSE_PORT or --port)."
            ),
            confidence="high",
        )

    if kind_hint == FailureKind.ENGINE_MISSING:
        return FailureClassification(
            kind=FailureKind.ENGINE_MISSING,
            reason="gb_synapse engine (llama.cpp build) is not installed.",
            next_step="Run: greenboost synapse build-engine",
            confidence="high",
        )

    if kind_hint == FailureKind.CAPACITY_EXCEEDED or (
        exception and "CPU offload" in str(exception)
    ):
        # _gate_cpu_offload() raises with a message that includes shim status
        # and T2 free. Downgrade confidence if T2 signal is not fresh.
        confidence = "high" if t2_freshly_read else "low"
        caveat = (
            ""
            if confidence == "high"
            else _stale_caveat("T2 memory availability")
        )
        return FailureClassification(
            kind=FailureKind.CAPACITY_EXCEEDED,
            reason=(
                f"Model '{model_name}' would require CPU offload, but "
                f"the shim is {'not active' if shim_status and not shim_status[0] else 'active'} "
                f"and T2 overflow capacity may be unavailable.{caveat}"
            ),
            next_step=(
                "Option 1: Ensure the shim is working (run `greenboost doctor` "
                "and check the shim status). Option 2: Free more T2 memory "
                "(check `greenboost dataflux-ui` for active tier allocations; "
                "serve a smaller model first to free space). Option 3: Use a "
                "smaller quant or model. Option 4 (debug only): set "
                "GB_SYNAPSE_ALLOW_CPU_OFFLOAD=1 to override this gate."
            ),
            confidence=confidence,
        )

    if kind_hint == FailureKind.SHIM_BROKEN or (
        shim_status and not shim_status[0]
    ):
        # Shim probe failed or returned False. Downgrade confidence if cache
        # is stale.
        confidence = (
            "high"
            if shim_cache_age_s is None or shim_cache_age_s < 3600
            else "low"
        )
        caveat = (
            ""
            if confidence == "high"
            else _stale_caveat("shim verdict", shim_cache_age_s)
        )
        reason_detail = (
            shim_status[1] if shim_status else "shim probe failed"
        )
        return FailureClassification(
            kind=FailureKind.SHIM_BROKEN,
            reason=(
                f"CUDA shim (libgreenboost_cuda.so) is not reliably working: "
                f"{reason_detail}.{caveat}"
            ),
            next_step=(
                "Run `greenboost synapse build-engine` to rebuild llama.cpp and "
                "refresh the engine's CUDA dependencies. If that doesn't help, "
                "run `greenboost doctor --json` to check the shim installation "
                "and CUDA compatibility. As a last resort (for testing only), "
                "set GB_SYNAPSE_SHIM=0 to disable the shim, but this will fall "
                "back to slower T2 access patterns."
            ),
            confidence=confidence,
        )

    # Fallback: unknown failure type.
    exception_message = str(exception) if exception else extra_context.get("message", "")
    return FailureClassification(
        kind=FailureKind.UNKNOWN,
        reason=(
            f"Serve failed with an unclassified error: {exception_message or 'no details available'}."
        ),
        next_step=(
            "Inspect the full error trace and logs (run the failed serve again "
            "with GB_DEBUG=1 if needed); consult `greenboost doctor` and "
            "`greenboost dataflux-ui` for system state. If the error is "
            "reproducible and not covered above, file a report with the full "
            "error message and context."
        ),
        confidence="low",
    )
