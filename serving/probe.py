#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""serving/probe.py — typed pre-serve probe suite for gb-synapse.

NemoClaw audit, Phase 5d. Design re-implemented from scratch (no code
copied) from NemoClaw's `probeLlamaCppAttachment`
(`src/lib/inference/llama-cpp/index.ts`) — its two structural moves, not
its literal failure-reason strings (GreenBoost's own serve sequence is
different, so the closed set below names GreenBoost's own steps):

  1. Every probe step is BOUNDED (a timeout, a byte cap) — no step may hang
     the caller indefinitely or process an unbounded response.
  2. Mixed or partial evidence FAILS CLOSED with a NAMED reason, never a
     caught exception logged and forgotten. This is the direct fix for the
     documented incident class (CLAUDE.md's Reference Workload Rule
     section): "error loading model hyperparameters: key not found in
     model: qwen3vlmoe.rope.dimension_sections" was a crash log to
     interpret by hand; this module's job is to turn that class of failure
     into `ProbeResult(ok=False, reason="gguf_malformed", message=...)`
     instead.

The five real steps this applies to gb-synapse's own serve sequence: GGUF
header read -> short load -> /health -> one 1-token completion -> one
tool-call completion. Each step is a plain callable so this module has no
hardware/network dependency of its own and is fully unit-testable with
fakes — the real callables live in gb_synapse.py's integration (see
`probe_serve_readiness`'s docstring for the wiring).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

# Closed set of failure reasons — a caller must never invent a new string
# outside this set; PROBE_FAILURE_REASONS is what makes that enforceable
# (see ProbeResult.__post_init__).
PROBE_FAILURE_REASONS = frozenset({
    # GGUF header read
    "gguf_unreadable",       # file missing, unopenable, or I/O error
    "gguf_malformed",        # header parses but is missing/wrong-typed a
                              # required key (the rope.dimension_sections
                              # incident class)
    # Short load
    "load_timeout",          # engine didn't report ready within the bound
    "load_crashed",          # engine process exited before reporting ready
    # /health
    "health_unreachable",    # no response at all within the bound
    "health_unhealthy",      # responded, but reported an unhealthy state
    # One 1-token completion
    "completion_timeout",
    "completion_malformed",  # response wasn't parseable as a completion
    "completion_empty",      # parsed fine, but produced zero tokens
    # One tool-call completion
    "tool_call_unsupported", # capabilities.toolCalls claimed but the
                              # engine/model didn't actually produce one
    "tool_call_malformed",   # produced *something* tool-call-shaped, but
                              # it didn't parse as a valid tool call
    # Aborted before a specific step could even run (e.g. the process
    # driving the probe itself was interrupted) — the catch-all that keeps
    # every other reason meaningfully specific rather than a dumping ground.
    "probe_aborted",
})

STEP_ORDER = (
    "gguf_header", "short_load", "health", "one_token_completion", "tool_call_completion",
)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    step: str
    reason: "str | None" = None
    message: str = ""
    elapsed_s: float = 0.0

    def __post_init__(self):
        if not self.ok and self.reason not in PROBE_FAILURE_REASONS:
            raise ValueError(
                f"probe failure reason {self.reason!r} is not in the closed set "
                f"PROBE_FAILURE_REASONS — add it there deliberately, don't "
                f"invent an ad-hoc string"
            )
        if self.ok and self.reason is not None:
            raise ValueError("a successful ProbeResult must not carry a reason")


class StepTimeout(Exception):
    """Raised by a step callable (or by the bounded-call wrapper) to signal
    it exceeded its bound. Caught internally and translated to the step's
    own *_timeout reason — callers never see this exception type."""


def _bounded_call(fn: Callable[[], object], timeout_s: float) -> object:
    """Call `fn()` and enforce `timeout_s` — a plain wall-clock check
    around the call, not a hard preemptive timeout (Python has no portable
    way to hard-kill an arbitrary callable). `fn` is expected to already
    respect its own timeout internally (a real HTTP probe passes its own
    `timeout=` to whatever client it uses); this is the outer belt-and-
    braces bound in case it doesn't.
    """
    started = time.monotonic()
    result = fn()
    elapsed = time.monotonic() - started
    if elapsed > timeout_s:
        raise StepTimeout(f"step exceeded its {timeout_s}s bound (took {elapsed:.2f}s)")
    return result


def probe_serve_readiness(
    steps: "dict[str, Callable[[], object]]",
    *,
    timeouts_s: "dict[str, float] | None" = None,
    require_tool_call: bool = True,
) -> "list[ProbeResult]":
    """Run the five-step probe sequence in STEP_ORDER, stopping at the
    first failure (mixed/partial evidence fails closed — a later step's
    success never excuses an earlier step's failure). Returns the list of
    ProbeResults produced so far (length 1..5); the LAST element's `.ok` is
    the overall verdict.

    `steps` maps step name -> a zero-arg callable. Each callable must
    return one of:
      - `True`/a truthy value meaning "this step succeeded, no further
        detail needed"
      - a `(ok: bool, message: str)` tuple for a step that wants to attach
        detail either way
      - raise `StepTimeout` to signal the step itself detected it ran too
        long (translated to the appropriate `*_timeout` reason)
      - raise any other exception, which this function catches and maps to
        the step's *_malformed/*_unreachable/*_crashed reason as
        appropriate — never propagates a bare traceback to the caller.

    `require_tool_call=False` skips the final step entirely (for a
    recipe/model whose `capabilities.toolCalls` is false — there is
    nothing to probe).

    Real wiring (gb_synapse.py's integration): `steps["gguf_header"]` reads
    the GGUF header via the same code `gguf_summary()` already uses;
    `steps["short_load"]` launches the engine with a short timeout and
    polls for readiness; `steps["health"]` hits the running engine's
    `/health`; `steps["one_token_completion"]`/`["tool_call_completion"]`
    send one real request each through the already-running server. None of
    that lives in THIS module — this module only knows the closed set of
    outcomes and the ordering/bounding contract, so it stays fully
    unit-testable without a GPU."""
    timeouts_s = timeouts_s or {}
    default_timeout = 30.0
    # (primary, secondary): primary is used when a step callable returns a
    # clean (False, message) — it ran and told us it failed; secondary is
    # used when the step callable raises an unexpected exception instead.
    # A step's dedicated *_timeout reason (see timeout_reason below) is
    # separate from both — it only fires via StepTimeout, never through
    # this table.
    reason_by_step = {
        "gguf_header": ("gguf_unreadable", "gguf_malformed"),
        "short_load": ("load_crashed", "load_crashed"),
        "health": ("health_unhealthy", "health_unreachable"),
        "one_token_completion": ("completion_empty", "completion_malformed"),
        "tool_call_completion": ("tool_call_unsupported", "tool_call_malformed"),
    }
    # Reason a step's TIMEOUT maps to. gguf_header has no dedicated timeout
    # reason of its own (a local file read hanging is, in practice, the
    # same failure as the file being unreadable) — mapped to
    # gguf_unreadable rather than inventing a reason string that isn't in
    # PROBE_FAILURE_REASONS.
    timeout_reason = {
        "gguf_header": "gguf_unreadable",
        "short_load": "load_timeout",
        "health": "health_unreachable",
        "one_token_completion": "completion_timeout",
        "tool_call_completion": "tool_call_unsupported",
    }

    order = STEP_ORDER if require_tool_call else STEP_ORDER[:-1]
    results: "list[ProbeResult]" = []

    for step in order:
        fn = steps.get(step)
        started = time.monotonic()
        if fn is None:
            results.append(ProbeResult(
                ok=False, step=step, reason="probe_aborted",
                message=f"no probe callable registered for step {step!r}",
            ))
            break
        timeout_s = timeouts_s.get(step, default_timeout)
        primary_reason, secondary_reason = reason_by_step[step]
        try:
            outcome = _bounded_call(fn, timeout_s)
        except StepTimeout as e:
            results.append(ProbeResult(
                ok=False, step=step, reason=timeout_reason[step], message=str(e),
                elapsed_s=time.monotonic() - started,
            ))
            break
        except Exception as e:
            results.append(ProbeResult(
                ok=False, step=step, reason=secondary_reason,
                message=f"{type(e).__name__}: {e}",
                elapsed_s=time.monotonic() - started,
            ))
            break

        elapsed = time.monotonic() - started
        if isinstance(outcome, tuple) and len(outcome) == 2:
            ok, message = outcome
        else:
            ok, message = bool(outcome), ""

        if ok:
            results.append(ProbeResult(ok=True, step=step, elapsed_s=elapsed))
            continue

        # A step that reports failure without raising picks the step's
        # PRIMARY reason (e.g. "unreachable"/"unhealthy" rather than the
        # exception-mapped secondary one — a step distinguishes these two
        # itself by returning (False, message) vs raising).
        results.append(ProbeResult(
            ok=False, step=step, reason=primary_reason, message=message,
            elapsed_s=elapsed,
        ))
        break

    return results


def overall_ok(results: "list[ProbeResult]") -> bool:
    """True iff every step that ran succeeded. probe_serve_readiness()
    stops at the first failure, so in practice this reduces to "the last
    result is ok" — spelled as `all()` so it stays correct even if a future
    caller assembles a results list a different way."""
    return bool(results) and all(r.ok for r in results)
