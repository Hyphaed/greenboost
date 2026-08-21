#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_tuner.py , the decision half of GreenBoost's control loop, with no hardware
in it.

Everything that reads a sensor or moves a lever already exists: `gb_monitor`
and `gb_telemetry` read, `gb_control.GbControl` writes (bounded, with baseline
capture and `restore_baseline()`), and `gb_mcp.optimize_inference()` sequences
the two. What was missing is the part in between , a policy that decides WHAT
should move, is testable without a GPU, and does not fool itself.

`decide(snapshot, state)` is a pure function: same inputs, same decisions, no
imports beyond the standard library. That is the whole design constraint, and
it is what lets the awkward cases (a measurement taken before an actuation
landed, a threshold crossed by one sample, a target that is not reachable on
this hardware at all) be written down as tests instead of discovered live.

Ported from GameNative's `TunerDecisionEngine.kt`, audited 2026-08-20. The
shape is the value; the thresholds are ours and are all fractions of live
readings, never absolute hardware figures.

Five things it carries that a plain threshold check does not:

* **Settle cycles.** A measurement taken before a lever's effect has landed
  describes the world before the change. `optimize_inference(measure=True)`
  re-read throughput immediately and could roll back on pre-change data.
* **Latching hysteresis.** Enter at one threshold, release at a lower one, so
  a state does not flap on a single sample crossing a line.
* **EMA baseline with baseline-relative degradation.** When the absolute
  target is unreachable on this box , and at 15.85 GB of weights against a
  12 GB card, it is , "below target" is true forever and says nothing. What
  matters is movement against this machine's own recent baseline. This is the
  fix for the n=3-population advisories the 2026-08-18 audit caught.
* **Frozen domains with re-probe.** A lever that was tried and made things
  worse is frozen for a while, then re-probed, rather than retried every tick
  or abandoned forever.
* **Harvest.** When decode is bandwidth-bound and the bottleneck cannot be
  moved, holding clocks and power at maximum buys nothing , give them back.
  This box's PCIe wall is exactly that case: two dedicated sessions confirmed
  no code lever remains in the transfer path, and the GPU still sits at full
  power waiting on DDR.

What it deliberately does NOT do: touch anything. `decide()` returns
`Decision` objects; applying them is the caller's job, under the existing
`GB_ORCH_ACTUATE=1` gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["TunerSnapshot", "TunerState", "TunerPolicy", "Decision",
           "decide", "decide_draft_depth", "Bottleneck"]


# ── vocabulary ─────────────────────────────────────────────────────────────
#
# Shared with the Gaming Suite's fan controller by intent (index chain 4): a
# "harvest" means the same thing in both , give a resource back because the
# bottleneck is elsewhere and holding it buys nothing.

class Bottleneck:
    UNKNOWN = "unknown"
    GPU_BOUND = "gpu_bound"          # the GPU is the limit; clocks matter
    BANDWIDTH_BOUND = "bandwidth_bound"  # weights/KV crossing PCIe is the limit
    IDLE = "idle"                    # nothing is being served


@dataclass(frozen=True)
class TunerPolicy:
    """Thresholds. Every one is a fraction or a percentage of a live reading ,
    no absolute watts, megahertz or megabytes, so the same policy is correct on
    the 12 GB host and the 8 GB feeder."""

    #: GPU utilisation that latches "GPU bound" on, and the lower one that
    #: releases it. Two numbers, not one, is the whole point.
    #:
    #: Utilisation is NOT sufficient on its own here , see _classify(). It
    #: counts time with a kernel resident, not time doing useful work, and a
    #: kernel stalled on a PCIe read counts fully. Measured on this box
    #: 2026-08-20 mid-decode: utilization.gpu 100% at 49.8 W of a 300 W limit.
    gpu_bound_enter_pct: float = 85.0
    gpu_bound_release_pct: float = 75.0

    #: Power draw, as a fraction of the board limit, that a genuinely
    #: compute-bound GPU sustains. Below this with weights streaming from T2,
    #: the SMs are waiting on the link however busy they look.
    compute_bound_power_frac: float = 0.55
    compute_bound_power_release_frac: float = 0.45

    #: Ticks that must pass after an actuation before a measurement counts.
    settle_cycles: int = 3

    #: EMA weight for the throughput baseline.
    ema_alpha: float = 0.25

    #: Samples required before the baseline is trusted at all. Below this the
    #: engine says "insufficient" rather than deciding on noise , the n=3
    #: advisory bug in one line.
    min_baseline_samples: int = 8

    #: Throughput drop against the EMA baseline that counts as a regression.
    regression_pct: float = 10.0

    #: How much of a lever to give back per harvest step, as a fraction of its
    #: current value.
    harvest_step_frac: float = 0.05

    #: Floor for a harvested lever, as a fraction of the value it started at.
    #: Below this the engine stops harvesting even if nothing has regressed ,
    #: the goal is to stop wasting headroom, not to run the card at minimum.
    harvest_floor_frac: float = 0.70

    #: Ticks a lever stays frozen after it made things worse, before re-probe.
    freeze_cycles: int = 40

    #: Draft acceptance below which the current speculative depth is wasting
    #: forward-pass slots, and above which there is room to go deeper.
    draft_accept_low_pct: float = 40.0
    draft_accept_high_pct: float = 75.0
    draft_depth_min: int = 1
    draft_depth_max: int = 8


@dataclass(frozen=True)
class TunerSnapshot:
    """One reading. Every field is optional , a missing sensor must produce
    "I don't know", never a zero that reads as a real measurement."""

    tok_s: float | None = None
    inter_token_p95_ms: float | None = None
    slow_token_ratio: float | None = None
    gpu_util_pct: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    sm_clock_mhz: float | None = None
    sm_clock_max_mhz: float | None = None
    vram_fill_pct: float | None = None
    t2_overflow_mb: float | None = None
    draft_accept_pct: float | None = None
    draft_depth: int | None = None
    serving: bool = True


@dataclass(frozen=True)
class Decision:
    lever: str
    action: str           # "harvest" | "restore" | "set" | "hold" | "observe"
    value: Any = None
    reason: str = ""
    #: True when the caller must re-measure after `settle_cycles` and roll back
    #: on a regression. False for decisions that cannot make throughput worse.
    verify: bool = False


@dataclass
class TunerState:
    """Carry-over between ticks. The caller owns one of these per served
    session and passes it back in unchanged , `decide()` returns the next one
    rather than mutating this."""

    baseline_tok_s: float | None = None
    baseline_samples: int = 0
    settle_remaining: int = 0
    bottleneck: str = Bottleneck.UNKNOWN
    frozen: dict[str, int] = field(default_factory=dict)     # lever -> ticks left
    harvest_level: dict[str, float] = field(default_factory=dict)  # lever -> frac of start
    last_action_tok_s: float | None = None                   # baseline at actuation time
    ticks: int = 0


# ── the engine ─────────────────────────────────────────────────────────────

def decide(snap: TunerSnapshot, state: TunerState,
           policy: TunerPolicy | None = None) -> tuple[list[Decision], TunerState]:
    """One tick. Returns (decisions, next_state).

    An empty decision list is a normal, common outcome and means "nothing
    should move right now" , which is not the same as "everything is fine".
    The reason is always carried on an `observe` decision so the caller can
    record WHY nothing happened, which is the part that is invisible in a log
    of actions taken.
    """
    p = policy or TunerPolicy()
    st = replace(state, ticks=state.ticks + 1,
                 frozen=dict(state.frozen), harvest_level=dict(state.harvest_level))

    # Thaw frozen levers one tick at a time; a lever that reaches zero is
    # re-probeable again.
    for lever in list(st.frozen):
        st.frozen[lever] -= 1
        if st.frozen[lever] <= 0:
            del st.frozen[lever]

    if not snap.serving:
        st.bottleneck = Bottleneck.IDLE
        return ([Decision("none", "observe", reason="nothing is being served")], st)

    # ── settle ────────────────────────────────────────────────────────────
    # Nothing is measured, nothing is decided, while a previous actuation is
    # still landing. Reporting the reading anyway is how a rollback ends up
    # triggered by data from before the change it is judging.
    if st.settle_remaining > 0:
        st.settle_remaining -= 1
        return ([Decision("none", "observe",
                          value=st.settle_remaining,
                          reason=f"settling after an actuation, "
                                 f"{st.settle_remaining} tick(s) to go , this "
                                 f"reading describes the previous state")], st)

    # ── baseline ──────────────────────────────────────────────────────────
    if snap.tok_s is not None and snap.tok_s > 0:
        if st.baseline_tok_s is None:
            st.baseline_tok_s = snap.tok_s
        else:
            st.baseline_tok_s = ((1 - p.ema_alpha) * st.baseline_tok_s
                                 + p.ema_alpha * snap.tok_s)
        st.baseline_samples += 1

    baseline_ready = st.baseline_samples >= p.min_baseline_samples

    # ── bottleneck classification, latching ───────────────────────────────
    st.bottleneck = _classify(snap, st.bottleneck, p)

    out: list[Decision] = []

    # ── regression check on a lever we moved ──────────────────────────────
    # Only meaningful against a baseline that existed BEFORE the actuation,
    # which is what last_action_tok_s holds.
    if (st.last_action_tok_s and baseline_ready and snap.tok_s
            and snap.tok_s < st.last_action_tok_s * (1 - p.regression_pct / 100.0)):
        for lever in list(st.harvest_level):
            out.append(Decision(
                lever, "restore",
                reason=f"throughput fell to {snap.tok_s:.2f} tok/s from a "
                       f"{st.last_action_tok_s:.2f} baseline "
                       f"(>{p.regression_pct:.0f}%) after harvesting , giving "
                       f"the lever back and freezing it",
                verify=False))
            st.frozen[lever] = p.freeze_cycles
            st.harvest_level.pop(lever, None)
        if out:
            st.last_action_tok_s = None
            st.settle_remaining = p.settle_cycles
            return (out, st)

    # ── harvest ───────────────────────────────────────────────────────────
    if st.bottleneck == Bottleneck.BANDWIDTH_BOUND:
        if not baseline_ready:
            out.append(Decision(
                "none", "observe",
                reason=f"bandwidth-bound, but only {st.baseline_samples} "
                       f"throughput sample(s) , need "
                       f"{p.min_baseline_samples} before a harvest can be "
                       f"judged"))
        else:
            out.extend(_harvest(snap, st, p))
    elif st.bottleneck == Bottleneck.GPU_BOUND:
        # The GPU is the limit: anything harvested earlier should come back.
        for lever in list(st.harvest_level):
            out.append(Decision(lever, "restore",
                                reason="GPU-bound again , the clocks are "
                                       "worth holding"))
            st.harvest_level.pop(lever, None)
        if out:
            st.settle_remaining = p.settle_cycles

    # ── speculative depth ─────────────────────────────────────────────────
    d = decide_draft_depth(snap, st, p)
    if d is not None:
        out.append(d)

    if not out:
        out.append(Decision("none", "hold",
                            reason=f"{st.bottleneck}, nothing to move"))
    return (out, st)


def _classify(snap: TunerSnapshot, current: str, p: TunerPolicy) -> str:
    """Latching bottleneck classification , power first, utilisation second.

    **`utilization.gpu` cannot tell waiting from working, and on this machine
    it is wrong in the direction that matters.** It reports the fraction of
    time a kernel was resident; a kernel stalled on a PCIe read of weights in
    T2 is resident the whole time. Measured live on this box 2026-08-20,
    mid-decode of the reference model: `utilization.gpu` **100%** while the
    board drew **49.8 W of a 300 W limit**. A classifier that trusted
    utilisation would call that GPU-bound, and the harvest rule , the one
    decision this engine exists to make , could then never fire on the exact
    workload it was written for.

    Power draw is the honest discriminator: a GPU actually executing dense
    matmuls pulls a large fraction of its board limit; one waiting on a link
    does not. Utilisation is kept only as the fallback for a node with no
    power telemetry, and the fallback says so by staying on `current` rather
    than inventing a verdict.

    Enter and release thresholds differ on purpose , our segments have flapped
    on single-sample crossings, and a controller that changes its mind every
    tick actuates more than it decides.
    """
    spilled = (snap.t2_overflow_mb or 0) > 0
    util = snap.gpu_util_pct
    frac = (snap.power_draw_w / snap.power_limit_w
            if snap.power_draw_w is not None and snap.power_limit_w
            else None)

    if frac is not None:
        if current == Bottleneck.GPU_BOUND:
            if frac >= p.compute_bound_power_release_frac:
                return Bottleneck.GPU_BOUND
            return (Bottleneck.BANDWIDTH_BOUND if spilled
                    else Bottleneck.UNKNOWN)
        if frac >= p.compute_bound_power_frac:
            return Bottleneck.GPU_BOUND
        if spilled:
            return Bottleneck.BANDWIDTH_BOUND
        # Idle-ish and nothing spilled: no claim either way.
        return Bottleneck.UNKNOWN

    if util is None:
        return current
    if current == Bottleneck.GPU_BOUND:
        if util < p.gpu_bound_release_pct:
            return Bottleneck.BANDWIDTH_BOUND if spilled else Bottleneck.UNKNOWN
        return Bottleneck.GPU_BOUND
    if util >= p.gpu_bound_enter_pct:
        return Bottleneck.GPU_BOUND
    return Bottleneck.BANDWIDTH_BOUND if spilled else current


def _harvest(snap: TunerSnapshot, st: TunerState, p: TunerPolicy) -> list[Decision]:
    """Give back what the bottleneck cannot use , the SM clock.

    The lever here is the clock, not the power limit, and that is a correction
    rather than a preference. Under power-first classification a card near its
    board limit is by definition compute-bound, so a power-limit harvest could
    only ever fire on a machine this engine had just called GPU-bound , dead
    code with a plausible docstring.

    What actually happens on a bandwidth-bound decode, measured on this box
    2026-08-20: the SMs sit at 2685 MHz drawing 49.8 W of a 300 W limit while
    they wait on PCIe reads of weights in T2. The watts are not the waste , the
    card is barely using them. The clock is: it is being held at boost to wait.

    Whether trimming it is free is a question this engine answers by
    experiment, not by assertion: the decision carries `verify`, the caller
    re-measures after the settle window, and a throughput regression restores
    the lever and freezes it. Dequantisation does run on those SMs, so "the
    clock is free because the link is the bottleneck" is a hypothesis, and the
    guard is what makes acting on it safe.
    """
    lever = "gpu_clocks_locked"
    if lever in st.frozen:
        return [Decision("none", "observe",
                         reason=f"{lever} frozen for {st.frozen[lever]} more "
                                f"tick(s) after an earlier harvest regressed")]
    if not snap.sm_clock_mhz or not snap.sm_clock_max_mhz:
        return [Decision("none", "observe",
                         reason="no clock telemetry , cannot tell whether the "
                                "card is holding boost while it waits")]
    # Nothing to give back if it is not actually holding a high clock.
    if snap.sm_clock_mhz < snap.sm_clock_max_mhz * p.harvest_floor_frac:
        return [Decision("none", "observe",
                         reason=f"SM clock {snap.sm_clock_mhz:.0f} MHz is "
                                f"already well under the "
                                f"{snap.sm_clock_max_mhz:.0f} MHz ceiling , "
                                f"nothing held that is not used")]

    level = st.harvest_level.get(lever, 1.0)
    nxt = level - p.harvest_step_frac
    if nxt < p.harvest_floor_frac:
        return [Decision("none", "hold",
                         reason=f"{lever} already harvested to "
                                f"{level:.0%} of baseline , floor reached")]
    target_mhz = int(snap.sm_clock_max_mhz * nxt)
    st.harvest_level[lever] = nxt
    st.settle_remaining = p.settle_cycles
    st.last_action_tok_s = st.baseline_tok_s
    _draw = (f"{snap.power_draw_w:.0f}W of {snap.power_limit_w:.0f}W"
             if snap.power_draw_w is not None and snap.power_limit_w
             else "power unknown")
    return [Decision(
        lever, "harvest", value=target_mhz,
        reason=f"decode is bandwidth-bound ({snap.t2_overflow_mb:.0f} MB "
               f"streaming from T2, card drawing {_draw}) while the SM clock "
               f"holds {snap.sm_clock_mhz:.0f} MHz , boost is being spent "
               f"waiting on the link, not computing",
        verify=True)]


def decide_draft_depth(snap: TunerSnapshot, st: TunerState,
                       policy: TunerPolicy | None = None) -> "Decision | None":
    """Speculative depth from measured acceptance, not from a swept constant.

    The 2026-08-05 sweep found the curve is non-monotonic (2:5.15, 3:5.58,
    4:6.50, 6:4.40, 8:5.76 tok/s), which is exactly what "the right depth is a
    function of state nobody is reading" looks like. Acceptance is that state,
    and the engine reports it on every response.

    Depth 0 is left alone: the engine reports acceptance 1.0 when nothing was
    drafted, and reading that as "drafting is working perfectly" is the
    confusion `gb_bench_spec.py` already had to guard against.

    Public because the proxy uses THIS rule on its own: it is the component
    that both sees every response's acceptance and can stamp a depth on the
    next request, and it has no business deciding anything about power limits.
    """
    p = policy or TunerPolicy()
    if "mtp_draft_n" in st.frozen:
        return None
    depth, acc = snap.draft_depth, snap.draft_accept_pct
    if not depth or acc is None:
        return None
    if acc < p.draft_accept_low_pct and depth > p.draft_depth_min:
        return Decision("mtp_draft_n", "set", value=depth - 1,
                        reason=f"only {acc:.0f}% of drafted tokens are being "
                               f"accepted at depth {depth} , the rejected "
                               f"ones cost a verification slot each",
                        verify=True)
    if (acc > p.draft_accept_high_pct and depth < p.draft_depth_max
            and st.bottleneck == Bottleneck.BANDWIDTH_BOUND):
        return Decision("mtp_draft_n", "set", value=depth + 1,
                        reason=f"{acc:.0f}% acceptance at depth {depth} and "
                               f"the pass is bandwidth-bound , a deeper draft "
                               f"rides the same forward pass",
                        verify=True)
    return None
