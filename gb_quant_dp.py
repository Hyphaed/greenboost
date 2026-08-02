# SPDX-License-Identifier: GPL-2.0-only
"""DP (dynamic-programming) budget-optimal mixed-precision layer planner.

missing_features.md item (c): `gb_quant.plan_quality()`'s per-layer ladder
walk is greedy — each layer independently picks the lowest precision that
meets ITS OWN error ceiling, so a low-sensitivity layer can never be traded
down to buy a high-sensitivity layer more precision within a fixed TOTAL
budget. This module adds an alternative planner that solves that as a real
budget-constrained optimization: minimize total quantization loss subject to
a total byte budget, via the Pareto-pruned knapsack DP ported from
AutoRound's `choose_bits_per_layer_with_path`
(`third_party/auto_round/auto_round/auto_scheme/delta_loss.py:1214`).

Opt-in only (`GB_QUANT_DP_PLAN=1` inside `gb_quant.plan_quality`) — the
default stays the greedy ladder walk. Pure Python, no torch/auto_round
import: the DP itself doesn't need either. It operates on the same
`{layer: {bits: rel_err}}` sensitivity dict
`gb_quant_calib.calibrate_sensitivity()` already produces, and on
`gb_quant._BYTES_PER_PARAM` for the cost axis.

The upstream `choose_bits_per_layer_with_path` docstring warns that an
unbounded state space (one DP state per distinct cumulative-bit sum) can
exceed 70 GB of RAM on models with many, size-incommensurate layers — the
same beam-width cap (`max_states`, Pareto-pruned then uniformly subsampled)
is carried over here, not just the algorithm.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

# DP state space is keyed by an integer "cost unit" — 1 unit == 1 MiB of
# quantized footprint. Fine enough not to distort a real budget, coarse
# enough to bound state count (a pure-float byte key would make every
# fractional GB distinct, blowing up cardinality for no benefit).
_UNITS_PER_GB = 1024


def _default_beam_width(n_layers: int) -> Optional[int]:
    """max_states cap — mirrors the upstream comment (unbounded search can
    exceed 70 GB of RAM). Scales down as layer count grows: more layers means
    more distinct cumulative sums, so a tighter per-layer cap holds total
    memory roughly constant. GB_QUANT_DP_MAX_STATES overrides explicitly."""
    override = os.environ.get("GB_QUANT_DP_MAX_STATES")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if n_layers <= 0:
        return 4096
    return max(256, min(4096, 200_000 // max(1, n_layers)))


def choose_bits_per_layer_with_path(
    layers: Dict[str, list], P: int, max_states: Optional[int] = None,
) -> "tuple[Optional[float], Optional[list]]":
    """Pareto-pruned knapsack DP (ported verbatim from AutoRound's
    `auto_scheme/delta_loss.py:choose_bits_per_layer_with_path` — see module
    docstring for the source).

    Args:
        layers: {layer_name: [(scheme, bits_cost:int, loss_cost:float,
                 layer_names:tuple), ...]} — one option list per layer.
        P: total integer cost-unit budget.
        max_states: beam width cap after Pareto-pruning each layer.

    Returns:
        (min_loss, best_path) where best_path is a list of
        (layer_names, scheme) tuples, or (None, None) if no combination of
        options across all layers fits within P.
    """
    dp: Dict[int, Tuple[float, tuple]] = {0: (0.0, ())}
    for layer_name, opts in layers.items():
        new_dp: Dict[int, Tuple[float, tuple]] = {}
        for cur_params, (cur_loss, cur_path) in dp.items():
            for opt in opts:
                scheme, bits_cost, loss_cost, layer_names = opt
                np_total = cur_params + bits_cost
                if np_total > P:
                    continue
                new_loss = cur_loss + loss_cost
                new_path = cur_path + ((layer_names, scheme),)
                if np_total not in new_dp or new_loss < new_dp[np_total][0]:
                    new_dp[np_total] = (new_loss, new_path)
        if not new_dp:
            return None, None
        # Pareto pruning: drop states dominated by a cheaper-or-equal state
        # with lower-or-equal loss.
        items = sorted(new_dp.items(), key=lambda x: x[0])
        pruned: Dict[int, Tuple[float, tuple]] = {}
        best_loss_so_far = float("inf")
        for params_val, (loss_val, path_val) in items:
            if loss_val < best_loss_so_far:
                pruned[params_val] = (loss_val, path_val)
                best_loss_so_far = loss_val
        if max_states is not None and len(pruned) > max_states:
            if max_states <= 1:
                best_k = min(pruned.keys(), key=lambda k: pruned[k][0])
                pruned = {best_k: pruned[best_k]}
            else:
                sorted_keys = sorted(pruned.keys())
                n = len(sorted_keys)
                step = (n - 1) / (max_states - 1)
                selected: Dict[int, Tuple[float, tuple]] = {}
                for i in range(max_states):
                    idx = int(round(i * step))
                    if idx >= n:
                        idx = n - 1
                    k = sorted_keys[idx]
                    selected[k] = pruned[k]
                pruned = selected
        dp = pruned

    best_params = min(dp.keys(), key=lambda k: dp[k][0])
    best_loss, best_path = dp[best_params]
    return best_loss, list(best_path)


def plan_bits_dp(
    sensitivity: Dict[str, Dict],
    layer_sizes: Dict[str, int],
    bytes_per_param: Dict,
    budget_gb: float,
    candidates: "tuple" = (16, "fp8", 8, 4, "tq3", "tq2"),
    excluded: "Optional[Dict[str, tuple]]" = None,
    max_states: "Optional[int]" = None,
) -> Dict[str, object]:
    """Solve the budget-constrained per-layer bit assignment via DP.

    Args:
        sensitivity: {layer: {bits: rel_err}} — from
            `gb_quant_calib.calibrate_sensitivity`. A layer/bits pair absent
            from this dict costs `loss=1.0` (the same "unknown = worst case"
            sentinel `plan_quality`'s greedy walk already uses via
            `layer_s.get(candidate, 1.0)`), except bits==16 which always
            costs `loss=0.0` (bf16 is the reference precision).
        layer_sizes: {layer: n_params}.
        bytes_per_param: `gb_quant._BYTES_PER_PARAM` (bits -> GiB/param scale).
        budget_gb: total byte budget across ALL layers (GiB).
        candidates: precision options considered per layer.
        excluded: {layer: (bits, ...)} — bits to drop for a specific layer
            (e.g. int4 when `in_features % group_size != 0`, mirroring the
            greedy walk's own group-size fallback).
        max_states: beam width; None -> `_default_beam_width(len(layer_sizes))`.

    Returns:
        {"per_layer_bits": {name: bits}, "total_loss": float | None,
         "total_gb": float | None, "feasible": bool}.
        On infeasibility (no combination fits `budget_gb`, e.g. every layer's
        cheapest option alone already exceeds it), per_layer_bits is {} and
        feasible is False — callers should fall back to another planner
        rather than treat this as a crash.
    """
    names = list(layer_sizes.keys())
    if max_states is None:
        max_states = _default_beam_width(len(names))
    excluded = excluded or {}

    P = max(0, int(round(budget_gb * _UNITS_PER_GB)))
    layers: Dict[str, list] = {}
    for name in names:
        n = layer_sizes[name]
        layer_s = sensitivity.get(name, {})
        skip_bits = excluded.get(name, ())
        opts = []
        for bits in candidates:
            if bits in skip_bits or bits not in bytes_per_param:
                continue
            gb = n * bytes_per_param[bits] / 2 ** 30
            cost_units = max(0, int(round(gb * _UNITS_PER_GB)))
            loss = 0.0 if bits == 16 else layer_s.get(bits, 1.0)
            opts.append((bits, cost_units, float(loss), (name,)))
        if not opts:
            continue
        layers[name] = opts

    if not layers:
        return {"per_layer_bits": {}, "total_loss": None, "total_gb": None,
               "feasible": False}

    min_loss, path = choose_bits_per_layer_with_path(layers, P, max_states=max_states)
    if path is None:
        return {"per_layer_bits": {}, "total_loss": None, "total_gb": None,
               "feasible": False}

    per_layer_bits: Dict[str, object] = {}
    total_gb = 0.0
    for layer_names, scheme in path:
        for name in layer_names:
            per_layer_bits[name] = scheme
            total_gb += layer_sizes[name] * bytes_per_param[scheme] / 2 ** 30

    return {"per_layer_bits": per_layer_bits, "total_loss": min_loss,
           "total_gb": round(total_gb, 3), "feasible": True}
