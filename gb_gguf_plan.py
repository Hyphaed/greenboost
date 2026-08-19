"""
gb_gguf_plan.py — GGUF per-tensor quantization plan (missing_features.md
item (j), component-sensitivity-gated mixed-precision quantization, GGUF/
llama-quantize path).

Why this file exists, and why it's separate from gb_quant.py/gb_quant_dp.py:
item (j) was originally filed against the torch calibration path
(gb_quant_calib.py/gb_quant_dp.py plan_bits_dp), but the reference decode-
throughput workload (research/notes/final_report_greenboost-decode-
throughput-3b65b4.md) is a GGUF served by llama.cpp — a disjoint code path
that never touches torch calibration. The mechanism this file wires up
already exists in the vendored engine and was simply never used:
llama-quantize's own `--tensor-type`/`--tensor-type-file` flags. See
gb_synapse._quantize_gguf() for the wiring; this module only builds the
plan that feeds it.

The Q8_0 source requirement: no f16/bf16 GGUF exists upstream for the
reference model family (highest available is Q8_0) — SSM precision cannot
be raised above the already-shipped Q4_K_M quant's own Q4_K/F32 mix without
requantizing from something higher, so plan_from_source() is designed to
run against a Q8_0 (or better) source file, not the Q4_K_M one. This has a
useful side effect: at a Q8_0 baseline, ssm/mtp tensors are ALREADY at
their floor (gb_quant_roles.ROLE_FLOORS_GGUF["ssm"] ==
ROLE_FLOORS_GGUF["mtp"] == "q8_0") — protecting them costs nothing, because
the plan simply never touches them; only the tolerant roles (ffn/attn/
output/embed) get pushed down the ladder to make room.

This is a greedy, role-priority allocator, NOT a byte-optimal search — see
plan_from_source()'s docstring for the exact algorithm and its stated scope
limitation relative to gb_quant_dp's DP core on the torch path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import gb_quant_roles

# Aggressiveness ladder for tolerant roles, MOST -> LEAST precise. Index 0
# MUST be "q8_0" — plan_from_source() assumes every tensor's SOURCE type is
# q8_0 (see module docstring) and computes every candidate as a byte-scale
# relative to that baseline, not an absolute size.
#
# Stops at IQ2_XXS — the decode-throughput report's own IQ2_M reference
# point (12.11 GB, already verified serving successfully) sits near this
# edge; going lower needs an imatrix (now buildable — see
# gb_synapse.build_engine()'s llama-imatrix target) and real measurement
# before it's worth adding as a planning candidate.
_TOLERANT_LADDER: tuple = (
    "q8_0", "q6_k", "q5_k", "q5_1", "q5_0", "q4_k", "q4_1", "q4_0",
    "iq4_nl", "iq4_xs", "q3_k", "iq3_s", "iq3_xxs", "q2_k",
    "iq2_s", "iq2_xs", "iq2_xxs",
)

# Roles pushed down the ladder when the budget is tight, most-tolerant-and-
# highest-byte-share role first. ffn is both the most tolerant role per the
# vault evidence AND the dominant byte contributor in the reference model
# (~56%, verified via gguf-py against the real served GGUF) — squeezing it
# first is the whole thesis of item (j), not an arbitrary ordering choice.
# ssm/mtp/norm are deliberately absent: their floor equals the Q8_0 source
# baseline, so they are never touched (see module docstring).
_COMPRESSIBLE_ROLE_PRIORITY: tuple = ("ffn", "attn", "output", "embed", "other")

# Approximate bytes/element for each ladder type, RELATIVE scaling only
# (plan_from_source uses ratios against "q8_0", never an absolute size) —
# nominal bits/8 the same way llama.cpp's own quantize size estimator
# rounds k-quants' true ~4.5/5.5/6.5-bit-effective packing. The actual
# assigned size is whatever llama-quantize really produces; verify with
# `llama-quantize --dry-run` before committing a real requantize pass (see
# gb_synapse._quantize_gguf()'s dry_run parameter).
_BYTES_PER_ELEM_GGUF: Dict[str, float] = {
    "q8_0": 1.0625, "q6_k": 0.820, "q5_k": 0.6875, "q5_1": 0.6875, "q5_0": 0.625,
    "q4_k": 0.5625, "q4_1": 0.625, "q4_0": 0.5625,
    "iq4_nl": 0.5625, "iq4_xs": 0.535, "q3_k": 0.4375,
    "iq3_s": 0.4375, "iq3_xxs": 0.343, "q2_k": 0.328,
    "iq2_s": 0.281, "iq2_xs": 0.266, "iq2_xxs": 0.258,
}


def _scale(ggml_type: str, ladder: tuple) -> float:
    """Byte-size ratio of `ggml_type` relative to `ladder[0]` (the assumed
    source/baseline type — see plan_from_source()'s docstring for why that
    must be "q8_0" in the default ladder). Takes `ladder` explicitly
    (rather than closing over the module-level default) so a caller — a
    test, or a future refinement — can pass a different ladder/baseline
    without _BYTES_PER_ELEM_GGUF silently normalizing against the wrong
    reference."""
    return _BYTES_PER_ELEM_GGUF[ggml_type] / _BYTES_PER_ELEM_GGUF[ladder[0]]


@dataclass
class GGUFQuantPlan:
    tensor_types: Dict[str, str]        # {tensor_name: ggml_type} — OVERRIDES ONLY; a
                                         # tensor absent from this dict keeps the source's
                                         # own type (llama-quantize's default per-tensor
                                         # behavior for any tensor not named in --tensor-type-file).
    role_breakdown: Dict[str, dict] = field(default_factory=dict)  # {role: {bytes_gb, floor_type}}
    estimated_bytes: int = 0
    budget_bytes: int = 0
    fits_budget: bool = False


def read_gguf_tensor_inventory(path: "str | Path") -> List[Tuple[str, int]]:
    """[(tensor_name, n_bytes)] for every tensor in the GGUF at `path`, via
    gguf-py. Pass a Q8_0 (or better) source — see this module's own
    docstring for why the Q4_K_M file this project already ships cannot be
    the source for this plan."""
    from gguf import GGUFReader
    r = GGUFReader(str(path))
    return [(t.name, int(t.n_bytes)) for t in r.tensors]


def plan_from_source(
    inventory: List[Tuple[str, int]],
    budget_bytes: int,
    role_priority: tuple = _COMPRESSIBLE_ROLE_PRIORITY,
    ladder: tuple = _TOLERANT_LADDER,
) -> GGUFQuantPlan:
    """Greedy role-priority ladder walk against a byte budget.

    Roles NOT in `role_priority` (ssm/mtp/norm) are never touched — their
    floor equals the Q8_0 source baseline this plan assumes, so leaving
    them alone already satisfies it (see module docstring).

    Algorithm: compute how many total bytes must be saved
    (`max(0, total_source - budget)`), then ask each role, IN PRIORITY
    ORDER, to give up as much of that as it can (capped at its own
    max-compression savings) before asking the next role at all — i.e. the
    first role in `role_priority` (ffn) is squeezed FIRST and hardest;
    later roles (attn, then output, then embed) are only touched once
    every earlier role has already given everything it safely can. Within
    a role, the LEAST aggressive ladder type that still delivers the
    required savings is picked (never over-compress past what's needed).
    This is a greedy allocator, not a byte-optimal search: unlike
    gb_quant_dp.plan_bits_dp on the torch path, it cannot trade bytes back
    from a later role to relax an earlier one once decided. Good enough
    for a first, measurable cut; a future refinement could generalize
    gb_quant_dp's DP core to the ggml-type vocabulary if this greedy
    result's quality/byte trade turns out not to be good enough once
    measured against the gb_aviary gates.
    """
    by_role: Dict[str, List[Tuple[str, int]]] = {}
    for name, n_bytes in inventory:
        role = gb_quant_roles.role_from_gguf_tensor(name)
        by_role.setdefault(role, []).append((name, n_bytes))

    role_bytes: Dict[str, float] = {}
    fixed_gb = 0.0
    for role, tensors in by_role.items():
        if role not in role_priority:
            b = sum(n for _, n in tensors) / 2 ** 30
            role_bytes[role] = b
            fixed_gb += b

    present_priority = [r for r in role_priority if by_role.get(r)]
    source_gb = {r: sum(n for _, n in by_role[r]) / 2 ** 30 for r in present_priority}
    total_source_gb = fixed_gb + sum(source_gb.values())
    budget_gb = budget_bytes / 2 ** 30
    savings_needed = max(0.0, total_source_gb - budget_gb)

    tensor_types: Dict[str, str] = {}
    for role in present_priority:
        src = source_gb[role]
        max_compressed = src * _scale(ladder[-1], ladder)
        take = min(savings_needed, src - max_compressed)
        target_gb = src - take

        chosen_idx, chosen_gb = len(ladder) - 1, max_compressed
        for idx in range(len(ladder)):
            candidate = src * _scale(ladder[idx], ladder)
            if candidate <= target_gb + 1e-9:
                chosen_idx, chosen_gb = idx, candidate
                break

        savings_needed = max(0.0, savings_needed - (src - chosen_gb))
        role_bytes[role] = chosen_gb
        if chosen_idx > 0:
            for name, _ in by_role[role]:
                tensor_types[name] = ladder[chosen_idx]

    estimated_bytes = int(sum(role_bytes.values()) * 2 ** 30)
    return GGUFQuantPlan(
        tensor_types=tensor_types,
        role_breakdown={
            role: {"bytes_gb": round(b, 3),
                  "floor_type": gb_quant_roles.ROLE_FLOORS_GGUF.get(role)}
            for role, b in role_bytes.items()
        },
        estimated_bytes=estimated_bytes,
        budget_bytes=budget_bytes,
        fits_budget=estimated_bytes <= budget_bytes,
    )


def write_tensor_type_file(plan: GGUFQuantPlan, path: "str | Path") -> Path:
    """llama-quantize --tensor-type-file format: one `tensor_name type`
    pair per line (space- or newline-separated per its own --help text)."""
    path = Path(path)
    path.write_text(
        "\n".join(f"{name} {ggml_type}" for name, ggml_type in sorted(plan.tensor_types.items()))
        + "\n"
    )
    return path
