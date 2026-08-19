"""
gb_quant_roles.py — one canonical tensor-role taxonomy, shared by both
quantization paths in this repo.

Why this exists: missing_features.md item (j) (component-sensitivity-gated
mixed-precision quantization) was filed against the torch calibration path
(gb_quant_calib.py/gb_quant_dp.py), but the reference decode-throughput
workload (research/notes/final_report_greenboost-decode-throughput-3b65b4.md)
is a GGUF served by llama.cpp — a disjoint code path that never touches
torch calibration at all. Both paths need the SAME answer to "what kind of
tensor is this, and how sensitive is it to aggressive quantization" — one
vocabulary here, one derivation per path:

    role_from_gguf_tensor()  — exact, from llama.cpp's own tensor naming
                                (blk.N.ssm_out, attn_qkv, ffn_gate, …).
                                Consumed by gb_gguf_plan.py (GGUF path).
    role_from_torch_module() — from the ancestor nn.Module's class, not a
                                name substring (item (j)'s own objection to
                                substring heuristics is valid HERE, where
                                role genuinely isn't encoded in the name).
                                Consumed by gb_quant_calib.py (torch path).

Evidence for the SSM/recurrent floor (see the vault notes cited in
research/notes/final_report_greenboost-decode-throughput-3b65b4.md §7-8):
Gated-DeltaNet/SSM tensors — the recurrent output projection and
update-coefficient weights specifically — are markedly more quantization-
sensitive than dense/attention tensors; a related-model community quant plan
flags the SSM output projection as "EXTREMELY quantization-sensitive" and
update-coefficient weights as "VERY quantization-sensitive", and a separate
SSM-quantization paper documents a zero-ratio-collapse failure mode under
standard post-training correction that plain transformer layers don't
exhibit, because errors compound through the recurrence rather than staying
local to one layer's output.

This module holds only FLOORS (a minimum precision below which a role must
not be pushed) — it does not decide the aggressive TARGET precision for
tolerant roles (ffn/attn/output/embed). That's the per-tensor byte-budget
search in gb_gguf_plan.py (GGUF path) or the existing DP planner
(gb_quant_dp.py, torch path); this module only tells them where the floor
is.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

# ---------------------------------------------------------------------------
# Canonical role vocabulary
# ---------------------------------------------------------------------------

ROLES: tuple = ("ssm", "mtp", "norm", "embed", "output", "attn", "ffn", "other")

# Minimum ggml type (llama-quantize's --tensor-type / --tensor-type-file
# vocabulary) a tensor of this role must not be pushed below. None = no
# floor — the byte-budget search may use the full candidate range.
#
# "ssm"/"mtp" get an explicit floor per the evidence above; "norm" is
# defensive documentation only (llama-quantize already refuses to quantize
# 1-D tensors below F32/F16 regardless of what's requested, per its own
# quantize.cpp — this floor can never actually bind, but records the
# intent so a future change to that behavior doesn't silently regress it).
ROLE_FLOORS_GGUF: dict = {
    "ssm": "q8_0",
    "mtp": "q8_0",
    "norm": "f32",
    "embed": None,
    "output": None,
    "attn": None,
    "ffn": None,
    "other": None,
}

# Same floors in gb-quant's own precision vocabulary (16/fp8/8/4/tq3/tq2),
# for the torch path (gb_quant.py/gb_quant_dp.py).
ROLE_FLOORS_BITS: dict = {
    "ssm": 8,
    "mtp": 8,
    "norm": 16,
    "embed": None,
    "output": None,
    "attn": None,
    "ffn": None,
    "other": None,
}

# Approximate most-precise → least-precise ordering for each vocabulary,
# used only to compare a candidate against a role's floor. Good enough for
# floor-comparison purposes; not a claim about exact relative quality
# between adjacent k-quant variants (e.g. Q5_K vs Q5_1 vary by real-world
# calibration, not just nominal bits).
_GGUF_TYPE_ORDER: tuple = (
    "f32", "f16", "bf16", "q8_0",
    "q6_k", "q5_k", "q5_1", "q5_0", "q4_k", "q4_1", "q4_0",
    "iq4_nl", "iq4_xs", "q3_k",
    "iq3_s", "iq3_xxs", "q2_k",
    "iq2_s", "iq2_xs", "iq2_xxs",
    "iq1_m", "iq1_s", "tq2_0", "tq1_0",
)

# Matches plan_bits_dp()'s own default `candidates` order (gb_quant_dp.py).
_BITS_ORDER: tuple = (16, "fp8", 8, 4, "tq3", "tq2")


def meets_floor_gguf(candidate_type: str, role: str) -> bool:
    """True if `candidate_type` (a ggml_type name, any case) is at or above
    `role`'s floor — i.e. safe to assign. No floor for the role, or an
    unrecognized type on either side (never seen before, e.g. a new ggml
    type), returns True rather than blocking on an unknown quantity."""
    floor = ROLE_FLOORS_GGUF.get(role)
    if floor is None:
        return True
    candidate_type = candidate_type.lower()
    try:
        return _GGUF_TYPE_ORDER.index(candidate_type) <= _GGUF_TYPE_ORDER.index(floor)
    except ValueError:
        return True


def meets_floor_bits(candidate_bits, role: str) -> bool:
    """Same comparison for gb-quant's bits vocabulary (16/fp8/8/4/tq3/tq2)."""
    floor = ROLE_FLOORS_BITS.get(role)
    if floor is None:
        return True
    try:
        return _BITS_ORDER.index(candidate_bits) <= _BITS_ORDER.index(floor)
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# GGUF path — exact, from llama.cpp's own tensor naming
# ---------------------------------------------------------------------------

def role_from_gguf_tensor(name: str) -> str:
    """Classify a GGUF tensor by role from its llama.cpp-assigned name.

    Names are either `blk.<N>.<root>[.<sub>].weight|bias` (per-layer) or a
    bare top-level root (`token_embd.weight`, `output.weight`,
    `output_norm.weight`). This is exact — llama.cpp's own tensor-naming
    convention already encodes role in the name — unlike the torch path,
    where a bare nn.Linear name carries no such information (see
    role_from_torch_module below).

    Order matters: more specific checks run before more general ones (e.g.
    "norm" — a substring match — must run before the "attn"/"ffn" prefix
    checks, or "attn_q_norm"/"ffn_norm"/"post_attention_norm" would be
    mis-bucketed as attn/ffn instead of norm).
    """
    parts = name.split(".")
    if parts and parts[0] == "blk" and len(parts) > 2:
        root = parts[2]
    else:
        # bare top-level tensor (token_embd.weight, output.weight,
        # output_norm.weight): the root is just the first dot-component.
        root = parts[0]

    root_l = root.lower()

    if root_l.startswith("ssm"):
        return "ssm"
    if root_l.startswith("nextn"):
        return "mtp"
    if "norm" in root_l:
        return "norm"
    if root_l.startswith("token_embd") or root_l == "embed":
        return "embed"
    if root_l == "output":
        return "output"
    if root_l.startswith("attn"):
        return "attn"
    if root_l.startswith("ffn"):
        return "ffn"
    return "other"


# ---------------------------------------------------------------------------
# Torch path — from the ancestor module's CLASS, not a name substring
# ---------------------------------------------------------------------------

# Class-name keyword → role, checked nearest-ancestor-first so a specific
# mixer class (e.g. "GatedDeltaNetMixer") wins over a generic outer class
# (e.g. "DecoderLayer") higher up the module tree.
_TORCH_CLASS_KEYWORDS: tuple = (
    ("mamba", "ssm"), ("deltanet", "ssm"), ("recurrent", "ssm"),
    ("mtp", "mtp"), ("nextn", "mtp"), ("draft", "mtp"),
    ("norm", "norm"),
    ("embed", "embed"),
    ("lmhead", "output"), ("lm_head", "output"), ("outputlayer", "output"),
    ("attention", "attn"), ("attn", "attn"),
    ("mlp", "ffn"), ("feedforward", "ffn"), ("ffn", "ffn"),
)

# Leaf-name substrings confident enough to decide role without walking the
# module tree at all (standard HF naming, not a guess — item (j)'s
# substring-heuristic objection was about SSM-vs-not being invisible in a
# name; it does not apply to these, which really are named this way
# everywhere).
_TORCH_NAME_KEYWORDS: tuple = (
    ("lm_head", "output"), ("embed_tokens", "embed"), ("wte", "embed"),
)


def role_from_torch_module(root: "nn.Module", name: str) -> str:
    """Classify an nn.Linear (or any leaf module) at dotted path `name`
    within `root` by walking up to its nearest ancestor module and
    inspecting that ancestor's CLASS name — not the leaf's own name string.

    This is the torch-path counterpart to role_from_gguf_tensor(); it exists
    because a torch checkpoint's `nn.Linear` names (e.g. "layers.3.linear1")
    carry no role information the way a GGUF tensor's name does. Role lives
    in which kind of block CONTAINS the linear (a Mamba2Mixer vs. an
    Attention vs. an MLP), so this walks the module tree rather than
    guessing from the leaf name — the gap item (j) actually identified.

    Falls back to "other" when no ancestor class or leaf-name keyword
    matches (e.g. a plain nn.Module hierarchy with generic class names)."""
    for keyword, role in _TORCH_NAME_KEYWORDS:
        if keyword in name:
            return role

    parts = name.split(".")
    modules_by_path = dict(root.named_modules())
    # Nearest ancestor first: drop trailing components one at a time.
    for i in range(len(parts) - 1, 0, -1):
        ancestor_path = ".".join(parts[:i])
        ancestor = modules_by_path.get(ancestor_path)
        if ancestor is None:
            continue
        class_name = type(ancestor).__name__.lower()
        for keyword, role in _TORCH_CLASS_KEYWORDS:
            if keyword in class_name:
                return role
    return "other"
