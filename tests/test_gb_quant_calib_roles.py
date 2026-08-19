"""Tests for gb_quant_calib.layer_roles() — the torch-path role lookup that
feeds gb_quant.plan_quality()'s DP branch (missing_features.md item (j)).

Kept in its own file (not embedded in calibrate_sensitivity's cache
round-trip tests) because layer_roles() is deliberately a SEPARATE,
un-cached call — see its own docstring for why folding a role string into
calibrate_sensitivity's {layer: {bits: rel_err}} dict would break that
dict's float()-only JSON cache serialization.
"""
import torch.nn as nn

import gb_quant_calib as gqc


class _MambaMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = nn.Linear(64, 64, bias=False)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(64, 64, bias=False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = _MambaMixer()
        self.attn = _Attn()
        self.lm_head = nn.Linear(64, 100)


def test_layer_roles_matches_iter_quantizable_linears():
    m = _Model()
    expected_names = {name for name, _ in gqc._iter_quantizable_linears(m, gqc._DEFAULT_SKIP_MODULES)}
    roles = gqc.layer_roles(m)

    assert set(roles.keys()) == expected_names


def test_layer_roles_classifies_by_ancestor_class():
    m = _Model()
    roles = gqc.layer_roles(m)

    assert roles["mixer.out_proj"] == "ssm"
    assert roles["attn.qkv"] == "attn"


def test_layer_roles_respects_skip_modules_default():
    # lm_head is in _DEFAULT_SKIP_MODULES, so _iter_quantizable_linears never
    # yields it — layer_roles() must not include it either.
    m = _Model()
    roles = gqc.layer_roles(m)

    assert "lm_head" not in roles
