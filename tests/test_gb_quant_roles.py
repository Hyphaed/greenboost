"""Tests for gb_quant_roles.py — the canonical tensor-role taxonomy shared by
the GGUF (llama-quantize) and torch (gb_quant_calib/gb_quant_dp) quantization
paths. CPU-only, no GGUF file, no CUDA.
"""
import torch
import torch.nn as nn

import gb_quant_roles as roles


# ---------------------------------------------------------------------------
# GGUF path — exact names, taken from a real read of the reference GGUF
# (Qwen3.6-27B-Fable-Fusion, Q4_K_M) via gguf-py.
# ---------------------------------------------------------------------------

def test_role_from_gguf_tensor_ssm_family():
    for name in ("blk.0.ssm_out.weight", "blk.0.ssm_alpha.weight",
                 "blk.0.ssm_beta.weight", "blk.0.ssm_conv1d.weight",
                 "blk.0.ssm_dt.bias", "blk.0.ssm_a", "blk.0.ssm_norm.weight"):
        assert roles.role_from_gguf_tensor(name) == "ssm", name


def test_role_from_gguf_tensor_mtp():
    assert roles.role_from_gguf_tensor("blk.64.nextn.eh_proj.weight") == "mtp"


def test_role_from_gguf_tensor_norm_catches_all_norm_variants():
    for name in ("blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight",
                 "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight",
                 "output_norm.weight"):
        assert roles.role_from_gguf_tensor(name) == "norm", name


def test_role_from_gguf_tensor_embed_and_output():
    assert roles.role_from_gguf_tensor("token_embd.weight") == "embed"
    assert roles.role_from_gguf_tensor("output.weight") == "output"
    # attn_output must NOT be mis-bucketed as "output" (generic-output check
    # is an exact-match, attn_output is caught by the attn prefix check).
    assert roles.role_from_gguf_tensor("blk.3.attn_output.weight") == "attn"


def test_role_from_gguf_tensor_attn_and_ffn():
    for name in ("blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
                 "blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight"):
        assert roles.role_from_gguf_tensor(name) == "attn", name
    for name in ("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"):
        assert roles.role_from_gguf_tensor(name) == "ffn", name


def test_role_from_gguf_tensor_unknown_falls_back_to_other():
    assert roles.role_from_gguf_tensor("blk.0.some_new_tensor.weight") == "other"
    assert roles.role_from_gguf_tensor("rope_freqs.weight") == "other"


# ---------------------------------------------------------------------------
# Floor comparisons
# ---------------------------------------------------------------------------

def test_meets_floor_gguf_protects_ssm():
    assert roles.meets_floor_gguf("q8_0", "ssm") is True
    assert roles.meets_floor_gguf("f16", "ssm") is True   # more precise than floor, fine
    assert roles.meets_floor_gguf("q4_k", "ssm") is False  # below the q8_0 floor
    assert roles.meets_floor_gguf("iq2_xxs", "ssm") is False


def test_meets_floor_gguf_no_floor_for_tolerant_roles():
    assert roles.meets_floor_gguf("iq2_xxs", "ffn") is True
    assert roles.meets_floor_gguf("iq1_s", "attn") is True


def test_meets_floor_bits_protects_ssm_and_mtp():
    assert roles.meets_floor_bits(8, "ssm") is True
    assert roles.meets_floor_bits(4, "ssm") is False
    assert roles.meets_floor_bits("tq2", "mtp") is False
    assert roles.meets_floor_bits(16, "norm") is True
    assert roles.meets_floor_bits(8, "norm") is False


def test_meets_floor_unknown_type_never_blocks():
    # An unrecognized ggml type/bits value shouldn't hard-block a caller —
    # only a *known* violation should.
    assert roles.meets_floor_gguf("some_future_type", "ssm") is True
    assert roles.meets_floor_bits("some_future_scheme", "ssm") is True


# ---------------------------------------------------------------------------
# Torch path — role from ancestor module CLASS, not the leaf's name string.
# ---------------------------------------------------------------------------

class _GatedDeltaNetMixer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.out_proj = nn.Linear(d, d)


class _SelfAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv_proj = nn.Linear(d, 3 * d)


class _MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_proj = nn.Linear(d, d)


class _DecoderLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mixer = _GatedDeltaNetMixer(d)
        self.self_attn = _SelfAttention(d)
        self.mlp = _MLP(d)
        self.input_layernorm = nn.LayerNorm(d)


class _TinyModel(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, d)
        self.layers = nn.ModuleList([_DecoderLayer(d)])
        self.lm_head = nn.Linear(d, 100)


def test_role_from_torch_module_uses_ancestor_class_not_leaf_name():
    m = _TinyModel()
    # None of these leaf names contain "ssm"/"attn"/"mlp" as a SUBSTRING in
    # a way item (j) could have detected — role must come from the ancestor
    # class, exactly the gap item (j) identified on the torch path.
    assert roles.role_from_torch_module(m, "layers.0.mixer.out_proj") == "ssm"
    assert roles.role_from_torch_module(m, "layers.0.self_attn.qkv_proj") == "attn"
    assert roles.role_from_torch_module(m, "layers.0.mlp.gate_proj") == "ffn"


def test_role_from_torch_module_leaf_name_keywords_win_for_standard_hf_names():
    m = _TinyModel()
    assert roles.role_from_torch_module(m, "lm_head") == "output"
    # embed_tokens is an nn.Embedding, not a Linear, but the classifier
    # should still resolve it correctly if ever called on one.
    assert roles.role_from_torch_module(m, "embed_tokens") == "embed"


def test_role_from_torch_module_unmatched_ancestor_is_other():
    class _Generic(nn.Module):
        def __init__(self):
            super().__init__()
            self.thing = nn.Linear(4, 4)

    m = _Generic()
    assert roles.role_from_torch_module(m, "thing") == "other"
