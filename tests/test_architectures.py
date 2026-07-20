import pytest
import torch
from torch import nn

from quipsharp.architectures import get_linear_groups, get_model_dims, MoEUnsupportedError


class _Attn(nn.Module):
    def __init__(self, has_k=True, has_v=True, v_is_none=False, has_norms=False, bias=False):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=bias)
        if has_k:
            self.k_proj = nn.Linear(8, 8, bias=bias)
        if has_v:
            self.v_proj = nn.Linear(8, 8, bias=bias) if not v_is_none else None
        self.o_proj = nn.Linear(8, 8, bias=False)
        if has_norms:
            self.q_norm = nn.RMSNorm(8)
            self.k_norm = nn.RMSNorm(8)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


class _MoEMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(8, 4)
        self.experts = nn.Parameter(torch.randn(4, 8, 16))


class _Layer(nn.Module):
    def __init__(self, attn=None, mlp=None):
        super().__init__()
        self.self_attn = attn if attn is not None else _Attn()
        self.mlp = mlp if mlp is not None else _MLP()


def _names(groups):
    return [[n for n, _ in g] for g in groups]


def test_llama_style_layer_groups():
    layer = _Layer()
    groups = _names(get_linear_groups(layer))
    assert groups == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]


def test_qwen2_style_bias_qkv_does_not_change_grouping():
    layer = _Layer(attn=_Attn(bias=True))
    groups = _names(get_linear_groups(layer))
    assert groups[0] == ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]


def test_qwen3_gemma_style_q_k_norm_are_skipped():
    layer = _Layer(attn=_Attn(has_norms=True))
    groups = get_linear_groups(layer)
    all_modules = [m for g in groups for _, m in g]
    assert all(isinstance(m, nn.Linear) for m in all_modules)
    assert _names(groups)[0] == ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]


def test_gemma4_kv_shared_layer_omits_k_and_v():
    layer = _Layer(attn=_Attn(has_k=False, has_v=False))
    groups = _names(get_linear_groups(layer))
    assert groups == [
        ["self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]


def test_gemma4_attention_k_eq_v_sets_v_proj_none():
    layer = _Layer(attn=_Attn(v_is_none=True))
    groups = _names(get_linear_groups(layer))
    assert groups[0] == ["self_attn.q_proj", "self_attn.k_proj"]


def test_moe_mlp_raises():
    layer = _Layer(mlp=_MoEMLP())
    with pytest.raises(MoEUnsupportedError):
        get_linear_groups(layer)


def test_moe_top_level_router_raises():
    layer = _Layer()
    layer.router = nn.Linear(8, 4)
    with pytest.raises(MoEUnsupportedError):
        get_linear_groups(layer)


def test_get_model_dims():
    class Cfg:
        num_hidden_layers = 12
        hidden_size = 256
        intermediate_size = 1024

    assert get_model_dims(Cfg()) == (12, 256, 1024)
