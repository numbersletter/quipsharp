"""Generic, introspection-based decoder-layer/linear-group discovery across
Llama2/3, Qwen2.5/Qwen3, and Gemma3/Gemma4(dense)-style architectures.

Doesn't import or branch on any specific transformers model class: every one
of these families names its attention/MLP Linear submodules identically
(self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj) and exposes
num_hidden_layers/hidden_size/intermediate_size on its config, so a single
attribute-presence-based walk covers all of them, including the
irregularities each family introduces (Qwen2's bias=True on q/k/v, Qwen3/
Gemma3/Gemma4's q_norm/k_norm, Gemma4's KV-shared layers that omit
k_proj/v_proj entirely) without needing per-family branches.
"""
from torch import nn


class MoEUnsupportedError(NotImplementedError):
    """Raised when a decoder layer has MoE routing (router/experts attrs).

    Expert weights in an MoE block are stored as raw nn.Parameter tensors,
    not nn.Linear, so a Linear-only quantizer would silently skip them,
    producing a model that looks fully quantized but isn't. Better to fail
    loudly here than let that happen quietly.
    """


def get_decoder_layers(model) -> list[nn.Module]:
    """The flat, ordered list of transformer decoder layers. All five
    target families expose this at model.model.layers."""
    return list(model.model.layers)


def get_model_dims(config) -> tuple[int, int, int]:
    """(num_layers, hidden_size, intermediate_size) -- attribute names
    confirmed consistent across LlamaConfig/Qwen2Config/Qwen3Config/
    Gemma3Config/Gemma4TextConfig."""
    return config.num_hidden_layers, config.hidden_size, config.intermediate_size


def _check_no_moe(layer: nn.Module, layer_index: int) -> None:
    mlp = getattr(layer, "mlp", None)
    if mlp is not None and (hasattr(mlp, "experts") or hasattr(mlp, "router")):
        raise MoEUnsupportedError(
            f"layer {layer_index}: mlp has an 'experts'/'router' attribute "
            "(MoE block) -- expert weights are packed nn.Parameter tensors, "
            "not nn.Linear, and quantizing them isn't implemented.")
    if hasattr(layer, "router") or hasattr(layer, "experts"):
        raise MoEUnsupportedError(
            f"layer {layer_index}: has a top-level 'router'/'experts' "
            "attribute (MoE block) -- not supported.")


def get_linear_groups(layer: nn.Module, layer_index: int = -1
                       ) -> list[list[tuple[str, nn.Linear]]]:
    """Group a decoder layer's target nn.Linear submodules by shared input
    activation: [q_proj,k_proj,v_proj] (only the ones present -- Gemma4's
    KV-shared layers omit k_proj/v_proj entirely, and attention_k_eq_v
    layers set v_proj=None), [o_proj], [gate_proj,up_proj], [down_proj].
    Names are relative to the layer, e.g. "self_attn.q_proj", so callers
    prefix them with the layer's own dotted path.

    Raises MoEUnsupportedError if the layer looks like an MoE block -- let
    it propagate rather than catching it (see the class docstring).
    """
    _check_no_moe(layer, layer_index)

    groups = []

    attn = getattr(layer, "self_attn", None)
    if attn is not None:
        qkv = []
        for name in ("q_proj", "k_proj", "v_proj"):
            mod = getattr(attn, name, None)
            if isinstance(mod, nn.Linear):
                qkv.append((f"self_attn.{name}", mod))
        if qkv:
            groups.append(qkv)
        o_proj = getattr(attn, "o_proj", None)
        if isinstance(o_proj, nn.Linear):
            groups.append([("self_attn.o_proj", o_proj)])

    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        gateup = []
        for name in ("gate_proj", "up_proj"):
            mod = getattr(mlp, name, None)
            if isinstance(mod, nn.Linear):
                gateup.append((f"mlp.{name}", mod))
        if gateup:
            groups.append(gateup)
        down_proj = getattr(mlp, "down_proj", None)
        if isinstance(down_proj, nn.Linear):
            groups.append([("mlp.down_proj", down_proj)])

    return groups
