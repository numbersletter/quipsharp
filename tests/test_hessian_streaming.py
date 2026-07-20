import torch
from torch import nn

from quipsharp.hessian import collect_hessians_streaming, HessianAccumulator


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class _MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)


class _ToyLayer(nn.Module):
    """A trivial stand-in for a transformer decoder layer: no attention math,
    no position embeddings -- just enough Linear structure to exercise the
    per-layer streaming/hooking/freeing mechanics without needing a real
    transformers model or a GPU."""

    def __init__(self, d, h):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MLP(d, h)

    def forward(self, x):
        attn_out = self.self_attn.o_proj(
            self.self_attn.q_proj(x) + self.self_attn.k_proj(x) + self.self_attn.v_proj(x))
        x = x + attn_out
        mlp_out = self.mlp.down_proj(torch.nn.functional.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x))
        return x + mlp_out


def _get_groups(layer, layer_index):
    return [
        [("self_attn.q_proj", layer.self_attn.q_proj),
         ("self_attn.k_proj", layer.self_attn.k_proj),
         ("self_attn.v_proj", layer.self_attn.v_proj)],
        [("self_attn.o_proj", layer.self_attn.o_proj)],
        [("mlp.gate_proj", layer.mlp.gate_proj), ("mlp.up_proj", layer.mlp.up_proj)],
        [("mlp.down_proj", layer.mlp.down_proj)],
    ]


def _forward_layer_fn(layer, hidden_states, layer_index):
    with torch.no_grad():
        return layer(hidden_states)


def test_streaming_matches_batch_hessian_and_frees_layers(tmp_path):
    torch.manual_seed(0)
    d, h, num_layers = 16, 32, 3
    layers = [_ToyLayer(d, h) for _ in range(num_layers)]
    hidden_states = torch.randn(20, 4, d)  # (N sequences, seqlen, hidden)

    model_name = "toy_model"
    out = collect_hessians_streaming(layers, hidden_states, _forward_layer_fn,
                                      _get_groups, device="cpu", model_name=model_name,
                                      root=str(tmp_path), batch_size=6)
    assert out.shape == hidden_states.shape

    # ground truth: replay the same layer-by-layer propagation manually,
    # accumulating H the straightforward (non-streaming) way, and confirm
    # collect_hessians_streaming's saved Hessians match exactly.
    x = hidden_states
    for layer_index, layer in enumerate(layers):
        acc_qkv = HessianAccumulator(d)
        acc_qkv.update(x)
        acc_o = HessianAccumulator(d)

        def make_hook(acc):
            def hook(module, inputs, output):
                acc.update(inputs[0])
            return hook

        h_o = layer.self_attn.o_proj.register_forward_hook(make_hook(acc_o))
        acc_down = HessianAccumulator(h)
        h_down = layer.mlp.down_proj.register_forward_hook(make_hook(acc_down))
        with torch.no_grad():
            x = layer(x)
        h_o.remove()
        h_down.remove()

        saved_qkv = torch.load(tmp_path / model_name / f"model.layers.{layer_index}.self_attn.q_proj.pt")
        saved_o = torch.load(tmp_path / model_name / f"model.layers.{layer_index}.self_attn.o_proj.pt")
        saved_down = torch.load(tmp_path / model_name / f"model.layers.{layer_index}.mlp.down_proj.pt")

        assert torch.allclose(saved_qkv, acc_qkv.hessian, atol=1e-4, rtol=1e-3)
        assert torch.allclose(saved_o, acc_o.hessian, atol=1e-4, rtol=1e-3)
        assert torch.allclose(saved_down, acc_down.hessian, atol=1e-4, rtol=1e-3)

        # k_proj/v_proj are aliases of q_proj's file -- no separate file written
        assert not (tmp_path / model_name / f"model.layers.{layer_index}.self_attn.k_proj.pt").exists()
        assert not (tmp_path / model_name / f"model.layers.{layer_index}.mlp.up_proj.pt").exists()

    assert torch.allclose(out, x, atol=1e-5)


def test_streaming_processes_in_chunks_matches_single_batch(tmp_path):
    """batch_size chunking shouldn't change the accumulated Hessian (H is a
    simple sum over all rows regardless of how they're grouped into
    forward-pass chunks)."""
    torch.manual_seed(1)
    d, h = 8, 16
    hidden_states = torch.randn(10, 3, d)

    layers_a = [_ToyLayer(d, h)]
    layers_b = [_ToyLayer(d, h)]
    layers_b[0].load_state_dict(layers_a[0].state_dict())

    collect_hessians_streaming(layers_a, hidden_states, _forward_layer_fn, _get_groups,
                                device="cpu", model_name="a", root=str(tmp_path / "a"),
                                batch_size=None)
    collect_hessians_streaming(layers_b, hidden_states, _forward_layer_fn, _get_groups,
                                device="cpu", model_name="b", root=str(tmp_path / "a"),
                                batch_size=3)

    h_a = torch.load(tmp_path / "a" / "a" / "model.layers.0.self_attn.q_proj.pt")
    h_b = torch.load(tmp_path / "a" / "b" / "model.layers.0.self_attn.q_proj.pt")
    assert torch.allclose(h_a, h_b, atol=1e-4, rtol=1e-3)
