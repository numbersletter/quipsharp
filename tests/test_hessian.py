import torch
from torch import nn

from quipsharp.hessian import collect_hessians, save_hessians, load_hessians


class TwoHeads(nn.Module):
    """Toy model where two independent nn.Linear layers consume the exact
    same input -- mirrors q_proj/k_proj/v_proj sharing a block's
    hidden_states, for testing the `groups` dedup path."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(5, 3, bias=False)
        self.b = nn.Linear(5, 3, bias=False)

    def forward(self, x):
        return self.a(x), self.b(x)


def test_collect_hessians_matches_manual_computation():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 4, bias=False), nn.ReLU(), nn.Linear(4, 3, bias=False))

    batches = [torch.randn(5, 6) for _ in range(4)]
    hessians = collect_hessians(model, batches)

    assert set(hessians.keys()) == {"0", "2"}
    assert hessians["0"].shape == (6, 6)
    assert hessians["2"].shape == (4, 4)

    all_x0 = torch.cat(batches, dim=0)
    expected_h0 = (all_x0.T @ all_x0) / all_x0.shape[0]
    assert torch.allclose(hessians["0"], expected_h0, atol=1e-4)

    with torch.no_grad():
        all_x2 = torch.relu(model[0](all_x0))
    expected_h2 = (all_x2.T @ all_x2) / all_x2.shape[0]
    assert torch.allclose(hessians["2"], expected_h2, atol=1e-4)


def test_collect_hessians_layer_name_filter():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    batches = [torch.randn(2, 4)]
    hessians = collect_hessians(model, batches, layer_names={"0"})
    assert set(hessians.keys()) == {"0"}


def test_collect_hessians_is_symmetric_psd():
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(8, 8, bias=False))
    batches = [torch.randn(20, 8) for _ in range(3)]
    H = collect_hessians(model, batches)["0"]
    assert torch.allclose(H, H.T, atol=1e-5)
    eigvals = torch.linalg.eigvalsh(H)
    assert eigvals.min() >= -1e-4


def test_save_and_load_hessians_round_trip(tmp_path):
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(6, 4, bias=False), nn.ReLU(), nn.Linear(4, 3, bias=False))
    batches = [torch.randn(5, 6) for _ in range(4)]
    hessians = collect_hessians(model, batches)

    save_hessians(hessians, model_name="toy-model", root=str(tmp_path))
    assert (tmp_path / "toy-model" / "0.pt").exists()
    assert (tmp_path / "toy-model" / "2.pt").exists()

    loaded = load_hessians("toy-model", layer_names=["0", "2"], root=str(tmp_path))
    assert set(loaded.keys()) == {"0", "2"}
    for name in ["0", "2"]:
        assert torch.allclose(loaded[name], hessians[name])


def test_load_hessians_returns_none_when_incomplete(tmp_path):
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    hessians = collect_hessians(model, [torch.randn(3, 4)])
    save_hessians(hessians, model_name="partial-model", root=str(tmp_path))

    # asking for a layer that was never saved should fail closed, not
    # silently return a partial dict
    result = load_hessians("partial-model", layer_names=["0", "does_not_exist"], root=str(tmp_path))
    assert result is None


def test_load_hessians_returns_none_when_missing_directory(tmp_path):
    result = load_hessians("never-saved-model", layer_names=["0"], root=str(tmp_path))
    assert result is None


def test_collect_hessians_with_groups_shares_hessian_and_skips_redundant_hook():
    torch.manual_seed(5)
    model = TwoHeads()
    batches = [torch.randn(4, 5) for _ in range(3)]
    hessians = collect_hessians(model, batches, groups=[["a", "b"]])

    assert set(hessians.keys()) == {"a", "b"}
    assert hessians["a"] is hessians["b"]  # same object, not a coincidentally-equal copy

    all_x = torch.cat(batches, dim=0)
    expected = (all_x.T @ all_x) / all_x.shape[0]
    assert torch.allclose(hessians["a"], expected, atol=1e-4)


def test_save_hessians_dedupes_grouped_names(tmp_path):
    torch.manual_seed(6)
    model = TwoHeads()
    batches = [torch.randn(4, 5) for _ in range(3)]
    hessians = collect_hessians(model, batches, groups=[["a", "b"]])

    save_hessians(hessians, model_name="grouped-model", root=str(tmp_path))
    out_dir = tmp_path / "grouped-model"
    assert (out_dir / "a.pt").exists()
    assert not (out_dir / "b.pt").exists()  # deduped: not written a second time
    assert (out_dir / "_aliases.json").exists()

    loaded = load_hessians("grouped-model", layer_names=["a", "b"], root=str(tmp_path))
    assert set(loaded.keys()) == {"a", "b"}
    assert loaded["a"] is loaded["b"]  # reconstructed as a shared object too
    assert torch.allclose(loaded["a"], hessians["a"])


def test_save_hessians_no_manifest_when_nothing_aliased(tmp_path):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    hessians = collect_hessians(model, [torch.randn(3, 4)])
    save_hessians(hessians, model_name="ungrouped-model", root=str(tmp_path))
    assert not (tmp_path / "ungrouped-model" / "_aliases.json").exists()
