import torch
import pytest

from quipsharp.hadamard import fwht, is_pow2, next_pow2, rademacher


def explicit_hadamard(n: int) -> torch.Tensor:
    """Reference O(n^2) construction via Sylvester's recursion, for testing only."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / (n ** 0.5)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 128])
def test_matches_explicit_matrix(n):
    torch.manual_seed(0)
    x = torch.randn(5, n)
    got = fwht(x, dim=-1)
    want = x @ explicit_hadamard(n).T
    assert torch.allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("n", [2, 8, 64])
def test_involution(n):
    torch.manual_seed(0)
    x = torch.randn(3, n)
    assert torch.allclose(fwht(fwht(x)), x, atol=1e-5)


@pytest.mark.parametrize("n", [2, 8, 64])
def test_orthogonal_norm_preserving(n):
    torch.manual_seed(0)
    x = torch.randn(4, n)
    assert torch.allclose(x.norm(dim=-1), fwht(x).norm(dim=-1), atol=1e-5)


def test_non_pow2_raises():
    with pytest.raises(ValueError):
        fwht(torch.randn(3))


def test_transform_along_arbitrary_dim():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 6)
    got = fwht(x, dim=1)
    want = torch.stack([fwht(x[:, :, j], dim=-1) for j in range(6)], dim=-1)
    assert torch.allclose(got, want, atol=1e-5)


def test_is_pow2_and_next_pow2():
    assert [is_pow2(n) for n in [1, 2, 3, 4, 5, 8, 15, 16]] == \
        [True, True, False, True, False, True, False, True]
    assert next_pow2(1) == 1
    assert next_pow2(5) == 8
    assert next_pow2(129) == 256
    assert next_pow2(256) == 256


def test_rademacher_is_pm1():
    s = rademacher(1000)
    assert set(s.unique().tolist()) <= {-1.0, 1.0}
