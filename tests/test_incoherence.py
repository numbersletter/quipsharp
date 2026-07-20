import torch

from quipsharp.incoherence import (
    incoherence_process, reconstruct_weight, reconstruct_hessian,
    linear_via_transformed, incoherence_mu,
)


def test_weight_round_trip():
    torch.manual_seed(0)
    m, n = 32, 16
    W = torch.randn(m, n)
    H = torch.randn(n, n)
    H = H @ H.T + torch.eye(n)  # PSD proxy Hessian

    what, hhat, signs = incoherence_process(W, H)
    assert what.shape == (m, n)
    assert signs.su.shape == (m,)
    assert signs.sv.shape == (n,)

    W_rec = reconstruct_weight(what, signs)
    assert torch.allclose(W_rec, W, atol=1e-4)


def test_hessian_round_trip():
    torch.manual_seed(1)
    n = 16
    H = torch.randn(n, n)
    H = H @ H.T + torch.eye(n)
    W = torch.randn(8, n)

    _, hhat, signs = incoherence_process(W, H)
    H_rec = reconstruct_hessian(hhat, signs)
    assert torch.allclose(H_rec, H, atol=1e-4)


def test_linear_via_transformed_matches_original_matmul():
    """This is the critical correctness check for Algorithm 2's activation
    transform order: Su/Sv sign flips must compose with Had in the exact
    order implemented, or this silently produces wrong outputs everywhere
    downstream. See incoherence.py's module docstring."""
    torch.manual_seed(2)
    m, n = 16, 32
    W = torch.randn(m, n)
    H = torch.eye(n)
    what, _, signs = incoherence_process(W, H)

    x = torch.randn(5, n)
    y = linear_via_transformed(x, what, signs)
    y_true = x @ W.T
    assert torch.allclose(y, y_true, atol=1e-4)


def test_incoherence_processing_suppresses_outliers():
    """Pedagogical check of Section 2.3: a matrix with one huge outlier entry
    should become far more incoherent (smaller mu) after RHT processing."""
    torch.manual_seed(3)
    m, n = 64, 64
    W = torch.randn(m, n) * 0.1
    W[3, 5] = 50.0  # inject an extreme outlier
    H = torch.eye(n)

    mu_before = incoherence_mu(W)
    what, _, _ = incoherence_process(W, H)
    mu_after = incoherence_mu(what)

    assert mu_after < mu_before / 5
