import torch
import pytest

from quipsharp.rfft import (
    to_transformed_space, from_transformed_space, random_phase,
    rfft_incoherence_process, reconstruct_weight, reconstruct_hessian, RFFTPhases,
)
from quipsharp.incoherence import incoherence_mu


def test_transform_round_trip():
    torch.manual_seed(0)
    n = 10
    phase = random_phase(n // 2)
    x = torch.randn(5, n)
    y = to_transformed_space(x, phase)
    x_rec = from_transformed_space(y, phase)
    assert torch.allclose(x, x_rec, atol=1e-5)


def test_transform_preserves_norm():
    torch.manual_seed(1)
    n = 12
    phase = random_phase(n // 2)
    x = torch.randn(4, n)
    y = to_transformed_space(x, phase)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_rejects_odd_dimension():
    m, n = 6, 7
    W = torch.randn(m, n)
    H = torch.randn(n, n)
    with pytest.raises(AssertionError):
        rfft_incoherence_process(W, H)


def test_weight_round_trip_non_pow2_dims():
    """The whole point of RFFT: works for dims that AREN'T powers of two,
    e.g. Gemma 3 1B's hidden_size=1152 (1152 = 2^7 * 9)."""
    torch.manual_seed(2)
    m, n = 18, 1152  # deliberately non-power-of-2, mirrors a real model dim
    W = torch.randn(m, n) * 0.1
    H = torch.randn(n, n)
    H = H @ H.T + torch.eye(n)

    what, hhat, phases = rfft_incoherence_process(W, H)
    assert what.shape == (m, n)
    assert phases.phase_u.shape == (m // 2,)
    assert phases.phase_v.shape == (n // 2,)

    W_rec = reconstruct_weight(what, phases)
    assert torch.allclose(W_rec, W, atol=1e-3)


def test_hessian_round_trip():
    torch.manual_seed(3)
    n = 18
    H = torch.randn(n, n)
    H = H @ H.T + torch.eye(n)
    W = torch.randn(10, n)

    _, hhat, phases = rfft_incoherence_process(W, H)
    H_rec = reconstruct_hessian(hhat, phases)
    assert torch.allclose(H_rec, H, atol=1e-3)


def test_transform_input_output_matches_original_matmul():
    """Critical correctness check (same role as the analogous RHT test):
    RFFTPhases.transform_input/transform_output must compose to exactly
    reproduce W @ x, or every downstream layer using RFFT silently breaks."""
    torch.manual_seed(4)
    m, n = 10, 18
    W = torch.randn(m, n)
    H = torch.eye(n)
    what, _, phases = rfft_incoherence_process(W, H)

    x = torch.randn(5, n)
    x2 = phases.transform_input(x)
    y1 = x2 @ what.T
    y = phases.transform_output(y1)

    y_true = x @ W.T
    assert torch.allclose(y, y_true, atol=1e-3)


def test_incoherence_processing_suppresses_outliers():
    torch.manual_seed(5)
    m, n = 40, 60
    W = torch.randn(m, n) * 0.1
    W[3, 5] = 50.0
    H = torch.eye(n)

    mu_before = incoherence_mu(W)
    what, _, _ = rfft_incoherence_process(W, H)
    mu_after = incoherence_mu(what)

    assert mu_after < mu_before / 5
