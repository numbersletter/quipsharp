import torch

from quipsharp.ldlq import block_ldl, block_diag_from_stack, block_ldlq
from quipsharp.codebooks.e8p import E8PCodebook


def random_psd(n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return A @ A.T + torch.eye(n)


def test_block_ldl_reconstructs_H():
    torch.manual_seed(0)
    n, g = 32, 8
    H = random_psd(n, 1)
    L, D = block_ldl(H, g, damping=0.0)
    Dfull = block_diag_from_stack(D)
    H_rec = L @ Dfull @ L.T
    assert torch.allclose(H_rec, H, atol=1e-2, rtol=1e-2)


def test_L_is_unit_block_lower_triangular():
    torch.manual_seed(1)
    n, g = 24, 8
    H = random_psd(n, 2)
    L, _ = block_ldl(H, g)
    T = n // g
    for i in range(T):
        block = L[i * g:(i + 1) * g, i * g:(i + 1) * g]
        assert torch.allclose(block, torch.eye(g), atol=1e-5)
    for i in range(T):
        for j in range(i + 1, T):
            block = L[i * g:(i + 1) * g, j * g:(j + 1) * g]
            assert torch.allclose(block, torch.zeros(g, g), atol=1e-6)


def test_block_ldlq_identity_quantizer_is_exact():
    torch.manual_seed(2)
    m, n, g = 12, 32, 8
    W = torch.randn(m, n)
    H = random_psd(n, 3)
    L, _ = block_ldl(H, g)
    Wq = block_ldlq(W, L, g, quantize_fn=lambda x, k: x)
    assert torch.allclose(Wq, W, atol=1e-4)


def _round_to_half_int(x: torch.Tensor, k: int = None) -> torch.Tensor:
    return (x * 2).round() / 2


def test_block_ldlq_beats_independent_rounding_under_correlated_hessian():
    """Core claim of Section 4.1: feeding back already-committed rounding
    error (using the real L from the correlated Hessian) should achieve
    lower H-weighted error than rounding each block independently (which is
    what you get by pretending L = I, i.e. no feedback)."""
    torch.manual_seed(3)
    m, n, g = 64, 64, 8
    W = torch.randn(m, n)
    H = random_psd(n, 4)
    L, _ = block_ldl(H, g)

    def h_weighted_error(Wq):
        E = Wq - W
        return torch.trace(E @ H @ E.T).item()

    Wq_feedback = block_ldlq(W, L, g, quantize_fn=_round_to_half_int)
    Wq_independent = block_ldlq(W, torch.eye(n), g, quantize_fn=_round_to_half_int)

    err_feedback = h_weighted_error(Wq_feedback)
    err_independent = h_weighted_error(Wq_independent)

    assert err_feedback < err_independent


def test_block_ldlq_with_e8p_codebook():
    """Plumbing check: BlockLDLQ composes with the real E8P vector
    quantizer (not just scalar rounding)."""
    torch.manual_seed(4)
    m, n, g = 16, 32, 8
    W = torch.randn(m, n) * 0.3
    H = random_psd(n, 5)
    L, _ = block_ldl(H, g)
    cb = E8PCodebook()

    def quantize_fn(target, k):
        vals, _ = cb.quantize(target)
        return vals

    Wq = block_ldlq(W, L, g, quantize_fn)
    assert Wq.shape == W.shape
    assert torch.isfinite(Wq).all()


def test_block_ldlq_refinement_is_noop_with_identity_quantizer():
    """With an exact (identity) quantizer, the base pass already reproduces
    W exactly, so the residual feeding the refinement sweeps is zero and
    refinement should leave the result unchanged."""
    torch.manual_seed(5)
    m, n, g = 12, 32, 8
    W = torch.randn(m, n)
    H = random_psd(n, 6)
    L, _ = block_ldl(H, g)
    Wq = block_ldlq(W, L, g, quantize_fn=lambda x, k: x, H=H, tune_iters=10)
    assert torch.allclose(Wq, W, atol=1e-4)


def test_block_ldlq_refinement_reduces_h_weighted_error():
    """The reference repo's extra tune_iters sweeps (not described in the
    paper's Section 4.1 text, but enabled by default -- 10 iters -- in the
    released code) should reduce, or at least not meaningfully worsen, the
    H-weighted quantization error versus the single base pass alone."""
    torch.manual_seed(6)
    m, n, g = 64, 64, 8
    W = torch.randn(m, n) * 0.3
    H = random_psd(n, 7)
    L, _ = block_ldl(H, g)
    cb = E8PCodebook()

    def quantize_fn(target, k):
        vals, _ = cb.quantize(target)
        return vals

    def h_weighted_error(Wq):
        E = Wq - W
        return torch.trace(E @ H @ E.T).item()

    Wq_base = block_ldlq(W, L, g, quantize_fn, H=None)
    Wq_refined = block_ldlq(W, L, g, quantize_fn, H=H, tune_iters=10)

    assert h_weighted_error(Wq_refined) <= h_weighted_error(Wq_base) * 1.01
