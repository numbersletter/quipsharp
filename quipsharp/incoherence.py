"""Incoherence processing via the Randomized Hadamard Transform.

Reference: QuIP# paper, Section 3, Algorithm 3 (IP-RHT), and Definition 2.1.

Forward (quantization time):
    What = H_m @ (Su[:,None] * W * Sv[None,:]) @ H_n
    Hhat = H_n @ (Sv[:,None] * H * Sv[None,:]) @ H_n
(H_m, H_n are the normalized, self-inverse Hadamard transforms of size m, n.)

Inverse (materializing W back, e.g. for testing):
    W = (H_m @ What @ H_n) * Su[:,None] * Sv[None,:]

Inference-time activation transforms (Algorithm 2), verified numerically to
require this exact operation order -- NOT the literal left-to-right reading
of the paper's "Had(S ⊙ x)" pseudocode, which is ambiguous about whether the
sign-flip or the transform happens first:
    x2 = Had(Sv ⊙ x)      i.e. fwht(x * sv)      -- scale THEN transform
    y  = Su ⊙ Had(y1)     i.e. fwht(y1) * su     -- transform THEN scale
This asymmetry is exactly what makes the two operations mutual inverses.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from .hadamard import fwht, is_pow2, rademacher


@dataclass
class IncoherenceSigns:
    su: Tensor  # (m,) row signs
    sv: Tensor  # (n,) column signs

    def transform_input(self, x: Tensor) -> Tensor:
        """Algorithm 2 step 1: x -> Had(Sv ⊙ x). Same protocol as
        rfft.RFFTPhases, so callers (quantize.py) don't need to know which
        transform a given layer used."""
        return transform_input(x, self.sv)

    def transform_output(self, y: Tensor) -> Tensor:
        """Algorithm 2 step 3: y -> Su ⊙ Had(y)."""
        return transform_output(y, self.su)

    def reconstruct_weight(self, what: Tensor) -> Tensor:
        return reconstruct_weight(what, self)


def incoherence_process(W: Tensor, H: Tensor, generator=None) -> tuple[Tensor, Tensor, IncoherenceSigns]:
    """Algorithm 3 (IP-RHT). W: (m, n) weight, H: (n, n) proxy Hessian.

    Both m and n must currently be powers of two (see rfft_incoherence for
    the general-dimension fallback described in Appendix A.2/Algorithm 4).
    """
    m, n = W.shape
    if not (is_pow2(m) and is_pow2(n)):
        raise ValueError(
            f"incoherence_process requires power-of-two dims, got ({m}, {n}); "
            "use rfft_incoherence for arbitrary sizes"
        )
    assert H.shape == (n, n)

    su = rademacher(m, device=W.device, dtype=W.dtype, generator=generator)
    sv = rademacher(n, device=W.device, dtype=W.dtype, generator=generator)

    Z = W * su.unsqueeze(1) * sv.unsqueeze(0)
    what = fwht(fwht(Z, dim=0), dim=1)

    Y = H * sv.unsqueeze(1) * sv.unsqueeze(0)
    hhat = fwht(fwht(Y, dim=0), dim=1)

    return what, hhat, IncoherenceSigns(su=su, sv=sv)


def reconstruct_weight(what: Tensor, signs: IncoherenceSigns) -> Tensor:
    """Invert incoherence_process's weight transform exactly (no quantization)."""
    Z = fwht(fwht(what, dim=0), dim=1)
    return Z * signs.su.unsqueeze(1) * signs.sv.unsqueeze(0)


def reconstruct_hessian(hhat: Tensor, signs: IncoherenceSigns) -> Tensor:
    """Invert incoherence_process's Hessian transform exactly. Testing/debug only."""
    Y = fwht(fwht(hhat, dim=0), dim=1)
    return Y * signs.sv.unsqueeze(1) * signs.sv.unsqueeze(0)


def transform_input(x: Tensor, sv: Tensor) -> Tensor:
    """Algorithm 2, step 1: x -> Had(Sv ⊙ x). x: (..., n)."""
    return fwht(x * sv, dim=-1)


def transform_output(y: Tensor, su: Tensor) -> Tensor:
    """Algorithm 2, step 3: y -> Su ⊙ Had(y). y: (..., m)."""
    return fwht(y, dim=-1) * su


def linear_via_transformed(x: Tensor, what: Tensor, signs: IncoherenceSigns) -> Tensor:
    """Reference (slow, dense) version of Algorithm 2 using the *unquantized*
    transformed weight `what`. Should reproduce W @ x exactly (up to fp error)
    -- this is the round-trip identity the real (quantized) inference path
    approximates. Useful only for testing.
    """
    x2 = transform_input(x, signs.sv)
    y1 = x2 @ what.T
    return transform_output(y1, signs.su)


def incoherence_mu(W: Tensor) -> float:
    """Definition 2.1's incoherence parameter mu for a weight matrix: smaller
    is more incoherent (less outlier-prone). mu = max|W_ij| * sqrt(mn) / ||W||_F.
    """
    m, n = W.shape
    return (W.abs().max() * (m * n) ** 0.5 / W.norm()).item()


def incoherence_process_auto(W: Tensor, H: Tensor, generator=None):
    """Dispatch to RHT (fast, needs power-of-two dims) when possible, else
    fall back to RFFT (needs only even dims) -- mirroring the paper's own
    primary-method-plus-fallback strategy (Section 3 vs. Appendix A.2).
    Real transformer widths are essentially never a power of two (e.g.
    Gemma 3 1B's hidden_size=1152), so most real layers take the RFFT path.

    Returns (what, hhat, transform) where `transform` is an IncoherenceSigns
    or an RFFTPhases -- both expose the same .transform_input/
    .transform_output/.reconstruct_weight methods, so callers don't need to
    know or care which one they got.
    """
    m, n = W.shape
    if is_pow2(m) and is_pow2(n):
        return incoherence_process(W, H, generator=generator)
    from .rfft import rfft_incoherence_process
    return rfft_incoherence_process(W, H, generator=generator)
