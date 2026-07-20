"""Incoherence processing via the Randomized FFT (RFFT).

Reference: QuIP# paper, Appendix A.2 and Algorithm 4 (IP-RFFT). This is the
fallback used when a dimension isn't a power of two (so the Hadamard
transform in hadamard.py/incoherence.py doesn't apply) -- RFFT only needs
the dimension to be EVEN, which covers essentially every real transformer
layer width.

The transform: reshape a real n-vector into n/2 complex numbers (adjacent
reals as (real, imag) pairs), multiply elementwise by a random unit-modulus
complex phase, apply the (unitary) DFT via FFT, then reshape the resulting
n/2 complex numbers back into n reals. Both the phase multiply and the DFT
are unitary, and the real<->complex reshaping preserves squared norm
(|a+bi|^2 = a^2+b^2), so the composite is an orthogonal map on R^n --
playing the same role the Hadamard transform does in the RHT.

Unlike the Hadamard transform, the (unitary) DFT is NOT self-inverse
(F^{-1} = F^H != F in general), so -- verified numerically, the same way
the RHT's activation-transform order was verified in incoherence.py --
"entering" the transformed space is scale(phase)-then-FFT, while "exiting"
is IFFT-then-scale(conjugate phase); note from_transformed_space already
folds that conjugate-phase multiply in, so it can be used directly as a
single step wherever incoherence.py uses a separate `fwht(...) * sign`.
"""
from dataclasses import dataclass

import torch
from torch import Tensor


def random_phase(n_half: int, device=None, generator=None) -> Tensor:
    """A uniform random unit-modulus complex vector of length n_half."""
    theta = torch.rand(n_half, device=device, generator=generator) * (2 * torch.pi)
    return torch.polar(torch.ones_like(theta), theta)


def to_transformed_space(x: Tensor, phase: Tensor, dim: int = -1) -> Tensor:
    """Enter the RFFT-transformed space: reshape to complex, multiply by
    phase, FFT, reshape back to real. Requires x.shape[dim] == 2*len(phase)."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    assert n == 2 * phase.shape[-1], f"expected last dim {2 * phase.shape[-1]}, got {n}"
    shape = x.shape
    xc = torch.view_as_complex(x.reshape(*shape[:-1], n // 2, 2).contiguous())
    Xc = torch.fft.fft(xc * phase, norm="ortho")
    y = torch.view_as_real(Xc).reshape(shape)
    return y.movedim(-1, dim)


def from_transformed_space(y: Tensor, phase: Tensor, dim: int = -1) -> Tensor:
    """Exit the RFFT-transformed space: IFFT, multiply by conjugate phase,
    reshape back to real. Exact inverse of to_transformed_space."""
    y = y.movedim(dim, -1)
    n = y.shape[-1]
    assert n == 2 * phase.shape[-1], f"expected last dim {2 * phase.shape[-1]}, got {n}"
    shape = y.shape
    yc = torch.view_as_complex(y.reshape(*shape[:-1], n // 2, 2).contiguous())
    Xc = torch.fft.ifft(yc, norm="ortho") * torch.conj(phase)
    x = torch.view_as_real(Xc).reshape(shape)
    return x.movedim(-1, dim)


@dataclass
class RFFTPhases:
    phase_u: Tensor  # (m//2,) complex
    phase_v: Tensor  # (n//2,) complex

    def transform_input(self, x: Tensor) -> Tensor:
        """Algorithm 2 step 1 analogue: x -> RFFT(phase_v applied to x)."""
        return to_transformed_space(x, self.phase_v, dim=-1)

    def transform_output(self, y: Tensor) -> Tensor:
        """Algorithm 2 step 3 analogue: y -> phase_u-conjugate-scaled IRFFT(y)."""
        return from_transformed_space(y, self.phase_u, dim=-1)

    def reconstruct_weight(self, what: Tensor) -> Tensor:
        return reconstruct_weight(what, self)


def rfft_incoherence_process(W: Tensor, H: Tensor, generator=None) -> tuple[Tensor, Tensor, RFFTPhases]:
    """Algorithm 4 (IP-RFFT). W: (m, n), H: (n, n). m and n must be even."""
    m, n = W.shape
    assert m % 2 == 0 and n % 2 == 0, "RFFT incoherence processing requires even dimensions"
    assert H.shape == (n, n)

    phase_u = random_phase(m // 2, device=W.device, generator=generator)
    phase_v = random_phase(n // 2, device=W.device, generator=generator)

    what = to_transformed_space(to_transformed_space(W, phase_u, dim=0), phase_v, dim=1)
    hhat = to_transformed_space(to_transformed_space(H, phase_v, dim=0), phase_v, dim=1)

    return what, hhat, RFFTPhases(phase_u=phase_u, phase_v=phase_v)


def reconstruct_weight(what: Tensor, phases: RFFTPhases) -> Tensor:
    """Invert rfft_incoherence_process's weight transform exactly."""
    y = from_transformed_space(what, phases.phase_v, dim=1)
    return from_transformed_space(y, phases.phase_u, dim=0)


def reconstruct_hessian(hhat: Tensor, phases: RFFTPhases) -> Tensor:
    """Invert rfft_incoherence_process's Hessian transform exactly."""
    y = from_transformed_space(hhat, phases.phase_v, dim=1)
    return from_transformed_space(y, phases.phase_v, dim=0)
