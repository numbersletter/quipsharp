"""Fast Walsh-Hadamard transform and the Randomized Hadamard Transform (RHT).

Reference: QuIP# paper, Section 3 and Algorithm 3 (IP-RHT).
"""
import torch
from torch import Tensor


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def fwht(x: Tensor, dim: int = -1) -> Tensor:
    """Orthonormal fast Walsh-Hadamard transform along `dim`.

    Computes H_n @ x where H_n is the (symmetric, orthogonal) Sylvester
    Hadamard matrix of size n, normalized by 1/sqrt(n) so that H_n is an
    involution (H_n @ H_n == I). Requires n = x.shape[dim] to be a power of
    two; runs in O(n log n) with no floating point multiplies in the
    butterfly stage (only the final 1/sqrt(n) scale).
    """
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    if not is_pow2(n):
        raise ValueError(f"fwht requires a power-of-two size along dim, got {n}")
    orig_shape = x.shape
    y = x.reshape(-1, n).contiguous()
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    y = (y * (n ** -0.5)).view(orig_shape)
    return y.movedim(-1, dim)


def rademacher(n: int, device=None, dtype=torch.float32, generator=None) -> Tensor:
    """Sample a uniform random +-1 vector of length n (S ~ U{-1,+1}^n)."""
    bits = torch.randint(0, 2, (n,), device=device, generator=generator)
    return bits.to(dtype) * 2 - 1
