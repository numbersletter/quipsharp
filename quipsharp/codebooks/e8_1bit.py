"""A 1-bit, 8-dimensional E8-lattice codebook, used as the 2nd RVQ stage to
build 3-bit quantization on top of the 2-bit E8P codebook (Section 4.3):
"we can also quantize to 3 bits by using the 2 bit E8P codebook and a 1-bit
E8 codebook (elements of E8 with norm <= 2 and 15 elements of E8 with
norm 4)".

Unlike E8P, this codebook has exactly 256 entries total (1 bit/dim x 8 dims
= 2^8), so it needs none of E8P's bit-packing/parity tricks -- we just
enumerate all 256 vectors directly and brute-force nearest-neighbor search
(trivial at this size).

E8 = (Z^8 u (Z^8+1/2)) n {x : sum(x) even} (paper, Section 4.2). Its
norm^2=2 shell has exactly 240 vectors -- this is the famous E8 kissing
number. Together with the zero vector that's 241; the paper pads to 256
with "15 elements of norm 4" but does not list them explicitly, so the 15
used here are an arbitrary (but verified-valid) choice of our own -- any
valid norm^2=4 E8 vectors would do, since their only role is filling out
the table, not the lattice's actual packing structure.
"""
import torch
from torch import Tensor

CODE_DIM = 8
TABLE_SIZE = 256


def _norm_sq_exact(grid: Tensor) -> Tensor:
    """Sum of squares, NOT norm().pow(2) -- squaring a sqrt round-trips
    through floating point and can miss exact integer equality checks."""
    return (grid ** 2).sum(-1)


def _integer_shell(norm_sq: int) -> Tensor:
    """All-integer E8 vectors (even coordinate sum) with the given squared norm."""
    # every coordinate of a norm^2=n integer vector has |value| <= sqrt(n)
    bound = int(norm_sq ** 0.5)
    vals = torch.arange(-bound, bound + 1, dtype=torch.float32)
    grid = torch.cartesian_prod(*([vals] * CODE_DIM))
    mask = (_norm_sq_exact(grid) == norm_sq) & (grid.sum(-1) % 2 == 0)
    return grid[mask]


def _half_integer_shell(norm_sq: int) -> Tensor:
    """All-half-integer E8 vectors (even coordinate sum) with the given
    squared norm. Search range generously bounds |each coord| <= sqrt(norm_sq)."""
    bound = int(norm_sq ** 0.5) + 1
    vals = torch.arange(-bound, bound, dtype=torch.float32) + 0.5
    grid = torch.cartesian_prod(*([vals] * CODE_DIM))
    mask = (_norm_sq_exact(grid) == norm_sq) & (grid.sum(-1) % 2 == 0)
    return grid[mask]


def _norm2_shell() -> Tensor:
    """The 240 minimal (norm^2=2) vectors of E8 -- the E8 kissing number."""
    shell = torch.cat([_integer_shell(2), _half_integer_shell(2)], dim=0)
    assert shell.shape == (240, CODE_DIM), shell.shape
    return shell


# An arbitrary but valid choice of 15 norm^2=4 E8 vectors used purely to pad
# the table to 256 entries -- see module docstring.
def _pad15_norm4() -> Tensor:
    shell4 = _integer_shell(4)  # plenty of all-integer norm^2=4 vectors exist
    return shell4[:15]


def build_table() -> Tensor:
    zero = torch.zeros(1, CODE_DIM)
    table = torch.cat([zero, _norm2_shell(), _pad15_norm4()], dim=0)
    assert table.shape == (TABLE_SIZE, CODE_DIM), table.shape
    return table


class E8OneBitCodebook:
    """1-bit (8 bits / 8 dims), 8-dimensional codebook: 256 entries, brute
    force nearest neighbor (the table is tiny, no packing tricks needed)."""

    def __init__(self, device=None):
        self.table = build_table().to(device=device)

    def quantize(self, x: Tensor):
        dist2 = (x.unsqueeze(-2) - self.table).pow(2).sum(-1)  # (..., 256)
        idx = dist2.argmin(-1)
        return self.table[idx], idx

    def dequantize(self, idx: Tensor) -> Tensor:
        return self.table[idx]
