"""The E8P ("E8 Padded") 2-bit, 8-dimensional lattice codebook.

Reference: QuIP# paper, Section 4.2 and Appendix C.1-C.2.

E8P packs a 2^16-entry codebook of points in E8 + 1/4 into a 16-bit code
without ever materializing all 65536 vectors:

    code = [abs_idx : 8 bits][sign_bits7 : 7 bits][shift_bit : 1 bit]

`abs_idx` selects one of 256 canonical "base" vectors (8 dims x 4 bits each
-> fits a tiny lookup table), `sign_bits7` gives 7 free per-coordinate sign
flips, and the 8th coordinate's sign is *inferred* from the parity of the
other 7 rather than stored -- this works because every base vector is
pre-normalized (see `_fix_parity_to_even`) to require an EVEN number of sign
flips to land back in D8-hat, uniformly across the whole table. `shift_bit`
then picks between the two cosets D8-hat -/+ 1/4 that together make up
E8 + 1/4.

The 256 base vectors are: the 227 elements of D8-hat = {x in Z^8+1/2 : sum(x)
even} with norm^2 <= 10, plus 29 hand-picked norm^2=12 "padding" vectors
given explicitly in the paper (Appendix C.1) to round 227 up to a clean
power of two.
"""
import torch
from torch import Tensor

CODE_DIM = 8
ABS_TABLE_SIZE = 256

# 29 padding vectors of norm^2 = 12 in E8 + 1/4, from Appendix C.1.
_PAD29_RAW = torch.tensor([
    [3, 1, 1, 1, 3, 3, 3, 3],
    [1, 3, 1, 1, 3, 3, 3, 3],
    [1, 1, 3, 1, 3, 3, 3, 3],
    [1, 1, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 3, 3, 1, 1],
    [3, 3, 3, 1, 3, 1, 3, 1],
    [3, 3, 3, 1, 1, 3, 3, 1],
    [3, 3, 3, 1, 3, 1, 1, 3],
    [3, 3, 3, 1, 1, 3, 1, 3],
    [3, 3, 3, 1, 1, 1, 3, 3],
    [3, 3, 1, 3, 3, 3, 1, 1],
    [3, 3, 1, 3, 3, 1, 3, 1],
    [3, 3, 1, 3, 1, 3, 3, 1],
    [3, 3, 1, 3, 3, 1, 1, 3],
    [3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 1, 3, 1, 1, 3, 3],
    [3, 1, 3, 3, 3, 3, 1, 1],
    [3, 1, 3, 3, 3, 1, 3, 1],
    [3, 1, 3, 3, 1, 3, 3, 1],
    [3, 1, 3, 3, 3, 1, 1, 3],
    [3, 1, 3, 3, 1, 3, 1, 3],
    [1, 3, 3, 3, 1, 1, 3, 3],
    [1, 3, 3, 3, 3, 3, 1, 1],
    [1, 3, 3, 3, 3, 1, 3, 1],
    [1, 3, 3, 3, 1, 3, 3, 1],
    [1, 3, 3, 3, 3, 1, 1, 3],
    [1, 3, 3, 3, 1, 3, 1, 3],
    [1, 1, 3, 3, 1, 3, 3, 3],
    [3, 3, 1, 1, 3, 3, 3, 1],
], dtype=torch.float32) / 2


def _enumerate_d8hat_abs_le_norm10() -> Tensor:
    """The 227 distinct elementwise-abs patterns of D8-hat = {x in Z^8+1/2 :
    sum(x) even} with squared norm <= 10."""
    half_ints = torch.arange(-4, 4, dtype=torch.float32) + 0.5  # {-3.5,...,3.5}
    grid = torch.cartesian_prod(*([half_ints] * CODE_DIM))  # (8**8, 8)
    even_sum = grid.sum(-1) % 2 == 0
    in_ball = grid.norm(dim=-1) ** 2 <= 10
    d8hat = grid[even_sum & in_ball]
    return torch.unique(d8hat.abs(), dim=0)


def _fix_parity_to_even(rows: Tensor) -> Tensor:
    """Negate the last coordinate on any row with odd coordinate sum, so
    every row has an even sum -- see module docstring."""
    rows = rows.clone()
    odd = rows.sum(-1) % 2 != 0
    rows[odd, -1] *= -1
    return rows


def build_abs_table() -> Tensor:
    """The (256, 8) table of canonical D8-hat representatives. 8 bits of an
    E8P code select a row of this table."""
    base = torch.cat([_enumerate_d8hat_abs_le_norm10(), _PAD29_RAW], dim=0)
    assert base.shape == (ABS_TABLE_SIZE, CODE_DIM), base.shape
    return _fix_parity_to_even(base)


def decode_from_table(abs_idx: Tensor, sign_bits7: Tensor, shift_bit: Tensor,
                       abs_table: Tensor) -> Tensor:
    """abs_idx: (...) long. sign_bits7: (..., 7) in {0,1}. shift_bit: (...)
    in {0,1}. abs_table: (K, 8). Returns (..., 8) decoded codewords."""
    base = abs_table[abs_idx]  # (..., 8)
    inferred = sign_bits7.sum(-1) % 2  # parity of the 7 free bits = 8th flip
    all_flip = torch.cat([sign_bits7, inferred.unsqueeze(-1)], dim=-1)
    signed = base * (1 - 2 * all_flip.to(base.dtype))
    shift = (2 * shift_bit.to(base.dtype) - 1) * 0.25  # 1 -> +0.25, 0 -> -0.25
    return signed + shift.unsqueeze(-1)


def _nearest_even_flip(x: Tensor, abs_table: Tensor):
    """For every row of abs_table, find the EVEN-popcount sign pattern that
    best matches x (elementwise), by taking the natural per-coordinate best
    sign and, if that has odd popcount, flipping whichever coordinate has
    the least confident (smallest |x_i * row_i|) sign to fix parity at
    minimal cost. This is the standard nearest-point trick for checkerboard
    (D_n-type) lattices: correct-and-repair is exact here because flipping
    any single coordinate is the cheapest possible parity fix.

    x: (..., 8). abs_table: (K, 8). Returns per-row (flip, vals, dist2) each
    broadcasting to (..., K, ...).
    """
    prod = x.unsqueeze(-2) * abs_table  # (..., K, 8)
    flip = (prod < 0).to(torch.int64)  # 1 where signs disagree
    parity_wrong = (flip.sum(-1) % 2 == 1)  # (..., K)

    min_coord = prod.abs().argmin(-1, keepdim=True)  # (..., K, 1)
    old_at_min = flip.gather(-1, min_coord).squeeze(-1)  # (..., K)
    new_at_min = old_at_min ^ parity_wrong.to(torch.int64)
    flip = flip.scatter(-1, min_coord, new_at_min.unsqueeze(-1))

    signs = 1 - 2 * flip  # (..., K, 8) in {-1, +1}
    vals = abs_table * signs
    dist2 = (x.unsqueeze(-2) - vals).pow(2).sum(-1)  # (..., K)
    return flip, vals, dist2


def encode_to_table(x: Tensor, abs_table: Tensor):
    """Nearest-neighbor encode x: (..., 8) against the full E8P grid (all 256
    base rows x 2 shifts x all valid even-parity sign patterns), by brute
    force over the 256 rows (K=256 is tiny, so this is fast enough on GPU or
    CPU without needing the production kernel's cache-optimized shortcuts).

    Returns (abs_idx, sign_bits7, shift_bit, vals) with vals the quantized
    approximation of x.
    """
    plus_flip, plus_vals, plus_d2 = _nearest_even_flip(x + 0.25, abs_table)
    minus_flip, minus_vals, minus_d2 = _nearest_even_flip(x - 0.25, abs_table)

    plus_idx = plus_d2.argmin(-1)
    minus_idx = minus_d2.argmin(-1)
    plus_best_d2 = plus_d2.gather(-1, plus_idx.unsqueeze(-1)).squeeze(-1)
    minus_best_d2 = minus_d2.gather(-1, minus_idx.unsqueeze(-1)).squeeze(-1)

    def gather_row(t, idx):  # t: (..., K, 8), idx: (...,) -> (..., 8)
        return t.gather(-2, idx[..., None, None].expand(*idx.shape, 1, CODE_DIM)).squeeze(-2)

    plus_flip_sel = gather_row(plus_flip, plus_idx)
    minus_flip_sel = gather_row(minus_flip, minus_idx)
    plus_vals_sel = gather_row(plus_vals, plus_idx) - 0.25
    minus_vals_sel = gather_row(minus_vals, minus_idx) + 0.25

    use_plus = plus_best_d2 <= minus_best_d2
    abs_idx = torch.where(use_plus, plus_idx, minus_idx)
    flip = torch.where(use_plus.unsqueeze(-1), plus_flip_sel, minus_flip_sel)
    shift_bit = (~use_plus).to(torch.int64)  # use_plus -> -0.25 -> shift_bit 0
    vals = torch.where(use_plus.unsqueeze(-1), plus_vals_sel, minus_vals_sel)

    return abs_idx, flip[..., :7], shift_bit, vals


def pack_code(abs_idx: Tensor, sign_bits7: Tensor, shift_bit: Tensor) -> Tensor:
    """Pack into one 16-bit int: [abs_idx:8][sign_bits7:7][shift_bit:1],
    matching the bit layout in the paper's Appendix C.2 worked example."""
    sign_int = torch.zeros_like(abs_idx)
    for i in range(7):
        sign_int = sign_int | (sign_bits7[..., i].to(torch.int64) << (6 - i))
    return (abs_idx.to(torch.int64) << 8) | (sign_int << 1) | shift_bit.to(torch.int64)


def unpack_code(code: Tensor):
    abs_idx = (code >> 8) & 0xFF
    shift_bit = code & 1
    sign_int = (code >> 1) & 0x7F
    bits = [(sign_int >> (6 - i)) & 1 for i in range(7)]
    sign_bits7 = torch.stack(bits, dim=-1)
    return abs_idx, sign_bits7, shift_bit


class E8PCodebook:
    """2-bit (16 bits / 8 dims), 8-dimensional vector quantization codebook
    based on the E8 lattice. See module docstring."""

    def __init__(self, device=None):
        self.abs_table = build_abs_table().to(device=device)

    def quantize(self, x: Tensor):
        """x: (..., 8) -> (vals, code16) where vals is the dequantized
        approximation and code16 is the packed 16-bit code."""
        abs_idx, sign_bits7, shift_bit, vals = encode_to_table(x, self.abs_table)
        code = pack_code(abs_idx, sign_bits7, shift_bit)
        return vals, code

    def dequantize(self, code: Tensor) -> Tensor:
        abs_idx, sign_bits7, shift_bit = unpack_code(code)
        return decode_from_table(abs_idx, sign_bits7, shift_bit, self.abs_table)
