import itertools

import torch
import pytest

from quipsharp.codebooks import e8p


@pytest.fixture(scope="module")
def abs_table():
    return e8p.build_abs_table()


def test_table_shape_and_counts():
    raw227 = e8p._enumerate_d8hat_abs_le_norm10()
    assert raw227.shape == (227, e8p.CODE_DIM)
    table = e8p.build_abs_table()
    assert table.shape == (256, e8p.CODE_DIM)


def test_table_rows_have_even_sum():
    table = e8p.build_abs_table()
    assert torch.all(table.sum(-1) % 2 == 0)


def test_table_rows_are_unique():
    table = e8p.build_abs_table()
    assert torch.unique(table, dim=0).shape[0] == 256


def test_expected_pattern_present(abs_table):
    """The abs pattern discussed in the paper's Appendix C.2 example (seven
    1/2's and one 3/2, here with the 3/2 in slot 3) should appear in the
    table, with the last coordinate pre-negated by this module's parity-fix
    (used directly in the worked-example test below)."""
    row = torch.tensor([0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5, -0.5])
    assert (abs_table.eq(row).all(dim=-1)).any()


def test_paper_worked_example_bit_decoding():
    """Appendix C.2's worked example, adapted to this module's convention
    (LAST coordinate carries the inferred sign, vs. the paper's prose which
    happens to infer the FIRST coordinate -- an arbitrary, equally valid
    choice; what matters is the parity arithmetic, which this checks).

    Paper: base s = {1/2,1/2,1/2,3/2,1/2,1/2,1/2,1/2} (odd sum, needs an odd
    number of flips); sign-decoded = {-1/2,-1/2,1/2,3/2,-1/2,1/2,-1/2,-1/2}.
    The paper's stated final vector (after "+1/4") is
    {-1/4,-3/4,3/4,7/4,-1/4,3/4,-1/4,-1/4} -- but -1/2+1/4=-1/4, not -3/4, at
    the 2nd coordinate; every other coordinate is consistent with a uniform
    "+1/4". This looks like an isolated arithmetic typo in the paper
    (arXiv:2402.04396v2, Appendix C.2) rather than a different intended rule,
    so this test checks against the arithmetically-consistent vector instead.

    Re-expressed with THIS module's even-sum canonical row (last coordinate
    pre-negated: {.5,.5,.5,1.5,.5,.5,.5,-.5}), the flips needed to reach the
    same sign-decoded vector are at coordinates {0,1,4,6} (0-indexed) among
    the free 7, with the 8th (last) coordinate correctly left unflipped
    since parity(4 flips) is even.
    """
    canon_row = torch.tensor([[0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5, -0.5]])
    sign_bits7 = torch.tensor([[1, 1, 0, 0, 1, 0, 1]])
    shift_bit = torch.tensor([1])

    out = e8p.decode_from_table(torch.tensor([0]), sign_bits7, shift_bit, canon_row)
    sign_decoded = torch.tensor([[-0.5, -0.5, 0.5, 1.5, -0.5, 0.5, -0.5, -0.5]])
    assert torch.allclose(sign_decoded + 0.25, torch.tensor(
        [[-0.25, -0.25, 0.75, 1.75, -0.25, 0.75, -0.25, -0.25]]))
    assert torch.allclose(out, sign_decoded + 0.25)


def test_pack_unpack_round_trip():
    torch.manual_seed(0)
    n = 500
    abs_idx = torch.randint(0, 256, (n,))
    sign_bits7 = torch.randint(0, 2, (n, 7))
    shift_bit = torch.randint(0, 2, (n,))
    code = e8p.pack_code(abs_idx, sign_bits7, shift_bit)
    assert code.min() >= 0 and code.max() < (1 << 16)
    a2, s2, sh2 = e8p.unpack_code(code)
    assert torch.equal(a2, abs_idx)
    assert torch.equal(s2, sign_bits7)
    assert torch.equal(sh2, shift_bit)


def test_quantize_dequantize_consistent(abs_table):
    torch.manual_seed(1)
    x = torch.randn(100, 8) * 0.5
    cb = e8p.E8PCodebook()
    vals, code = cb.quantize(x)
    redecoded = cb.dequantize(code)
    assert torch.allclose(vals, redecoded, atol=1e-6)


def test_decoded_vectors_are_in_e8_plus_quarter(abs_table):
    """Every decoded codeword, shifted by -+1/4 back down, must land exactly
    on the E8 lattice (all-integer or all-half-integer coords summing even)."""
    torch.manual_seed(2)
    x = torch.randn(200, 8)
    cb = e8p.E8PCodebook()
    vals, code = cb.quantize(x)
    _, _, shift_bit = e8p.unpack_code(code)
    unshifted = vals - torch.where(shift_bit.bool(), 0.25, -0.25).unsqueeze(-1)
    is_int = torch.allclose(unshifted, unshifted.round(), atol=1e-5)
    is_half_int = torch.allclose((unshifted * 2), (unshifted * 2).round(), atol=1e-5)
    assert is_int or is_half_int
    sums = unshifted.sum(-1)
    assert torch.allclose(sums.round() % 2, torch.zeros_like(sums), atol=1e-5)


def _brute_force_nearest_even_flip(x_row: torch.Tensor, table_row: torch.Tensor):
    """O(2^7) ground truth for a single (x, table_row) pair: try every
    even-popcount sign pattern explicitly."""
    best_d2 = None
    best_vals = None
    for bits in itertools.product([0, 1], repeat=7):
        parity = sum(bits) % 2
        flips = list(bits) + [parity]
        signs = torch.tensor([1 - 2 * f for f in flips], dtype=torch.float32)
        vals = table_row * signs
        d2 = (x_row - vals).pow(2).sum().item()
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_vals = vals
    return best_d2, best_vals


def test_nearest_even_flip_matches_brute_force(abs_table):
    """Validate the O(8)-per-row greedy parity-fix trick in _nearest_even_flip
    against true brute force over all 128 valid sign patterns, for a handful
    of random query vectors and table rows."""
    torch.manual_seed(3)
    row_ids = torch.randint(0, 256, (10,))
    for x in torch.randn(6, 8):
        for r in row_ids:
            table_row = abs_table[r]
            _, vals_all, dist2_all = e8p._nearest_even_flip(x.unsqueeze(0), table_row.unsqueeze(0))
            got_d2 = dist2_all[0, 0].item()
            want_d2, _ = _brute_force_nearest_even_flip(x, table_row)
            assert got_d2 == pytest.approx(want_d2, abs=1e-5), (x, table_row)


def _fit_lloyd_max_4level(samples: torch.Tensor, iters: int = 100) -> torch.Tensor:
    """1D Lloyd-Max (= 1D k-means) quantizer levels for scalar 2-bit (4
    level) quantization -- the correct, bit-budget-matched scalar baseline
    for E8P (16 bits / 8 dims = 2 bits/coordinate)."""
    levels = torch.quantile(samples, torch.tensor([0.125, 0.375, 0.625, 0.875]))
    for _ in range(iters):
        d2 = (samples.unsqueeze(-1) - levels.unsqueeze(0)) ** 2
        assign = d2.argmin(-1)
        for k in range(4):
            mask = assign == k
            if mask.any():
                levels[k] = samples[mask].mean()
    return levels


def test_quantize_beats_bitrate_matched_scalar_quantizer():
    """Figure 3's core claim: at a matched bit budget, the E8 lattice
    codebook (16 bits / 8 dims = 2 bits/coordinate) achieves lower MSE than
    the best possible SCALAR (1D) quantizer at the same 2 bits/coordinate.
    (A naive "round to nearest half-integer" baseline would be unfair here
    since it has unbounded levels, i.e. an unbounded bitrate.)"""
    torch.manual_seed(4)
    x = torch.randn(2000, 8)
    cb = e8p.E8PCodebook()
    vals, _ = cb.quantize(x)
    e8p_mse = (x - vals).pow(2).mean().item()

    levels = _fit_lloyd_max_4level(x.reshape(-1))
    d2 = (x.reshape(-1, 1) - levels.unsqueeze(0)) ** 2
    scalar_quantized = levels[d2.argmin(-1)].reshape(x.shape)
    scalar_mse = (x - scalar_quantized).pow(2).mean().item()

    assert e8p_mse < scalar_mse
