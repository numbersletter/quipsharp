import torch

from quipsharp.codebooks.e8p import E8PCodebook
from quipsharp.codebooks.e8_1bit import E8OneBitCodebook, build_table, _norm2_shell
from quipsharp.rvq import RVQStage, fit_stage_scale, rvq_quantize, rvq_dequantize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_norm2_shell_is_e8_kissing_number():
    shell = _norm2_shell()
    assert shell.shape[0] == 240
    assert torch.allclose((shell ** 2).sum(-1), torch.full((240,), 2.0))


def test_1bit_table_has_256_unique_rows():
    table = build_table()
    assert table.shape == (256, 8)
    assert torch.unique(table, dim=0).shape[0] == 256


def test_1bit_quantize_dequantize_round_trip():
    cb = E8OneBitCodebook(device=DEVICE)
    torch.manual_seed(0)
    x = torch.randn(50, 8, device=DEVICE)
    vals, idx = cb.quantize(x)
    assert torch.allclose(cb.dequantize(idx), vals)


def test_rvq_4bit_beats_2bit_e8p():
    """4-bit RVQ (E8P applied twice, per Section 4.3) should meaningfully
    reduce MSE vs. plain 2-bit E8P alone (a specific ratio isn't claimed by
    the paper, so this checks a comfortable margin rather than a precise
    theoretical bound)."""
    torch.manual_seed(1)
    x = torch.randn(3000, 8, device=DEVICE)

    e8p = E8PCodebook(device=DEVICE)
    s0 = fit_stage_scale(e8p, seed=1, device=DEVICE)
    s1 = fit_stage_scale(e8p, seed=2, device=DEVICE)
    stages = [RVQStage(e8p, s0), RVQStage(e8p, s1)]

    vals_4bit, codes = rvq_quantize(x, stages)
    mse_4bit = (x - vals_4bit).pow(2).mean().item()

    vals_2bit, _ = e8p.quantize(x / s0)
    mse_2bit = (x - vals_2bit * s0).pow(2).mean().item()

    assert mse_4bit < mse_2bit * 0.85


def test_rvq_dequantize_matches_quantize_output():
    torch.manual_seed(2)
    x = torch.randn(200, 8, device=DEVICE)
    e8p = E8PCodebook(device=DEVICE)
    onebit = E8OneBitCodebook(device=DEVICE)
    s0 = fit_stage_scale(e8p, seed=3, device=DEVICE)
    s1 = fit_stage_scale(onebit, seed=4, device=DEVICE)
    stages = [RVQStage(e8p, s0), RVQStage(onebit, s1)]

    vals, codes = rvq_quantize(x, stages)
    redecoded = rvq_dequantize(codes, stages)
    assert torch.allclose(vals, redecoded, atol=1e-6)


def test_rvq_3bit_beats_2bit_e8p():
    """3-bit RVQ (2-bit E8P + 1-bit E8, per Section 4.3) should reduce MSE
    vs. plain 2-bit E8P alone, though by less than the 4-bit (E8P x2) case."""
    torch.manual_seed(3)
    x = torch.randn(3000, 8, device=DEVICE)

    e8p = E8PCodebook(device=DEVICE)
    onebit = E8OneBitCodebook(device=DEVICE)
    s0 = fit_stage_scale(e8p, seed=5, device=DEVICE)
    s1 = fit_stage_scale(onebit, seed=6, device=DEVICE)
    stages = [RVQStage(e8p, s0), RVQStage(onebit, s1)]

    vals_3bit, _ = rvq_quantize(x, stages)
    mse_3bit = (x - vals_3bit).pow(2).mean().item()

    vals_2bit, _ = e8p.quantize(x / s0)
    mse_2bit = (x - vals_2bit * s0).pow(2).mean().item()

    assert mse_3bit < mse_2bit
