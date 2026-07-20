"""Residual Vector Quantization (RVQ), Section 4.3.

RVQ(x, p, q) reaches p bits/vector using a sequence of q_i-bit codebooks by
repeatedly quantizing the residual left over from previous stages:
    delta_i = Q_{q_i}((x - sum_{j<i} delta_j) / s_i) * s_i
    RVQ(x) = sum_i delta_i
Each stage has its own scale s_i, fit (per Appendix F.5) by minimizing
quantization error of Gaussian samples against that stage's codebook -- not
derived analytically, so we do the same coarse sweep here rather than
hardcoding the paper's specific reported constants (which are tied to their
exact codebook construction).

QuIP# builds 4-bit from two 2-bit E8P stages, and 3-bit from one 2-bit E8P
stage plus one 1-bit E8 stage (see codebooks/e8_1bit.py).
"""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RVQStage:
    codebook: object  # has .quantize(x) -> (vals, code) and .dequantize(code) -> vals
    scale: float


def _sweep_scale(x: Tensor, codebook, candidates) -> float:
    """The scale s among `candidates` minimizing E[||x - s*dequant(quantize(x/s))||^2]."""
    best_scale, best_mse = None, float("inf")
    for s in candidates:
        s = s.item()
        vals, _ = codebook.quantize(x / s)
        mse = (x - vals * s).pow(2).mean().item()
        if mse < best_mse:
            best_mse, best_scale = mse, s
    return best_scale


def fit_stage_scale(codebook, dim: int = 8, n_samples: int = 200_000,
                     candidates=None, seed: int = 0, device=None) -> float:
    """Coarse-sweep the scale s minimizing E[||x - s*dequant(quantize(x/s))||^2]
    for x ~ N(0, I_dim), matching the methodology in Appendix F.5."""
    if candidates is None:
        candidates = torch.linspace(0.5, 4.0, 71)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, dim, generator=g)
    if device is not None:
        x = x.to(device)
    return _sweep_scale(x, codebook, candidates)


def fit_residual_stage_scale(prev_stages: list["RVQStage"], codebook, dim: int = 8,
                              n_samples: int = 200_000, candidates=None, seed: int = 0,
                              device=None) -> float:
    """Fit a next RVQ stage's scale on the residual left by `prev_stages`
    on x ~ N(0, I_dim) -- Appendix F.5 / the reference repo's
    opt_resid_scale (1/2.04 for 3-bit in divide-by-s convention)."""
    if candidates is None:
        candidates = torch.linspace(0.05, 2.0, 79)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, dim, generator=g)
    if device is not None:
        x = x.to(device)
    vals, _ = rvq_quantize(x, prev_stages)
    return _sweep_scale(x - vals, codebook, candidates)


def rvq_quantize(x: Tensor, stages: list[RVQStage]):
    """x: (..., d). Returns (vals, codes) where vals is the summed
    dequantized approximation and codes is a list of per-stage codes."""
    residual = x
    total = torch.zeros_like(x)
    codes = []
    for stage in stages:
        vals, code = stage.codebook.quantize(residual / stage.scale)
        total = total + vals * stage.scale
        residual = residual - vals * stage.scale
        codes.append(code)
    return total, codes


def rvq_dequantize(codes: list, stages: list[RVQStage]) -> Tensor:
    total = None
    for stage, code in zip(stages, codes):
        vals = stage.codebook.dequantize(code) * stage.scale
        total = vals if total is None else total + vals
    return total
