"""Algorithm 1 (QuIP# without fine-tuning) and Algorithm 2 (inference for a
quantized linear layer), tying together incoherence processing + BlockLDLQ
+ a lattice codebook (E8P, or RVQ-wrapped E8P for 3/4 bit).

This is a *reference* implementation of Algorithm 2: it densely
dequantizes the whole weight matrix and does a normal matmul, rather than
the paper's custom CUDA kernel that decodes codewords on the fly inside the
GEMV without ever materializing the dense weight. That kernel is a separate,
GPU-systems-engineering project; this module is about getting the
quantization *algorithm* correct first.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from .incoherence import incoherence_process_auto
from .ldlq import block_ldl, block_ldlq
from .rvq import RVQStage, rvq_quantize, rvq_dequantize


@dataclass
class QuantizedLayer:
    codes: Tensor  # (..., m, n // g): per-block codes from the codebook
    signs: object  # IncoherenceSigns (RHT) or RFFTPhases -- see incoherence_process_auto
    g: int
    shape: tuple[int, int]  # original (m, n), for sanity checks
    scale: float  # see quantize_linear's docstring on codebook_scale


class RVQCodebookAdapter:
    """Wraps a list of RVQStages behind the single-codebook .quantize/
    .dequantize interface, so 3/4-bit RVQ is a drop-in replacement for plain
    2-bit E8P in quantize_linear/dequantize_linear/linear_forward."""

    def __init__(self, stages: list[RVQStage]):
        self.stages = stages

    def quantize(self, x: Tensor):
        vals, codes = rvq_quantize(x, self.stages)
        return vals, torch.stack(codes, dim=0)  # (num_stages, ...)

    def dequantize(self, code: Tensor) -> Tensor:
        codes = [code[i] for i in range(code.shape[0])]
        return rvq_dequantize(codes, self.stages)


def quantize_linear(W: Tensor, H: Tensor, codebook, g: int = 8,
                     damping: float = 1e-2, generator=None,
                     codebook_scale: float = 1.0, tune_iters: int = 10):
    """Algorithm 1: QuIP# without fine-tuning, for one nn.Linear's weight.

    W: (m, n) weight. H: (n, n) proxy Hessian (E[x x^T] over calibration
    activations). codebook: anything with .quantize(x: (m,g)) -> (vals,
    code) / .dequantize(code) -> vals (E8PCodebook, E8OneBitCodebook, or
    RVQCodebookAdapter all satisfy this).

    codebook_scale: the codebook's own preferred input scale, i.e. the `s`
    from `rvq.fit_stage_scale(codebook)` -- codebooks like E8P are built
    around roughly-unit-variance Gaussian input (their smallest nonzero
    magnitude is 0.25), so a layer's actual (usually much smaller or
    larger) weight magnitude must be normalized to match before quantizing,
    then undone after (Appendix F.5's "scale by rho"). Computing this is a
    one-time cost per *codebook* (not per layer), so callers should compute
    it once (e.g. via fit_stage_scale) and pass it in rather than
    recomputing it on every quantize_linear call.

    tune_iters: extra BlockLDLQ refinement sweeps using the full Hessian
    (see ldlq.block_ldlq's docstring); matches the reference repo's
    --quip_tune_iters default of 10. Pass 0 to disable and only run the
    base single pass.

    Returns (layer, what_q) where `layer` is the compact representation to
    store/ship, and `what_q` is the dense quantized transformed weight
    (handy for immediately checking error without a round trip).
    """
    # damp H once up front (regularize_H): the refinement sweeps must see
    # the damped H too, not just the LDL
    H = H + damping * H.diagonal().mean() * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    what, hhat, signs = incoherence_process_auto(W, H, generator=generator)
    L, _ = block_ldl(hhat, g, damping=0.0)

    rms = what.pow(2).mean().sqrt().item()
    layer_scale = rms * codebook_scale
    what_scaled = what / layer_scale

    T = what.shape[1] // g
    codes = [None] * T

    def quantize_fn(target, k):
        vals, code = codebook.quantize(target)
        codes[k] = code  # keyed by block index, not call order: block_ldlq
        return vals       # visits blocks last-to-first and revisits during refinement

    what_q_scaled = block_ldlq(what_scaled, L, g, quantize_fn, H=hhat, tune_iters=tune_iters)
    what_q = what_q_scaled * layer_scale
    # each per-block code has shape (..., m) (plain codebook) or
    # (num_stages, m) (RVQCodebookAdapter); stacking a new LAST axis gives
    # (..., m, T) either way, so layer.codes[..., k] recovers block k's code.
    codes_stacked = torch.stack(codes, dim=-1)

    layer = QuantizedLayer(codes=codes_stacked, signs=signs, g=g,
                            shape=tuple(W.shape), scale=layer_scale)
    return layer, what_q


def dequantize_linear(layer: QuantizedLayer, codebook) -> Tensor:
    """Reconstruct the transformed weight What (NOT the original W) from
    stored per-block codes."""
    m, n = layer.shape
    T = n // layer.g
    blocks = [codebook.dequantize(layer.codes[..., k]) for k in range(T)]
    return torch.cat(blocks, dim=-1) * layer.scale  # (m, n)


def linear_forward(x: Tensor, layer: QuantizedLayer, codebook) -> Tensor:
    """Algorithm 2 (reference/dense version): apply a QuIP#-quantized linear
    layer to input x: (..., n) -> (..., m). Works regardless of whether the
    layer used RHT or RFFT incoherence processing, via the common
    transform_input/transform_output protocol (see incoherence_process_auto)."""
    what_q = dequantize_linear(layer, codebook)
    x2 = layer.signs.transform_input(x)
    y1 = x2 @ what_q.T
    return layer.signs.transform_output(y1)
