"""BlockLDLQ: adaptive-rounding vector quantization with linear feedback.

Reference: QuIP# paper, Section 4.1, Theorem 4.1, and Appendix B (Block LDL).
This module's decomposition and rounding order match the released QuIP#
repo's actual implementation (lib/utils/math_utils.py:block_LDL and
lib/algo/quip.py:LDLQ) rather than a literal reading of the paper's H =
L^T D L pseudocode -- both are valid, self-consistent parameterizations of
the same general scheme, but matching the repo lets this module include the
repo's additional iterative refinement pass (see block_ldlq's `tune_iters`),
which is not described in the paper's Section 4.1 text.

Decomposition: H = L @ blockdiag(D) @ L^T, L unit block-lower-triangular
(standard LDL^T, L lower). Computed by taking an ordinary (scalar) Cholesky
factor H = L0 @ L0^T and rescaling each block-COLUMN of L0 by the inverse
of its own diagonal block so every diagonal block becomes I:
    D_i = L0_ii @ L0_ii^T
    L[:, i, :] = L0[:, i, :] @ L0_ii^{-1}

Rounding then proceeds LAST block to FIRST (block T-1 first, using no
feedback since nothing follows it), each block's target adjusted by the
rounding error already committed on later blocks, weighted by L's
sub-diagonal entries:
    What_k = Q(W_k + sum_{j>k} (W_j - What_j) @ L_{j,k})

Optionally followed by `tune_iters` additional sweeps (default 10, matching
the repo's `--quip_tune_iters` default) that revisit every block using the
FULL Hessian and the *current* global residual:
    What_k <- Q(What_k + (W - What) @ H[:, k] @ H[k, k]^{-1})
This is a block-coordinate-descent / Gauss-Seidel step: holding every other
block's current (quantized) value fixed, it's the exact minimizer of
tr((What-W) H (What-W)^T) over block k's continuous value, before
requantizing that optimum. It's not part of the paper's stated algorithm or
Theorem 4.1's bound, but is enabled by default in the released code.
"""
import torch
from torch import Tensor


def block_ldl(H: Tensor, g: int, damping: float = 1e-2) -> tuple[Tensor, Tensor]:
    """g-block LDL decomposition H = L @ blockdiag(D) @ L^T.

    Returns L: (n, n) dense, unit block-lower-triangular.
            D: (T, g, g) stacked diagonal blocks, T = n // g.
    """
    n = H.shape[0]
    assert n % g == 0, f"block size {g} must evenly divide n={n}"
    T = n // g

    M = H.clone()
    mean_diag = M.diagonal().mean()
    if damping:
        M += damping * mean_diag * torch.eye(n, device=H.device, dtype=H.dtype)

    try:
        L0 = torch.linalg.cholesky(M)  # M = L0 @ L0^T, L0 lower triangular (not yet block-unit)
    except torch.linalg.LinAlgError:
        # ill-conditioned H can defeat fp32 Cholesky: retry in fp64,
        # escalating damping only if fp64 alone isn't enough
        M = M.double()
        extra = 0.0
        for attempt in range(4):
            try:
                L0 = torch.linalg.cholesky(M)
                break
            except torch.linalg.LinAlgError:
                if attempt == 3:
                    raise
                step = float(mean_diag) * (damping if damping else 1e-2) * 10 ** attempt
                extra += step
                M += step * torch.eye(n, device=H.device, dtype=torch.float64)
        print(f"block_ldl: fp32 Cholesky failed (n={n}); succeeded in fp64 with "
              f"{extra / float(mean_diag):.0e} x mean_diag extra damping", flush=True)

    DL = torch.diagonal(L0.reshape(T, g, T, g), dim1=0, dim2=2).permute(2, 0, 1).clone()  # (T,g,g)
    D = DL @ DL.transpose(-1, -2)
    DL_inv = torch.linalg.inv(DL)

    L = L0.reshape(n, T, g)
    for i in range(T):
        L[:, i, :] = L[:, i, :] @ DL_inv[i]
    L = L.reshape(n, n)

    # downcast in case the fp64 retry path ran (callers matmul L against fp32 W)
    return L.to(H.dtype), D.to(H.dtype)


def block_diag_from_stack(D: Tensor) -> Tensor:
    """(T, g, g) -> (n, n) block-diagonal matrix. Testing/debug convenience."""
    T, g, _ = D.shape
    n = T * g
    out = torch.zeros(n, n, device=D.device, dtype=D.dtype)
    for j in range(T):
        out[j * g:(j + 1) * g, j * g:(j + 1) * g] = D[j]
    return out


def block_ldlq(W: Tensor, L: Tensor, g: int, quantize_fn, H: Tensor = None,
               tune_iters: int = 10) -> Tensor:
    """Round W (m, n) block-column-wise with linear feedback, per Section 4.1,
    processing blocks last-to-first (see module docstring).

    quantize_fn(target: (m, g), k: int) -> (m, g): applies a vector quantizer
    independently to each of the m rows (each row is one g-dim vector to
    quantize, e.g. E8PCodebook.quantize wrapped to return just the values).
    `k` is the block index being quantized (0-indexed into W's column
    blocks) -- callers that need to remember per-block state (e.g. storing
    codes for later dequantization) must key it off `k`, not off call order:
    blocks are visited last-to-first, and refinement revisits every block
    `tune_iters` more times, so call order does not match block order and
    a block is generally quantized more than once.

    H: the same proxy Hessian used to build L. If given (and tune_iters >
    0), runs the reference repo's additional iterative refinement sweeps
    after the base pass (see module docstring); if None, only the base
    single pass runs.
    """
    m, n = W.shape
    assert n % g == 0
    T = n // g

    Wq = torch.zeros_like(W)

    for k in reversed(range(T)):
        cols_k = slice(k * g, (k + 1) * g)
        tail = slice((k + 1) * g, n)
        feedback = (W[:, tail] - Wq[:, tail]) @ L[tail, cols_k]
        target = W[:, cols_k] + feedback
        Wq[:, cols_k] = quantize_fn(target, k)

    if H is not None and tune_iters > 0:
        diag_blocks = torch.stack([H[k * g:(k + 1) * g, k * g:(k + 1) * g] for k in range(T)])
        diag_inv = torch.linalg.inv(diag_blocks)  # (T, g, g), precomputed once (H is fixed)
        for _ in range(tune_iters):
            for k in reversed(range(T)):
                cols_k = slice(k * g, (k + 1) * g)
                resid = W - Wq
                adjustment = resid @ H[:, cols_k] @ diag_inv[k]
                target = Wq[:, cols_k] + adjustment
                Wq[:, cols_k] = quantize_fn(target, k)

    return Wq
