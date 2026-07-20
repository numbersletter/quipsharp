import pytest
import torch

from quipsharp.codebooks.e8p import E8PCodebook
from quipsharp.rvq import RVQStage, fit_stage_scale
from quipsharp.quantize import quantize_linear, dequantize_linear, linear_forward, RVQCodebookAdapter
from quipsharp.ldlq import block_ldlq
from quipsharp.incoherence import reconstruct_weight
from quipsharp.rfft import RFFTPhases

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _correlated_hessian(n, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(n, n, generator=g).to(device)
    return A @ A.T + torch.eye(n, device=device)


@pytest.fixture(scope="module")
def e8p_cb():
    return E8PCodebook(device=DEVICE)


@pytest.fixture(scope="module")
def e8p_scale(e8p_cb):
    """fit_stage_scale is a ~7s sweep; a codebook's optimal scale doesn't
    depend on the layer being quantized, so fit it once and reuse."""
    return fit_stage_scale(e8p_cb, seed=42, device=DEVICE)


def test_quantize_then_dequantize_matches_what_q(e8p_cb, e8p_scale):
    torch.manual_seed(0)
    m, n, g = 16, 32, 8
    W = torch.randn(m, n, device=DEVICE) * 0.3
    H = _correlated_hessian(n, 1, DEVICE)

    layer, what_q = quantize_linear(W, H, e8p_cb, g=g, codebook_scale=e8p_scale)
    what_rebuilt = dequantize_linear(layer, e8p_cb)
    assert torch.allclose(what_rebuilt, what_q)


def test_linear_forward_approximates_original_layer(e8p_cb, e8p_scale):
    torch.manual_seed(1)
    m, n, g = 64, 64, 8
    W = torch.randn(m, n, device=DEVICE) * 0.2
    H = _correlated_hessian(n, 2, DEVICE)

    layer, _ = quantize_linear(W, H, e8p_cb, g=g, codebook_scale=e8p_scale)

    x = torch.randn(32, n, device=DEVICE)
    y_true = x @ W.T
    y_quant = linear_forward(x, layer, e8p_cb)

    rel_err = (y_quant - y_true).norm() / y_true.norm()
    assert rel_err < 0.5  # 2-bit/weight is lossy; this just guards against a broken pipeline


def test_full_pipeline_beats_naive_quantization_with_outlier(e8p_cb, e8p_scale):
    """The end-to-end value proposition of the whole paper: incoherence
    processing + adaptive (BlockLDLQ) rounding + E8P should beat naively
    E8P-quantizing each block of W independently with no incoherence
    processing, especially once W has an outlier and H has real
    correlation structure -- exactly the regime the paper targets. Both
    paths get the same RMS-based scale-to-codebook treatment (Appendix
    F.5), so the comparison isolates incoherence+feedback, not a scale
    mismatch."""
    torch.manual_seed(2)
    m, n, g = 128, 128, 8
    W = torch.randn(m, n, device=DEVICE) * 0.15
    W[7, 40] = 8.0  # inject an outlier weight
    H = _correlated_hessian(n, 3, DEVICE)

    # full pipeline: incoherence processing + BlockLDLQ + E8P. Compare error
    # back in the ORIGINAL W-basis (reconstruct_weight undoes the lossless
    # RHT/sign transform exactly), so it's directly comparable to the naive
    # baseline below regardless of which random signs quantize_linear drew.
    layer, what_q = quantize_linear(W, H, e8p_cb, g=g, codebook_scale=e8p_scale)
    W_q_full = reconstruct_weight(what_q, layer.signs)
    err_full = torch.trace((W_q_full - W) @ H @ (W_q_full - W).T).item()

    # naive baseline: quantize W directly with E8P, block by block, no
    # incoherence processing and no adaptive-rounding feedback (L = I) --
    # but still scaled to the codebook's expected RMS, same as the full path.
    L_identity = torch.eye(n, device=DEVICE)
    naive_scale = W.pow(2).mean().sqrt().item() * e8p_scale

    def quantize_fn(target, k):
        vals, _ = e8p_cb.quantize(target / naive_scale)
        return vals * naive_scale

    W_q_naive = block_ldlq(W, L_identity, g, quantize_fn)
    err_naive = torch.trace((W_q_naive - W) @ H @ (W_q_naive - W).T).item()

    assert err_full < err_naive * 0.5


def test_full_pipeline_with_4bit_rvq_beats_2bit(e8p_cb, e8p_scale):
    torch.manual_seed(3)
    m, n, g = 64, 64, 8
    W = torch.randn(m, n, device=DEVICE) * 0.2
    H = _correlated_hessian(n, 4, DEVICE)

    s1 = fit_stage_scale(e8p_cb, seed=11, device=DEVICE)
    rvq_cb = RVQCodebookAdapter([RVQStage(e8p_cb, e8p_scale), RVQStage(e8p_cb, s1)])
    rvq_scale = fit_stage_scale(rvq_cb, seed=12, device=DEVICE)

    layer_2bit, what_q_2bit = quantize_linear(W, H, e8p_cb, g=g, codebook_scale=e8p_scale)
    layer_4bit, what_q_4bit = quantize_linear(W, H, rvq_cb, g=g, codebook_scale=rvq_scale)

    W_q_2bit = reconstruct_weight(what_q_2bit, layer_2bit.signs)
    W_q_4bit = reconstruct_weight(what_q_4bit, layer_4bit.signs)

    err_2bit = (W_q_2bit - W).pow(2).mean().item()
    err_4bit = (W_q_4bit - W).pow(2).mean().item()

    assert err_4bit < err_2bit * 0.85

    x = torch.randn(16, n, device=DEVICE)
    y_true = x @ W.T
    y_4bit = linear_forward(x, layer_4bit, rvq_cb)
    rel_err = (y_4bit - y_true).norm() / y_true.norm()
    assert rel_err < 0.5


def test_tune_iters_wiring_reduces_error_and_codes_still_roundtrip(e8p_cb, e8p_scale):
    """quantize_linear used to never pass H/tune_iters into block_ldlq, so
    BlockLDLQ's refinement sweeps were dead code from this entry point.
    Confirm tune_iters>0 (a) actually changes/improves the H-weighted
    reconstruction error vs tune_iters=0, and (b) codes still round-trip
    correctly through dequantize_linear (guards against the codes-array
    being keyed by call order instead of block index, which silently
    permutes columns once blocks are revisited more than once each)."""
    torch.manual_seed(7)
    m, n, g = 64, 64, 8
    W = torch.randn(m, n, device=DEVICE) * 0.2
    H = _correlated_hessian(n, 8, DEVICE)

    layer_base, what_q_base = quantize_linear(W, H, e8p_cb, g=g, codebook_scale=e8p_scale,
                                               tune_iters=0)
    layer_refined, what_q_refined = quantize_linear(W, H, e8p_cb, g=g, codebook_scale=e8p_scale,
                                                     tune_iters=10)

    W_q_base = reconstruct_weight(what_q_base, layer_base.signs)
    W_q_refined = reconstruct_weight(what_q_refined, layer_refined.signs)
    err_base = torch.trace((W_q_base - W) @ H @ (W_q_base - W).T).item()
    err_refined = torch.trace((W_q_refined - W) @ H @ (W_q_refined - W).T).item()
    assert err_refined <= err_base * 1.01

    what_rebuilt = dequantize_linear(layer_refined, e8p_cb)
    assert torch.allclose(what_rebuilt, what_q_refined)


def test_full_pipeline_dispatches_to_rfft_for_non_pow2_dims(e8p_cb, e8p_scale):
    """Every real transformer layer width in the wild (e.g. Gemma 3 1B's
    hidden_size=1152) is NOT a power of two, so quantize_linear/
    linear_forward must transparently take the RFFT path (via
    incoherence_process_auto) and still work end to end."""
    torch.manual_seed(5)
    m, n, g = 48, 144, 8  # both even, neither a power of two (144 = 16*9)
    W = torch.randn(m, n, device=DEVICE) * 0.2
    H = _correlated_hessian(n, 6, DEVICE)

    layer, what_q = quantize_linear(W, H, e8p_cb, g=g, codebook_scale=e8p_scale)
    assert isinstance(layer.signs, RFFTPhases)

    W_q = layer.signs.reconstruct_weight(what_q)
    assert W_q.shape == W.shape

    x = torch.randn(20, n, device=DEVICE)
    y_true = x @ W.T
    y_quant = linear_forward(x, layer, e8p_cb)
    rel_err = (y_quant - y_true).norm() / y_true.norm()
    assert rel_err < 0.5


def test_full_pipeline_with_3bit_rvq_beats_2bit():
    """End-to-end 3-bit-beats-2-bit guard through build_codebook itself (the
    exact construction quantize_model.py and eval use). Regression: the 2nd
    RVQ stage was once scale-fit on N(0,1) instead of stage-1 residuals
    (Appendix F.5), leaving it effectively dead -- real 3-bit checkpoints
    evaluated worse than 2-bit. Non-power-of-two n forces the RFFT path."""
    from quipsharp.checkpoint import build_codebook

    torch.manual_seed(11)
    m, n, g = 64, 144, 8
    W = torch.randn(m, n, device=DEVICE) * 0.2
    H = _correlated_hessian(n, 12, DEVICE)

    cb2, scale2 = build_codebook("2bit", DEVICE, seed=0)
    cb3, scale3 = build_codebook("3bit", DEVICE, seed=0)

    # a healthy 1-bit residual stage must actually fire, not decode to zero
    layer_3bit, what_q_3bit = quantize_linear(W, H, cb3, g=g, codebook_scale=scale3)
    stage2_nonzero = (layer_3bit.codes[1] != 0).float().mean().item()
    assert stage2_nonzero > 0.5, f"1-bit stage decodes to zero on {1 - stage2_nonzero:.0%} of blocks"

    layer_2bit, what_q_2bit = quantize_linear(W, H, cb2, g=g, codebook_scale=scale2)
    W_q_2bit = layer_2bit.signs.reconstruct_weight(what_q_2bit)
    W_q_3bit = layer_3bit.signs.reconstruct_weight(what_q_3bit)

    err_2bit = torch.trace((W_q_2bit - W) @ H @ (W_q_2bit - W).T).item()
    err_3bit = torch.trace((W_q_3bit - W) @ H @ (W_q_3bit - W).T).item()
    assert err_3bit < err_2bit * 0.75, (
        f"3-bit H-weighted error {err_3bit:.4f} not meaningfully below 2-bit {err_2bit:.4f}")

    # and the extra stage must survive the code round trip (int16 buffers etc.)
    what_rebuilt = dequantize_linear(layer_3bit, cb3)
    assert torch.allclose(what_rebuilt, what_q_3bit)


def test_residual_stage_scale_fits_residual_not_unit_gaussian():
    """fit_residual_stage_scale on E8P residuals must land near the
    reference repo's 1/2.04 ~= 0.49 -- far below any N(0,1) fit."""
    from quipsharp.codebooks.e8_1bit import E8OneBitCodebook
    from quipsharp.rvq import fit_residual_stage_scale

    e8p = E8PCodebook(device=DEVICE)
    s1 = fit_stage_scale(e8p, seed=0, device=DEVICE, n_samples=20_000)
    onebit = E8OneBitCodebook(device=DEVICE)
    s2 = fit_residual_stage_scale([RVQStage(e8p, s1)], onebit, seed=1, device=DEVICE,
                                   n_samples=20_000)
    assert 0.2 < s2 <= 0.6, f"residual-fitted 1-bit stage scale {s2} not in the expected range"


def test_quantized_linear_preserves_bias(e8p_cb, e8p_scale):
    """The replaced nn.Linear's bias must be carried through QuantizedLinear
    verbatim and added in forward -- it was once silently dropped, which
    destroys bias-carrying models (Qwen2.5's q/k/v)."""
    from quipsharp.checkpoint import QuantizedLinear

    torch.manual_seed(13)
    m, n, g = 16, 32, 8
    W = torch.randn(m, n, device=DEVICE) * 0.2
    bias = torch.randn(m, device=DEVICE)
    H = _correlated_hessian(n, 14, DEVICE)

    layer, _ = quantize_linear(W, H, e8p_cb, g=g, codebook_scale=e8p_scale)
    qlin = QuantizedLinear.from_quantized_layer(layer, bias=bias).to(DEVICE)
    qlin.codebook = e8p_cb

    x = torch.randn(8, n, device=DEVICE)
    expected = linear_forward(x, layer, e8p_cb) + bias
    assert torch.allclose(qlin(x), expected, atol=1e-5)

    # the bias must survive a state_dict round trip into a freshly built module
    fresh = QuantizedLinear(n, m, g, qlin.transform_kind, has_bias=True).to(DEVICE)
    fresh.load_state_dict(qlin.state_dict())
    fresh.codebook = e8p_cb
    assert torch.allclose(fresh(x), expected, atol=1e-5)

    # and a no-bias module must behave exactly as before
    qlin_nobias = QuantizedLinear.from_quantized_layer(layer).to(DEVICE)
    qlin_nobias.codebook = e8p_cb
    assert qlin_nobias.bias is None
    assert "bias" not in qlin_nobias.state_dict()
    assert torch.allclose(qlin_nobias(x), expected - bias, atol=1e-5)


def test_quantized_linear_forward_handles_bf16_activations_on_rfft_path(e8p_cb, e8p_scale):
    """Every non-Llama-2 target checkpoint is bfloat16, and torch.view_as_complex
    (the first op in the RFFT input transform) rejects bfloat16 outright
    (fp16/fp32/fp64 only) -- this crashed a real Llama-3-8B quantized eval.
    QuantizedLinear.forward must explicitly upcast activations to float32
    before the transform and cast the result back to the caller's dtype."""
    from quipsharp.checkpoint import QuantizedLinear

    torch.manual_seed(9)
    m, n, g = 16, 24, 8  # n even but not a power of two -> forces the RFFT path
    W = torch.randn(m, n, device=DEVICE) * 0.2
    H = _correlated_hessian(n, 10, DEVICE)

    layer, _ = quantize_linear(W, H, e8p_cb, g=g, codebook_scale=e8p_scale)
    assert isinstance(layer.signs, RFFTPhases)

    qlin = QuantizedLinear.from_quantized_layer(layer).to(DEVICE)
    qlin.codebook = e8p_cb

    x = torch.randn(4, n, device=DEVICE)
    y_fp32 = qlin(x)
    y_bf16 = qlin(x.bfloat16())  # crashed with RuntimeError before the explicit upcast

    assert y_bf16.dtype == torch.bfloat16
    # identical up to the bf16 rounding of the input and output
    rel_err = (y_bf16.float() - y_fp32).norm() / y_fp32.norm()
    assert rel_err < 0.05
