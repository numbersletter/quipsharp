"""Compact on-disk format for QuIP#-quantized checkpoints: per-layer codes
(the same packed representation quipsharp/quantize.py's QuantizedLayer
already produces -- e.g. one 16-bit code per 8-dim block for 2-bit E8P)
instead of a dense dequantized weight.

This matters for memory, not just disk. Before this module existed,
scripts/quantize_model.py immediately dequantized each layer back to a
dense (fp16-sized) weight and saved via a stock `model.save_pretrained` --
so a "quantized" 70B checkpoint was still ~140GB on disk and needed the
same ~140GB resident to load, defeating the entire point of quantizing on a
machine with 83GB RAM / 2x24GB GPUs. Storing the compact codes instead
makes a 70B 2-bit checkpoint ~17GB, all of which comfortably fits on a
single 24GB GPU -- `QuantizedLinear.forward` dequantizes that one layer's
dense weight on the fly for the matmul (freed again right after), rather
than ever holding the whole model densely at once. This is the
memory-bounded analogue of the paper's custom fused CUDA kernel, just
without the fusion (see quantize.py's module docstring for why the kernel
itself is out of scope for this project).

On-disk layout (a directory):
    quip_config.json    -- manifest: source model_id, codebook bits, block
                            size, seed, and per-quantized-module shape info
    quantized_state.pt   -- torch.save of the whole (post-surgery) model's
                            state_dict: every QuantizedLinear's compact
                            codes/signs/scale buffers, plus every
                            NON-quantized tensor (embeddings, lm_head,
                            layernorms) verbatim -- i.e. everything needed
                            to reconstruct a runnable model, nothing more.
    tokenizer files      -- from tokenizer.save_pretrained, unchanged.

Deliberately NOT a stock HF checkpoint (no safe_serialization/config.json
tied to a registered AutoModel class): the point is a small, simple, fully
self-contained format for this project's own load_quantized_checkpoint,
not interop with arbitrary HF tooling. Plain torch.save also sidesteps
safetensors' lack of complex-dtype support, which the RFFT transform's
phase_u/phase_v buffers would otherwise hit for every non-power-of-two
layer (i.e. nearly every real layer -- see incoherence.py's docstring).
"""
import json
from pathlib import Path

import torch
from torch import nn, Tensor

from .incoherence import IncoherenceSigns
from .rfft import RFFTPhases
from .quantize import QuantizedLayer, linear_forward, RVQCodebookAdapter
from .codebooks.e8p import E8PCodebook
from .codebooks.e8_1bit import E8OneBitCodebook
from .rvq import RVQStage, fit_stage_scale, fit_residual_stage_scale

MANIFEST_NAME = "quip_config.json"
STATE_NAME = "quantized_state.pt"


def num_rvq_stages(bits: str) -> int:
    return 1 if bits == "2bit" else 2


def build_codebook(bits: str, device, seed: int = 0):
    """Builds the codebook + fitted input scale for `bits`: plain E8P
    (2-bit) or an RVQ of [E8P, E8-1bit] (3-bit) / [E8P, E8P] (4-bit).
    Deterministic given `seed`, so (bits, seed) alone rebuilds the
    identical codebook at load time."""
    e8p = E8PCodebook(device=device)
    e8p_scale = fit_stage_scale(e8p, seed=seed, device=device, n_samples=20_000)
    if bits == "2bit":
        return e8p, e8p_scale

    second_cb = E8OneBitCodebook(device=device) if bits == "3bit" else E8PCodebook(device=device)
    # fit on stage 1's residuals (Appendix F.5); changing this invalidates
    # previously saved 3/4-bit checkpoints
    second_scale = fit_residual_stage_scale([RVQStage(e8p, e8p_scale)], second_cb,
                                             seed=seed + 1, device=device, n_samples=20_000)
    rvq_cb = RVQCodebookAdapter([RVQStage(e8p, e8p_scale), RVQStage(second_cb, second_scale)])
    rvq_scale = fit_stage_scale(rvq_cb, seed=seed + 2, device=device, n_samples=20_000)
    return rvq_cb, rvq_scale


class QuantizedLinear(nn.Module):
    """Drop-in replacement for one quantized nn.Linear: holds compact codes
    + transform signs + scale as buffers, dequantizes -> matmuls in forward.
    `codebook` is stateless and shared across layers, so the caller builds
    it once and assigns it after construction (never serialized)."""

    def __init__(self, in_features: int, out_features: int, g: int, transform_kind: str,
                 num_stages: int = 1, has_bias: bool = False):
        super().__init__()
        assert transform_kind in ("rht", "rfft")
        self.in_features = in_features
        self.out_features = out_features
        self.g = g
        self.transform_kind = transform_kind
        self.num_stages = num_stages
        self.codebook = None  # assigned by the caller after construction
        # the replaced nn.Linear's bias, verbatim (only the weight is quantized)
        self.register_buffer("bias", torch.zeros(out_features) if has_bias else None)

        T = in_features // g
        codes_shape = (out_features, T) if num_stages == 1 else (num_stages, out_features, T)
        # int16: packed E8P codes are 16-bit (int64 would 4x the checkpoint);
        # _quantized_layer recovers the unsigned value via `.to(int64) & 0xFFFF`
        self.register_buffer("codes", torch.zeros(codes_shape, dtype=torch.int16))
        self.register_buffer("scale", torch.zeros(()))
        if transform_kind == "rht":
            self.register_buffer("su", torch.zeros(out_features))
            self.register_buffer("sv", torch.zeros(in_features))
        else:
            self.register_buffer("phase_u", torch.zeros(out_features // 2, dtype=torch.complex64))
            self.register_buffer("phase_v", torch.zeros(in_features // 2, dtype=torch.complex64))

    @classmethod
    def from_quantized_layer(cls, layer: QuantizedLayer, bias: Tensor = None) -> "QuantizedLinear":
        m, n = layer.shape
        kind = "rht" if isinstance(layer.signs, IncoherenceSigns) else "rfft"
        stages = layer.codes.shape[0] if layer.codes.dim() == 3 else 1
        mod = cls(in_features=n, out_features=m, g=layer.g, transform_kind=kind, num_stages=stages,
                  has_bias=bias is not None)
        with torch.no_grad():
            mod.codes.copy_(layer.codes)
            mod.scale.copy_(torch.as_tensor(layer.scale, dtype=torch.float32))
            if bias is not None:
                mod.bias.copy_(bias.float())
            if kind == "rht":
                mod.su.copy_(layer.signs.su)
                mod.sv.copy_(layer.signs.sv)
            else:
                mod.phase_u.copy_(layer.signs.phase_u)
                mod.phase_v.copy_(layer.signs.phase_v)
        return mod

    def _quantized_layer(self) -> QuantizedLayer:
        signs = (IncoherenceSigns(su=self.su, sv=self.sv) if self.transform_kind == "rht"
                 else RFFTPhases(phase_u=self.phase_u, phase_v=self.phase_v))
        # undo the int16 compaction (indexing also rejects int16 tensors)
        codes = self.codes.to(torch.int64) & 0xFFFF
        return QuantizedLayer(codes=codes, signs=signs, g=self.g,
                               shape=(self.out_features, self.in_features),
                               scale=self.scale.item())

    def forward(self, x: Tensor) -> Tensor:
        # explicit fp32 upcast (view_as_complex rejects bf16), restore dtype at the end
        y = linear_forward(x.float(), self._quantized_layer(), self.codebook)
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)


def replace_with_quantized_linear_(parent: nn.Module, attr: str, layer: QuantizedLayer,
                                    codebook, bias: Tensor = None) -> QuantizedLinear:
    """Swap `parent`'s `attr` submodule (an nn.Linear) for a QuantizedLinear
    built from `layer` + `codebook`, carrying over `bias` verbatim."""
    qlin = QuantizedLinear.from_quantized_layer(layer, bias=bias)
    qlin.codebook = codebook
    setattr(parent, attr, qlin)
    return qlin


def save_quantized_checkpoint(model: nn.Module, model_id: str, save_path, bits: str,
                               block_size: int, seed: int, quantized_modules: list[dict],
                               tokenizer=None) -> None:
    """Save `model` (with its target nn.Linear submodules already replaced
    by QuantizedLinear, e.g. via replace_with_quantized_linear_) in this
    module's compact format. `quantized_modules`: list of
    {"name": <dotted path from model root, e.g. "model.layers.0.self_attn.q_proj">,
     "in_features": int, "out_features": int, "transform_kind": "rht"|"rfft"}
    for every replaced module -- load_quantized_checkpoint needs these to
    reconstruct each QuantizedLinear's buffer shapes before it can
    load_state_dict into them."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path / STATE_NAME)

    manifest = {
        "model_id": model_id,
        "bits": bits,
        "block_size": block_size,
        "seed": seed,
        "quantized_modules": quantized_modules,
    }
    (save_path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)


def is_quantized_checkpoint(path) -> bool:
    return (Path(path) / MANIFEST_NAME).exists()


def load_quantized_checkpoint(save_path, device="cuda", gpu_devices=None,
                               gpu_budget_bytes=None, cpu_budget_bytes=0):
    """Inverse of save_quantized_checkpoint: rebuilds the meta-device
    architecture skeleton for the manifest's source model_id (never
    re-downloads/re-materializes the ORIGINAL dense checkpoint -- only this
    project's own compact saved state), swaps in empty QuantizedLinear
    placeholders of the right shape at each quantized module, then loads
    the saved state_dict with assign=True (which, unlike the default
    copy_-based load, can replace meta tensors outright instead of trying
    to copy data into them). Returns the ready-to-use nn.Module.

    Placement: everything lands on `device` unless the checkpoint exceeds
    `gpu_budget_bytes` (e.g. a ~35GB 3-bit 70B), in which case decoder
    layers are bin-packed across `gpu_devices` (then CPU) via
    accelerate.dispatch_model. Each QuantizedLinear gets a codebook built on
    its own device -- advanced indexing needs codes and tables co-located."""
    from .lazy_weights import build_meta_model

    save_path = Path(save_path)
    manifest = json.loads((save_path / MANIFEST_NAME).read_text())
    state = torch.load(save_path / STATE_NAME, map_location="cpu")

    model, _text_config, _full_config = build_meta_model(manifest["model_id"])

    stages = num_rvq_stages(manifest["bits"])
    for info in manifest["quantized_modules"]:
        parent_name, _, attr = info["name"].rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        qlin = QuantizedLinear(info["in_features"], info["out_features"], manifest["block_size"],
                                info["transform_kind"], num_stages=stages,
                                has_bias=info.get("has_bias", False))
        setattr(parent, attr, qlin)

    model.load_state_dict(state, strict=True, assign=True)

    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    fits_one_device = (gpu_devices is None or gpu_budget_bytes is None
                       or total_bytes <= gpu_budget_bytes)

    if fits_one_device:
        device = gpu_devices[0] if gpu_devices else device
        model.to(device)
        device_of_module = {info["name"]: device for info in manifest["quantized_modules"]}
    else:
        from accelerate import dispatch_model

        layer_bytes: dict[int, int] = {}
        for key, t in state.items():
            if key.startswith("model.layers."):
                idx = int(key.split(".")[2])
                layer_bytes[idx] = layer_bytes.get(idx, 0) + t.numel() * t.element_size()

        devices = list(gpu_devices) + (["cpu"] if cpu_budget_bytes else [])
        budgets = [gpu_budget_bytes] * len(gpu_devices) + \
            ([cpu_budget_bytes] if cpu_budget_bytes else [])
        main_device = gpu_devices[0]

        device_map = {}
        dev_idx, used = 0, 0
        for i in sorted(layer_bytes):
            while dev_idx < len(devices) - 1 and used + layer_bytes[i] > budgets[dev_idx]:
                dev_idx, used = dev_idx + 1, 0
            device_map[f"model.layers.{i}"] = devices[dev_idx]
            used += layer_bytes[i]
        for name, _ in model.model.named_children():
            if name != "layers":
                device_map[f"model.{name}"] = main_device
        if hasattr(model, "lm_head"):
            device_map["lm_head"] = main_device

        model = dispatch_model(model, device_map=device_map, main_device=main_device)
        device_of_module = {
            info["name"]: device_map[".".join(info["name"].split(".")[:3])]
            for info in manifest["quantized_modules"]}

    codebook_per_device = {}
    for info in manifest["quantized_modules"]:
        dev = device_of_module[info["name"]]
        if dev not in codebook_per_device:
            codebook_per_device[dev], _ = build_codebook(manifest["bits"], dev, manifest["seed"])
        model.get_submodule(info["name"]).codebook = codebook_per_device[dev]

    model.eval()
    return model
