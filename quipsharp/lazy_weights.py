"""Shard-aware lazy weight streaming for causal LMs too large to
materialize fully in CPU RAM (this machine has 83GB free RAM; a 70B-class
fp16/bf16 checkpoint is ~140GB, so `AutoModelForCausalLM.from_pretrained`
with no device_map -- which loads every tensor onto CPU at once -- would
OOM before quantization even starts). Instead:

  1. `LazyCheckpoint` indexes a HF repo's safetensors shards (via the repo's
     own `model.safetensors.index.json`) and lets callers pull out ONE named
     tensor at a time (`safetensors.safe_open` memory-maps the shard and
     only realizes the requested tensor -- the rest of that shard's tensors
     are never read into memory).
  2. `build_text_skeleton` constructs the model architecture on the `meta`
     device (via `accelerate.init_empty_weights`) -- zero real memory for
     any weight -- using `config.get_text_config()` so multimodal wrappers
     (e.g. Gemma4's `Gemma4ForConditionalGeneration`, which bundles a vision
     tower + audio tower this project has no use for) resolve to their
     text-only decoder class instead of the full multimodal model.
  3. `KeyMap` translates between the skeleton's state_dict key names and the
     checkpoint's tensor names. For plain causal LMs the two already match
     (identity map). For a multimodal-wrapped checkpoint they don't: e.g.
     Gemma4's checkpoint stores decoder weights under
     "model.language_model.layers.0...." while the text-only skeleton
     (`Gemma4ForCausalLM`) expects "model.layers.0....". Rather than
     hardcoding "language_model" (or any other family-specific string), the
     common prefix in front of ".layers.0." is discovered empirically on
     both sides and diffed -- the same introspection-based approach
     architectures.py uses for module discovery.
  4. `materialize_module_` / `release_module_` fill a submodule's
     parameters/buffers in from the checkpoint (via
     `accelerate.utils.set_module_tensor_to_device`) and, once a caller is
     done with it, collapse it back to `meta` (zero real memory) -- this is
     what makes it safe to walk all N decoder layers of even a 70B model one
     at a time, the same peak-memory pattern hessian.py's
     collect_hessians_streaming and quantize_model.py already use for GPU
     residency, just extended one level further down to loading itself.
"""
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open


class LazyCheckpoint:
    """Tensor-name -> tensor accessor over a (possibly multi-shard)
    safetensors checkpoint, without ever loading more than one tensor into
    memory at a time."""

    def __init__(self, model_id: str, revision: str = None):
        from huggingface_hub import snapshot_download

        self.local_dir = Path(snapshot_download(
            model_id, revision=revision,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"]))

        index_path = self.local_dir / "model.safetensors.index.json"
        if index_path.exists():
            self.weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        else:
            single = self.local_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no safetensors weights found under {self.local_dir}")
            with safe_open(single, framework="pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}

        self._open_files: dict[str, "safe_open"] = {}

    def keys(self):
        return self.weight_map.keys()

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    _DTYPE_NBYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2,
                      "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}

    def tensor_nbytes(self, name: str) -> int:
        """Size of tensor `name` in bytes, read from the shard's metadata
        only (safetensors' `get_slice` doesn't materialize the tensor's
        data) -- used to bin-pack layers across devices before loading
        anything real."""
        shard = self.weight_map[name]
        if shard not in self._open_files:
            self._open_files[shard] = safe_open(self.local_dir / shard, framework="pt", device="cpu")
        sl = self._open_files[shard].get_slice(name)
        shape = sl.get_shape()
        nbytes_per_elem = self._DTYPE_NBYTES.get(sl.get_dtype(), 4)
        n = 1
        for d in shape:
            n *= d
        return n * nbytes_per_elem

    def get_tensor(self, name: str, device=None) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._open_files:
            self._open_files[shard] = safe_open(self.local_dir / shard, framework="pt", device="cpu")
        t = self._open_files[shard].get_tensor(name)
        return t.to(device) if device is not None else t

    def close(self) -> None:
        self._open_files.clear()


def _wrap_prefix(names) -> str:
    """The substring BEFORE ".layers." in whichever name contains
    ".layers.0." -- e.g. "model." (plain causal LM) or
    "model.language_model." (Gemma4's multimodal-wrapped checkpoint, whose
    decoder lives at model.language_model.layers). This is the prefix that
    also sits in front of every OTHER top-level submodule sharing the same
    wrapping (embed_tokens, norm, ...), not just `.layers.` itself -- so
    diffing on this, rather than on the full "....layers." string, is what
    lets KeyMap remap embed_tokens/norm/lm_head correctly too, not only
    decoder-layer weights. Raises if no name matches (every causal LM this
    project targets has a decoder `.layers` list)."""
    for name in names:
        idx = name.find(".layers.0.")
        if idx != -1:
            return name[:idx + 1]  # include the dot right before "layers"
    raise ValueError("no '.layers.0.' found in any name -- not a decoder-style causal LM")


class KeyMap:
    """Translates skeleton state_dict names <-> checkpoint tensor names by
    diffing each side's wrap prefix (see _wrap_prefix). Checkpoint names
    outside that common wrapping (e.g. a multimodal checkpoint's
    vision_tower/audio_tower/embed_vision weights, which the text-only
    skeleton has no submodule for at all) map to None."""

    def __init__(self, checkpoint_names, skeleton_names):
        self.src_wrap = _wrap_prefix(checkpoint_names)
        self.dst_wrap = _wrap_prefix(skeleton_names)

    def to_checkpoint(self, skeleton_name: str) -> str:
        if skeleton_name.startswith(self.dst_wrap):
            return self.src_wrap + skeleton_name[len(self.dst_wrap):]
        return skeleton_name  # e.g. "lm_head.weight" -- outside the wrap, same on both sides

    def to_skeleton(self, checkpoint_name: str) -> str | None:
        if checkpoint_name.startswith(self.src_wrap):
            return self.dst_wrap + checkpoint_name[len(self.src_wrap):]
        top_segment = self.src_wrap.split(".")[0] + "."
        if not checkpoint_name.startswith(top_segment):
            return checkpoint_name  # outside the wrapper namespace entirely (e.g. "lm_head.weight")
        return None  # under the wrapper's top segment but not the text decoder's own sub-wrap
                     # (e.g. "model.vision_tower...." -- a branch the text-only skeleton lacks)


def build_meta_model(model_id: str):
    """Returns (model, text_config, full_config): `model` is on the meta
    device (zero real memory) with the architecture for `model_id`'s text
    decoder -- `config.get_text_config()` is a no-op for plain causal LMs
    and unwraps multimodal wrappers (e.g. Gemma4's ForConditionalGeneration)
    down to their text-only class. Does not touch any checkpoint weights;
    use `build_text_skeleton` when you also need a `LazyCheckpoint` to fill
    it from (e.g. for quantization/hessian collection). This half is reused
    on its own by checkpoint.py's loader, which fills the skeleton from this
    project's own compact saved state instead of the original checkpoint."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights

    full_config = AutoConfig.from_pretrained(model_id)
    text_config = full_config.get_text_config()

    # `from_config` defaults to float32 unless told otherwise -- since
    # set_module_tensor_to_device (see materialize_module_) casts whatever
    # it's given to the SKELETON's existing dtype, building the skeleton in
    # fp32 would silently upcast every bf16/fp16 checkpoint tensor on
    # materialization, roughly DOUBLING real memory use versus what the
    # checkpoint's own on-disk size (and any budget computed from it, e.g.
    # load_dense_causal_lm's bin-packing) assumes. `get_text_config()` may
    # not carry the dtype field even when the wrapping multimodal config
    # does (empirically true for Gemma4), so read it off `full_config`.
    dtype = getattr(full_config, "dtype", None) or getattr(full_config, "torch_dtype", None) or torch.bfloat16

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(text_config, dtype=dtype)

    return model, text_config, full_config


def build_text_skeleton(model_id: str):
    """Returns (model, text_config, full_config, checkpoint) -- see
    `build_meta_model`, plus a `LazyCheckpoint` for `model_id` to fill it
    from."""
    model, text_config, full_config = build_meta_model(model_id)
    checkpoint = LazyCheckpoint(model_id)
    return model, text_config, full_config, checkpoint


def materialize_module_(module: nn.Module, module_prefix: str, checkpoint: LazyCheckpoint,
                         key_map: KeyMap, device) -> None:
    """Fill every parameter/buffer of `module` (a submodule of the full
    skeleton, reached via dotted path `module_prefix`) from `checkpoint`,
    in place, onto `device`. Names present in the skeleton but absent from
    the checkpoint (e.g. a non-persistent rotary_emb buffer that's computed,
    not stored) are silently left as-is."""
    from accelerate.utils import set_module_tensor_to_device

    names = [n for n, _ in module.named_parameters(recurse=True)]
    names += [n for n, _ in module.named_buffers(recurse=True)]
    for local_name in names:
        full_skeleton_name = f"{module_prefix}.{local_name}" if module_prefix else local_name
        ckpt_name = key_map.to_checkpoint(full_skeleton_name)
        if ckpt_name not in checkpoint:
            continue
        tensor = checkpoint.get_tensor(ckpt_name, device=device)
        set_module_tensor_to_device(module, local_name, device, value=tensor)


def release_module_(module: nn.Module) -> None:
    """Collapse every real tensor in `module` back to the `meta` device
    (zero memory), freeing whatever materialize_module_ allocated."""
    module.to("meta")


def _layer_nbytes(layer: nn.Module, layer_index: int, checkpoint: LazyCheckpoint,
                   key_map: KeyMap) -> int:
    """Exact weight-byte size of decoder `layer` (summed from the
    checkpoint's own shard metadata via LazyCheckpoint.tensor_nbytes -- no
    tensor data is read), used to bin-pack layers across devices before
    loading anything real. A per-architecture analytical formula (e.g.
    `4*hidden^2` for qkvo) is NOT used here: head_dim*num_attention_heads
    frequently differs from hidden_size (true for e.g. Gemma4's 12B config:
    hidden_size=3840 but head_dim=256 x num_attention_heads=16=4096), which
    silently breaks any such formula's assumptions -- reading the actual
    checkpoint shapes sidesteps needing to know each family's projection
    dimensions at all."""
    total = 0
    for local_name, _ in layer.named_parameters(recurse=True):
        ckpt_name = key_map.to_checkpoint(f"model.layers.{layer_index}.{local_name}")
        if ckpt_name in checkpoint:
            total += checkpoint.tensor_nbytes(ckpt_name)
    return total


def load_dense_causal_lm(model_id: str, gpu_devices: list[str], gpu_budget_bytes: int,
                          cpu_budget_bytes: int = None):
    """Load `model_id`'s full-precision text-only causal LM, spreading
    decoder layers across `gpu_devices` (filling each to `gpu_budget_bytes`
    before moving to the next) and then, if it still doesn't fit, onto CPU
    -- for a dense (unquantized) baseline eval of a model too big for a
    single GPU (e.g. Gemma4-31B's ~62GB bf16 vs. one 4090's 24GB).

    This is this project's own loader rather than
    `AutoModelForCausalLM.from_pretrained(..., device_map="auto")` because
    that path can't handle a multimodal-wrapped checkpoint like Gemma4's
    (its keys live under "model.language_model...." while the text-only
    skeleton this function builds expects "model...." -- see KeyMap); for
    plain (non-wrapped) causal LMs the two approaches are equivalent, but
    for models this large the standard `from_pretrained(device_map="auto")`
    path is almost always the better-supported (and better-tested) choice,
    so prefer that where the config is already a plain text config --
    reach for this function specifically because of the wrapper mismatch,
    not as a general replacement.

    After materializing every submodule directly onto its assigned device,
    wires up `accelerate.dispatch_model` (which -- given a device_map --
    inserts forward hooks that move activations across devices between
    submodules automatically) so the resulting model behaves like a normal
    single-device model to callers: they can just call `model(input_ids)`.
    """
    from accelerate import dispatch_model

    model, text_config, full_config = build_meta_model(model_id)
    checkpoint = LazyCheckpoint(model_id)
    skeleton_names = [n for n, _ in model.named_parameters()] + [n for n, _ in model.named_buffers()]
    key_map = KeyMap(list(checkpoint.keys()), skeleton_names)

    devices = list(gpu_devices) + (["cpu"] if cpu_budget_bytes else [])
    budgets = [gpu_budget_bytes] * len(gpu_devices) + ([cpu_budget_bytes] if cpu_budget_bytes else [])
    main_device = gpu_devices[0] if gpu_devices else "cpu"
    if len(gpu_devices) > 1:
        # main device also holds embed/norm/lm_head + logits; halve its layer budget
        budgets[0] = gpu_budget_bytes // 2

    layers = model.get_submodule("model.layers")

    device_map = {}
    dev_idx, used = 0, 0
    for i, layer in enumerate(layers):
        layer_bytes = _layer_nbytes(layer, i, checkpoint, key_map)
        while dev_idx < len(devices) - 1 and used + layer_bytes > budgets[dev_idx]:
            dev_idx, used = dev_idx + 1, 0
        target = devices[dev_idx]
        materialize_module_(layer, f"model.layers.{i}", checkpoint, key_map, target)
        device_map[f"model.layers.{i}"] = target
        used += layer_bytes

    for name, child in model.model.named_children():
        if name == "layers":
            continue
        materialize_module_(child, f"model.{name}", checkpoint, key_map, main_device)
        device_map[f"model.{name}"] = main_device

    if hasattr(model, "lm_head"):
        materialize_module_(model.lm_head, "lm_head", checkpoint, key_map, main_device)
        device_map["lm_head"] = main_device

    model.tie_weights()  # re-establish e.g. lm_head<->embed_tokens tying broken by the
                         # in-place tensor replacement above (see set_module_tensor_to_device)
    checkpoint.close()

    model = dispatch_model(model, device_map=device_map, main_device=main_device)
    model.eval()
    return model
