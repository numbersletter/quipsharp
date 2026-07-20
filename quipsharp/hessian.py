"""Calibration Hessian collection (Appendix F.2): H = E_x[x x^T] over a
calibration dataset's activations, one proxy Hessian per linear layer's
input. This is the H used throughout incoherence processing and BlockLDLQ.
"""
import json
from pathlib import Path

import torch
from torch import nn, Tensor

ALIAS_MANIFEST_NAME = "_aliases.json"


class HessianAccumulator:
    """Accumulates H = sum_i x_i x_i^T for one layer's input activations."""

    def __init__(self, n: int, device=None, dtype=torch.float32):
        self.n = n
        self.H = torch.zeros(n, n, device=device, dtype=dtype)
        self.count = 0

    def update(self, x: Tensor) -> None:
        x = x.reshape(-1, self.n).to(self.H.dtype)
        self.H += x.T @ x
        self.count += x.shape[0]

    @property
    def hessian(self) -> Tensor:
        return self.H / max(self.count, 1)


def collect_hessians(model: nn.Module, calibration_batches, layer_names=None,
                      groups=None, device=None) -> dict[str, Tensor]:
    """Register forward hooks on nn.Linear submodules and accumulate
    H = E[x x^T] over `calibration_batches`.

    calibration_batches: iterable of model-ready inputs, each fed through
    `model(batch)` with grad disabled.
    layer_names: if given, only hook these submodule names (default: all
    nn.Linear layers in the model).
    groups: optional list of lists of layer names that are known to consume
    the exact same input activations (e.g. a block's q_proj/k_proj/v_proj,
    which all take the same hidden_states -- or gate_proj/up_proj in a
    SwiGLU-style MLP). Only the first name in each group is actually
    hooked; every other name in the group shares that Hessian by object
    identity (not a copy) in the returned dict, avoiding redundant
    compute *and* letting save_hessians store it once instead of once per
    alias.
    """
    representative_of: dict[str, str] = {}
    if groups:
        for group in groups:
            rep = group[0]
            for name in group:
                representative_of[name] = rep

    accumulators: dict[str, HessianAccumulator] = {}
    hooks = []

    def make_hook(name, n):
        acc = HessianAccumulator(n, device=device)
        accumulators[name] = acc

        def hook(module, inputs, output):
            acc.update(inputs[0].detach())

        return hook

    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and (layer_names is None or name in layer_names)):
            continue
        if name in representative_of and representative_of[name] != name:
            continue  # non-representative alias: skip hooking, will share below
        hooks.append(module.register_forward_hook(make_hook(name, module.in_features)))

    if not hooks:
        raise ValueError("no matching nn.Linear layers found to hook")

    model.eval()
    with torch.no_grad():
        for batch in calibration_batches:
            model(batch)

    for h in hooks:
        h.remove()

    hessians = {name: acc.hessian for name, acc in accumulators.items()}
    for name, rep in representative_of.items():
        if name != rep:
            hessians[name] = hessians[rep]  # same tensor object, not a copy
    return hessians


def save_hessians(hessians: dict[str, Tensor], model_name: str, root: str = "hess") -> None:
    """Persist collected Hessians to <root>/<model_name>/<layer_name>.pt,
    mirroring the reference QuIP# repo's on-disk layout (its hess/<model>/
    directory of per-layer .pt files). Hessians only depend on the
    calibration data and model, not on bitrate/codebook choices, so caching
    them lets later runs skip recomputation entirely.

    Names that share the exact same tensor object (e.g. produced by
    collect_hessians' `groups` argument) are written to disk once; the
    others are recorded in an alias manifest so load_hessians can point
    them back at that one file instead of duplicating it.

    Safe to call incrementally (e.g. once per transformer layer while
    streaming): the alias manifest is merged with, not replaced by, whatever
    this call contributes, so earlier layers' aliases survive later calls.
    """
    out_dir = Path(root) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / ALIAS_MANIFEST_NAME
    aliases: dict[str, str] = {}
    if manifest_path.exists():
        aliases = json.loads(manifest_path.read_text())

    written_for_id: dict[int, str] = {}
    for name, H in hessians.items():
        key = id(H)
        if key in written_for_id:
            aliases[name] = written_for_id[key]
            continue
        written_for_id[key] = name
        torch.save(H.cpu(), out_dir / f"{name}.pt")

    if aliases:
        manifest_path.write_text(json.dumps(aliases, indent=2))


def collect_hessians_streaming(layers, hidden_states: Tensor, forward_layer_fn,
                                get_groups_fn, device, model_name: str,
                                root: str = "hess", batch_size: int = None,
                                load_layer_fn=None, unload_layer_fn=None) -> Tensor:
    """Per-layer-streaming Hessian collection for models too large to fit on
    one GPU at once (matches the reference repo's hessian_offline_llama.py
    memory pattern): only ONE decoder layer is resident on `device` at a
    time -- the rest of `layers` stay wherever the caller put them
    (typically CPU, for a model loaded with low_cpu_mem_usage=True and no
    device_map). Each layer's Hessians are saved to disk immediately after
    that layer finishes (via save_hessians) rather than held in memory
    across layers, and the layer is moved off `device` before the next one
    comes on, so peak GPU memory is one-layer, not whole-model.

    layers: ordered decoder layers, e.g. architectures.get_decoder_layers(model).
    hidden_states: (N, seqlen, hidden) calibration activations already
    embedded (the model's embed_tokens output) -- becomes layer 0's input;
    each layer's output becomes the next layer's input.
    forward_layer_fn(layer, hidden_states_chunk, layer_index) -> hidden_states_chunk:
    architecture-specific glue (building position_ids/attention_mask/
    position_embeddings and calling the layer) supplied by the caller, so
    this function itself stays architecture-agnostic. layer_index is passed
    through so e.g. Gemma-style sliding/global attention alternation (which
    depends on which layer this is) can be handled by the caller.
    get_groups_fn(layer, layer_index) -> list[list[(name, nn.Linear)]]:
    e.g. architectures.get_linear_groups. Names are qualified with
    "model.layers.<i>." before hooking/saving, matching the on-disk layout
    load_hessians/quantize scripts expect.
    batch_size: if given, hidden_states is processed in chunks of this many
    sequences at a time (per layer) instead of all at once, bounding peak
    activation memory; default None processes everything in one chunk.
    load_layer_fn(layer, layer_index) / unload_layer_fn(layer, layer_index):
    override how a layer's real weights come onto/off `device` for the
    duration it's being processed. Default (None) is `layer.to(device)` /
    `layer.to("cpu")`, which assumes `layers` are already real (e.g. a
    normally-loaded, CPU-resident model) -- pass these when `layers` start
    on the meta device instead (e.g. lazy_weights.build_text_skeleton's
    output for a model too big to fully materialize on CPU at once), to
    stream each layer's weights in directly from disk via
    lazy_weights.materialize_module_/release_module_ instead.

    Returns the final layer's output hidden states (on CPU).
    """
    n = hidden_states.shape[0]
    bs = batch_size or n
    load_layer_fn = load_layer_fn or (lambda layer, layer_index: layer.to(device))
    unload_layer_fn = unload_layer_fn or (lambda layer, layer_index: layer.to("cpu"))

    for layer_index, layer in enumerate(layers):
        load_layer_fn(layer, layer_index)
        prefix = f"model.layers.{layer_index}."
        groups = get_groups_fn(layer, layer_index)

        representative_of: dict[str, str] = {}
        for group in groups:
            rep_name = prefix + group[0][0]
            for name, _ in group:
                representative_of[prefix + name] = rep_name

        accumulators: dict[str, HessianAccumulator] = {}
        hooks = []
        for group in groups:
            rep_name, rep_module = prefix + group[0][0], group[0][1]
            acc = HessianAccumulator(rep_module.in_features, device=device)
            accumulators[rep_name] = acc

            def hook(module, inputs, output, acc=acc):
                acc.update(inputs[0].detach())

            hooks.append(rep_module.register_forward_hook(hook))

        # Write each chunk's output directly into a preallocated destination tensor rather
        # than collecting a list of chunks and torch.cat-ing them afterward: the list
        # approach transiently holds the OLD hidden_states, the fully-populated chunk list,
        # AND the freshly concatenated tensor all at once (3x the activation tensor's size --
        # for a 70B-class model at devset_size=256/ctx_size=4096/hidden=8192, that's the
        # difference between a ~34GB and a ~52GB peak), which is exactly what OOM-killed a
        # real Llama-2-70B Hessian collection run on this machine's shared 93GB RAM.
        new_hidden_states = torch.empty_like(hidden_states)
        for i in range(0, n, bs):
            chunk = forward_layer_fn(layer, hidden_states[i:i + bs].to(device), layer_index)
            new_hidden_states[i:i + bs] = chunk.cpu()
            del chunk

        for h in hooks:
            h.remove()

        layer_hessians = {name: acc.hessian for name, acc in accumulators.items()}
        for name, rep in representative_of.items():
            if name != rep:
                layer_hessians[name] = layer_hessians[rep]
        save_hessians(layer_hessians, model_name, root=root)

        unload_layer_fn(layer, layer_index)
        del accumulators, layer_hessians
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        hidden_states = new_hidden_states

    return hidden_states


def load_hessians(model_name: str, layer_names, root: str = "hess",
                   device=None) -> dict[str, Tensor] | None:
    """Load previously-saved Hessians for `model_name`. Returns None (rather
    than partial results) if the cache directory is missing or doesn't have
    every name in `layer_names`, so the caller knows to recompute from
    scratch instead of silently mixing stale and fresh data.

    Names recorded as aliases (see save_hessians) are loaded from their
    representative's file and share that tensor object, mirroring what
    collect_hessians' `groups` argument produces.
    """
    out_dir = Path(root) / model_name
    if not out_dir.exists():
        return None

    aliases = {}
    manifest_path = out_dir / ALIAS_MANIFEST_NAME
    if manifest_path.exists():
        aliases = json.loads(manifest_path.read_text())

    loaded_for_rep: dict[str, Tensor] = {}
    hessians = {}
    for name in layer_names:
        rep = aliases.get(name, name)
        if rep not in loaded_for_rep:
            path = out_dir / f"{rep}.pt"
            if not path.exists():
                return None
            loaded_for_rep[rep] = torch.load(path, map_location=device)
        hessians[name] = loaded_for_rep[rep]
    return hessians
