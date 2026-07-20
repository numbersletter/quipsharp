"""Generic per-layer-streaming QuIP# quantization (no fine-tuning) for any
Llama2/3, Qwen2.5/Qwen3, or Gemma3/Gemma4(dense/multimodal)-style causal LM,
given Hessians already collected by collect_hessians.py.

Weights stream in from disk one decoder layer at a time (lazy_weights.py) --
never the whole model at once, the same reasoning as collect_hessians.py's
switch away from a full CPU load -- get quantized, and are immediately
replaced (checkpoint.replace_with_quantized_linear_) with a QuantizedLinear
holding only the compact per-block codes (2 bits/weight for 2-bit E8P, not
a dense dequantized weight): the ORIGINAL weight is freed right after, so
peak memory stays one-layer regardless of model size. The final
save_quantized_checkpoint is consequently also compact on disk (see
checkpoint.py's module docstring for why this replaced this project's
earlier dense-checkpoint approach) -- a 70B 2-bit checkpoint is ~17GB, not
~140GB, and comfortably fits on a single 24GB GPU at eval time with no
custom kernel needed (see quipsharp/quantize.py's module docstring for why
the kernel itself is out of scope).
"""
import argparse
import time

import torch
from transformers import AutoTokenizer

from quipsharp.architectures import get_decoder_layers, get_linear_groups, get_model_dims
from quipsharp.checkpoint import build_codebook, replace_with_quantized_linear_, save_quantized_checkpoint
from quipsharp.hessian import load_hessians
from quipsharp.lazy_weights import build_text_skeleton, KeyMap, materialize_module_, release_module_
from quipsharp.quantize import quantize_linear

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", required=True, type=str)
parser.add_argument("--hessian_path", default="hess", type=str)
parser.add_argument("--save_path", required=True, type=str)
parser.add_argument("--codebook", default="2bit", choices=["2bit", "3bit", "4bit"], type=str)
parser.add_argument("--block_size", default=8, type=int)
parser.add_argument("--damping", default=1e-2, type=float)
parser.add_argument("--tune_iters", default=10, type=int)
parser.add_argument("--seed", default=0, type=int)


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    # Hard requirement, not a fallback: a real Llama-2-70B quantize run once hit
    # "CUDA driver initialization failed", silently continued on CPU, and burned
    # hours producing nothing usable. Fail immediately instead.
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable (driver init failure or bad "
                         "CUDA_VISIBLE_DEVICES?) -- refusing to silently run this "
                         "job on CPU; check nvidia-smi and retry.")
    device = "cuda"
    model_name = args.model_id.split("/")[-1]

    print(f"building meta skeleton + lazy checkpoint index for {args.model_id} ...", flush=True)
    model, text_config, full_config, checkpoint = build_text_skeleton(args.model_id)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    skeleton_names = ([n for n, _ in model.named_parameters()] +
                       [n for n, _ in model.named_buffers()])
    key_map = KeyMap(list(checkpoint.keys()), skeleton_names)

    print(f"building {args.codebook} codebook ...", flush=True)
    cb, scale = build_codebook(args.codebook, device, args.seed)

    layers = get_decoder_layers(model)
    num_layers, hidden_size, intermediate_size = get_model_dims(model.config)
    print(f"quantizing {num_layers} layers (hidden={hidden_size}, intermediate={intermediate_size}) ...",
          flush=True)

    quantized_modules = []
    t0 = time.time()
    for layer_index, layer in enumerate(layers):
        prefix = f"model.layers.{layer_index}."
        materialize_module_(layer, f"model.layers.{layer_index}", checkpoint, key_map, device)
        groups = get_linear_groups(layer, layer_index)
        qualified_names = [prefix + name for group in groups for name, _ in group]

        hessians = load_hessians(model_name, qualified_names, root=args.hessian_path, device=device)
        if hessians is None:
            raise FileNotFoundError(
                f"missing cached Hessians for layer {layer_index} under "
                f"{args.hessian_path}/{model_name}/ -- run collect_hessians.py first")

        for group in groups:
            for name, module in group:
                qname = prefix + name
                W = module.weight.data.to(device).float()
                # captured before release_module_ collapses the module to meta
                bias = module.bias.data.detach().float().clone() if module.bias is not None else None
                H = hessians.pop(qname)  # pop: aliased names may share this tensor
                qlayer, _what_q = quantize_linear(W, H, cb, g=args.block_size, damping=args.damping,
                                                   codebook_scale=scale, tune_iters=args.tune_iters)
                del W, H

                parent_name, _, attr = name.rpartition(".")
                parent = layer.get_submodule(parent_name) if parent_name else layer
                release_module_(module)  # free the dense weight before attaching the replacement
                qlin = replace_with_quantized_linear_(parent, attr, qlayer, cb, bias=bias)
                qlin.to("cpu")
                kind = "rht" if hasattr(qlayer.signs, "su") else "rfft"
                quantized_modules.append({
                    "name": f"model.layers.{layer_index}.{name}",
                    "in_features": qlayer.shape[1], "out_features": qlayer.shape[0],
                    "transform_kind": kind, "has_bias": bias is not None,
                })
                del qlayer
                # per-group: keeps large late-group Hessian allocations from fragmenting
                if device == "cuda":
                    torch.cuda.empty_cache()
        if (layer_index + 1) % 5 == 0 or layer_index == len(layers) - 1:
            print(f"  quantized {layer_index + 1}/{len(layers)} layers "
                  f"({time.time() - t0:.1f}s elapsed)", flush=True)

    print("materializing non-quantized weights (embeddings/norms/lm_head) ...", flush=True)
    materialize_module_(model, "", checkpoint, key_map, "cpu")
    model.tie_weights()  # re-establish e.g. lm_head<->embed_tokens tying broken by the
                         # in-place tensor replacement above (see set_module_tensor_to_device)
    checkpoint.close()

    print(f"quantization complete in {time.time() - t0:.1f}s, saving to {args.save_path} ...",
          flush=True)
    save_quantized_checkpoint(model, args.model_id, args.save_path, bits=args.codebook,
                               block_size=args.block_size, seed=args.seed,
                               quantized_modules=quantized_modules, tokenizer=tokenizer)
    print("done.", flush=True)


if __name__ == "__main__":
    main(parser.parse_args())
