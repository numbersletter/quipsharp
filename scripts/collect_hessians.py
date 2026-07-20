"""Generic calibration-Hessian collection for any Llama2/3, Qwen2.5/Qwen3, or
Gemma3/Gemma4(dense/multimodal)-style causal LM, streaming both the WEIGHTS
(from disk, one decoder layer at a time -- see lazy_weights.py) and GPU
residency (one layer at a time) so models that don't fit in CPU RAM at all
(e.g. Llama-2/3-70B's ~140GB fp16/bf16, on a machine with 83GB free RAM)
never need more than one layer's worth of real weights materialized
anywhere at once. This goes one step further than the reference QuIP# repo's
hessian_offline_llama.py, which streams GPU residency but still loads the
WHOLE model onto CPU first (fine for their hardware; not for a 70B model
here).

NOTE on forward_layer_fn / build_forward_layer_fn below: this is the part of
the whole pipeline that has NOT been exercised against a live model this
session. It was written directly against the installed transformers==5.13.0
source (masking_utils.create_causal_mask / create_sliding_window_causal_mask,
and each model's own rotary_emb calling convention, incl. Gemma3/Gemma4's
extra layer_type argument), but treat the first real run against each
architecture family as this function's actual validation, not this code
review.
"""
import argparse

import torch
from transformers import AutoTokenizer

from quipsharp.architectures import get_decoder_layers, get_linear_groups
from quipsharp.hessian import collect_hessians_streaming
from quipsharp.lazy_weights import build_text_skeleton, KeyMap, materialize_module_, release_module_

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", required=True, type=str)
parser.add_argument("--save_path", default="hess", type=str)
parser.add_argument("--devset_size", default=256, type=int)
parser.add_argument("--ctx_size", default=4096, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--calibration_dataset", default="HuggingFaceFW/fineweb-edu", type=str)
parser.add_argument("--calibration_config", default="sample-10BT", type=str)


def sample_calibration_ids(tokenizer, dataset_name, dataset_config, devset_size, ctx_size, seed):
    """Stream `devset_size` documents from the calibration dataset, each
    tokenized and truncated/padded to exactly ctx_size tokens (matching the
    reference repo's fixed-length-window calibration recipe, just against
    FineWeb-Edu instead of their RedPajama-1T sample -- see the earlier
    decision to move off WikiText2 for calibration to keep it disjoint from
    the WikiText2 perplexity eval set)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    rows = []
    for example in ds:
        ids = tokenizer(example["text"], return_tensors="pt").input_ids[0]
        if ids.shape[0] < ctx_size:
            continue
        rows.append(ids[:ctx_size])
        if len(rows) == devset_size:
            break
    return torch.stack(rows, dim=0)


def build_forward_layer_fn(model):
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    config = model.config
    layer_types = getattr(config, "layer_types", None)

    def forward_layer_fn(layer, hidden_states, layer_index):
        bsz, seqlen, _ = hidden_states.shape
        position_ids = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
        mask_kwargs = dict(config=config, inputs_embeds=hidden_states, attention_mask=None,
                            past_key_values=None, position_ids=position_ids)

        extra_kwargs = {}
        if layer_types is not None:
            layer_type = layer_types[layer_index]
            mask = (create_sliding_window_causal_mask(**mask_kwargs) if layer_type == "sliding_attention"
                    else create_causal_mask(**mask_kwargs))
            try:
                position_embeddings = model.model.rotary_emb(hidden_states, position_ids, layer_type)
            except TypeError:
                position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
            # Gemma4/Gemma4Unified's attention unconditionally writes
            # shared_kv_states[self.layer_type] = ... when store_full_length_kv
            # is set on a layer -- even when config.num_kv_shared_layers == 0
            # means nothing ever actually READS from it -- so it must be a
            # dict, not the None a plain (non-Gemma4) decoder layer would
            # default to. A fresh {} per layer (rather than one shared across
            # the whole model) is correct here specifically because our
            # calibration set's target checkpoints have no kv-shared
            # consumer layers to begin with; a model that actually used kv
            # sharing would need this threaded across layers instead.
            extra_kwargs["shared_kv_states"] = {}
        else:
            mask = create_causal_mask(**mask_kwargs)
            position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        with torch.no_grad():
            out = layer(hidden_states, attention_mask=mask, position_embeddings=position_embeddings,
                        position_ids=position_ids, past_key_values=None, use_cache=False, **extra_kwargs)
        return out[0] if isinstance(out, tuple) else out

    return forward_layer_fn


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

    print(f"sampling {args.devset_size} calibration sequences of {args.ctx_size} tokens from "
          f"{args.calibration_dataset}/{args.calibration_config} ...", flush=True)
    calib_ids = sample_calibration_ids(tokenizer, args.calibration_dataset, args.calibration_config,
                                        args.devset_size, args.ctx_size, args.seed)

    embed_tokens = model.get_submodule("model.embed_tokens")
    materialize_module_(embed_tokens, "model.embed_tokens", checkpoint, key_map, device)
    with torch.no_grad():
        hidden_states = embed_tokens(calib_ids.to(device)).cpu()
    release_module_(embed_tokens)
    if device == "cuda":
        torch.cuda.empty_cache()

    def load_layer_fn(layer, layer_index):
        materialize_module_(layer, f"model.layers.{layer_index}", checkpoint, key_map, device)

    def unload_layer_fn(layer, layer_index):
        release_module_(layer)
        if device == "cuda":
            torch.cuda.empty_cache()

    layers = get_decoder_layers(model)
    print(f"streaming Hessian collection over {len(layers)} layers ...", flush=True)
    collect_hessians_streaming(layers, hidden_states, build_forward_layer_fn(model),
                                get_linear_groups, device=device, model_name=model_name,
                                root=args.save_path, batch_size=args.batch_size,
                                load_layer_fn=load_layer_fn, unload_layer_fn=unload_layer_fn)
    checkpoint.close()
    print(f"done. Hessians saved to {args.save_path}/{model_name}/", flush=True)


if __name__ == "__main__":
    main(parser.parse_args())
