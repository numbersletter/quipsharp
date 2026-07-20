"""Generic zero-shot ARC-Challenge / ARC-Easy (or any lm-eval-harness task)
eval for any causal LM checkpoint (quantized or not).

Unlike the reference repo (which needs a custom BaseLM adaptor + a custom
CUDA-kernel nn.Module, because its packed-codebook format can't be loaded by
a stock transformers model), this project's quantized output is a plain
dense HF checkpoint -- so it can be wrapped directly with modern
lm-evaluation-harness's built-in HFLM and passed straight to
lm_eval.simple_evaluate, no custom adaptor needed.

Requires `pip install lm_eval` in this project's environment -- it is not
one of the base dependencies (torch/transformers/datasets/accelerate) and is
not installed by default.

Loading (quipsharp.model_loading.load_eval_model) transparently handles this
project's own compact quantized checkpoints, plain dense HF models too big
for one GPU, and dense multimodal-WRAPPED HF repos like Gemma4's -- see
eval_ppl.py's module docstring / lazy_weights.py for why the split exists.
"""
import argparse
import json

import torch
from transformers import AutoTokenizer

from quipsharp.model_loading import load_eval_model

parser = argparse.ArgumentParser()
parser.add_argument("--model_id_or_path", required=True, type=str)
parser.add_argument("--tasks", default="arc_challenge,arc_easy", type=str)
parser.add_argument("--num_fewshot", default=0, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--gpu_devices", default=None, type=str,
                     help="devices to spread an oversized model over "
                          "(default: all visible GPUs; pass to restrict to a subset)")
parser.add_argument("--gpu_budget_gb", default=20.0, type=float)
parser.add_argument("--cpu_budget_gb", default=60.0, type=float)
parser.add_argument("--offload_folder", default=None, type=str)


def main(args):
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError(
            "lm_eval is required for zero-shot eval but isn't installed in "
            "this environment; run `pip install lm_eval`.") from e

    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        gpu_devices = ["cpu"]
    elif args.gpu_devices:
        gpu_devices = args.gpu_devices.split(",")
    else:
        gpu_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    print(f"loading {args.model_id_or_path} ...", flush=True)
    model, device = load_eval_model(args.model_id_or_path, gpu_devices=gpu_devices,
                                     gpu_budget_bytes=int(args.gpu_budget_gb * 1024**3),
                                     cpu_budget_bytes=int(args.cpu_budget_gb * 1024**3),
                                     offload_folder=args.offload_folder)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id_or_path)

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    task_names = args.tasks.split(",")

    results = simple_evaluate(model=lm, tasks=task_names, num_fewshot=args.num_fewshot)

    print(json.dumps(results["results"], indent=2))
    if args.output_path is not None:
        results["config"]["model"] = args.model_id_or_path
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main(parser.parse_args())
