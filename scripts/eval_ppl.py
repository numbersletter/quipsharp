"""Generic WikiText2 perplexity eval for any causal LM checkpoint (quantized
or not) -- same computation as the reference repo's eval/eval_ppl.py:
concatenate the whole test split with "\n\n", tokenize once, slice into
non-overlapping ctx_size windows, and report exp(mean per-window NLL).

Loading (quipsharp.model_loading.load_eval_model) transparently handles
three cases: this project's own compact quantized checkpoints (which fit on
one GPU regardless of original model size), plain dense HF models too big
for one GPU (via device_map="auto", spread/offloaded across the given GPUs
and CPU), and dense multimodal-WRAPPED HF repos like Gemma4's (which need
this project's own key-remapping loader -- see lazy_weights.py)."""
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from quipsharp.model_loading import load_eval_model, resolve_gpu_budget_bytes, resolve_cpu_budget_bytes

parser = argparse.ArgumentParser()
parser.add_argument("--model_id_or_path", required=True, type=str)
parser.add_argument("--ctx_size", default=4096, type=int)
parser.add_argument("--gpu_devices", default=None, type=str,
                     help="devices to spread an oversized model over "
                          "(default: all visible GPUs; pass to restrict to a subset)")
# budgets: a number (GB) or "auto" to size from detected free VRAM / available RAM.
parser.add_argument("--gpu_budget_gb", default="auto", type=str)
parser.add_argument("--cpu_budget_gb", default="auto", type=str)
parser.add_argument("--offload_folder", default=None, type=str)


def build_eval_windows(tokenizer, ctx_size: int) -> torch.Tensor:
    # namespaced id: datasets>=5 rejects the bare "wikitext"
    test_text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(test_text, return_tensors="pt").input_ids
    nsamples = ids.shape[1] // ctx_size
    return ids[:, :nsamples * ctx_size].view(nsamples, ctx_size)


@torch.no_grad()
def evaluate_ppl(model, windows: torch.Tensor, device, loss_chunk: int = 512) -> float:
    """Per-window mean NLL -> exp(mean over windows). The shifted cross
    entropy runs in `loss_chunk`-token slices: one-shot needs a contiguous
    (ctx_size x vocab) fp32 copy of the logits, multi-GB extra for
    152k-262k vocabs. Chunked sum/count computes the identical mean."""
    total_loss = 0.0
    n = windows.shape[0]
    for i in range(n):
        inp = windows[i:i + 1].to(device)
        logits = model(inp, use_cache=False).logits
        seqlen = inp.shape[1]
        loss_sum, count = 0.0, 0
        for s in range(0, seqlen - 1, loss_chunk):
            e = min(s + loss_chunk, seqlen - 1)
            chunk_logits = logits[0, s:e, :].float()
            chunk_labels = inp[0, s + 1:e + 1]
            loss_sum += torch.nn.functional.cross_entropy(
                chunk_logits, chunk_labels, reduction="sum").item()
            count += e - s
        del logits
        total_loss += loss_sum / count
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  eval window {i + 1}/{n}, running avg loss {total_loss / (i + 1):.4f}", flush=True)
    return float(torch.exp(torch.tensor(total_loss / n)))


def main(args):
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        gpu_devices = ["cpu"]
    elif args.gpu_devices:
        gpu_devices = args.gpu_devices.split(",")
    else:
        gpu_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    gpu_budget_bytes = resolve_gpu_budget_bytes(args.gpu_budget_gb)
    cpu_budget_bytes = resolve_cpu_budget_bytes(args.cpu_budget_gb)
    print(f"loading {args.model_id_or_path} (gpu budget {gpu_budget_bytes/1024**3:.1f}GB, "
          f"cpu {cpu_budget_bytes/1024**3:.1f}GB) ...", flush=True)
    model, device = load_eval_model(args.model_id_or_path, gpu_devices=gpu_devices,
                                     gpu_budget_bytes=gpu_budget_bytes,
                                     cpu_budget_bytes=cpu_budget_bytes,
                                     offload_folder=args.offload_folder)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id_or_path)

    print("loading wikitext2 test split ...", flush=True)
    windows = build_eval_windows(tokenizer, args.ctx_size)
    print(f"  {windows.shape[0]} windows of {args.ctx_size} tokens", flush=True)

    ppl = evaluate_ppl(model, windows, device)
    print(f"\nWikiText2 perplexity ({args.model_id_or_path}): {ppl:.4f}")


if __name__ == "__main__":
    main(parser.parse_args())
