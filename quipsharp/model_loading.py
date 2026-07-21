"""Single entry point eval scripts use to load either a compact
quantized checkpoint (checkpoint.py) or a dense baseline model (any HF repo
id or a plain `model.save_pretrained` directory) too big for one GPU,
picking the right strategy so callers (eval_ppl.py/eval_zeroshot.py) don't
need to know which case they're in.
"""
from pathlib import Path

from .checkpoint import is_quantized_checkpoint, load_quantized_checkpoint


def resolve_gpu_budget_bytes(arg, headroom_frac: float = 0.15) -> int:
    """Per-GPU weight budget in bytes. A number is taken as GB; "auto" is the
    free VRAM of the least-free visible device minus `headroom_frac` of its
    total (headroom for activations/logits)."""
    if arg != "auto":
        return int(float(arg) * 1024**3)
    import torch
    budgets = [free - int(headroom_frac * total)
               for free, total in (torch.cuda.mem_get_info(i)
                                    for i in range(torch.cuda.device_count()))]
    return min(budgets) if budgets else 0


def resolve_cpu_budget_bytes(arg, use_frac: float = 0.8) -> int:
    """CPU offload budget in bytes. A number is taken as GB; "auto" is
    `use_frac` of currently-available RAM (MemAvailable)."""
    if arg != "auto":
        return int(float(arg) * 1024**3)
    with open("/proc/meminfo") as f:
        avail_kb = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable"))
    return int(avail_kb * 1024 * use_frac)


def load_eval_model(model_id_or_path: str, gpu_devices: list[str] = ("cuda:0",),
                     gpu_budget_bytes: int = 20 * 1024**3, cpu_budget_bytes: int = 60 * 1024**3,
                     offload_folder: str = None):
    """Returns (model, main_device) ready for `model(input_ids)` calls.

    Three cases:
      1. `model_id_or_path` is one of this project's own compact quantized
         checkpoints (checkpoint.save_quantized_checkpoint's output) --
         codes-only, so it comfortably fits on a single GPU regardless of
         original model size (e.g. ~17GB for a 70B model at 2 bits/weight).
      2. A plain (non-multimodal) HF causal LM, local or a hub repo id --
         `AutoModelForCausalLM.from_pretrained(..., device_map="auto")` is
         the standard, best-tested way to spread/offload a model too big
         for one GPU, so use it directly.
      3. A multimodal-WRAPPED HF repo (e.g. Gemma4's *ForConditionalGeneration*,
         whose checkpoint keys live under "model.language_model...." while
         the text-only class needed for these text-only evals expects
         "model...."): `from_pretrained(device_map="auto")` can't match
         those keys (see lazy_weights.py's module docstring), so use this
         project's own lazy_weights.load_dense_causal_lm instead.
    """
    if Path(model_id_or_path).is_dir() and is_quantized_checkpoint(model_id_or_path):
        model = load_quantized_checkpoint(model_id_or_path, device=gpu_devices[0],
                                           gpu_devices=list(gpu_devices),
                                           gpu_budget_bytes=gpu_budget_bytes,
                                           cpu_budget_bytes=cpu_budget_bytes)
        return model, gpu_devices[0]

    from transformers import AutoConfig
    full_config = AutoConfig.from_pretrained(model_id_or_path)
    text_config = full_config.get_text_config()

    if text_config is full_config:
        from transformers import AutoModelForCausalLM

        max_memory = {i: gpu_budget_bytes for i in range(len(gpu_devices))}
        if len(gpu_devices) > 1:
            # leave logits headroom on device 0
            max_memory[0] = int(gpu_budget_bytes * 0.6)
        if cpu_budget_bytes:
            max_memory["cpu"] = cpu_budget_bytes
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path, dtype="auto", low_cpu_mem_usage=True, device_map="auto",
            max_memory=max_memory, offload_folder=offload_folder)
        model.eval()
        main_device = gpu_devices[0] if gpu_devices else "cpu"
        return model, main_device

    from .lazy_weights import load_dense_causal_lm
    model = load_dense_causal_lm(model_id_or_path, gpu_devices=list(gpu_devices),
                                  gpu_budget_bytes=gpu_budget_bytes, cpu_budget_bytes=cpu_budget_bytes)
    main_device = gpu_devices[0] if gpu_devices else "cpu"
    return model, main_device
