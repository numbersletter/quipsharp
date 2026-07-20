"""Orchestrates the full benchmark across all target models: for each,
collect_hessians (once -- Hessians are bit-width-independent) ->
quantize_model (once per requested bit-width) -> eval_ppl (baseline +
each bit-width) -> eval_zeroshot (baseline + each bit-width), logging
every stage's stdout to `--log_root` and every eval result into one JSON
summary at `--results_path`.

Results JSON shape, per model key:
    {"model_id": ..., "baseline_ppl": ..., "baseline_zeroshot": {...},
     "2bit": {"quantized_ppl": ..., "quantized_zeroshot": {...}},
     "3bit": {...}, "wall_time_s": ...}
Baseline numbers live at the model level (they don't depend on bit-width);
quantized numbers are nested under their codebook name.

`--gpu` is the single GPU-placement knob: it becomes CUDA_VISIBLE_DEVICES
for every child, and children use whatever it exposes (renumbered from
cuda:0) -- all of it for big-model evals, one GPU each for concurrently
scheduled small evals (via a per-child CUDA_VISIBLE_DEVICES override).

Resumable: a stage is skipped if its expected output already exists (the
Hessian dir, the quantized checkpoint dir, or a results entry already in
the summary JSON), so a crashed or interrupted run can just be re-invoked
with the same arguments to pick up where it left off, instead of redoing
already-finished (and often the most expensive) stages.

Runs everything through subprocess rather than importing and calling the
other scripts' main() in-process: each stage gets a fresh Python process
(and hence fresh, fully-released CUDA/CPU memory) before the next one
starts, which matters here because a 70B-class stage can use most of this
machine's GPU+CPU memory -- reusing one long-lived process across stages
risks fragmentation/leaked allocations compounding across 9 models.
"""
import argparse
import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# HF repo id per model. Gemma4's sizes/names were confirmed directly against
# the HF Hub (google/gemma-4-12B, google/gemma-4-31B) rather than guessed.
MODELS = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "gemma4-12b": "google/gemma-4-12B",
    "gemma4-31b": "google/gemma-4-31B",
}

# Models whose dense baseline doesn't fit one 24GB GPU (weights + big-vocab
# logits), so its eval gets every GPU in --gpu. Quantized-checkpoint evals
# only need that when the checkpoint file itself exceeds one GPU's budget
# (decided by file size in run_model).
LARGE_MODELS = {"llama2-70b", "llama3-70b", "qwen3-32b", "gemma4-31b", "gemma4-12b"}

# halved devset for the largest-hidden-dim models: collection holds
# ~devset*ctx*hidden*2bytes*2 of activations in CPU RAM
DEVSET_SIZE_OVERRIDES = {"llama2-70b": 128, "llama3-70b": 128}

SCRIPTS_DIR = Path(__file__).parent
HESSIAN_LOCK_PATH = SCRIPTS_DIR.parent / ".hessian_collection.lock"


@contextlib.contextmanager
def hessian_collection_lock():
    """Serializes hessian collection (the CPU-RAM-heavy phase) across any
    concurrently-running orchestrators."""
    HESSIAN_LOCK_PATH.touch(exist_ok=True)
    with open(HESSIAN_LOCK_PATH, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

parser = argparse.ArgumentParser()
parser.add_argument("--models", default=",".join(MODELS), type=str,
                     help="comma-separated keys from MODELS to run (default: all)")
parser.add_argument("--codebooks", default="2bit,3bit", type=str,
                     help="comma-separated bit-widths to quantize+eval per model "
                          "(any subset of 2bit,3bit,4bit); Hessians are collected once "
                          "and reused across all of them")
parser.add_argument("--gpu", default="0,1", type=str,
                     help="CUDA_VISIBLE_DEVICES for every child process; single-GPU stages use the "
                          "first listed GPU, oversized evals spread across all of them")
parser.add_argument("--hess_root", default=str(SCRIPTS_DIR.parent / "hess"), type=str)
parser.add_argument("--quant_root", default=str(SCRIPTS_DIR.parent / "quantized"), type=str)
parser.add_argument("--log_root", default=str(SCRIPTS_DIR.parent / "logs"), type=str)
parser.add_argument("--results_path", default=str(SCRIPTS_DIR.parent / "results.json"), type=str)
parser.add_argument("--devset_size", default=256, type=int)
parser.add_argument("--ctx_size", default=4096, type=int)
parser.add_argument("--ppl_ctx_size", default=4096, type=int)
parser.add_argument("--zeroshot_tasks", default="arc_challenge,arc_easy,winogrande", type=str)
parser.add_argument("--zeroshot_batch_size", default=8, type=int,
                     help="batch size for dense (baseline) zeroshot eval")
parser.add_argument("--zeroshot_batch_size_quantized", default=16, type=int,
                     help="batch size for quantized-checkpoint zeroshot eval; larger than the dense "
                          "default to amortize per-forward dequantization, but capped at 16 so a "
                          "batch's logits still fit a 24GB GPU for 262k-vocab models")
parser.add_argument("--gpu_budget_gb", default=20.0, type=float)
parser.add_argument("--cpu_budget_gb", default=42.0, type=float,
                     help="CPU weight budget for offloaded dense evals (set ~20GB below total RAM)")
parser.add_argument("--offload_folder", default=str(SCRIPTS_DIR.parent / "offload"), type=str)
parser.add_argument("--skip_baseline", action="store_true",
                     help="skip dense (unquantized) baseline evals -- just quantize + eval those")
parser.add_argument("--delete_hessians_after", action="store_true",
                     help="delete a model's Hessian cache once it's quantized (Hessians can be tens "
                          "of GB per model for large hidden/intermediate sizes; only needed transiently "
                          "between collect_hessians and quantize_model)")


def load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_results(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2))


def run(cmd: list[str], log_path: Path, extra_env: dict = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"+ {' '.join(cmd)}", flush=True)
    env = {**os.environ, **extra_env} if extra_env else None
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)


def _hessians_complete(hess_dir: Path, model_id: str) -> bool:
    """Complete iff the LAST decoder layer's file exists (an interrupted
    collection leaves a valid-looking but partial directory)."""
    if not (hess_dir / "_aliases.json").exists():
        return False
    from transformers import AutoConfig
    num_layers = AutoConfig.from_pretrained(model_id).get_text_config().num_hidden_layers
    last_layer_prefix = f"model.layers.{num_layers - 1}."
    # every group's REPRESENTATIVE name gets a physical .pt file (aliased names are only
    # ever recorded in the manifest, never written as their own file), so a file glob is a
    # sufficient completeness check without needing to also inspect the alias manifest.
    return any(p.name.startswith(last_layer_prefix) for p in hess_dir.glob("*.pt"))


def stage_collect_hessians(model_key: str, model_id: str, args) -> Path:
    hess_dir = Path(args.hess_root) / model_id.split("/")[-1]
    if hess_dir.exists() and _hessians_complete(hess_dir, model_id):
        print(f"[{model_key}] Hessians already present at {hess_dir}, skipping collection", flush=True)
        return hess_dir
    log_path = Path(args.log_root) / model_key / "collect_hessians.log"
    devset_size = DEVSET_SIZE_OVERRIDES.get(model_key, args.devset_size)
    print(f"[{model_key}] waiting for hessian-collection lock (serialized across GPUs) ...", flush=True)
    with hessian_collection_lock():
        print(f"[{model_key}] acquired hessian-collection lock", flush=True)
        run([sys.executable, str(SCRIPTS_DIR / "collect_hessians.py"),
             "--model_id", model_id, "--save_path", args.hess_root,
             "--devset_size", str(devset_size), "--ctx_size", str(args.ctx_size)], log_path)
    return hess_dir


def quantized_checkpoint_path(model_id: str, codebook: str, args) -> Path:
    return Path(args.quant_root) / f"{model_id.split('/')[-1]}-{codebook}"


def stage_quantize(model_key: str, model_id: str, codebook: str, args) -> Path:
    save_path = quantized_checkpoint_path(model_id, codebook, args)
    if (save_path / "quip_config.json").exists():
        print(f"[{model_key}] {codebook} checkpoint already present at {save_path}, skipping", flush=True)
        return save_path
    log_path = Path(args.log_root) / model_key / f"quantize_{codebook}.log"
    run([sys.executable, str(SCRIPTS_DIR / "quantize_model.py"),
         "--model_id", model_id, "--hessian_path", args.hess_root,
         "--save_path", str(save_path), "--codebook", codebook], log_path)
    return save_path


def _pinned_eval_opts(args, gpu_override: str):
    """(cpu_budget_gb, extra_env) for an eval child. Pinning to one physical
    GPU is just a CUDA_VISIBLE_DEVICES override (children use whatever is
    visible); the CPU budget is split across concurrent slots."""
    if gpu_override is None:
        return args.cpu_budget_gb, None
    n_slots = max(len(args.gpu.split(",")), 1)
    return args.cpu_budget_gb / n_slots, {"CUDA_VISIBLE_DEVICES": gpu_override}


def stage_eval_ppl(model_key: str, model_id_or_path: str, tag: str, args,
                   gpu_override: str = None) -> float:
    log_path = Path(args.log_root) / model_key / f"eval_ppl_{tag}.log"
    cpu_budget_gb, extra_env = _pinned_eval_opts(args, gpu_override)
    run([sys.executable, str(SCRIPTS_DIR / "eval_ppl.py"),
         "--model_id_or_path", model_id_or_path, "--ctx_size", str(args.ppl_ctx_size),
         "--gpu_budget_gb", str(args.gpu_budget_gb),
         "--cpu_budget_gb", str(cpu_budget_gb), "--offload_folder", args.offload_folder],
        log_path, extra_env)
    for line in log_path.read_text().splitlines()[::-1]:
        if "perplexity" in line.lower():
            return float(line.strip().split(":")[-1])
    raise RuntimeError(f"couldn't parse perplexity from {log_path}")


def stage_eval_zeroshot(model_key: str, model_id_or_path: str, tag: str, args,
                        gpu_override: str = None) -> dict:
    log_path = Path(args.log_root) / model_key / f"eval_zeroshot_{tag}.log"
    out_json = Path(args.log_root) / model_key / f"eval_zeroshot_{tag}.json"
    batch_size = args.zeroshot_batch_size if tag == "baseline" else args.zeroshot_batch_size_quantized
    cpu_budget_gb, extra_env = _pinned_eval_opts(args, gpu_override)
    run([sys.executable, str(SCRIPTS_DIR / "eval_zeroshot.py"),
         "--model_id_or_path", model_id_or_path, "--tasks", args.zeroshot_tasks,
         "--output_path", str(out_json), "--batch_size", str(batch_size),
         "--gpu_budget_gb", str(args.gpu_budget_gb),
         "--cpu_budget_gb", str(cpu_budget_gb), "--offload_folder", args.offload_folder],
        log_path, extra_env)
    return json.loads(out_json.read_text())["results"]


def run_model(model_key: str, codebooks: list[str], results: dict, args) -> None:
    model_id = MODELS[model_key]
    size_note = " (large model -- dense baseline eval will offload to CPU/disk, expect this to be slow)" \
        if model_key in LARGE_MODELS and not args.skip_baseline else ""
    print(f"\n=== {model_key} ({model_id}){size_note} ===", flush=True)
    entry = results.setdefault(model_key, {"model_id": model_id})
    t0 = time.time()

    # collect hessians once (bit-width-independent), quantize the missing
    # bit-widths, then optionally delete
    missing = [cb for cb in codebooks
               if not (quantized_checkpoint_path(model_id, cb, args) / "quip_config.json").exists()]
    if missing:
        hess_dir = stage_collect_hessians(model_key, model_id, args)
        for cb in missing:
            stage_quantize(model_key, model_id, cb, args)
        if args.delete_hessians_after:
            print(f"[{model_key}] deleting Hessian cache {hess_dir}", flush=True)
            shutil.rmtree(hess_dir, ignore_errors=True)
    else:
        print(f"[{model_key}] all requested checkpoints ({', '.join(codebooks)}) already "
              "present, skipping hessians + quantization", flush=True)

    # pending evals as (store, key, kind, path, tag) jobs: single-GPU-sized
    # ones run concurrently (one per GPU), all-GPU ones run alone first
    jobs = []
    if not args.skip_baseline:
        for key, kind in (("baseline_ppl", "ppl"), ("baseline_zeroshot", "zeroshot")):
            if key not in entry:
                jobs.append((entry, key, kind, model_id, "baseline"))
    for cb in codebooks:
        quant_path = quantized_checkpoint_path(model_id, cb, args)
        sub = entry.setdefault(cb, {})
        for key, kind in (("quantized_ppl", "ppl"), ("quantized_zeroshot", "zeroshot")):
            if key not in sub:
                jobs.append((sub, key, kind, str(quant_path), cb))

    gpu_ids = args.gpu.split(",")
    gpu_budget_bytes = int(args.gpu_budget_gb * 1024**3)

    def needs_all_gpus(job) -> bool:
        _store, _key, _kind, path, tag = job
        if len(gpu_ids) <= 1:
            return False  # nothing to pin; everything already runs serially below
        if tag == "baseline":
            return model_key in LARGE_MODELS
        state = Path(path) / "quantized_state.pt"
        return state.exists() and state.stat().st_size > gpu_budget_bytes

    def run_eval(job, gpu_override=None):
        store, key, kind, path, tag = job
        fn = stage_eval_ppl if kind == "ppl" else stage_eval_zeroshot
        return store, key, fn(model_key, path, tag, args, gpu_override=gpu_override)

    exclusive = [j for j in jobs if needs_all_gpus(j)]
    parallel = [j for j in jobs if not needs_all_gpus(j)]

    for job in exclusive:
        store, key, value = run_eval(job)
        store[key] = value
        save_results(Path(args.results_path), results)

    if len(gpu_ids) <= 1 or len(parallel) <= 1:
        for job in parallel:
            store, key, value = run_eval(job)
            store[key] = value
            save_results(Path(args.results_path), results)
    else:
        import queue
        from concurrent.futures import ThreadPoolExecutor, as_completed

        free_gpus = queue.Queue()
        for g in gpu_ids:
            free_gpus.put(g)

        def run_pinned(job):
            gpu = free_gpus.get()
            try:
                return run_eval(job, gpu_override=gpu)
            finally:
                free_gpus.put(gpu)

        # results/save_results are only touched from this thread, so no
        # locking; a failed eval doesn't cancel siblings, but the model is
        # still reported failed.
        errors = []
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
            futures = [ex.submit(run_pinned, job) for job in parallel]
            for fut in as_completed(futures):
                try:
                    store, key, value = fut.result()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    errors.append(e)
                    continue
                store[key] = value
                save_results(Path(args.results_path), results)
        if errors:
            raise errors[0]

    entry["wall_time_s"] = entry.get("wall_time_s", 0) + (time.time() - t0)
    save_results(Path(args.results_path), results)
    print(f"=== {model_key} done in {time.time() - t0:.1f}s ===", flush=True)


def main(args):
    # Every child process (and any library call in this process) sees only the
    # chosen physical GPU, as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    codebooks = args.codebooks.split(",")
    unknown = [cb for cb in codebooks if cb not in ("2bit", "3bit", "4bit")]
    if unknown:
        raise SystemExit(f"unknown codebook(s) {unknown}; valid: 2bit, 3bit, 4bit")

    Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
    results = load_results(Path(args.results_path))

    # one model's failure shouldn't abort the sweep: record, continue, report
    failures = []
    for model_key in args.models.split(","):
        try:
            run_model(model_key, codebooks, results, args)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"=== {model_key} FAILED (see traceback above) -- continuing with next model ===",
                  flush=True)
            failures.append(model_key)

    if failures:
        print(f"\nDone, but these models failed and need attention: {', '.join(failures)}. "
              f"Results so far at {args.results_path}", flush=True)
        sys.exit(1)
    print(f"\nAll requested models complete. Results at {args.results_path}", flush=True)


if __name__ == "__main__":
    main(parser.parse_args())
