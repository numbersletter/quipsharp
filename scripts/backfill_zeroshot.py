"""Backfill extra zeroshot tasks (default: winogrande) into results.json for
models whose evals already ran. Idempotent: skips entries that already have
every requested task. Run with the orchestrator STOPPED (it holds results.json
in memory and would clobber concurrent edits).

    python scripts/backfill_zeroshot.py            # all models, winogrande
    python scripts/backfill_zeroshot.py --models llama2-7b --tasks winogrande
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_benchmark import MODELS, quantized_checkpoint_path

ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--models", default=",".join(MODELS), type=str)
parser.add_argument("--tasks", default="winogrande", type=str)
parser.add_argument("--results_path", default=str(ROOT / "results.json"), type=str)
parser.add_argument("--quant_root", default=str(ROOT / "quantized"), type=str)
parser.add_argument("--batch_size", default=16, type=int)


def eval_tasks(model_id_or_path: str, tasks: str, tag: str, model_key: str, args) -> dict:
    out_json = ROOT / "logs" / model_key / f"backfill_{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(Path(__file__).parent / "eval_zeroshot.py"),
                    "--model_id_or_path", model_id_or_path, "--tasks", tasks,
                    "--output_path", str(out_json), "--batch_size", str(args.batch_size)],
                   check=True)
    return json.loads(out_json.read_text())["results"]


def main(args):
    results_path = Path(args.results_path)
    results = json.loads(results_path.read_text())
    wanted = args.tasks.split(",")

    for key in args.models.split(","):
        entry = results.get(key)
        if not entry:
            continue
        targets = [("baseline_zeroshot", entry, MODELS[key])]
        for cb in ("2bit", "3bit"):
            if cb in entry:
                targets.append(("quantized_zeroshot", entry[cb],
                                str(quantized_checkpoint_path(MODELS[key], cb, args))))
        for field, store, path in targets:
            existing = store.get(field)
            if existing is None or all(t in existing for t in wanted):
                continue
            missing = ",".join(t for t in wanted if t not in existing)
            print(f"[{key}] {field} += {missing} ({path})", flush=True)
            existing.update(eval_tasks(path, missing, f"{field}_{Path(path).name}", key, args))
            results_path.write_text(json.dumps(results, indent=2))

    print("backfill complete")


if __name__ == "__main__":
    main(parser.parse_args())
