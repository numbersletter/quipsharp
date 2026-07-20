#!/bin/bash
# Detached launch of the full benchmark sweep, smallest models first, with
# per-model Hessian cleanup (disk can't hold the HF cache plus several
# hundred GB of Hessians). Resumable: re-run after any crash/reboot.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

# the default caching allocator fragments under the quantizer's multi-GB
# allocations (fp64 Cholesky blocks); expandable segments avoid that
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD=(python scripts/run_benchmark.py
  --models llama2-7b,llama3-8b,qwen2.5-7b,qwen3-8b,gemma4-12b,qwen3-32b,gemma4-31b,llama2-70b,llama3-70b
  --delete_hessians_after
  "$@")
"${CMD[@]}" && exit 0
