# QuIP# benchmark

QuIP# implementation (2-bit E8P + 3-bit RVQ, no fine-tuning),
benchmarked on WikiText2 ppl + ARC zeroshot. **6/9 models are done** (see
`results.json`): llama2-7b, llama3-8b, qwen2.5-7b, qwen3-8b, qwen3-32b,
gemma4-12b. Remaining: gemma4-31b, llama3-70b.

## Install

```bash
conda create -n quipsharp -y python=3.12 && conda activate quipsharp
pip install torch --index-url https://download.pytorch.org/whl/cu126  # match your driver
pip install transformers==5.13.0 datasets accelerate lm-eval pytest sentencepiece protobuf
pip install -e .
hf auth login          # token needs gated meta-llama/google access
```

## Run

```bash
conda activate quipsharp
nohup setsid bash scripts/launch_full_benchmark.sh \
  --models gemma4-31b,llama3-70b \
  --gpu 0,1 --gpu_budget_gb 20 --cpu_budget_gb 42 > logs/orchestrator.log 2>&1 &
```

- Size the flags to the machine: `--gpu` = GPUs to use, `--gpu_budget_gb`
  ≈ VRAM − 4GB, `--cpu_budget_gb` ≈ RAM − 20GB (or whatever is suitable).
- Progress: `logs/orchestrator.log`; per-stage logs in `logs/<model>/`.
