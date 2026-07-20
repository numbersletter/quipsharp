# QuIP# benchmark

QuIP# implementation (2-bit E8P + 3-bit RVQ, no fine-tuning),
benchmarked on WikiText2 ppl + ARC zeroshot. **6/8 models are done** (see
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
  --gpu 0,1 --gpu_budget_gb 60 --cpu_budget_gb 48 > logs/orchestrator.log 2>&1 &
```

- Size the flags to the machine: `--gpu` = GPUs to use, `--gpu_budget_gb`.
  Choose allocation that is appropriate given the config (with 1 H100, for example, you can
  opt to use 80% of the VRAM which would be 64GB and then 50% of available RAM for cpu_budget_gb).
  This is just a safeguard against OOM errors.
- Progress: `logs/orchestrator.log`; per-stage logs in `logs/<model>/`.
