"""From-scratch QuIP# (2/3/4-bit lattice-codebook post-training quantization)
implementation. See each module's docstring; scripts/ contains the runnable
pipeline (collect_hessians -> quantize_model -> eval_ppl / eval_zeroshot,
orchestrated across models and bit-widths by run_benchmark.py).
"""
