# Colab Day 1 — QLoRA Intro: LoRA Sizing & Quantization Footprint (`day1_qlora_intro.ipynb`)

## What it accomplishes
The first Colab notebook (T4 GPU) in the QLoRA fine-tuning phase. Builds hands-on intuition for two things before any real training happens: (1) exactly how many trainable parameters each LoRA configuration produces for Llama-3.2-3B, and (2) how much GPU memory the base model actually needs at each quantization level. Nothing here trains or modifies the model — it's pure sizing/exploration, matching the doc's Week-7-derived "Day 1" curriculum.

## How it works, step by step

1. **Lite LoRA sizing (`r=32`, attention layers only)** — hand-computes trainable parameter count from Llama-3.2-3B's actual dimensions: hidden size 3072, KV dimension 1024 (grouped-query attention means `k_proj`/`v_proj` are smaller than `q_proj`/`o_proj`), across 28 transformer layers.

2. **Full LoRA sizing (`r=256`, attention + MLP layers)** — same approach, but adds `gate_proj`/`up_proj`/`down_proj` (each `3072 × r + 8192 × r`, since Llama-3.2-3B's MLP intermediate dimension is 8192) on top of the attention-layer terms. Reserved for the eventual `cars_full` run — not used for the lite fine-tune.

3. **Auth + install** — installs `transformers`/`accelerate`/`bitsandbytes`, logs into Hugging Face Hub via Colab's Secrets manager (`HF_TOKEN`, since Colab doesn't have access to the local `.env`).

4. **Quantization footprint — three levels, one at a time** (per the doc's instruction to restart the runtime between loads to avoid GPU memory buildup on the T4's 15GB):
   - No quantization (native bf16): baseline memory footprint
   - 8-bit (`BitsAndBytesConfig(load_in_8bit=True)`)
   - 4-bit (`nf4`, double quantization, `bnb_4bit_compute_dtype=torch.bfloat16`) — the actual config QLoRA training will use

## Results achieved

**LoRA parameter sizing:**

| Config | Trainable params | Adapter size (fp32) | vs. lite |
|---|---|---|---|
| Lite (`r=32`, attention only) | 18,350,080 | 73.4 MB | — |
| Full (`r=256`, attention + MLP) | 389,021,696 | 1,556.1 MB | **21.2x** |

**Quantization memory footprint (Llama-3.2-3B, ~3.2B params):**

| Precision | Memory | vs. baseline |
|---|---|---|
| bf16 (no quant) | 6.4 GB | — |
| 8-bit | 3.6 GB | ~44% smaller |
| 4-bit (nf4, double-quant) | 2.2 GB | ~66% smaller |

## Key takeaway
The full config isn't just a bigger `r` — it's over 20x more trainable capacity than lite, which is exactly why the instructions doc gates it behind the full ~270k-row dataset (enough signal to justify that much trainable capacity without overfitting). On the quantization side, 4-bit at 2.2GB leaves comfortable headroom inside the T4's 15GB, even before accounting for activations, gradients, and the LoRA adapter itself.

**Known gotcha carried forward to Colab Day 3 (training):** this notebook's 4-bit config used `bnb_4bit_compute_dtype=torch.bfloat16`. That setting turned out to be the root cause of a severe training slowdown later (~0.04 it/s, 100x slower than expected) — **T4 GPUs (Turing architecture) lack native bf16 tensor core support**, so bf16 compute silently falls back to a much slower emulated path. `torch.cuda.is_bf16_supported()` misleadingly returns `True` on a T4 (it only checks CUDA-level availability, not tensor-core acceleration). The fix used in training was switching to `bnb_4bit_compute_dtype=torch.float16` and `dtype=torch.float16` at model load — worth updating this notebook's Cell 5 to match if it's re-run, and worth remembering this is a T4-specific issue that won't apply once training moves to an A100 for the `cars_full` run.
