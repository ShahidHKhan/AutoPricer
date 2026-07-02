# Colab Day 3 — QLoRA Training, cars_lite (`day3_training.ipynb`)

## What it accomplishes
Runs the actual lite QLoRA fine-tune of Llama-3.2-3B on the `cars_lite_prompts` dataset (`r=32`, attention-only LoRA, 1 epoch), then pushes the resulting adapter to Hugging Face Hub. This is the core training step of the whole project — the point where the model starts learning to predict used-car prices from the prompt format, rather than relying on pretraining knowledge alone.

## How it works, step by step

1. **Setup** — installs `transformers`/`accelerate`/`bitsandbytes`/`datasets`/`peft`/`trl`, authenticates via Colab Secrets, loads the tokenizer and `cars_lite_prompts` (train/val).

2. **Model load with T4-specific dtype fixes** — `bnb_4bit_compute_dtype=torch.float16` and an explicit `dtype=torch.float16` at `from_pretrained`. Confirmed via a dtype check (`torch.float16`) before proceeding.

3. **LoRA config + `SFTTrainer` setup** — `LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj","v_proj","k_proj","o_proj"])`, `SFTConfig` with `fp16=True, bf16=False` explicitly set, `max_length=128`, `report_to="none"` and `push_to_hub=False` (both deliberately deferred until training was verified working).

4. **Trainable-parameter dtype fix** — after building the trainer, an inspection loop found every LoRA `lora_A`/`lora_B` matrix (224 tensors total: 28 layers × 4 projections × 2 matrices) was `torch.bfloat16` despite the base model correctly loading as `torch.float16`. Cast all 224 trainable parameters to `torch.float32` (not `float16` — AMP's `GradScaler` requires trainable weights in fp32; only the frozen base model can safely stay fp16).

5. **Train** — `fine_tuning.train()`, 817 steps, 1 epoch.

6. **Push adapter to Hub** — `fine_tuning.model.push_to_hub("ShahidHKhan/autopricer-lite", private=True)` plus the tokenizer, once training was confirmed successful.

## The debugging saga (three stacked root causes, worth remembering)

Getting from "trainer built" to "training actually runs" required diagnosing three distinct, unrelated dtype issues in sequence — each produced the *same* symptom category (crash or catastrophic slowdown) but had genuinely different causes:

1. **T4 lacks bf16 tensor cores** — the original `bnb_4bit_compute_dtype=torch.bfloat16` (copied from Day 1's exploration notebook) caused training to run at **0.04 it/s** (~6 hour ETA for 816 steps) instead of a sane rate. `torch.cuda.is_bf16_supported()` misleadingly reports `True` on a T4 — it only checks CUDA-level compatibility, not actual tensor-core acceleration. Fix: switch to `float16` compute dtype.
2. **`SFTConfig`'s `bf16` autodetect** — even after fixing (1) and setting `fp16=True`, training crashed with `NotImplementedError: ... not implemented for 'BFloat16'`. Root cause: `bf16` silently defaults to `True` when the hardware is detected as capable, *regardless* of `fp16=True` being set. Fix: explicitly pass `bf16=False`.
3. **LoRA adapter dtype mismatch** — same crash persisted even with the base model correctly fp16. Direct inspection (`named_parameters()`) proved every LoRA matrix was bf16, inherited from the model checkpoint's `config.json` metadata rather than the explicit `dtype=torch.float16` used at load. First fix attempt (casting adapters to fp16) produced a *new* error, `ValueError: Attempting to unscale FP16 gradients.` — AMP's `GradScaler` requires trainable parameters in fp32. Final fix: cast trainable params to `float32`, not `float16`.

**Also encountered along the way:** `trl` 1.7.0 removed `SFTConfig`'s `group_by_length` argument (replaced conceptually by `packing`, not used here) — simply dropped, no functional impact at this dataset/sequence-length scale.

## Results achieved

**Training run:**

| Step | Training Loss | Validation Loss | Mean Token Accuracy |
|---|---|---|---|
| 50 | 1.436 | 1.434 | 0.685 |
| 400 | 1.344 | 1.336 | 0.695 |
| 817 (final) | 1.307 | 1.307 | 0.698 |

- **Total runtime: 1h 20m** (4,831.96s), 817 steps, 1 epoch — slower than the early-step it/s reading suggested (settled lower once fp32-adapter + 4-bit dequant overhead was fully warmed up).
- **Training loss and validation loss tracked closely the entire run** (never diverging) — a strong signal of no overfitting on a single epoch over 26,132 examples.
- **Adapter pushed to `ShahidHKhan/autopricer-lite`**: 73.4MB — matching the hand-calculated Colab Day 1 estimate (18,350,080 params × 4 bytes) exactly.

## Key takeaway
The actual training mechanics (LoRA config, trainer setup, `.train()` call) matched the instructions doc closely — the real complexity was T4-specific dtype plumbing that the doc's original snippets didn't fully anticipate (they conditionally chose `bf16`/`fp16` based on hardware, which this project initially hardcoded and had to unwind in three separate steps). The close train/val loss tracking is the most encouraging result: for the eventual full-dataset run on an A100, the doc's original `bf16=True` should be restored (A100 natively supports it) rather than reusing this T4-specific `fp16`/fp32-adapter workaround.

**Still pending:** evaluating the fine-tuned adapter against the same test set and outlier-handling used in Colab Day 2 ($6,723.06 MAE baseline) — this is Colab's `day5_test_finetuned.ipynb`, the next step in the doc's file tree, not yet run.
