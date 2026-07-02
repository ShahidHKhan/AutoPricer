# Colab Day 5 — Fine-Tuned Model Evaluation (`day5_test_finetuned.ipynb`)

## What it accomplishes
Evaluates the QLoRA-fine-tuned Llama-3.2-3B adapter (`ShahidHKhan/autopricer-lite`, trained in Colab Day 3) against the exact same held-out test set and methodology used for the base model in Colab Day 2. This produces the direct "did fine-tuning help" comparison — the entire point of the QLoRA phase of the project — and closes out the lite-mode pipeline end to end.

## How it works, step by step

1. **Load base model + apply adapter** — loads Llama-3.2-3B in 4-bit (same T4-safe dtype settings carried over from Day 3: `bnb_4bit_compute_dtype=torch.float16`, explicit `dtype=torch.float16`), then wraps it with `PeftModel.from_pretrained(base_model, "ShahidHKhan/autopricer-lite")`. Tokenizer loaded from the same Hub repo (confirms Day 3's tokenizer push worked). `fine_tuned_model.eval()` disables the LoRA config's 0.1 dropout for inference.

2. **Load test set** — `cars_lite_prompts`'s test split, same 1,452 rows used in Colab Day 2, ensuring an apples-to-apples comparison.

3. **Single-example sanity check** — ran on `test_ds[0]`, the identical row used in Day 2's base-model spot check (2020 Kia Soul LX, actual $15,995). Reused the exact same `predict_price`/`extract_price` functions from Day 2 for consistency.

4. **Full test-set loop** — same pattern as Day 2: `predict_price` + `extract_price` over all 1,452 rows via `tqdm`. Zero parse failures.

5. **Error metrics + proactive outlier check** — computed MAE/median/RMSE, and *this time* checked the top-5 largest errors immediately (rather than reactively, as Day 2 required) given the known hallucination failure mode discovered in the base model. No hallucination outliers found — all worst-case errors were on genuinely hard cases (higher-value cars), with sane predicted numbers.

## Results achieved

**Single-example comparison (`test_ds[0]`, actual $15,995):**

| | Raw output | Predicted | Error |
|---|---|---|---|
| Base model (Day 2) | `'12,000.00. What is'` | $12,000 | $3,995 |
| **Fine-tuned** | `'16995.00'` | $16,995 | **$1,000** |

Fine-tuned output stopped cleanly after the price — no trailing hallucinated text, unlike the base model. This is direct evidence the fine-tune taught correct output-format behavior, not just better numbers.

**Full test set (1,452 rows):**

| Metric | Base model (Day 2) | Fine-tuned (Day 5) |
|---|---|---|
| Parse failures | 0 / 1,452 | 0 / 1,452 |
| Hallucination outliers | 1 (one $8B prediction) | 0 |
| MAE | $6,723.06 | **$2,471.41** |
| Median absolute error | $4,985.01 | **$1,702.50** |
| RMSE | $10,460.88 | **$3,941.56** |
| Inference time (T4) | 21m17s (1.14 it/s) | 13m55s (1.74 it/s) |

**MAE improved 63%** ($6,723.06 → $2,471.41) after fine-tuning. Inference also got faster, consistent with the model producing shorter, cleaner, more consistent outputs.

## Full project comparison (all models measured so far)

| Model | MAE | Source |
|---|---|---|
| Random baseline | $54,959 | Day 3 |
| Constant (mean) | $9,781 | Day 3 |
| Llama-3.2-3B (base, 4-bit, zero-shot) | $6,723.06 | Colab Day 2 |
| Best numeric-only Linear Regression | $5,374 | Day 3 |
| Text-only Linear Regression | $4,480 | Day 3 |
| Gemini 2.5 Flash (zero-shot) | $2,996.45 | Day 4 |
| Random Forest (text) | $3,327 | Day 3 |
| XGBoost (text-only) | $3,570 | Day 3 |
| XGBoost (text + numeric) | $2,798.88 | Day 3 — previous best |
| **Llama-3.2-3B (fine-tuned, lite QLoRA)** | **$2,471.41** | **Colab Day 5 — new best** |

## Key takeaway
The lite QLoRA fine-tune — 1 epoch, ~26k training examples, `r=32` attention-only LoRA, run on a free Colab T4 — **beat every other model in the project**, including XGBoost with direct access to explicit numeric features (mileage, horsepower, accident history) that the LLM never saw, and zero-shot Gemini 2.5 Flash. This is a meaningful proof point for the value of fine-tuning over both traditional ML and prompting a larger frontier model, achieved with a comparatively small, cheap training run. It also validates the entire debugging effort from Colab Day 3 — the T4 dtype fixes weren't just about getting training to run, they produced a model that generalizes well to unseen data.

This closes out the full lite-mode pipeline (Day 1 → Colab Day 5), per the project's "prove it works end-to-end on lite before touching `cars_full`" strategy. The `cars_full` rerun (Day 2 reprocessing, Day 3/4 re-evaluation, and the A100 full QLoRA fine-tune with `r=256` attention+MLP) remains deferred, per that same plan.
