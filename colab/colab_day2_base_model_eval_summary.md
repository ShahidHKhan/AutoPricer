# Colab Day 2 — Base Model Zero-Shot Evaluation (`day2_base_model_eval.ipynb`)

## What it accomplishes
Establishes the "before fine-tuning" baseline for the QLoRA phase: how well does the raw, untrained Llama-3.2-3B (4-bit, zero-shot) price a used car when given the exact prompt format built in Day 5? This becomes the `"Llama-3.2-3B (base, 4-bit)"` row in the final results chart, and the number the fine-tuned model (Colab Day 3+) needs to beat to prove QLoRA training was worthwhile.

## How it works, step by step

1. **Setup** — installs deps, authenticates via Colab Secrets, loads the tokenizer and the 4-bit base model (`nf4`, double quant, `bnb_4bit_compute_dtype=torch.bfloat16`), loads `cars_lite_prompts`'s **test split only** (1,452 cars — held out, never seen in training).

2. **`predict_price(prompt, model, tokenizer)`** — tokenizes the prompt, calls `model.generate(max_new_tokens=8)` under `torch.no_grad()`, decodes only the newly generated tokens (slicing off the prompt length first).

3. **`extract_price(text)`** — regex-parses the first numeric-looking substring out of the model's raw text output (handles trailing hallucinated text, commas, decimals).

4. **Single-example sanity check** — ran on one test row (2020 Kia Soul LX, actual $15,995) before scaling up. Raw output `'12,000.00. What is'` — coherent price, but trails into a hallucinated continuation (expected for an untrained base model; it has no learned stopping behavior after the price).

5. **Full test-set loop** — `predict_price` + `extract_price` run over all 1,452 test rows via `tqdm`. **Zero parse failures** — every row produced an extractable number, even where the underlying prediction was wildly wrong.

6. **Initial MAE calculation** — computed mean/median/RMSE across all 1,452 predictions. The raw MAE (**$5,516,356.86**) was absurd, while the median (**$4,985.01**) looked completely reasonable — a classic signature of one or a few extreme outliers dominating the mean.

7. **Outlier diagnosis** — printed the 10 largest-error predictions with their raw model output. Found **one single hallucination**: the model predicted `'8,000,000,000.'` (eight billion dollars) for a car actually worth $4,999. Confirmed via arithmetic that this one row alone accounted for essentially the entire MAE distortion ($8B / 1,452 rows ≈ $5.5M, matching the inflated MAE almost exactly). The other 9 rows in the worst-10 list were genuine, unremarkable zero-shot misses.

8. **Robust re-calculation** — excluded predictions above a generous $1M sanity bound (only 1 row excluded) and recomputed MAE/RMSE on the clean set.

## Results achieved

| Metric | Value |
|---|---|
| Parse failures | 0 / 1,452 |
| Outliers excluded (>$1M) | 1 / 1,452 |
| **MAE (outliers excluded)** | **$6,723.06** |
| RMSE (outliers excluded) | $10,460.88 |
| Median absolute error | $4,985.01 |

Full test-set inference took **21m17s** on a T4 (≈1.14 it/s).

## Key takeaway
A single hallucinated 8-billion-dollar prediction inflated the naive MAE by nearly 1000x — the median error told the true story the whole time, and stayed completely unaffected by the outlier. This is the baseline the fine-tuned model needs to clear: **MAE ≈ $6,723 / median ≈ $4,985**. It's also a preview of a needed consistency practice for evaluating the fine-tuned model later — the same outlier-exclusion logic (or at minimum, a check for it) should be applied there too, so one future hallucination can't make an actually-improved fine-tuned model look worse than the base model on MAE alone.

**For comparison against other stages already measured:** this zero-shot base-model MAE ($6,723) is meaningfully worse than Day 3's best traditional ML model (XGBoost text+numeric, $2,798.88) and Day 4's zero-shot Gemini 2.5 Flash ($2,996.45) — expected, since an untrained 3B open-weight model has neither task-specific training nor Gemini-scale pretraining on pricing-adjacent text. The QLoRA fine-tune's job is to close (or beat) that gap using only ~26k lite training examples.
