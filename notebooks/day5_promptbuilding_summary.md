# Day 5 — Prompt/Completion Dataset Building (`5_promptbuilding.ipynb`)

## What it accomplishes
Converts the cleaned, Gemini-summarized `Car` objects from `cars_lite_processed` into the `prompt`/`completion` format required by `SFTTrainer` for QLoRA fine-tuning, and pushes that as a new HF Hub dataset, `cars_lite_prompts`. This is the last local, CPU-only, no-cost step before the pipeline moves to Google Colab for Day 6-7 fine-tuning.

## How it works, step by step

1. **Load processed data** — pulls `cars_lite_processed` train/val/test via `Car.from_hub`. Confirms **0 missing summaries** across all three splits before proceeding — no cleanup needed, Day 2 finished clean.

2. **Load the Llama-3.2-3B tokenizer** — `AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token=...)`, using the local `.env`'s `HF_TOKEN`. Requires the gated-model license to have been accepted on Hugging Face beforehand. Used only for token counting/truncation — no model weights loaded, CPU-only.

3. **Build prompts** — `car.make_prompts(tokenizer, max_tokens=120, do_round=True)` run over train+val+test. Populates:
   - `prompt`: `"What does this used car cost to the nearest dollar?\n\n{summary}\n\nPrice is $"`, truncating `summary` to 120 tokens if needed.
   - `completion`: the rounded price as a string (e.g. `"77695.00"`).

4. **Truncation risk check** — computed token-length distribution across all 26,132 train summaries before pushing anything: min 41, max 146, mean 58.5. Only **1 row** actually exceeded the 120-token cap — negligible, no upstream fix needed.

5. **Push to Hub** — `Car.push_prompts_to_hub("ShahidHKhan/cars_lite_prompts", train, val, test)`. Note: this method names the splits `train`/`val`/`test` (not `validation`) — a naming quirk specific to this method vs. the earlier `push_to_hub`, worth remembering when referencing `DATASET_NAME` splits later in Colab training code.

6. **Verify round-trip** — reloaded via `load_dataset(...)`, confirmed row counts and column names match expectations exactly.

## Results achieved
- `cars_lite_prompts` pushed to HF Hub: **26,132 train / 1,452 val / 1,452 test**, exactly matching `cars_lite_processed`'s split sizes.
- Each row has exactly two columns: `prompt`, `completion` — all other `Car` fields (`title`, `id`, `full`, etc.) correctly stripped by `to_datapoint()`.
- Sample verified end-to-end: a 2008 Porsche 911 Turbo Convertible prompt/completion pair matched byte-for-byte between the in-memory object (pre-push) and the reloaded Hub dataset (post-push) — confirms no corruption in the round trip.

## Key takeaway
This notebook is pure data-format transformation — no model training, no LLM API calls, entirely free and CPU-only. It's the bridge between Day 2's human-readable `summary` text and the exact input format `SFTTrainer` expects. The near-total absence of truncation (1/26,132 rows) confirms `max_tokens=120` was a well-chosen ceiling given Day 2's `SYSTEM_PROMPT` format constraints — no need to revisit that setting.
