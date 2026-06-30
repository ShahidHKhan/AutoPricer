# Day 2 — Data Pre-processing / LLM Rewriting (`2_preprocessing.ipynb`)

## What it accomplishes
Takes the raw `Car` objects from Day 1 (specifically their noisy `full` text blob) and rewrites each one into a clean, consistently-formatted natural-language `summary` using Gemini 2.5 Flash-Lite. This summary becomes the primary text feature used in every later modeling stage (Day 3 baselines, Day 4 deep learning, Day 5+ fine-tuning prompts). Also ports the `Tester`/`evaluator.py` class needed for Day 3.

## How it works, step by step

1. **Setup & auth** — loads `.env`, logs into Hugging Face Hub, confirms `GOOGLE_API_KEY` (Gemini key) is set.

2. **Load raw data** — pulls `cars_lite_raw` train/val/test splits via `Car.from_hub`, inspects one sample's `full` field to confirm what's about to be sent to the LLM.

3. **Define the rewrite prompt** — a fixed `SYSTEM_PROMPT` instructing Gemini to respond only in a strict 5-line format: `Title`, `Category`, `Make`, `Description` (1 sentence on condition/history), `Details` (1 sentence on mileage/fuel/transmission). Uses `litellm`'s `completion()` so the same code could swap providers if needed.

4. **`Preprocessor` class** — wraps the Gemini call with retry logic: up to 5 attempts per car, exponential backoff capped at 10s between attempts, tracks running totals for input tokens, output tokens, and dollar cost (`response._hidden_params["response_cost"]`). On exhausting retries, leaves `car.summary = None` rather than crashing the whole run.

5. **Concurrent batch runner (`run_preprocessing`)** — uses a `ThreadPoolExecutor` (network-bound, not CPU-bound, so threads not processes) with `max_workers=10` to fire off many Gemini calls in parallel, with a live `tqdm` progress bar showing running failure count.

6. **Full run on `cars_lite`** — processes all 29,036 cars (train+val+test combined). Result: **zero failures**, ~16.84 it/s average, **$1.43 total cost** — in line with the project's earlier single-call cost estimate (~$0.000052/car).

7. **Verification** — confirms `sum(c.summary is None for c in lite_cars) == 0`, spot-checks a couple of summaries for format consistency.

8. **Push processed data to Hub** — strips the now-unneeded `full` and `id` fields (kept only for internal bookkeeping) and pushes `cars_lite_processed` as a `DatasetDict` (train/validation/test).

9. **Port `evaluator.py`** — writes the `Tester` class (operates on `.price`/`.title`/`.summary`, i.e. `Car` objects directly) to `auto_pricer/evaluator.py` via `%%writefile`, since this is needed before Day 3 baselines can be evaluated.

## Results achieved
- `cars_lite_processed` fully populated and pushed to Hugging Face Hub (29,036 cars, zero missing summaries).
- Total Gemini cost for the lite run: **$1.43**.
- The full ~258k `cars_full` run was scoped out but deliberately deferred — the project's chosen strategy is to validate the entire pipeline end-to-end on lite first, then redo this step (along with Day 3/4 and the full QLoRA fine-tune) on `cars_full_raw` once everything downstream is proven to work.

## Key operational notes worth remembering
- AI Studio's free tier caps Gemini Flash models at 1,500 requests/day — billing must be enabled for any full-scale run.
- Gemini's Flash-Lite endpoint can occasionally degrade for hours (503 "high demand" errors) — symptom is throughput collapsing to ~1.7-2.2 it/s. This is a provider issue, not a code bug; the fix is to stop and resume later, not tune retry parameters further.
- `max_workers=8-10` worked well; going higher (15-20) was associated with worse 503 rates in earlier informal testing.
