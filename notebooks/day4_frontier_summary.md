# Day 4 — Deep Learning and Frontier LLMs (`4_frontier.ipynb`)

## What it accomplishes
Establishes a **zero-shot frontier LLM baseline** for used-car pricing: how accurately can Gemini 2.5 Flash price a car from its Day 2-generated `summary` alone, with no fine-tuning and no numeric features. This number serves two purposes: (1) it's the bar Day 5-7's QLoRA-fine-tuned Llama-3.2-3B needs to clear to justify the fine-tuning effort, and (2) it informs the later Chrome-extension deployment decision — if zero-shot API calls are already accurate enough, that's a viable (simpler) deployment path alongside or instead of a locally-hosted fine-tuned model.

**Scope note:** The instructions doc's original Day 4 also included a from-scratch Deep Neural Network (`DeepNeuralNetwork` + `ResidualBlock` on `HashingVectorizer` + numeric features). This was deliberately skipped — it doesn't produce anything reusable toward the QLoRA/extension pipeline, and the project's actual goal is the fine-tuned Llama model, not a standalone PyTorch net. Day 4 was narrowed to **frontier zero-shot only**.

## How it works, step by step

1. **Setup** — `sys.path.append("..")` to make the `auto_pricer` package importable from `notebooks/`, then loads `cars_lite_processed` test split (1,452 cars) via `Car.from_hub`.

2. **`messages_for(car)`** — builds a single user-role prompt: *"Estimate the price of this used car. Respond with the price, no explanation"* followed by the car's `summary` text.

3. **`gemini_2_5_flash(car)`** — calls `litellm.completion()` with `model="gemini/gemini-2.5-flash"`, returns the raw text response.

4. **Single-row test** — confirmed Gemini does *not* strictly follow "no explanation": it returns formatted strings like `'$17,800'` rather than a bare number, so raw output can't be compared to price directly.

5. **`parse_price(text)`** — regex-based parser that strips `$`/`,` and extracts the numeric value (e.g. `'$17,800'` → `17800.0`). Handles `None` inputs (returns `0.0`) for cases where retries are exhausted.

6. **10-car sample test with cost tracking** — added `gemini_2_5_flash_tracked(car)`, which also captures `response._hidden_params["response_cost"]` per call (same pattern as Day 2's `Preprocessor`). Used to project full-test-set cost before committing to a paid run at scale.
   - Sample cost: $0.0197 / 10 cars → projected **$2.86** for the full 1,452-car test set.
   - One notable miss in the sample: a 2007 Subaru B9 Tribeca with a self-contradictory summary ("Carfax certified with a clean title, though it has a previously frame-damaged history") — Gemini undershot the actual price by $6,825, likely over-weighting the "frame-damaged" phrase.

7. **Full concurrent batch run (`run_zero_shot_batch`)** — `ThreadPoolExecutor` with `max_workers=8` (same worker count the Day 2 doc found to be the 503-error sweet spot), live `tqdm` progress bar, running cost accumulator. Runs all 1,452 test cars through `gemini_2_5_flash_tracked`.

8. **Error metrics** — computes Mean Absolute Error, Median Absolute Error, max/min error, and counts how many predictions failed to parse (`0.0` values) across the full test set — the parse-failure check matters because a garbled response (e.g. a price range) would otherwise silently skew the mean.

9. **Worst-case inspection** — pulls the single highest-error prediction (`np.argmax(errors)`) and prints its summary/actual/predicted to confirm whether outliers are genuine hard cases (rare/exotic vehicles Gemini has little market intuition for) or pipeline bugs (bad parsing).

## Results achieved

**Cost:**
- 10-car sample: $0.0197 → projected $2.86 for full run
- Actual full run (1,452 cars): **$3.52** (~23% above the small-sample projection — worth remembering that small samples can undershoot real cost by ~20-25%, likely due to longer responses on more complex/unusual cars)

**Accuracy (full 1,452-car test set):**

| Metric | Value |
|---|---|
| Mean Absolute Error | $2,996.45 |
| Median Absolute Error | $2,082.50 |
| Max error | $33,100.00 |
| Min error | $0.00 |
| Parse failures | 0 / 1,452 |

**Worst-case outlier:** 2014 Aston Martin Vanquish Coupe — actual $116,900, predicted $150,000 (error $33,100). Confirmed as a genuine hard case, not a parsing bug: exotic/rare vehicles fall outside the dense middle of the used-car market (Camrys, Civics, Explorers) that Gemini has abundant pricing intuition for from training data.

## Day 3 vs Day 4 comparison

| Model | Mean Absolute Error | R² | Notes |
|---|---|---|---|
| XGBoost (text + numeric, Day 3) | **$2,798.88** | 85.7% | Best traditional ML model; combines Gemini-summary text vector with mileage/car_age/horsepower/has_accidents |
| Gemini 2.5 Flash (zero-shot, Day 4) | $2,996.45 | — | Full 1,452-car test set. No training, no explicit numeric features — priced from summary text alone |
| **Gap (XGBoost vs Flash)** | **~$197.57 (~7% worse)** | | |

| Model | Mean Absolute Error | Sample size | Cost |
|---|---|---|---|
| Gemini 2.5 Flash-Lite (zero-shot) | — | — | — *(not yet run — optional)* |
| Gemini 2.5 Pro (zero-shot) | $3,326.40 | 50 cars (`test[:50]`) | $0.6848 |

**Note on the Pro result:** this is measured on a 50-car sample only, not the full 1,452-car test set, so it is **not directly comparable** to XGBoost's or Flash's full-set MAE — a small sample carries much higher variance (e.g. a single exotic/rare car can swing the average significantly). A same-sample comparison (Flash vs Pro on the identical 50 cars) is needed before concluding Pro under- or out-performs Flash — pending.

## Key takeaway
XGBoost with explicit numeric features currently **edges out zero-shot Gemini by ~7%**, consistent with the instructions doc's prediction that cars carry a much stronger numeric price prior than the course's original Amazon-products dataset — an LLM reading text alone can't fully substitute for direct access to mileage/age/horsepower. This sets a concrete target for Day 5-7: the QLoRA-fine-tuned Llama-3.2-3B needs to beat (or meaningfully approach) **$2,798.88** to prove fine-tuning was worth the effort over both traditional ML and zero-shot prompting. It also gives a real cost/accuracy data point relevant to the eventual Chrome-extension deployment decision — a $3.52-per-1,452-cars zero-shot API cost is trivial, but the ~7% accuracy gap and the exotic-car failure mode are worth weighing against the complexity of shipping a locally-hosted fine-tuned model.
