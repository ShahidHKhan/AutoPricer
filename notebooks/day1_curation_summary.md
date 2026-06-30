# Day 1 — Data Curation (`1_curation.ipynb`)

## What it accomplishes
Takes the raw Kaggle "US Used Cars" CSV (66 columns, scraped from cars.com) and turns it into clean, structured `Car` objects, split into train/val/test, and pushed to Hugging Face Hub as the raw dataset for the rest of the pipeline. This notebook is run twice — once with `LITE_MODE = True` (50k rows → `cars_lite_raw`), once with `LITE_MODE = False` (500k rows → `cars_full_raw`).

## How it works, step by step

1. **Config & imports** — `LITE_MODE` flag controls `NROWS` (50k or 500k) and the target HF dataset name. Logs into Hugging Face Hub using `HF_TOKEN` from `.env`.

2. **Load raw CSV** — `pd.read_csv` with `nrows=NROWS`.

3. **Drop dead columns** — removes columns that are 100% null (`is_certified`, `combine_fuel_economy`, `vehicle_damage_category`), truck-specific with >90% null (`bed`, `cabin`, `bed_length`, `bed_height`), or pure listing/dealer metadata with no predictive value (`is_oemcpo`, `isCab`, `main_picture_url`, `listing_id`, `sp_id`, `sp_name`, `dealer_zip`, `latitude`, `longitude`, `vin`).

4. **Drop new cars** — filters out `is_new == True` rows, since this is a used-car pricer and new-car listings follow MSRP/dealer-markup logic rather than depreciation signals.

5. **Impute missing values**:
   - Damage/history cluster (`salvage`, `theft_title`, `frame_damaged`, `fleet`, `has_accidents`) → `False`. These five columns share the same ~40% missingness pattern, traced to one underlying vehicle-history-report data source; imputing `False` is safe since sellers with clean histories are more likely to have the report available.
   - `owner_count` → median.
   - Numeric columns (`horsepower`, `torque`, `power`, `highway_fuel_economy`, `city_fuel_economy`, `engine_displacement`) → median. Note: `torque` and `power` arrive as composite strings (e.g. `'200 lb-ft @ 1,750 RPM'`), so the leading number is regex-extracted before imputing.

6. **Drop rows missing critical fields** — `price`, `mileage`, `year`, `make_name`, `model_name` have no sensible imputation, so rows missing any of these are dropped outright.

7. **Filter outliers** — `price` between 500 and 150,000 (excludes salvage-only and exotic outliers), `year >= 1990` (pre-1990 cars are collectibles, not daily drivers), `mileage <= 300,000` (beyond this, price signal collapses).

8. **Build the `full` text blob** — combines structured fields (title, body type, mileage, horsepower, fuel, transmission, accident/frame-damage flags) with a truncated, scrubbed `description` field (cars.com scraping artifacts stripped via regex, capped to 400 chars so options-lists don't dominate). This blob is what Day 2's Gemini rewrite step will consume.

9. **Build `Car` pydantic objects** — one per surviving row, with sequential `id` assigned.

10. **Train/val/test split** — 90/5/5, shuffled with `random.seed(42)` for reproducibility.

11. **Push to Hugging Face Hub** — `Car.push_to_hub(...)` writes the three splits as a `DatasetDict` to `ShahidHKhan/cars_lite_raw` or `ShahidHKhan/cars_full_raw`.

12. **Verify round-trip** — reloads from the Hub via `Car.from_hub(...)` and checks row counts and a sample repr match expectations.

## Results achieved
- Lite run (50k raw rows): **29,036 cars** survived cleaning → pushed as `cars_lite_raw` (26,132 train / 1,452 val / 1,452 test).
- Full run (500k raw rows): **258k cars** survived cleaning → pushed as `cars_full_raw`. (Doc estimated ~270-290k; actual came in slightly lower but well within reason.)

## Key takeaway
This notebook is purely structural/numeric cleaning — no LLM calls, no text rewriting. Its output (`full` field) is raw and somewhat noisy; Day 2 is responsible for turning it into a clean natural-language summary.
