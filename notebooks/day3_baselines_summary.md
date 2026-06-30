# Day 3 — Baselines & Traditional ML (`3_baselines.ipynb`)

## What it accomplishes
Establishes a full ladder of price-prediction baselines on `cars_lite_processed`, from naive (random/constant guessing) up through traditional ML using both numeric features and text-vectorized summaries. This produces the comparison numbers that every later, more sophisticated model (deep learning, frontier LLMs, fine-tuned LLM) needs to beat in order to be considered worthwhile — and the doc's prediction that "cars have a much stronger numeric prior than generic products" is directly tested and confirmed here.

## How it works, step by step

1. **Setup** — imports the ported `Tester`/`evaluate` from `auto_pricer/evaluator.py`, loads `cars_lite_processed` train/val/test from the Hub (26,132 / 1,452 / 1,452).

2. **`random_pricer`** — returns a uniform random guess between $500 and $150,000, ignoring all features. Establishes the absolute floor.

3. **`constant_pricer`** — always predicts the training set's mean price (~$21,973). By definition this produces r² ≈ 0%, since r² measures variance explained relative to the mean.

4. **Linear Regression v1** — features: `mileage`, `car_age` (2026 − year), `text_length` (length of the Gemini summary). First model with real signal.

5. **Linear Regression v2** — adds `horsepower` and `has_accidents` to the v1 feature set. Horsepower in particular drives a large jump in performance, confirming cars have stronger numeric price signal than the course's original Amazon-products dataset.

6. **CountVectorizer + Linear Regression** — vectorizes the `summary` text (`max_features=5000`, English stop words removed) and fits a linear model directly on the resulting sparse bag-of-words matrix, with no numeric features at all. Performs *better* than the numeric-only v2 model — likely because make/model/trim tokens (e.g. "Porsche 911") carry strong price signal on their own.

7. **Random Forest on vectorized text** — same `CountVectorizer` matrix, but with a `RandomForestRegressor` (100 trees) instead of a linear model. Captures non-linear interactions between word tokens; clear train/test gap (overfitting) but still strong test performance.

8. **XGBoost, text-only** — `XGBRegressor` (200 estimators, max_depth=6) on the same text vector. Comparable test performance to the random forest but with a notably smaller train/test gap, suggesting more efficient generalization.

9. **XGBoost, text + numeric combined** — concatenates the sparse text vector with `mileage`, `car_age`, `horsepower`, `has_accidents` via `scipy.sparse.hstack`, then fits `XGBRegressor` on the combined matrix. This is the best-performing traditional ML model in the project, since it captures both the brand/model signal from text and the strong numeric depreciation priors.

All models are evaluated identically via `evaluate(predictor_fn, test)`, which scores 200 randomly-drawn test points, prints color-coded per-point errors, and renders an error-trend chart (cumulative mean ± 95% CI) plus a truth-vs-predicted scatter plot with MSE/r²/average-error in the title.

## Results achieved (test set, $ average absolute error / r²)

| Model | Error | R² |
|---|---|---|
| Random | $54,959 | -2497% |
| Constant (mean) | $9,781 | -0.0% |
| Linear (mileage/age/text_len) | $8,007 | 29.9% |
| Linear (+horsepower/accidents) | $5,374 | 62.6% |
| Text-only Linear Regression | $4,480 | 74.4% |
| Random Forest (text) | $3,327 | 82.9% |
| XGBoost (text-only) | $3,570 | 82.8% |
| **XGBoost (text + numeric)** | **$2,799** | **85.7%** |

## Key takeaway
Adding horsepower to the numeric linear model produced the single biggest jump among the numeric-only models, confirming cars carry much stronger numeric price priors than generic e-commerce products. But text features (via the Gemini-cleaned `summary`) consistently outperformed numeric-only features alone, and the best result combined both — text for brand/model/trim signal, numeric for depreciation-driven priors. This combined XGBoost model ($2,799 error) is the bar that Day 4's deep learning and frontier LLM zero-shot models need to clear.
