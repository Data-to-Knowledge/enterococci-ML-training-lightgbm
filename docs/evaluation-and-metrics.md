# Evaluation and Metrics

This document explains how models are evaluated, why we use time-series cross-validation, and what each metric means.

## Why Not Standard Cross-Validation?

In standard k-fold cross-validation, data is randomly split into folds. This works well for i.i.d. (independent and identically distributed) data, but water quality data is a **time series**. Tomorrow's Enterococci count depends on today's rainfall, yesterday's wind, and the recent contamination history.

If we randomly split the data, the model could train on future observations and predict past ones, which is unrealistic. In production, the model will only ever have access to past data when making predictions.

**Time-series cross-validation** respects this constraint: training data always comes **before** test data in time.

## How the Folds Work

The evaluation uses **3 temporal folds**, each covering a one-year test window:

| Fold | Training Period | Test Period |
|------|----------------|-------------|
| 1 | Everything before 1 Oct 2021 | 1 Oct 2021 to 1 Oct 2022 |
| 2 | Everything before 1 Oct 2022 | 1 Oct 2022 to 1 Oct 2023 |
| 3 | Everything before 1 Oct 2023 | 1 Oct 2023 to 1 Oct 2024 |

For each fold:
1. The model is trained from scratch on all data up to the fold's start date
2. The model predicts on the test period
3. Metrics are calculated on the test predictions
4. Results are aggregated across all folds

This means each fold has progressively more training data, which mirrors how the model would be retrained in practice.

### Excluded Sites

Six sites are excluded from the test evaluation because they are not part of the production LAWA API:

- Scarborough Beach by clock tower
- Sumner Beach Surf club
- Caroline Bay - mid beach
- Timaru Coast Caroline Bay at Virtue Avenue
- Timaru Coast at yacht club jetty
- Taylors Mistake Beach Surf club

These sites may still appear in training data but are not used to judge model performance.

## Metrics Explained

The evaluation computes both regression metrics (how close are the numbers?) and classification metrics (did we correctly flag unsafe conditions?).

### Regression Metrics

#### RMSE (Root Mean Squared Error)

The square root of the average squared prediction error. Lower is better.

```
RMSE = sqrt( mean( (actual - predicted)^2 ) )
```

RMSE is sensitive to large errors, which is important here because missing a high Enterococci value (e.g. predicting 100 when the actual is 2000) is much worse than a small error on a safe sample.

Three variants are reported:
- **RMSE** — across all samples
- **RMSE_safe** — only for samples below 280 MPN/100mL
- **RMSE_exceedance** — only for samples at or above 280 MPN/100mL

#### WMAPE (Weighted Mean Absolute Percentage Error)

A percentage-based error metric that gives more weight to larger values.

```
WMAPE = sum(|actual - predicted|) / sum(actual) × 100
```

Two variants:
- **WMAPE_safe** — for safe samples (< 280)
- **WMAPE_exceedance** — for exceedance samples (>= 280)

These are reported separately because the model needs to perform well in both regimes. A model that is accurate on safe samples but terrible on exceedances is not useful.

#### MAE (Mean Absolute Error)

The average absolute difference between predicted and actual values. Easier to interpret than RMSE but less sensitive to large errors.

### Classification Metrics

For classification purposes, each sample is labelled as either **safe** (< 280 MPN/100mL) or **exceedance** (>= 280 MPN/100mL). The model's point prediction is classified the same way.

#### Confusion Matrix

|  | Predicted Safe | Predicted Exceedance |
|--|---------------|---------------------|
| **Actually Safe** | True Negative (TN) | False Positive (FP) |
| **Actually Exceedance** | False Negative (FN) | True Positive (TP) |

In this domain:
- **False Negative (FN)** = the model says the water is safe, but it is actually contaminated. This is the most dangerous error — people might swim in unsafe water.
- **False Positive (FP)** = the model says the water is unsafe, but it is actually fine. This causes unnecessary beach closures, which is costly but not dangerous.

#### Sensitivity (Recall for Exceedances)

```
Sensitivity = TP / (TP + FN)
```

The proportion of actual exceedances that the model correctly identified. **This is the most important metric** because missing an exceedance puts public health at risk. We aim for high sensitivity, ideally above 70%.

#### Specificity (Recall for Safe Conditions)

```
Specificity = TN / (TN + FP)
```

The proportion of actually safe conditions that the model correctly identified as safe. High specificity means fewer unnecessary beach closures.

#### The Sensitivity-Specificity Trade-off

There is always a tension between these two metrics. Making the model more cautious (predicting higher values) increases sensitivity but decreases specificity, leading to more false alarms. The probabilistic framework helps manage this trade-off because decision-makers can choose which quantile to act on:

- Use **q_0.50** (median) for a balanced prediction
- Use **q_0.80** or **q_0.90** for a more cautious prediction that catches more exceedances
- Use **q_0.95** for a very conservative upper bound

#### Precautionary Sensitivity

Similar to sensitivity but uses the 140 MPN/100mL threshold instead of 280. This measures how well the model detects conditions that are approaching unsafe levels.

## Where Results Are Saved

Evaluation results are saved to `reports/evaluation_results.json`. The structure looks like:

```json
{
  "probabilistic_framework": {
    "fold_1": {
      "rmse": 437,
      "sensitivity": 0.55,
      "specificity": 0.91,
      "wmape_exceedance": 0.62,
      "tp": 16, "fp": 25, "tn": 245, "fn": 13
    },
    "fold_2": { ... },
    "fold_3": { ... },
    "overall": {
      "rmse": 647,
      "sensitivity": 0.68,
      "specificity": 0.92,
      ...
    }
  }
}
```

## Interpreting Results

When reviewing evaluation results, focus on:

1. **Sensitivity** — Is the model catching enough exceedances? Below 60% is concerning.
2. **Specificity** — Is the model generating too many false alarms? Below 80% means too many unnecessary closures.
3. **RMSE_exceedance** — How far off are the predictions when it matters most?
4. **Fold consistency** — Are results stable across folds, or does the model perform well in one period but poorly in another?

A good model for this application typically has:
- Sensitivity around 65 to 75%
- Specificity around 85 to 95%
- Reasonable RMSE_exceedance (lower is better, but exceedance values are inherently noisy)

## Code Structure

| Class | Location | Responsibility |
|-------|----------|---------------|
| `Evaluator` | `src/evaluation/evaluator.py` | Computes all metrics for a given set of predictions |
| `TimeSeriesEvaluator` | `src/evaluation/time_series_evaluator.py` | Runs time-series CV, calls Evaluator per fold, aggregates results |
| `TimeSeriesCV` | `src/evaluation/cross_validation.py` | Generates train/test splits based on date ranges |
