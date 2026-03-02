# Model Architecture

This document explains how each model framework works, how models are trained and saved, and how the model factory pattern ties everything together.

## Overview

The repo contains three model frameworks:

| Framework | Purpose | Output |
|-----------|---------|--------|
| **Probabilistic Framework** | Production model with uncertainty quantification | Multiple quantile predictions + point forecast |
| **Matrix Decomposition** | Alternative approach capturing spatial-temporal patterns | Single point prediction |
| **LightGBM Benchmark** | Baseline comparison model | Single point prediction |

All models follow a common interface: they have `train(data)`, `predict(X)`, `save(path)`, and `load(path)` methods. The `ModelFactory` class in `src/models/model_factory.py` instantiates the correct model based on a name string and the corresponding config.

## Model Factory

The factory pattern means the pipeline does not need to know which specific model class it is working with. It simply calls:

```python
model = ModelFactory(config).get_model("probabilistic_framework")
model.train(data)
predictions = model.predict(X)
```

Supported model names:
- `"probabilistic_framework"` — returns a `ProbabilisticForecastingModel`
- `"lightgbm"` — returns a `LightGBMModel`
- `"matrix_decomposition_framework"` — returns a `MatrixDecompositionFramework`

---

## 1. Probabilistic Framework (Production Model)

**Location:** `src/models/probabilistic/`

This is the most important model in the repo. It produces a full predictive distribution rather than just a single number, which is critical for decision-making (e.g. "is there a 90% chance this beach is safe?").

### What Is Quantile Regression?

Standard regression predicts the average (mean) of the target variable. Quantile regression instead predicts specific percentiles. For example:

- The **0.50 quantile** (median) is the value where 50% of observations fall below
- The **0.90 quantile** is the value where 90% of observations fall below
- The **0.95 quantile** gives a conservative upper estimate

By training multiple quantile models, we get a picture of the full distribution, not just the centre.

### Three-Stage Architecture

#### Stage 1: Quantile Ensemble

**Location:** `src/models/probabilistic/quantile_modeling.py`

The first stage trains **12 independent LightGBM models**, one for each quantile level:

```
Quantiles: [0.20, 0.35, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975]
```

Each model uses `objective='quantile'` with `alpha` set to the quantile level. They all receive the same input features but learn to predict different parts of the distribution.

The output is a DataFrame with columns like `q_0.20`, `q_0.35`, ..., `q_0.975`.

**Out-of-fold (OOF) predictions:** During training, 2-fold cross-validation is used to generate predictions on the training data without data leakage. Each fold trains on half the data and predicts the other half. These OOF predictions are used as input for the optional meta-learner (Stage 2).

#### Stage 2: Meta-Learner (Optional)

The meta-learner is a stacking model that takes the quantile predictions as input features and learns to produce a refined point forecast. It is trained on the OOF predictions from Stage 1 to avoid overfitting.

Key details:
- It is a LightGBM regressor trained on the quantile columns
- It uses early stopping with a custom metric: recall on exceedance samples (those >= 280 MPN/100mL)
- A 15% validation split is used for early stopping
- It can be enabled or disabled via the `meta_learner` flag in config

When disabled (the current default), the median quantile (q_0.50) is used as the point prediction.

#### Stage 3: Calibration and Monotonicity

Two post-processing steps are applied:

**Isotonic monotonicity enforcement:** Quantile predictions should always be monotonically increasing (q_0.20 <= q_0.35 <= ... <= q_0.975). In practice, independently trained models can occasionally violate this. Scikit-learn's `IsotonicRegression` is applied per row to enforce this constraint.

**Conformalized quantile regression:** Uses the MAPIE library (`ConformalizedQuantileRegressor`) to calibrate prediction intervals. This splits training data into a fit set and a conformalization set, fits the model on one, and calibrates the uncertainty on the other.

### Output Format

The probabilistic model returns a DataFrame with these columns:

```
q_0.20  q_0.35  q_0.50  q_0.60  q_0.70  q_0.75  q_0.80  q_0.85  q_0.90  q_0.925  q_0.95  q_0.975  predictions
```

The `predictions` column contains the point forecast (either from the meta-learner or the median quantile).

### Domain Constraints

All predictions are post-processed:
- **Minimum:** 5 MPN/100mL (Enterococci are never truly absent)
- **Rounding:** Values are rounded to the nearest integer
- These constraints are applied to all quantile columns and the point prediction

---

## 2. Matrix Decomposition Framework

**Location:** `src/models/matrix_decomp/`

This framework takes a fundamentally different approach. Instead of modelling each sample independently, it captures the **joint spatial-temporal structure** of contamination across all sites simultaneously.

### What Is NMF?

Non-negative Matrix Factorisation (NMF) decomposes a large matrix into two smaller matrices whose product approximates the original:

```
X ≈ W × H^T

Where:
  X = data matrix (rows = timestamps, columns = sites)
  W = temporal factors (how each latent pattern varies over time)
  H = spatial factors (how each latent pattern varies across sites)
```

All values are non-negative, which is appropriate for Enterococci counts (you cannot have negative bacteria).

### Training Pipeline

1. **Pivot the data** into a matrix: rows are timestamps, columns are sites, values are Enterococci counts
2. **Fill missing values** using forward-fill within each day and row means
3. **Apply NMF** with 4 latent components to get W (temporal) and H (spatial) matrices
4. **Train a temporal model** (RandomForest) to predict the W matrix from time-varying features (rainfall, wind, tides)
5. **Train a spatial model** (RandomForest) to predict the H matrix from site-level features (location, soil type, land cover)

### Prediction

To predict for new data:
1. Use the temporal model to predict W from the new environmental features
2. Use the spatial model to predict H from site features
3. Reconstruct: X_predicted = W_predicted × H_predicted^T
4. Reshape back from matrix form to a single-column Series

### Why This Model?

It captures correlations between sites. If two beaches tend to spike together after heavy rain, NMF can learn that shared latent factor. The main probabilistic model treats each sample independently and does not model these spatial relationships.

---

## 3. LightGBM Benchmark

**Location:** `src/models/benchmarks/lightgbm_models.py`

A single LightGBM model with a **Poisson objective function**. Poisson regression is appropriate because Enterococci counts are non-negative integers (count data).

This model serves as a simple baseline. It produces point predictions only, with no uncertainty quantification. The same domain constraints apply (minimum 5, maximum 10,000, rounded to integers).

It inherits from scikit-learn's `BaseEstimator` and `RegressorMixin`, so it is compatible with standard scikit-learn workflows (cross-validation, grid search, etc.).

---

## Hyperparameters

All hyperparameters are defined in `src/config/main_config.yaml` under the `models` section. Key parameters for the probabilistic framework:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_estimators` | 250 | Number of boosting rounds |
| `learning_rate` | 0.04 | Step size for gradient descent |
| `num_leaves` | 31 | Maximum leaves per tree (controls complexity) |
| `min_data_in_leaf` | 10 | Minimum samples in a leaf node |
| `max_depth` | 5 | Maximum tree depth (prevents overfitting) |
| `colsample_bytree` | 0.8 | Fraction of features used per tree |
| `reg_alpha` | 0.05 | L1 regularisation |
| `reg_lambda` | 0.65 | L2 regularisation |

These were tuned to balance sensitivity (detecting exceedances) with specificity (avoiding false alarms).

## Saving and Loading Models

Models are serialised using `joblib` and saved to the `models/` directory:

```
models/
├── probabilistic_framework.joblib
├── lightgbm.joblib
└── matrix_decomposition_framework.joblib
```

Each model class has `save(path)` and `load(path)` class methods. The pipeline saves models after training and can reload them for prediction without retraining.

## Class Relationships

```
ModelFactory
    │
    ├── get_model("probabilistic_framework")
    │       └── ProbabilisticForecastingModel
    │               └── uses ProbabilisticQuantileEnsembleModel (Stage 1)
    │               └── uses LGBMRegressor (Stage 2, meta-learner)
    │               └── uses IsotonicRegression (Stage 3)
    │               └── uses ConformalizedQuantileRegressor (Stage 3)
    │
    ├── get_model("lightgbm")
    │       └── LightGBMModel (extends BaseEstimator, RegressorMixin)
    │
    └── get_model("matrix_decomposition_framework")
            └── MatrixDecompositionFramework
                    └── uses MatrixDecompositionModel (NMF core)
                    └── uses RandomForestRegressor (temporal + spatial)
```
