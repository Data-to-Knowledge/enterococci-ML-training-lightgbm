# Project Overview

## What Are Enterococci?

Enterococci are a type of bacteria found in the intestines of warm-blooded animals. Their presence in recreational water (rivers, beaches, estuaries) indicates faecal contamination, which can cause illness in swimmers. Enterococci are the standard indicator organism for marine water quality in New Zealand and many other countries.

Concentrations are measured in **MPN/100mL** (Most Probable Number per 100 millilitres of water).

## The Problem We Are Solving

Regional councils in New Zealand collect water samples at coastal beaches to test for Enterococci. However, lab results take 24 to 48 hours. By the time a result comes back, the conditions may have changed entirely.

This project builds **machine learning models that predict Enterococci concentrations in near-real-time** using readily available environmental data (rainfall, wind, tides, site characteristics). The goal is to provide timely warnings so councils can issue or lift beach advisories faster.

## Key Thresholds

These thresholds are based on New Zealand recreational water quality guidelines:

| Threshold | Value (MPN/100mL) | Meaning |
|-----------|-------------------|---------|
| **Exceedance** | 280 | Water is considered unsafe for swimming |
| **Precautionary** | 140 | Early warning level; conditions may be deteriorating |
| **Minimum prediction** | 5 | Lower bound applied to model outputs (bacteria are never truly zero) |
| **Maximum cap** | 10,000 | Upper bound applied during preprocessing |

## Study Area

The model covers **15 coastal monitoring sites** in Canterbury, New Zealand. These are beaches where Environment Canterbury regularly samples water quality. Each site has different characteristics (harbour vs open coast, soil type, catchment land use) that influence contamination patterns.

Six additional sites are excluded from test evaluation because they are not part of the production LAWA (Land Air Water Aotearoa) API.

## High-Level Pipeline Flow

```
┌─────────────────┐
│  Configuration   │  main_config.yaml drives all settings
│  (YAML)          │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Data Loading    │  Load CSVs from data/processed/
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Preprocessing   │  Clean, deduplicate, cap target, convert types
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Feature         │  Temporal, wind-shore, lagged Enterococci features
│  Engineering     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Model Training  │  Train one or more model frameworks
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Evaluation      │  Time-series cross-validation, metrics
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Prediction &    │  Generate forecasts, interactive plots,
│  Visualisation   │  Dash dashboard
└─────────────────┘
```

## The Three Model Frameworks

The repo contains three distinct modelling approaches. Each is a separate framework with its own training and prediction logic.

### 1. Probabilistic Framework (Production Model)

This is the **primary model used in production**. Rather than predicting a single number, it produces a full probability distribution via quantile regression. This means it outputs predictions at multiple confidence levels (e.g. "there is a 90% chance the value is below X").

It has a three-stage architecture:
1. **Quantile ensemble** — 12 independent LightGBM models, each trained to predict a different quantile
2. **Meta-learner** (optional) — a stacker that combines quantile predictions into a refined point forecast
3. **Calibration** — isotonic monotonicity enforcement and conformalized quantile regression

See [model-architecture.md](model-architecture.md) for full details.

### 2. Matrix Decomposition Framework

This is an alternative approach that uses **Non-negative Matrix Factorisation (NMF)** to decompose the Enterococci data into latent temporal and spatial patterns. It then uses RandomForest models to predict these latent factors from environmental features, and reconstructs the full prediction matrix.

This captures spatial correlations between sites that the other models do not explicitly model.

### 3. LightGBM Benchmark

A straightforward single LightGBM model with a Poisson objective function. It serves as a **baseline for comparison** against the more complex frameworks. It produces point predictions only (no quantiles or uncertainty).

## Configuration

All pipeline behaviour is driven by `src/config/main_config.yaml`. The key sections are:

| Section | What it controls |
|---------|-----------------|
| `data` | Paths to training/test/prediction CSVs, column names |
| `feature_engineering` | Lag windows for rainfall, wind, and tides |
| `models` | Hyperparameters for each model framework |
| `evaluation` | Which metrics to compute, cross-validation fold dates |
| `pipeline` | Which models to train and evaluate |
| `prediction` | Output paths, whether to generate visualisations |

The config is loaded by `src/config/config_loader.py` and path constants are defined in `src/config/paths.py`.

## Module Map

| Module | Location | Responsibility |
|--------|----------|---------------|
| Pipeline | `src/pipeline/` | Orchestrates the full workflow |
| Config | `src/config/` | Configuration loading and path constants |
| Data | `src/data/` | Loading, cleaning, and feature engineering |
| Models | `src/models/` | Model implementations and factory |
| Evaluation | `src/evaluation/` | Metrics, cross-validation, reporting |
| Visualisation | `src/visualization/` | Dashboards and plots |
| Utils | `src/utils/` | Logging helpers |

## Further Reading

- [getting-started.md](getting-started.md) — How to set up and run everything
- [data-and-features.md](data-and-features.md) — Data pipeline and feature engineering
- [model-architecture.md](model-architecture.md) — Detailed model design
- [evaluation-and-metrics.md](evaluation-and-metrics.md) — How models are evaluated
- [visualisation.md](visualisation.md) — Dashboards and plots
- [backlog.md](backlog.md) — Planned improvements and experiments, prioritised
