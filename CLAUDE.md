# CLAUDE.md

## Project

Enterococci water quality prediction model for Environment Canterbury (ECan). LightGBM quantile regression predicting bacterial concentrations at coastal monitoring sites in Canterbury, New Zealand. The model is live in production, serving predictions via the LAWA API.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
python src/pipeline/main_pipeline.py --mode all
```

Python 3.14. No version pins in requirements.txt (intentional, for compatibility).

## Repo structure

- `src/pipeline/main_pipeline.py` -- entry point, orchestrates everything
- `src/config/main_config.yaml` -- all pipeline settings (hyperparams, CV folds, paths, which models to run)
- `src/config/paths.py` -- path constants
- `src/data/` -- DataLoader, Preprocessor, FeatureEngineer
- `src/models/probabilistic/` -- production model (quantile ensemble + optional meta-learner + calibration)
- `src/models/matrix_decomp/` -- NMF-based alternative framework
- `src/models/benchmarks/` -- single LightGBM baseline with Poisson objective
- `src/models/model_factory.py` -- factory pattern, instantiates models by name
- `src/evaluation/` -- Evaluator base class + TimeSeriesEvaluator with TSCV
- `src/visualization/` -- Dash dashboard, interactive Plotly plots
- `docs/` -- onboarding documentation and development backlog

## Key conventions

- **British English** in all documentation (e.g. "modelling", "visualisation", "colour")
- **No em dashes** in docs -- use double hyphens (`--`) instead
- Target variable is `Enterococci` (MPN/100mL)
- Exceedance threshold: 280, precautionary threshold: 140, minimum prediction: 5, maximum cap: 10,000
- Model outputs are constrained: minimum 5 MPN/100mL, rounded to nearest integer
- TSCV uses 3 annual folds defined in config; no random splits

## Three model frameworks

1. **Probabilistic framework** (production) -- 12 quantile LightGBM models -> optional meta-learner -> isotonic calibration
2. **Matrix decomposition** -- NMF spatial-temporal decomposition + RandomForest. Not the focus right now.
3. **LightGBM benchmark** -- single Poisson-objective model, baseline comparison only

## Config-driven behaviour

Almost everything is controlled via `src/config/main_config.yaml`:
- `pipeline.models_to_process` controls which frameworks run
- `models.meta_learner` toggles the stacker (currently `false`)
- `evaluation.cross_validation` defines fold date boundaries
- Hyperparameters for all models live here

## MAPIE dependency

MAPIE 1.3.0+ restructured its API. The correct import is:
```python
from mapie.regression import ConformalizedQuantileRegressor
```
The new API flow is `fit()` -> `conformalize()` -> `predict_interval()` (not the old `fit()` -> `predict(alpha=)`).

## Current state (as of March 2025)

- 13 active monitoring sites in production
- Cass Bay and Governers Bay excluded (couldn't meet 50% sensitivity threshold)
- 2025/26 season performance: ~80% sensitivity, specificity, and accuracy across ~500 samples
- Meta-learner is built but disabled
- Baseline hyperparameters (v2 aggressive tuning was reverted)
- Training data goes up to late 2024; retraining with 2024/25 and 2025/26 data is top priority

## Branch conventions

- `main` -- production-ready code
- `dev/prep_josh` -- current working branch for intern onboarding prep (docs, cleanup, backlog)

## Things to watch out for

- Feature engineering changes must be mirrored in the inference pipeline (separate repo) to avoid training/serving skew
- The dashboard previously had a hardcoded local path for a banner image (now removed)
- `preprocessing.py` converts DateTime through multiple format passes -- this is intentional to handle mixed date formats in the raw data
- The `performance_evaluation_old_pipeline` method in evaluator.py is called from time_series_evaluator.py and prints coloured terminal output -- it's legacy but functional

## Documentation

All docs are in `docs/`:
- `getting-started.md` -- setup, running the pipeline, troubleshooting
- `project-overview.md` -- domain context, architecture, config
- `data-and-features.md` -- raw data, preprocessing, feature engineering
- `model-architecture.md` -- all 3 frameworks in detail
- `evaluation-and-metrics.md` -- TSCV, metrics, interpreting results
- `visualisation.md` -- plots and dashboard
- `backlog.md` -- prioritised development roadmap
