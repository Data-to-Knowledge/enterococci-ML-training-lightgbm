# Getting Started

This guide walks you through setting up your environment, running the pipeline, and understanding what outputs to expect.

## Prerequisites

- **Python 3.11+** (3.12 or 3.13 recommended; 3.14 works but some pinned package versions may need updating)
- **Git**
- **VS Code** (recommended) or any Python IDE
- Access to 1Password for any credentials or environment variables required by the team

## Clone and Branch Setup

```bash
git clone <repo-url>
cd enterococci-ML-training-lightgbm
git checkout dev/prep_josh
```

## Creating a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **macOS/Linux:** `source .venv/bin/activate`

## Installing Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you hit build errors (especially with `pandas` or `numpy`), it is likely a Python version mismatch. Try removing version pins in `requirements.txt` or using a slightly older Python version (3.12).

## Project Structure

```
enterococci-ML-training-lightgbm/
├── data/
│   ├── raw/                  # Original unprocessed data
│   ├── interim/              # Intermediate processed data
│   └── processed/            # Final model-ready datasets
│       ├── training_data.csv
│       ├── validation_data.csv
│       └── test_data.csv
├── models/                   # Saved trained models (.joblib files)
├── reports/                  # Evaluation results (JSON)
├── results/
│   ├── predictions.csv       # Model predictions output
│   ├── graphs_plots/         # Static visualisations
│   ├── interactive/          # Interactive HTML plots
│   └── visualizations/       # Additional visual outputs
├── scripts/                  # Standalone utility scripts
├── src/                      # Main source code
│   ├── config/               # YAML config + path constants
│   ├── data/                 # Data loading, preprocessing, feature engineering
│   ├── models/               # Model implementations
│   │   ├── benchmarks/       # LightGBM benchmark model
│   │   ├── matrix_decomp/    # NMF matrix decomposition framework
│   │   └── probabilistic/    # Quantile regression framework (production)
│   ├── evaluation/           # Metrics, cross-validation, evaluators
│   ├── visualization/        # Dashboards and plotting
│   ├── pipeline/             # Main pipeline orchestration
│   └── utils/                # Logging and helpers
├── docs/                     # Documentation (you are here)
├── requirements.txt
└── README.md
```

## Running the Pipeline

The main entry point is `src/pipeline/main_pipeline.py`. It accepts a `--mode` argument:

```bash
# Run the full pipeline (train + evaluate + predict)
python src/pipeline/main_pipeline.py --mode all

# Train models only
python src/pipeline/main_pipeline.py --mode train

# Evaluate models only (runs time-series cross-validation)
python src/pipeline/main_pipeline.py --mode evaluate

# Generate predictions and visualisations only
python src/pipeline/main_pipeline.py --mode predict
```

You can also point to a different config file:

```bash
python src/pipeline/main_pipeline.py --config config/main_config.yaml --mode all
```

## What Each Mode Does

| Mode | What happens | Key outputs |
|------|-------------|-------------|
| `train` | Loads data, preprocesses, engineers features, trains models | `models/*.joblib` |
| `evaluate` | Runs time-series cross-validation across 3 annual folds | `reports/evaluation_results.json` |
| `predict` | Generates predictions on test data, creates visualisations | `results/predictions.csv`, interactive HTML plots |
| `all` | Runs all three in sequence | All of the above |

## Expected Outputs

After a full `--mode all` run, you should see:

- **`models/`** — Serialised model files (e.g. `probabilistic_framework.joblib`)
- **`reports/evaluation_results.json`** — Per-fold and aggregate metrics (RMSE, sensitivity, specificity, etc.)
- **`results/predictions.csv`** — Predictions with quantile columns
- **`results/interactive/`** — Interactive Plotly HTML plots you can open in a browser
- A **Dash dashboard** will launch on `http://localhost:8514` during the predict phase

## Configuration

All pipeline behaviour is controlled by `src/config/main_config.yaml`. This includes:

- Data paths
- Feature engineering settings (lag windows)
- Model hyperparameters
- Which models to train and evaluate
- Cross-validation test periods
- Evaluation metrics

See [project-overview.md](project-overview.md) for more on how configuration drives the pipeline.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` for a package | Install it: `pip install <package-name>` and add it to `requirements.txt` |
| Build errors when installing packages | Check your Python version; try 3.12 if on 3.14 |
| `No module named 'src'` | Make sure you are running from the repo root directory |
| Pipeline crashes during predict | Ensure models have been trained first (`--mode train`) |
| Dash dashboard not loading | Check the port (default 8514) is not already in use |

## Next Steps

- Read [project-overview.md](project-overview.md) to understand the domain and architecture
- Read [data-and-features.md](data-and-features.md) to understand the data pipeline
- Read [model-architecture.md](model-architecture.md) for a deep dive into how the models work
