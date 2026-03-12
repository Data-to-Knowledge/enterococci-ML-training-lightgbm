# Inference Demo Scripts

A self-contained, read-only copy of the production inference pipeline for learning and experimentation. This folder runs the Enterococci water quality prediction model end-to-end -- fetching live weather, tides, and historical data, then generating predictions for all 15 coastal monitoring sites across Lyttelton and Akaroa harbours.

All database components have been removed. Predictions are written to a local CSV file instead of the production database.

---

## Prerequisites

- Python 3.13+ (tested on 3.14)
- API credentials for MetService and NIWA (see [Environment variables](#environment-variables))
- Internet access (the script calls external APIs in real time)

---

## Setup

### 1. Create the virtual environment

From within the `inference-demo-scripts/` folder:

```bash
python -m venv .venv-inference
```

### 2. Activate it

**Windows (PowerShell):**
```powershell
.venv-inference\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
.venv-inference\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv-inference/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements-inference.txt
```

### 4. Set up environment variables

Open `inference-env.env` and fill in the three API credentials (see [Environment variables](#environment-variables) below). The script loads this file automatically on startup via `python-dotenv`.

---

## Running the script

With the virtual environment activated:

```bash
python inference-demo.py
```

The script will:

1. Load the pre-trained probabilistic model from `trained_models_inference/`
2. Fetch live weather data from MetService (Lyttelton) and NIWA Mintaka (Akaroa)
3. Fetch tide predictions from LINZ static charts
4. Fetch historical Enterococci samples from ECan's Hilltop server
5. Engineer all features (rolling rainfall, wind vectors, tidal state, seasonal stats)
6. Run the model and compute SHAP values for interpretability
7. Write predictions to `inference_outputs/inference_predictions.csv`
8. Append a detailed log to `inference_outputs/inference.log`

A typical successful run takes 30--60 seconds, mostly spent waiting on API responses.

---

## Environment variables

The script reads credentials from `inference-env.env` (loaded first) or any `.env.local` / `.env` file in the working directory.

| Variable | Purpose | Source |
|---|---|---|
| `METSERVICE_KEY` | API key for MetService hourly weather (Lyttelton) | MetService developer portal |
| `NIWA_USERNAME` | Username for NIWA Mintaka 10-minute weather (Akaroa) | NIWA / ECan agreement |
| `NIWA_KEY` | Password for NIWA Mintaka | NIWA / ECan agreement |

If the NIWA credentials are missing or invalid, Akaroa sites will be excluded by health gating and only Lyttelton harbour predictions will be produced. MetService failures similarly exclude Lyttelton sites. The script will not crash -- it gates each harbour independently.

---

## Folder structure

```
inference-demo-scripts/
├── inference-demo.py                # Main entry point -- orchestrates everything
├── inference-env.env                # API credentials (gitignored)
├── requirements-inference.txt       # Python dependencies
├── .venv-inference/                 # Virtual environment (gitignored)
│
├── helpers/
│   ├── WeatherAPI_Functions.py      # Weather + tide API clients
│   └── inference_data_preparation.py  # Site metadata, Hilltop fetch, feature helpers, SHAP
│
├── src_inference/                   # Model and pipeline code (mirrored from src/)
│   ├── config/
│   │   ├── config_loader.py         # YAML config loader
│   │   ├── constants.py             # Site codes and holiday date ranges
│   │   ├── main_config.yaml         # Hyperparameters and pipeline settings
│   │   └── paths.py                 # All file paths (PROJECT_ROOT, model paths, output paths)
│   ├── data/
│   │   ├── data_loader.py           # DataLoader class
│   │   ├── feature_engineering.py   # FeatureEngineer (wind-shore interactions, temporal features)
│   │   └── preprocessing.py         # Preprocessor (date parsing, encoding, normalisation)
│   ├── models/
│   │   ├── model_factory.py         # Factory that instantiates models by name
│   │   ├── probabilistic/
│   │   │   ├── main_probabilistic.py    # Probabilistic framework (12 quantile LightGBM ensemble)
│   │   │   └── quantile_modeling.py     # Individual quantile model wrapper
│   │   ├── matrix_decomp/               # NMF spatial-temporal framework (not used for inference)
│   │   └── benchmarks/                  # Single LightGBM baseline (not used for inference)
│   └── utils/
│       └── logging.py               # Shared logger setup
│
├── trained_models_inference/        # Pre-trained model artefacts
│   ├── probabilistic_framework.joblib   # Serialised model (12 quantile LightGBM models)
│   ├── probabilistic_framework.txt      # LightGBM text format (used for feature names + categoricals)
│   ├── reference_schema.pkl             # Training DataFrame schema (column order + dtypes)
│   └── training_data/
│       └── training_data.csv            # Historical training data
│
└── inference_outputs/               # Generated at runtime (gitignored)
    ├── inference_predictions.csv     # Prediction output (upserted each run)
    └── inference.log                 # Timestamped log file (appended each run)
```

---

## How the script works

### Data flow

```
External APIs ──► Raw data ──► Feature engineering ──► Model prediction ──► CSV output
```

### Step-by-step walkthrough

#### 1. Environment and module setup

The script loads `inference-env.env` via `python-dotenv`, then sets up Python path aliasing so that `joblib` can unpickle the model. The `.joblib` file was serialised with `src.*` module paths, but this folder uses `src_inference.*` -- the aliasing block maps every `src.X` to `src_inference.X` in `sys.modules`.

#### 2. Timestamp initialisation

Three timestamps are computed: NZ Standard Time (fixed UTC+12), NZ Local Time (DST-aware), and UTC. All downstream logic uses these -- weather APIs expect UTC, site metadata uses local time, and the output CSV records local time.

#### 3. Tide data (LINZ)

Tide predictions are fetched from LINZ static tide charts for both Akaroa and Lyttelton harbours. The tide data provides three features per site:
- **Tidal state** -- rising, falling, high, or low
- **Hours to high tide** -- how far away the next high tide is
- **High tide height** -- predicted height in metres

These are computed by `WeatherAPI_Functions.add_tide_variables()`.

#### 4. Weather data

**Lyttelton** -- MetService hourly observations via their JSON API. The script fetches 3 days of data and extracts rainfall (mm), wind speed, and wind direction. Wind is decomposed into eastward (Ve) and northward (Vn) vector components.

**Akaroa** -- NIWA Mintaka 10-minute observations via their REST API (HTTP Basic Auth). The 10-minute data is aggregated to hourly: rainfall is summed, wind vectors are averaged. A "zeros-only guard" checks that the data contains genuine non-zero values in the last 12 hours -- if everything is exactly zero, it is treated as a sensor outage and the data is discarded.

#### 5. Rolling features

From the raw hourly weather, rolling accumulations are computed:
- **Rainfall**: 3h, 6h, 12h, 24h, 48h, 72h cumulative sums
- **Wind speed**: 3h, 6h, 12h rolling means of the vector magnitude

These capture antecedent conditions -- e.g., how much rain fell in the last 72 hours, which strongly influences bacterial runoff.

#### 6. Health gating

Before predictions are generated, each harbour is checked for data completeness. The required features are the rolling rainfall and wind columns (3H, 6H, 12H, wind_speed_3h, wind_speed_6h, wind_speed_12h). If any are missing or NaN for a harbour, that harbour's sites are excluded from the prediction batch. This prevents the model from producing misleading predictions when input data is unavailable.

The health policy is **per-harbour** by default -- so Lyttelton sites can still get predictions even if Akaroa weather is down, and vice versa.

#### 7. Site metadata and historical Enterococci

Each of the 15 sites has static metadata: soil type, shallowness, catchment slope, land cover, GPS coordinates, beach orientation, and flags for watercraft use, sewage discharge, and high-intensity agriculture.

Historical Enterococci samples are fetched from ECan's Hilltop server for the current bathing season (October to March). Two seasonal features are computed per site:
- **Site season average** -- rolling mean of the last 5 samples
- **Site historical exceedance rate** -- proportion of samples exceeding 280 MPN/100mL this season

#### 8. Final feature engineering

The `FeatureEngineer` class adds:
- **Wind-shore interaction** -- dot product of wind vector with beach orientation, capturing whether wind is onshore or offshore
- **Temporal features** -- hour of day, day of week, month, season, weekend flag, holiday flag

The inference DataFrame is then aligned to the training schema using `conform_features_to_training()`, which drops extra columns, adds any missing ones as NaN, and reorders to match exactly.

#### 9. Categorical feature alignment

LightGBM requires categorical features to have the exact same levels and encoding as during training. The script reads categorical metadata from the `.txt` model file and applies `pd.CategoricalDtype` with the training-time levels. This is handled by `get_cat_info_from_model_txt()` and `apply_model_categoricals()`.

#### 10. Model prediction

The probabilistic framework produces 12 quantile predictions per site (at quantiles 0.20 through 0.975). The point forecast is the median across these quantiles, landing around the 78th percentile -- a deliberately risk-averse estimate for public health.

#### 11. SHAP values

SHAP (SHapley Additive exPlanations) values are computed for each of the 12 quantile models, then the median SHAP value across quantiles is taken per feature. This tells you which features drove each prediction up or down. If SHAP computation fails for any reason, the script continues without it.

#### 12. CSV output (upsert)

Predictions are written to `inference_outputs/inference_predictions.csv`. On subsequent runs, the script performs an **upsert** -- existing rows with the same `(DateTime, SITE_NAME)` key are replaced, new rows are appended. This means you can run the script multiple times without creating duplicates.

Each row is stamped with a `generated_at` timestamp so you can see when predictions were produced.

---

## Key concepts

### Quantile regression

Unlike standard regression which predicts a single number, the model predicts a full distribution. Each of the 12 quantile models answers: "What value will Enterococci be below with probability tau?" For example, the q_0.90 model predicts the value that 90% of outcomes should fall below. This gives us uncertainty estimates and enables exceedance probability calculations.

### Exceedance threshold

The regulatory exceedance threshold is **280 MPN/100mL**. A precautionary threshold of **140 MPN/100mL** is also used. Predictions are constrained to a minimum of 5 and capped at 10,000.

### Two harbours, two weather sources

Lyttelton and Akaroa harbours have independent weather stations and are treated separately throughout the pipeline. A failure in one weather source does not affect the other harbour's predictions.

### Module aliasing for joblib

The trained model (`.joblib`) was serialised in the production repo where the package is called `src`. In this demo folder the package is renamed to `src_inference` to avoid clashing with the training pipeline. The aliasing block at the top of `inference-demo.py` maps all `src.*` module paths to `src_inference.*` so that `joblib.load()` can reconstruct the model objects.

---

## Outputs

### inference_predictions.csv

Each row is one site at one hour. Key columns:

| Column | Description |
|---|---|
| `DateTime` | Prediction timestamp (NZ local, hourly) |
| `SITE_NAME` | Beach name |
| `predictions` | Point forecast (MPN/100mL) |
| `q_0.20` -- `q_0.975` | Quantile predictions |
| `SHAP_*` | SHAP contribution of each feature |
| `generated_at` | When the prediction was produced |

### inference.log

Timestamped log of every run, including:
- Which APIs were called and how long they took
- Health gating decisions per harbour
- Number of historical records fetched
- Model prediction shape
- Any warnings or errors

Read this file to understand what happened during a run and to learn how logging is structured in production Python applications.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Only Lyttelton sites in output | NIWA credentials invalid or expired | Check `NIWA_USERNAME` and `NIWA_KEY` in `inference-env.env` |
| Only Akaroa sites in output | MetService key invalid | Check `METSERVICE_KEY` in `inference-env.env` |
| No predictions at all | Both weather sources down or credentials wrong | Check the log file for health gating messages |
| `ModuleNotFoundError: No module named 'src'` | Module aliasing failed | Make sure you are running from within the `inference-demo-scripts/` directory |
| LightGBM "Model format error" warnings | Minor version mismatch between model training and current LightGBM | Safe to ignore -- the model still loads and predicts correctly |
| `ImportError: mapie` | MAPIE version mismatch | Ensure `mapie >= 1.3.0` is installed; the import is `from mapie.regression import ConformalizedQuantileRegressor` |
| Empty Hilltop data | ECan server down or outside bathing season (Apr--Sep) | Check the log; seasonal features will be NaN but predictions still run |

---

## What this folder does NOT include

- **Database writes** -- the production pipeline writes to Azure SQL via MSSQL upserts; this demo writes to a local CSV
- **Run ID tracking** -- production assigns a UUID to each inference run for traceability
- **Structured telemetry** -- production emits JSON event objects for monitoring dashboards
- **Passwords.ini** -- production historically read credentials from a config file; this demo uses environment variables only

These were removed to keep the demo simple and self-contained. If you are curious about any of them, ask and we can walk through the production code.
