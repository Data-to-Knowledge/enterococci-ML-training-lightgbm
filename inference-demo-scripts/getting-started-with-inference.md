# Getting Started with the Inference Demo

Welcome to the Enterococci inference demo. This is your entry point.

The purpose of this folder is to let you run the full prediction pipeline locally -- from live weather and tide data through to model predictions -- in a self-contained environment with no connection to any production database.

Read this file first. Then follow the links at the bottom once you are up and running.

---

## What this folder does

The script `inference-demo.py` fetches live data from three external APIs, assembles a feature matrix for each of the 15 coastal monitoring sites around Lyttelton and Akaroa harbours, runs it through a trained LightGBM quantile model, and writes predictions to a local CSV file.

Every time you run it you get a fresh set of predictions for the current hour. If the CSV already exists, new rows are upserted (matched on site and datetime) rather than appended blindly.

---

## How this differs from the production pipeline

Here is what is stripped out compared to production:

| Production feature | Demo replacement |
|---|---|
| Azure SQL database writes (MSSQL upsert) | CSV file write with upsert logic |
| Azure SQL tide table read | LINZ static chart API (public, no auth) |
| Run ID generation (`uuid`) and orchestration | Not present |
| Structured event sourcing (`ev_*` emitters, KQL telemetry) | Not present |
| `Passwords.ini` credential file | Environment variables only via `.env` file |
| Container-level run bookkeeping | Not present |

The core data flow -- API fetches, feature engineering, model inference, SHAP values -- is identical to production. The model file itself (`trained_models_inference/probabilistic_framework.joblib`) is the same serialised model that serves live predictions via the LAWA API.

If you want to understand the full production architecture (run ID tracking, event sourcing, database schema) have a chat with Atman or Nina after you have run the demo a few times.

---

## Prerequisites

- Python 3.13 or later
- API credentials for MetService and NIWA (provided via 1Password -- see below)
- Internet access (the script calls external APIs in real time)

---

## Setup

All commands below should be run from inside the `inference-demo-scripts/` folder.

**Use `python -m pip` throughout.** Plain `pip` is blocked on ECan machines.

### 1. Create the virtual environment

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

### 3. Install dependencies

```bash
python -m pip install -r requirements-inference.txt
```

### 4. Set up your credentials

Copy the `inference-env.env` template and fill in your credentials:

```
METSERVICE_KEY=<your key here>
NIWA_USERNAME=<your username here>
NIWA_KEY=<your password here>
```

Your credentials will be sent to you via a secure 1Password note. **Open the note and copy the values immediately** -- 1Password self-destructs the note after you view it, so do not close it before you have copied everything.

---

## Environment hygiene -- important

These rules protect you and the team from hard-to-fix problems.

**Never commit your credentials or outputs to Git.** The following are already in `.gitignore` and must stay that way:

```
.venv-inference/          <-- your virtual environment
inference_outputs/        <-- prediction CSVs and log files
inference-env.env         <-- your credentials
```

**Set up your own venv locally.**

**Keep your credentials out of code.** If you are experimenting and need to test credentials, always use the `.env` file -- never hardcode them in a script, even temporarily.

---

## Running the script

With your venv activated and credentials in place:

```bash
python inference-demo.py
```

The script takes roughly 18--22 seconds to run. All external API calls run concurrently so the wall time is the slowest single call, not the sum of all calls. While it runs you will see log output in the terminal. After it completes:

- Predictions: `inference_outputs/inference_predictions.csv`
- Full log: `inference_outputs/inference.log`

The log is worth reading in full at least once. It shows every phase of the pipeline, which APIs responded successfully, what the health gate decided for each harbour, and how many rows were written.

---

## What to expect

**Lyttelton sites** run unless the MetService API is down. There are two stations with automatic fallback, so this is robust.

**Akaroa sites** require NIWA Mintaka credentials. If the credentials are wrong you will see a 401 warning in the log and Akaroa sites will be excluded by the health gate -- this is expected behaviour, not a script error.

If both harbours fail the health gate, the script exits cleanly with a log message and writes nothing to the CSV.

---

## Folder structure

```
inference-demo-scripts/
|
|-- inference-demo.py              Main script -- run this
|-- inference-demo-documentation.md   Full technical walkthrough
|-- getting-started-with-inference.md This file
|-- requirements-inference.txt     Dependencies for this folder only
|-- inference-env.env              Your credentials (gitignored, not committed)
|
|-- helpers/
|   |-- WeatherAPI_Functions.py       Weather and tide API integration
|   |-- inference_data_preparation.py Feature engineering, SHAP, Hilltop fetch
|   |-- WeatherAPI_Functions_GUIDE.md  Reference docs for WeatherAPI_Functions.py
|   |-- inference_data_preparation_GUIDE.md  Reference docs for inference_data_preparation.py
|   |-- helpers_README.md             Overview of the helpers directory
|
|-- src_inference/                 Model source code (renamed from 'src' in production)
|   |-- config/                    Paths, constants, config loader
|   |-- data/                      Preprocessing and feature engineering
|   |-- models/                    Probabilistic framework, benchmarks, matrix decomp
|   |-- utils/                     Logging utilities
|
|-- trained_models_inference/      Trained model files (not for editing)
|   |-- probabilistic_framework.joblib
|   |-- probabilistic_framework.txt
|   |-- reference_schema.pkl
|   |-- training_data/training_data.csv
|
|-- inference_outputs/             Generated at runtime (gitignored)
    |-- inference_predictions.csv
    |-- inference.log
```

---

## Suggested reading order

Once the script is running, work through the documentation in this order:

1. [inference-demo-documentation.md](inference-demo-documentation.md) -- Full technical walkthrough of what the script does, phase by phase
2. [helpers/helpers_README.md](helpers/helpers_README.md) -- Overview of the two helper modules
3. [helpers/WeatherAPI_Functions_GUIDE.md](helpers/WeatherAPI_Functions_GUIDE.md) -- How the weather and tide API calls work
4. [helpers/inference_data_preparation_GUIDE.md](helpers/inference_data_preparation_GUIDE.md) -- Feature engineering, SHAP, and Hilltop data
5. [../docs/model-architecture.md](../docs/model-architecture.md) -- How the LightGBM quantile ensemble model works
6. [../docs/data-and-features.md](../docs/data-and-features.md) -- The full feature set and what each variable represents
7. [../docs/project-overview.md](../docs/project-overview.md) -- Domain context and system architecture

The `docs/` folder covers the training pipeline. The inference demo and the training pipeline share the same feature engineering and model classes, so understanding one helps with the other.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
You forgot to activate your venv, or the venv does not have dependencies installed. Run `python -m pip install -r requirements-inference.txt` again.

**`401 Unauthorized` for NIWA**
Your `NIWA_USERNAME` or `NIWA_KEY` in `inference-env.env` are wrong or expired. Check with Nina or Atman.

**Akaroa sites excluded, Lyttelton predictions written**
This is the expected behaviour when NIWA credentials fail or the sensor returns suspect data. Check the log for the specific reason.

**`inference-env.env` is not being read**
Make sure the file is in the `inference-demo-scripts/` folder (same directory as `inference-demo.py`) and that the variable names match exactly (no quotes around values, no trailing spaces).

**LightGBM model format warnings on startup**
These are non-fatal warnings from a minor version mismatch between the LightGBM version that saved the model and the one installed locally. The model loads and runs correctly -- you can ignore them.

---

## Questions

If anything is unclear or not working, ask Atman or Nina on Teams. No question is too basic -- getting this running is the point.
