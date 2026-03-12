# --- Load .env BEFORE any imports use env vars ---
try:
    from dotenv import load_dotenv, find_dotenv
    for env_name in ("inference-env.env", ".env.local", ".env"):
        p = find_dotenv(env_name, usecwd=True)
        if p:
            load_dotenv(p, override=False)
except Exception:
    pass


# --- Imports ---
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Alias 'src' -> 'src_inference' so joblib can unpickle the model.
# The .joblib file was serialised in the production repo where the package was named 'src'.
# This folder renames it 'src_inference' to avoid clashes. Before calling joblib.load(),
# we register every 'src.X' alias in sys.modules pointing at the corresponding
# 'src_inference.X' module. If the submodule is already imported we reuse it directly;
# otherwise we import it on-demand. ImportError is silenced for optional submodules
# (e.g. matrix_decomp) that may not be present in a trimmed deployment.
import src_inference
sys.modules["src"] = src_inference
for attr in ("models", "models.probabilistic", "models.probabilistic.main_probabilistic",
             "models.probabilistic.quantile_modeling", "models.matrix_decomp",
             "models.matrix_decomp.main_matrix", "models.matrix_decomp.matrix_decomposition_model",
             "models.benchmarks", "models.benchmarks.lightgbm_models",
             "data", "data.feature_engineering", "data.preprocessing",
             "config", "config.constants", "config.paths", "utils", "utils.logging"):
    full = f"src_inference.{attr}"   # real module path in this repo
    alias = f"src.{attr}"            # legacy path baked into the .joblib pickle
    if full in sys.modules:
        sys.modules[alias] = sys.modules[full]
    else:
        try:
            mod = __import__(full, fromlist=[attr.split(".")[-1]])
            sys.modules[alias] = mod
        except ImportError:
            pass  # optional submodule not present -- skip silently

import pandas as pd
import datetime as dt
import numpy as np
from helpers import WeatherAPI_Functions as f
import time
import logging
import joblib
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor
from src_inference.data.feature_engineering import FeatureEngineer
from helpers.inference_data_preparation import *
from src_inference.config.constants import SITE_CODES as site_codes
from src_inference.config.paths import *
from pathlib import Path


# --- Logger setup (console + log file) ---
logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    # Console output
    console_h = logging.StreamHandler(sys.stdout)
    console_h.setFormatter(fmt)
    console_h.setLevel(logging.INFO)
    logger.addHandler(console_h)

    # Log file (appends each run)
    log_dir = Path(os.path.dirname(__file__)) / "inference_outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_h = logging.FileHandler(log_dir / "inference.log", mode="a", encoding="utf-8")
    file_h.setFormatter(fmt)
    file_h.setLevel(logging.INFO)
    logger.addHandler(file_h)

logging.getLogger().setLevel(logging.WARNING)


# --- Time setup ---
timestamps = get_current_datetimes()
DateTime_NZST = timestamps["DateTime_NZST"]
DateTime_NZLT = timestamps["DateTime_NZLT"]
DateTime_UTC  = timestamps["DateTime_UTC"]
DateTime_UTC_H = DateTime_UTC.replace(minute=0, second=0, microsecond=0)

NZ_LOCAL = DateTime_NZLT
if getattr(NZ_LOCAL, "tzinfo", None) is not None:
    NZ_LOCAL = pd.Timestamp(NZ_LOCAL).tz_localize(None)

if getattr(DateTime_UTC, 'tzinfo', None) is not None:
    DateTime_UTC = pd.Timestamp(DateTime_UTC).tz_localize(None)
DateTime_UTC_H = DateTime_UTC.replace(minute=0, second=0, microsecond=0)
TARGET_LOCAL_H = pd.Timestamp(NZ_LOCAL).tz_localize(None).floor("h")

logger.info("Script started | NZ_LOCAL=%s | UTC=%s", NZ_LOCAL, DateTime_UTC)


# --- Health gating config ---
HEALTH_REQUIRED_FEATURES = [
    s.strip() for s in os.getenv(
        "HEALTH_REQUIRED_FEATURES",
        "3H,6H,12H,wind_speed_3h,wind_speed_6h,wind_speed_12h"
    ).split(",") if s.strip()
]
HEALTH_STRICT = os.getenv("HEALTH_STRICT", "0") == "1"
HEALTH_POLICY = os.getenv("HEALTH_POLICY", "per_harbour").strip().lower()


def _harbour_health_ok(df_at_hour: pd.DataFrame, *, required: list[str], strict: bool) -> tuple[bool, list[str]]:
    """
    Check health for a harbour's target-hour row AFTER rolling features,
    BEFORE we drop Ve/Vn/DateTime. Tides are not considered.
    Returns (ok, missing_features).
    """
    if df_at_hour is None or df_at_hour.empty:
        return False, (required + (["Rainfall", "Ve", "Vn"] if strict else []))

    row = df_at_hour.iloc[0]
    missing = []
    for feat in required:
        if feat not in row or pd.isna(row[feat]):
            missing.append(feat)
    if strict:
        for base in ("Rainfall", "Ve", "Vn"):
            if base not in row or pd.isna(row[base]):
                missing.append(base)

    ordered, seen = [], set()
    for x in (required + (["Rainfall","Ve","Vn"] if strict else [])):
        if x in missing and x not in seen:
            seen.add(x); ordered.append(x)
    return (len(ordered) == 0, ordered)


# --- MetService station config ---
station_candidates = [s.strip() for s in os.getenv("METSERVICE_STATIONS", "93786,93951").split(",") if s.strip()]


# --- Paths & Config ---
reference_df = pd.read_pickle(REFERENCE_SCHEMA_PATH)
site_data = create_site_metadata(NZ_LOCAL)
logger.info("Loaded reference schema and site metadata (%d sites).", len(site_data))

# Pre-compute the Hilltop season window here so it is available for the concurrent fetch below.
# It depends only on NZ_LOCAL which is already set.
_season_start_year = NZ_LOCAL.year if NZ_LOCAL.month >= 10 else NZ_LOCAL.year - 1
_hilltop_from = f"{_season_start_year}-10-01"
_hilltop_to   = NZ_LOCAL.strftime("%Y-%m-%d")

site_list = ['Akaroa', 'Lyttelton']


def _fetch_lyt_day(i):
    """Fetch one 24-hour window of Lyttelton MetService data (used in thread pool)."""
    single_date = DateTime_UTC + dt.timedelta(days=-i)
    try:
        return f.get_hourly_weather_data_Lyttelton(
            datetime=single_date,
            station_candidates=tuple(station_candidates),
            logger=logger
        )
    except Exception as e:
        logger.warning("MetService fetch raised on %s: %s", single_date.date(), e)
        return None


# Run all six I/O-bound fetches concurrently.
# Every fetch is independent and network-bound, so threads give near-linear
# speedup without requiring an async rewrite of the requests-based HTTP layer.
# Expected wall time: ~10-15s instead of ~40-70s sequential.
logger.info(
    "Fetching all data concurrently: MetService (3 days), LINZ tides, NIWA Akaroa, Hilltop..."
)
_t_fetch = time.perf_counter()

with ThreadPoolExecutor(max_workers=6) as _pool:
    _fut_tides   = _pool.submit(f.get_tide_data, site_list=site_list, year=False)
    _fut_lyt     = [_pool.submit(_fetch_lyt_day, i) for i in range(3)]
    _fut_akaroa  = _pool.submit(f.get_10min_weather_data_Akaroa, DateTime_UTC, 3, logger=logger)
    _fut_hilltop = _pool.submit(
        fetch_enterococci_data_for_sites,
        site_codes=site_codes,
        from_date=_hilltop_from,
        to_date=_hilltop_to,
        logger=None,
        save_path=None,
        timeout_s=10,
        max_retries=2,
    )
    # Collect results -- blocks here until every thread is done
    _tides_raw              = _fut_tides.result()
    _lyt_results            = [fut.result() for fut in _fut_lyt]
    lyttelton_weather_parts = [r for r in _lyt_results if r is not None]
    df_akaroa_10min_weather = _fut_akaroa.result()
    hist_df                 = _fut_hilltop.result()

logger.info("All fetches complete in %.1fs.", time.perf_counter() - _t_fetch)


# --- Process tides ---
tides_df = _tides_raw
if tides_df is None or tides_df.empty:
    tides_df = pd.DataFrame(columns=["DateTime", "Tidal_height", "Tidal_state", "Harbour"])
    logger.error("LINZ tides returned empty DataFrame; tide features will be NaN.")
else:
    logger.info("Tides fetched from LINZ (%d rows).", len(tides_df))
    try:
        tides_df["DateTime"]    = pd.to_datetime(tides_df["DateTime"]).dt.tz_localize(None)
        tides_df["Harbour"]     = tides_df["Harbour"].astype(str).str.strip()
        tides_df["Tidal_state"] = tides_df["Tidal_state"].astype(str).str.strip().str.casefold()
    except Exception as e:
        logger.warning("Failed to normalise tides_df: %s", e)


# --- Process Lyttelton weather ---
df_lyttelton_weather = (
    pd.concat(lyttelton_weather_parts, ignore_index=True)
    if lyttelton_weather_parts else
    pd.DataFrame(columns=["DateTime", "Rainfall", "wind_direction", "wind_speed", "Ve", "Vn", "StationId"])
)
df_lyttelton_weather = df_lyttelton_weather.drop(
    ['wind_direction', 'wind_speed', 'StationId'], axis=1, errors='ignore')
if not df_lyttelton_weather.empty:
    df_lyttelton_weather["DateTime"] = (
        pd.to_datetime(df_lyttelton_weather["DateTime"], errors="coerce").dt.tz_localize(None)
    )
else:
    logger.warning("No Lyttelton weather returned from MetService; using NaNs for this hour.")
    df_lyttelton_weather = pd.DataFrame(
        [{"DateTime": DateTime_UTC, "Rainfall": np.nan, "Ve": np.nan, "Vn": np.nan}]
    )


# --- Process Akaroa weather ---
if df_akaroa_10min_weather is None or df_akaroa_10min_weather.empty:
    df_akaroa_10min_weather = pd.DataFrame(columns=["DateTime", "Rainfall", "Ve", "Vn"])
    logger.warning("NIWA Akaroa weather returned empty.")

# Floor timestamps to the hour, then aggregate 10-min observations to hourly totals/means
df_akaroa_10min_weather["DateTime"] = (
    pd.to_datetime(df_akaroa_10min_weather["DateTime"], errors="coerce")
      .dt.tz_localize(None)
      .dt.floor("h")
)
df_akaroa_10min_weather = df_akaroa_10min_weather.drop(
    ['wind_direction', 'wind_speed'], axis=1, errors='ignore')
df_akaroa_weather = df_akaroa_10min_weather.groupby('DateTime', as_index=False).agg(
    Rainfall=('Rainfall', 'sum'),   # sum gives hourly accumulation
    Ve=('Ve', 'mean'),
    Vn=('Vn', 'mean')
)
df_akaroa_weather["DateTime"] = (
    pd.to_datetime(df_akaroa_weather["DateTime"], errors="coerce").dt.tz_localize(None)
)


# --- ZERO-ONLY GUARD for Akaroa ---
def _zeros_only_guard(df_hourly: pd.DataFrame, now_utc: pd.Timestamp, lookback_h: int = 12) -> bool:
    """Detect a likely NIWA sensor outage by checking for all-zero data.

    A genuine calm day (no rain, no wind) is possible but rare. When every
    Rainfall, Ve, and Vn value in the most recent window is exactly zero, it
    is more likely that the sensor is reporting a flat-line rather than real
    observations. In that case we treat the data as missing so that the health
    gate excludes Akaroa rather than feeding the model suspect inputs.

    Parameters
    ----------
    df_hourly : pd.DataFrame
        Hourly aggregated Akaroa weather data with DateTime, Rainfall, Ve, Vn columns.
    now_utc : pd.Timestamp
        Current UTC timestamp (tz-naive).
    lookback_h : int
        Number of recent hours to inspect. Default 12, overridable via NIWA_ZERO_GUARD_H.

    Returns
    -------
    bool
        True if all values in the window are zero or NaN (guard triggered), False otherwise.
    """
    if df_hourly is None or df_hourly.empty:
        return True
    start_ts = now_utc - dt.timedelta(hours=lookback_h)
    window = df_hourly[df_hourly["DateTime"] > start_ts]
    if window.empty:
        return True
    for col in ("Rainfall", "Ve", "Vn"):
        s = pd.to_numeric(window[col], errors="coerce")
        if s.notna().any() and (s.fillna(0).abs() > 0).any():
            return False
    return True

NIWA_ZERO_GUARD_H = int(os.getenv("NIWA_ZERO_GUARD_H", "12"))
if _zeros_only_guard(df_akaroa_weather, DateTime_UTC, NIWA_ZERO_GUARD_H):
    logger.warning("Akaroa weather failed zeros-only guard; using NaNs.")
    df_akaroa_weather = pd.DataFrame([{"DateTime": DateTime_UTC, "Rainfall": np.nan, "Ve": np.nan, "Vn": np.nan}])


# --- Rolling features for Rainfall/Wind ---
logger.info("Calculating rolling features for rainfall and wind...")

def _ensure_min_weather_frame(df, label):
    """Guarantee that a weather DataFrame has the minimum columns needed for rolling feature computation.

    If the DataFrame is None or empty (e.g. the API returned nothing), a
    single-row placeholder is returned with NaN values. This ensures that
    downstream rolling operations and the health gate can always run without
    attribute errors, and that missing data results in NaN features rather than
    a script crash.

    Parameters
    ----------
    df : pd.DataFrame or None
        Weather DataFrame to validate. Expected columns: DateTime, Rainfall, Ve, Vn.
    label : str
        Human-readable harbour name used in log messages ("Akaroa" or "Lyttelton").

    Returns
    -------
    pd.DataFrame
        The input DataFrame with any missing required columns added as NaN,
        or a single-row NaN placeholder if the input was empty.
    """
    need = ["DateTime", "Rainfall", "Ve", "Vn"]
    if df is None or df.empty:
        logger.warning(f"No {label} weather available pre-rolling; using NaNs for this hour.")
        return pd.DataFrame([{"DateTime": DateTime_UTC, "Rainfall": np.nan, "Ve": np.nan, "Vn": np.nan}])
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce").dt.tz_localize(None)
    return df

df_akaroa_weather    = _ensure_min_weather_frame(df_akaroa_weather, "Akaroa")
df_lyttelton_weather = _ensure_min_weather_frame(df_lyttelton_weather, "Lyttelton")

df_akaroa_weather    = df_akaroa_weather.sort_values(by="DateTime", ascending=True)
df_lyttelton_weather = df_lyttelton_weather.sort_values(by="DateTime", ascending=True)

df_akaroa_weather    = f.add_rainfall_variables(df_akaroa_weather)
df_lyttelton_weather = f.add_rainfall_variables(df_lyttelton_weather)
for i in [3, 6, 12]:
    df_akaroa_weather    = f.add_wind_variables(df_akaroa_weather, i)
    df_lyttelton_weather = f.add_wind_variables(df_lyttelton_weather, i)


# --- Select the target hour rows ---
akaroa_upto = df_akaroa_weather[df_akaroa_weather["DateTime"] <= DateTime_UTC]
if not akaroa_upto.empty:
    akaroa_at_hour = akaroa_upto.sort_values("DateTime", ascending=False).head(1).copy()
else:
    logger.warning(f"No Akaroa weather record found before or at {DateTime_UTC}; using NaNs for this hour.")
    akaroa_at_hour = pd.DataFrame([{"DateTime": DateTime_UTC, "Rainfall": np.nan, "Ve": np.nan, "Vn": np.nan}])

lyttelton_upto = df_lyttelton_weather[df_lyttelton_weather["DateTime"] <= DateTime_UTC]
if not lyttelton_upto.empty:
    lyttelton_at_hour = lyttelton_upto.sort_values("DateTime", ascending=False).head(1).copy()
else:
    logger.warning(f"No Lyttelton weather record found before or at {DateTime_UTC}; using NaNs for this hour.")
    lyttelton_at_hour = pd.DataFrame([{"DateTime": DateTime_UTC, "Rainfall": np.nan, "Ve": np.nan, "Vn": np.nan}])


# --- Health gate per harbour ---
ak_ok, ak_missing = _harbour_health_ok(
    akaroa_at_hour, required=HEALTH_REQUIRED_FEATURES, strict=HEALTH_STRICT
)
lt_ok, lt_missing = _harbour_health_ok(
    lyttelton_at_hour, required=HEALTH_REQUIRED_FEATURES, strict=HEALTH_STRICT
)

logger.info("Health check: Lyttelton=%s (missing: %s), Akaroa=%s (missing: %s)",
            lt_ok, lt_missing or "none", ak_ok, ak_missing or "none")

EARLY_SKIP = False
harbours_to_write: list[str] = []
if HEALTH_POLICY == "per_harbour":
    if lt_ok: harbours_to_write.append("Lyttelton")
    if ak_ok: harbours_to_write.append("Akaroa")
    if not harbours_to_write:
        EARLY_SKIP = True
else:
    if lt_ok and ak_ok:
        harbours_to_write = ["Lyttelton", "Akaroa"]
    else:
        EARLY_SKIP = True

if EARLY_SKIP:
    logger.warning("Health check failed for all harbours; skipping predictions.")


# --- Add tide features ---
_ak_cols = ["tidal_state", "hours_to_high_tide", "high_tide_height",
            "high_tide_height_X_hours_to_high_tide"]
ak_vals = f.add_tide_variables(TARGET_LOCAL_H, "Akaroa", tides_df)
akaroa_at_hour.loc[:, _ak_cols] = [list(ak_vals)]

_lt_cols = ["tidal_state", "hours_to_high_tide", "high_tide_height",
            "high_tide_height_X_hours_to_high_tide"]
lt_vals = f.add_tide_variables(TARGET_LOCAL_H, "Lyttelton", tides_df)
lyttelton_at_hour.loc[:, _lt_cols] = [list(lt_vals)]


# --- Final shape for merge ---
def _prep_row(df_row: pd.DataFrame, harbour_name: str) -> pd.DataFrame:
    """Prepare a single-hour weather row for merging with site metadata.

    Drops intermediate columns that were needed for rolling computations but
    are not model inputs (Ve, Vn, DateTime), then stamps the Harbour name so
    the row can be joined onto the site metadata table.

    Parameters
    ----------
    df_row : pd.DataFrame
        Single-row DataFrame containing the target hour's weather and rolling features.
    harbour_name : str
        Harbour label to attach ("Akaroa" or "Lyttelton").

    Returns
    -------
    pd.DataFrame
        Cleaned single-row DataFrame ready for merging, or an empty DataFrame
        if the input was None or empty.
    """
    if df_row is None or df_row.empty:
        return pd.DataFrame(columns=["Harbour"])
    out = df_row.copy()
    out = out.drop(['Ve','Vn','DateTime'], axis=1, errors='ignore')
    out['Harbour'] = harbour_name
    return out

akaroa_prepped    = _prep_row(akaroa_at_hour, "Akaroa")       if "Akaroa" in harbours_to_write else pd.DataFrame(columns=["Harbour"])
lyttelton_prepped = _prep_row(lyttelton_at_hour, "Lyttelton") if "Lyttelton" in harbours_to_write else pd.DataFrame(columns=["Harbour"])

df_weather = pd.concat([lyttelton_prepped, akaroa_prepped], ignore_index=True)


# --- Merge With Site Metadata ---
if EARLY_SKIP:
    nowcast = pd.DataFrame(columns=["SITE_NAME", "DateTime"])
    output_df = pd.DataFrame(columns=["DateTime","SITE_NAME"])
else:
    site_data_filtered = site_data[site_data["Harbour"].isin(harbours_to_write)].copy()
    nowcast = pd.merge(site_data_filtered, df_weather, on='Harbour', how='left')

    if set(harbours_to_write) == {"Lyttelton", "Akaroa"}:
        all_sites = [
            'Akaroa at main beach', 'Cass Bay at boat ramp', 'Charteris Bay Paradise Beach',
            'Church Bay Beach', 'Corsair Bay Beach', 'Diamond Harbour Beach',
            'Duvauchelle Bay by camping ground', 'French Farm Bay', 'Glen Bay boat ramp',
            'Governors Bay Sandy Beach', 'Purau Bay Beach', 'Rapaki Bay Beach',
            'Takamatua Bay Boat ramp', 'Tikao Bay mid beach', 'Wainui Beach Wainui Beach'
        ]
        site_hour_df = pd.DataFrame({
            'SITE_NAME': all_sites,
            'DateTime': [NZ_LOCAL.replace(tzinfo=None)] * len(all_sites)
        })
        nowcast = pd.merge(site_hour_df, nowcast, on=['SITE_NAME','DateTime'], how='left', sort=False)
        if nowcast.shape[0] != 15:
            logger.warning(f"Expected 15 sites in nowcast but got {nowcast.shape[0]}")
        else:
            logger.info("All 15 sites present in nowcast after full merge.")

    nowcast['DateTime'] = pd.to_datetime(nowcast['DateTime'], errors='coerce').dt.tz_localize(None)

    # --- Historical Enterococci (Hilltop) ---
    # hist_df was already fetched concurrently with the weather data above.
    logger.info("Using Hilltop data for season %s to %s.", _hilltop_from, _hilltop_to)
    if not hist_df.empty:
        hist_df['DateTime'] = pd.to_datetime(hist_df['DateTime'], errors='coerce').dt.tz_localize(None)
        logger.info("Fetched %d records from Hilltop.", len(hist_df))
    else:
        logger.warning("Hilltop fetch returned empty dataframe.")

    nowcast = add_seasonal_features(nowcast, hist_df, NZ_LOCAL, logger=None)

    # --- Final Feature Engineering ---
    feature_engineer = FeatureEngineer()
    nowcast = feature_engineer.wind_shoretype_feature(nowcast)
    nowcast = feature_engineer.temporal_features(nowcast)

    INFERENCE_DATA = nowcast.copy()
    logger.info("Data ready: %d rows, %d columns.", INFERENCE_DATA.shape[0], INFERENCE_DATA.shape[1])

    # Save the real DateTime for output BEFORE any column drops
    TRUE_DATETIME = INFERENCE_DATA["DateTime"].copy()

    # Remove extra columns not in training
    INFERENCE_DATA = conform_features_to_training(INFERENCE_DATA, reference_df, logger=None)

    # --- Load model metadata and set categorical features ---
    cat_indices, cat_levels = get_cat_info_from_model_txt(MODEL_TXT_PATH)
    booster = lgb.Booster(model_file=MODEL_TXT_PATH)
    feature_names = booster.feature_name()

    INFERENCE_DATA = apply_model_categoricals(
        df=INFERENCE_DATA,
        feature_names=feature_names,
        cat_indices=cat_indices,
        cat_levels=cat_levels,
        logger=None
    )

    # Force dummy DateTime for LGBM/SHAP compatibility
    if "DateTime" in INFERENCE_DATA.columns:
        INFERENCE_DATA["DateTime"] = 0
        INFERENCE_DATA["DateTime"] = INFERENCE_DATA["DateTime"].astype("int64")

    # --- Load model & generate predictions ---
    logger.info("Generating predictions using the Probabilistic Forecasting Model.")

    model_data = joblib.load(TRAINED_MODEL_PATH)
    pred_df = model_data.predict(INFERENCE_DATA)
    logger.info(f"Model predictions completed. Output shape: {pred_df.shape}")

    lower_bounds = pred_df.get("q_0.2", pred_df.iloc[:, 0])
    upper_bounds = pred_df.get("q_0.975", pred_df.iloc[:, -2])
    point_forecast = pred_df.get("predictions", pred_df.iloc[:, -1])

    # --- SHAP Values ---
    try:
        logger.info("Computing SHAP values...")

        shap_input = prepare_shap_input(
            df=INFERENCE_DATA,
            feature_names=feature_names,
            cat_indices=cat_indices,
            cat_levels=cat_levels,
            reference_df=reference_df,
            logger=logger
        )

        quantiles = [0.20, 0.35, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975]
        shap_dfs = []
        base_values = []

        for q in quantiles:
            rf_model = model_data.quantile_ensemble.models[q]
            shap_df_q, base_val = compute_shap_df(rf_model, shap_input, logger=None)
            shap_dfs.append(shap_df_q)
            base_values.append(base_val)

        shap_stack = np.stack([df.values for df in shap_dfs], axis=2)
        median_shap_values = np.median(shap_stack, axis=2)

        shap_df = pd.DataFrame(median_shap_values, columns=shap_dfs[0].columns)
        shap_df["SHAP_base_value"] = np.median(base_values)

        export_df = INFERENCE_DATA.copy()
        export_df["DateTime"] = TRUE_DATETIME.astype(str)

        output_df = pd.concat([
            export_df.reset_index(drop=True),
            pred_df.reset_index(drop=True),
            shap_df.reset_index(drop=True)
        ], axis=1)

        logger.info("SHAP values computed successfully.")

    except Exception as e:
        logger.warning(f"SHAP calculation skipped or failed: {e}", exc_info=True)

        export_df = INFERENCE_DATA.copy()
        export_df["DateTime"] = TRUE_DATETIME.astype(str)
        output_df = pd.concat([
            export_df.reset_index(drop=True),
            pred_df.reset_index(drop=True)
        ], axis=1)


# ---------- CSV write (upsert: merge new predictions into existing CSV) ----------
KEYS = ["DateTime", "SITE_NAME"]

# Normalize keys
if "DateTime" in output_df.columns:
    output_df["DateTime"] = (
        pd.to_datetime(output_df["DateTime"], errors="coerce")
          .dt.tz_localize(None)
          .dt.floor("h")
    )
output_df = output_df.dropna(subset=[k for k in KEYS if k in output_df.columns])

if output_df.duplicated(subset=KEYS, keep=False).any():
    logger.warning("Batch contains duplicate (DateTime, SITE_NAME); keeping last per key.")
    output_df = (
        output_df.sort_values(KEYS)
                 .drop_duplicates(subset=KEYS, keep="last")
    )

# Stamp when this row was generated
output_df["generated_at"] = pd.Timestamp.now().floor("s")

# Ensure the output directory exists
csv_path = PREDICTION_STORAGE_PATH
csv_path.parent.mkdir(parents=True, exist_ok=True)

if not output_df.empty:
    if csv_path.exists():
        # Upsert: load existing, update matching keys, append new
        existing_df = pd.read_csv(csv_path, parse_dates=["DateTime"])
        existing_df["DateTime"] = pd.to_datetime(existing_df["DateTime"], errors="coerce").dt.tz_localize(None).dt.floor("h")

        # Align columns (existing may have fewer/more columns than new)
        all_cols = list(dict.fromkeys(list(existing_df.columns) + list(output_df.columns)))
        existing_df = existing_df.reindex(columns=all_cols)
        output_df = output_df.reindex(columns=all_cols)

        # Remove existing rows that match new keys (upsert = delete + insert)
        merge_keys = existing_df.set_index(KEYS).index
        new_keys = output_df.set_index(KEYS).index
        keep_mask = ~merge_keys.isin(new_keys)
        merged_df = pd.concat([existing_df[keep_mask], output_df], ignore_index=True)

        merged_df = merged_df.sort_values(KEYS)
        merged_df.to_csv(csv_path, index=False)
        logger.info(
            "Upserted %d rows into %s (total rows now: %d).",
            len(output_df), csv_path.name, len(merged_df)
        )
    else:
        # First run: just write
        output_df.to_csv(csv_path, index=False)
        logger.info("Wrote %d rows to new file %s.", len(output_df), csv_path.name)
else:
    reason = "health_check_failed" if EARLY_SKIP else "empty_batch"
    logger.info("No rows to write (%s).", reason)

logger.info("Script complete.")
