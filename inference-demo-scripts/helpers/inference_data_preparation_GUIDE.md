# inference_data_preparation.py -- Function Reference

**File:** `helpers/inference_data_preparation.py`

Data preparation, feature engineering, Hilltop API integration, SHAP computation, and schema alignment for the inference pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Execution Flow](#execution-flow)
3. [Timezone and Datetime Functions](#timezone-and-datetime-functions)
4. [Site Metadata](#site-metadata)
5. [Historical Enterococci Fetch](#historical-enterococci-fetch)
6. [Seasonal Feature Engineering](#seasonal-feature-engineering)
7. [Schema Conformance](#schema-conformance)
8. [Categorical Encoding](#categorical-encoding)
9. [SHAP Preparation and Computation](#shap-preparation-and-computation)
10. [Exceedance Probability](#exceedance-probability)
11. [Common Errors and Solutions](#common-errors-and-solutions)

---

## Overview

This module contains all data preparation and feature engineering functions used by `inference-demo.py` during the prediction pipeline. Functions are called in a specific sequence to:

1. Generate current timestamps in multiple timezones
2. Create site metadata with temporal features
3. Fetch historical Enterococci measurements from Hilltop
4. Compute seasonal features (rolling averages, exceedance rates)
5. Align inference data to training schema
6. Apply categorical encodings from the trained model
7. Prepare SHAP-compatible input for explainability
8. Compute SHAP values for all quantile models

**Critical design principle:** All functions maintain tz-naive datetime handling to ensure compatibility with LightGBM models.

---

## Execution Flow

The functions in this module are called in the following sequence during `inference-demo.py` execution:

```
PHASE 1: INITIALISATION
  get_current_datetimes()        --> timestamps in UTC, NZST, NZLT
  create_site_metadata()         --> 15-row DataFrame with site characteristics

PHASE 2: HISTORICAL DATA
  fetch_enterococci_data_for_sites()  --> current-season samples from Hilltop

PHASE 3: SEASONAL FEATURES
  add_seasonal_features()        --> rolling average + exceedance rate per site

PHASE 4: SCHEMA ALIGNMENT
  conform_features_to_training() --> match columns to training schema
  get_cat_info_from_model_txt()  --> extract categorical metadata from .txt
  apply_model_categoricals()     --> set CategoricalDtype on columns

PHASE 5: SHAP
  prepare_shap_input()           --> dtype validation for TreeExplainer
  compute_shap_df()              --> SHAP values per quantile model (x12)
```

---

## Timezone and Datetime Functions

### `get_named_timezones()`

Returns a dictionary of pytz timezone objects.

```python
def get_named_timezones() -> dict
```

**Returns:**
```python
{
    "NZST": pytz.timezone('Etc/GMT-12'),      # Fixed UTC+12 (no DST)
    "NZLT": pytz.timezone('Pacific/Auckland'), # UTC+12/13 (DST-aware)
    "UTC": pytz.timezone('UTC')                # UTC+0
}
```

**Why two NZ timezones?**
- `NZST` ensures consistency with training data (fixed offset)
- `NZLT` aligns with actual NZ civil time for seasonal logic and holiday checks

---

### `get_current_datetimes()`

Get the current time in multiple timezones, floored to the hour.

```python
def get_current_datetimes() -> dict
```

**Returns:**
```python
{
    "DateTime_UTC":  datetime,  # e.g. 2026-03-12 01:00:00
    "DateTime_NZST": datetime,  # e.g. 2026-03-12 13:00:00 (fixed UTC+12)
    "DateTime_NZLT": datetime,  # e.g. 2026-03-12 14:00:00 (DST-aware)
}
```

All returned datetimes are **tz-aware** (they carry timezone info from pytz). The caller in `inference-demo.py` strips timezone info to make them tz-naive before downstream use, because LightGBM does not support `datetime64[ns, tz]` dtypes.

---

### `get_current_season()`

Determine the bathing season label based on a date. Seasons run October to September.

```python
def get_current_season(today: pd.Timestamp) -> str
```

**Examples:**
```python
get_current_season(pd.Timestamp("2026-03-12"))  # --> "2025-2026"
get_current_season(pd.Timestamp("2025-10-01"))  # --> "2025-2026" (season starts)
get_current_season(pd.Timestamp("2025-09-30"))  # --> "2024-2025" (last day of prev season)
```

---

## Site Metadata

### `create_site_metadata()`

Create a 15-row DataFrame with site characteristics and temporal features for the target prediction hour.

```python
def create_site_metadata(datetime_ts: pd.Timestamp) -> pd.DataFrame
```

**Parameters:**
- `datetime_ts` -- Reference datetime for temporal features (typically NZ local time)

**Returns:** DataFrame with 15 rows (one per site) containing:

**Static columns (never change):**

| Column | Type | Description |
|---|---|---|
| `SITE_NAME` | str | Full site name (e.g. "Akaroa at main beach") |
| `Harbour` | str | "Akaroa" or "Lyttelton" |
| `Shallowness` | str | "shallow" / "medium" / "deep" |
| `Soil_type` | str | "sandy" / "muddy" / "stony" |
| `Catchment_slope` | int | Slope rating 0--10 |
| `Landcover_catchment` | str | "urban" / "grass" / "forest" |
| `watercraft_use` | int | 0=none, 1=medium, 2=high |
| `sewage_discharge_beach` | int | 0=no, 1=yes (nearby discharge) |
| `high_intensity_agri_beach` | int | 0=no, 1=yes (agricultural runoff) |
| `beach_orientation_angle` | int | Degrees 0--360 (used for wind-shore calculation) |
| `Latitude` | float | Decimal degrees south (negative) |
| `Longitude` | float | Decimal degrees east (positive) |

**Temporal columns (computed from `datetime_ts`):**

| Column | Type | Description |
|---|---|---|
| `DateTime` | datetime64 | Reference datetime (tz-naive) |
| `season` | str | Current season "YYYY-YYYY" |
| `YEAR` | int | Year |
| `MONTH` | int | Month 1--12 |
| `WEEK` | int | ISO week 1--53 |
| `DAY_OF_WEEK` | int | 0=Monday, 6=Sunday |
| `WEEKEND` | int | 1 if Saturday/Sunday, else 0 |
| `TIME_OF_DAY` | int | Hour 0--23 |
| `HOLIDAY_FLAG` | int | 1 if NZ public holiday period, else 0 |

**15 sites (order matters for the model):**

| # | Site | Harbour |
|---|---|---|
| 1 | Akaroa at main beach | Akaroa |
| 2 | Cass Bay at boat ramp | Lyttelton |
| 3 | Charteris Bay Paradise Beach | Lyttelton |
| 4 | Church Bay Beach | Lyttelton |
| 5 | Corsair Bay Beach | Lyttelton |
| 6 | Diamond Harbour Beach | Lyttelton |
| 7 | Duvauchelle Bay by camping ground | Akaroa |
| 8 | French Farm Bay | Akaroa |
| 9 | Glen Bay boat ramp | Akaroa |
| 10 | Governors Bay Sandy Beach | Lyttelton |
| 11 | Purau Bay Beach | Lyttelton |
| 12 | Rapaki Bay Beach | Lyttelton |
| 13 | Takamatua Bay Boat ramp | Akaroa |
| 14 | Tikao Bay mid beach | Akaroa |
| 15 | Wainui Beach Wainui Beach | Akaroa |

**Holiday flag logic:** Uses `HOLIDAY_DATE_DATA` from `src_inference.config.constants`, which defines date ranges for Labour Weekend, Canterbury Anniversary, Christmas, Waitangi, Easter, Anzac Day, and King's Birthday.

---

## Historical Enterococci Fetch

### `fetch_enterococci_data_for_sites()`

Fetch historical Enterococci measurements from ECan's Hilltop webservice for all sites in the current bathing season.

```python
def fetch_enterococci_data_for_sites(
    site_codes: dict,           # {SITE_NAME: SQ_CODE}
    from_date: str,             # "YYYY-MM-DD"
    to_date: str,               # "YYYY-MM-DD"
    logger=None,
    save_path=None,
    *,
    event_fn=None,              # Optional event logging callback
    timeout_s=20,               # Per-request timeout
    max_retries=3,              # Attempts per site
    backoff_base=2.0            # Exponential backoff base
) -> pd.DataFrame
```

**Returns:** DataFrame with columns `[SITE_NAME, DateTime, Enterococci]`. Typically 50--150 rows (15 sites x ~5--10 samples each over the season so far).

**API details:**
- Endpoint: `http://wateruse.ecan.govt.nz/WQSurfaceWater.hts`
- Protocol: Hilltop XML
- One request per site, with a 250ms politeness delay between requests

**Example URL:**
```
http://wateruse.ecan.govt.nz/WQSurfaceWater.hts?Service=Hilltop&Request=GetData&Site=SQ32610&Measurement=Enterococci&TimeInterval=2025-10-01/2026-03-12
```

**Retry logic:** Exponential backoff with jitter -- `sleep = backoff_base^(attempt-1) + random(0, 0.5)`. The demo uses `timeout_s=10` and `max_retries=2`.

**Failure modes:**
1. Individual site timeout -- logs warning, continues with other sites
2. XML parse error -- skips site, logs warning
3. No data for site in date range -- 0 rows for that site (normal for some beaches)
4. All sites fail -- returns empty DataFrame

---

### `_to_float_with_censor()`

Convert Hilltop string values (including censor markers) to floats. Lab results sometimes come back as `<10` (below detection limit) or `>280` (above measurement range).

```python
def _to_float_with_censor(text: str | None) -> float | None
```

**Censor rules:**

| Input | Output | Rule |
|---|---|---|
| `"<10"` | `5.0` | 0.5 x limit (conservative lower bound) |
| `">280"` | `308.0` | 1.1 x limit (conservative upper bound) |
| `"≤50"` | `50.0` | Threshold itself |
| `"≥100"` | `110.0` | 1.1 x limit |
| `"125"` | `125.0` | Plain number |
| `"nan"` / `None` / `""` | `None` | Unparseable |

**Why these rules?** Lab detection limits mean some results can only be expressed as ranges. Using half the lower limit and 1.1x the upper limit preserves the sample for statistics without overstating certainty.

---

## Seasonal Feature Engineering

### `add_seasonal_features()`

Compute site-specific seasonal features from historical Enterococci measurements.

```python
def add_seasonal_features(
    nowcast_df: pd.DataFrame,      # Current sites (15 rows)
    hist_df: pd.DataFrame,         # Historical Enterococci
    reference_time: pd.Timestamp,  # Current datetime
    logger=None
) -> pd.DataFrame
```

Adds two columns to `nowcast_df`:

#### `Site_Season_Average`

Rolling mean of the last 5 Enterococci samples for this site in the current season (minimum 1 sample required).

**Example:**
```python
# Site: Akaroa at main beach
# Historical samples this season: [52, 308, 45, 67, 89]
# Rolling mean (window=5, min_periods=1):
#   After sample 1: mean([52]) = 52.0
#   After sample 2: mean([52, 308]) = 180.0
#   After sample 3: mean([52, 308, 45]) = 135.0
#   After sample 4: mean([52, 308, 45, 67]) = 118.0
#   After sample 5: mean([52, 308, 45, 67, 89]) = 112.2  <-- final value
```

#### `Site_Historical_Exceedance_Rate`

Proportion of all historical samples exceeding 280 MPN/100mL for this site in the current season.

**Example:**
```python
# 10 samples total, 2 exceed 280 (values: 308, 340)
# Exceedance rate = 2/10 = 0.20 (20%)
```

**Edge cases:**
- Off-season (April--September) -- both features set to NaN
- Empty Hilltop response -- both features set to NaN
- No samples for a particular site -- NaN for that site only

**Why these features?** Both are in the top 10 most important predictors (from SHAP analysis). They capture site-specific contamination patterns and recent bacterial trends.

---

### `is_in_bathing_season()`

Check if a date falls within the NZ bathing season (October--March).

```python
def is_in_bathing_season(date: pd.Timestamp) -> bool
```

Returns `True` for months 10, 11, 12, 1, 2, 3.

---

### `assign_bathing_season()`

Return the season label as `"YYYY-YYYY+1"` for a given datetime. Functionally identical to `get_current_season()` but accepts a plain datetime.

```python
def assign_bathing_season(dt) -> str
```

---

## Schema Conformance

### `conform_features_to_training()`

Align the inference DataFrame to the exact training schema -- same columns, same order.

```python
def conform_features_to_training(
    df: pd.DataFrame,           # Inference data
    reference_df: pd.DataFrame, # Training schema snapshot (from reference_schema.pkl)
    logger=None
) -> pd.DataFrame
```

**What it does:**
1. Identifies extra columns in inference data (not in training) -- drops them
2. Identifies missing columns (in training but not inference) -- adds them as NaN
3. Reorders columns to match training exactly

**Why necessary?** LightGBM expects features in the exact same order as training. Any mismatch causes prediction errors or silent misalignment.

**Must be called BEFORE** categorical encoding, so that categorical columns are present when `apply_model_categoricals()` runs.

---

## Categorical Encoding

### `get_cat_info_from_model_txt()`

Parse the LightGBM model `.txt` file to extract categorical feature metadata.

```python
def get_cat_info_from_model_txt(model_txt_path) -> tuple
```

**Returns:** `(cat_indices, cat_levels)` where:
- `cat_indices` -- list of feature indices (0-based) that are categorical
- `cat_levels` -- list of category lists (one per categorical feature)

**Parses these lines from the `.txt` file:**
```
[categorical_feature:2,5,7,9,10,11,12,13]
pandas_categorical:[['Akaroa', 'Lyttelton'], ['Alongshore', 'Offshore', 'Onshore'], ...]
```

**Why parse the .txt?** LightGBM saves categorical metadata to its text dump (not the `.joblib`). We need the exact category order so that codes 0, 1, 2, etc. match what the model was trained on.

---

### `apply_model_categoricals()`

Set `pd.CategoricalDtype` on DataFrame columns using model metadata.

```python
def apply_model_categoricals(
    df: pd.DataFrame,
    feature_names: list,       # From booster.feature_name()
    cat_indices: list,         # From get_cat_info_from_model_txt()
    cat_levels: list,          # From get_cat_info_from_model_txt()
    logger=None
) -> pd.DataFrame
```

**What it does for each categorical column:**
1. Looks up the column name from `feature_names[idx]`
2. Sets `pd.CategoricalDtype(categories=cats, ordered=False)`
3. Validates that all observed values are in the training categories
4. Logs a warning if unseen categories are found (they become NaN automatically)

**Example:**
```python
# Before
df['Harbour'].dtype       # --> object (string)
df['Harbour'].unique()    # --> ['Akaroa', 'Lyttelton']

# After
df['Harbour'].dtype       # --> category
df['Harbour'].cat.codes   # --> [0, 1, 1, 1, ...]
```

---

## SHAP Preparation and Computation

### `prepare_shap_input()`

Prepare a SHAP-compatible DataFrame by handling DateTime and validating all dtypes.

```python
def prepare_shap_input(
    df: pd.DataFrame,
    feature_names: list,
    cat_indices: list,
    cat_levels: list,
    reference_df: pd.DataFrame,
    logger=None
) -> pd.DataFrame
```

**What it does:**
1. Copies and selects only the model's feature columns
2. Re-applies categorical dtypes
3. Replaces `DateTime` with `0` (int64) -- SHAP's TreeExplainer does not support `datetime64`
4. Casts non-categorical columns to match training dtypes
5. Validates no `object` dtypes remain (raises `ValueError` if any do)

---

### `compute_shap_df()`

Compute SHAP values for a single LightGBM model using TreeExplainer.

```python
def compute_shap_df(
    model,                    # Single quantile LightGBM model
    shap_input: pd.DataFrame, # From prepare_shap_input()
    logger=None
) -> tuple[pd.DataFrame, float]
```

**Returns:** `(shap_df, base_value)` where:
- `shap_df` -- DataFrame of SHAP values (columns named `SHAP_<feature>`)
- `base_value` -- the model's expected prediction when all features are at baseline

**SHAP values interpretation:**
- Positive SHAP -- feature pushed prediction higher
- Negative SHAP -- feature pushed prediction lower
- `prediction = base_value + sum(all SHAP values)`

**In inference-demo.py**, this is called once per quantile model (12 times). The median SHAP value across all 12 quantiles is taken per feature, giving a single representative explanation.

---

## Exceedance Probability

### `prob_exceed_vectorised()`

Calculate P(Y > threshold) from quantile predictions using piecewise-linear CDF inversion.

```python
def prob_exceed_vectorised(
    Q: np.ndarray,          # Quantile predictions, shape (n_obs, n_quantiles)
    p: np.ndarray,          # Quantile levels, shape (n_quantiles,), ascending
    y_threshold: float,     # e.g. 280
    debug: bool = False
) -> np.ndarray            # Exceedance probabilities, shape (n_obs,)
```

**How it works:**
1. For each observation, find the segment where `Q[k] <= threshold < Q[k+1]`
2. Linearly interpolate the CDF at the threshold: `F(y) = p_lo + (y - Q_lo)/(Q_hi - Q_lo) * (p_hi - p_lo)`
3. Return `P(Y > y) = 1 - F(y)`

**Example:**
```python
Q = np.array([[100, 150, 200, 250]])   # 1 observation, 4 quantiles
p = np.array([0.25, 0.50, 0.75, 0.95])
y_threshold = 220

# Segment: 200 <= 220 < 250 (k=2)
# F(220) = 0.75 + (220-200)/(250-200) * (0.95-0.75) = 0.83
# P(Y > 220) = 1 - 0.83 = 0.17 (17% chance of exceeding 220)
```

**Edge cases:**
- Threshold below all quantiles -- `P(exceed) = 1.0`
- Threshold above all quantiles -- `P(exceed) = 0.0`
- Degenerate segment (`Q_hi == Q_lo`) -- falls back to upper quantile level

**Note:** This function is not called directly in `inference-demo.py`. It is available for downstream analysis -- e.g. computing `P(Enterococci > 280)` from the quantile predictions in the output CSV.

---

## Common Errors and Solutions

### Timezone TypeError

```
TypeError: Cannot compare tz-naive and tz-aware datetimes
```

**Cause:** Mixing tz-aware and tz-naive datetimes in comparisons or merges.

**Solution:** Strip timezone info before use:
```python
if getattr(DateTime_UTC, 'tzinfo', None) is not None:
    DateTime_UTC = pd.Timestamp(DateTime_UTC).tz_localize(None)
```

---

### Categorical Mismatch

```
WARNING: Harbour: Inference data has unseen categories: {'Unknown'}
```

**Cause:** Inference data contains a category not seen during training.

**Handling:** `apply_model_categoricals()` automatically coerces unknown categories to NaN and logs a warning. No action needed unless the unknown category is expected (e.g. a new site was added).

---

### Empty Hilltop Response

```
WARNING: Hilltop fetch returned empty dataframe.
```

**Cause:** No samples collected in date range (normal early in a new season), or Hilltop server is unreachable.

**Impact:** `Site_Season_Average` and `Site_Historical_Exceedance_Rate` will be NaN for all sites. Predictions still run -- these features are useful but not critical.

---

### SHAP TreeExplainer DateTime Error

```
TypeError: TreeExplainer does not support datetime64
```

**Cause:** DateTime column was not replaced with an integer placeholder before SHAP computation.

**Handling:** `inference-demo.py` replaces DateTime with `0` (int64) before calling the model, and `prepare_shap_input()` does the same defensively. If you see this error, the replacement was somehow skipped.

---

### Schema Column Mismatch

```
KeyError: "column_name"
```

**Cause:** A required column is missing from the inference data, or an unexpected column name has changed.

**Prevention:** `conform_features_to_training()` ensures all training columns are present (adding missing ones as NaN) and drops extras. Make sure it is called before prediction.

---

## Related Documentation

- [inference-demo-documentation.md](../inference-demo-documentation.md) -- Full inference demo overview and setup
- [WeatherAPI_Functions_GUIDE.md](WeatherAPI_Functions_GUIDE.md) -- Weather and tide API function reference
