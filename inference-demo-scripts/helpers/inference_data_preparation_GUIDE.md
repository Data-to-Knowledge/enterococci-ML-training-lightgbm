# inference_data_preparation.py Function Reference

**File:** [helpers/inference_data_preparation.py](helpers/inference_data_preparation.py)
**Lines:** 741 total
**Purpose:** Data preparation, feature engineering, Hilltop API integration, SHAP preparation, schema alignment for inference pipeline

**Last Updated:** 2025-11-29

---

## Table of Contents

1. [Overview](#overview)
2. [Execution Flow in inference.py](#execution-flow-in-inferencepy)
3. [Phase 1: Timezone & Datetime Setup](#phase-1-timezone--datetime-setup)
4. [Phase 2: Site Metadata Creation](#phase-2-site-metadata-creation)
5. [Phase 3: Historical Enterococci Fetch](#phase-3-historical-enterococci-fetch)
6. [Phase 4: Seasonal Feature Engineering](#phase-4-seasonal-feature-engineering)
7. [Phase 5: Schema Conformance](#phase-5-schema-conformance)
8. [Phase 6: Categorical Encoding](#phase-6-categorical-encoding)
9. [Phase 7: SHAP Preparation & Computation](#phase-7-shap-preparation--computation)
10. [Helper Functions](#helper-functions)
11. [Probability Calculations](#probability-calculations)
12. [Common Errors & Solutions](#common-errors--solutions)

---

## Overview

This module contains all data preparation and feature engineering functions used by [inference.py](../inference.py) during the hourly prediction pipeline. Functions are called in a specific sequence to:

1. Generate current timestamps in multiple timezones
2. Create site metadata with temporal features
3. Fetch historical Enterococci measurements from Hilltop
4. Compute seasonal features (rolling averages, exceedance rates)
5. Align inference data to training schema
6. Apply categorical encodings from trained model
7. Prepare SHAP-compatible input for explainability
8. Compute SHAP values for all quantile models

**Critical Design Principle:** All functions maintain tz-naive datetime handling to ensure compatibility with LightGBM models and Azure SQL Server storage.

---

## Execution Flow in inference.py

The functions in this module are called in the following sequence during `inference.py` execution:

```python
# PHASE 1: INITIALIZATION (Line 68)
timestamps = get_current_datetimes()
# → Returns: {DateTime_UTC, DateTime_NZST, DateTime_NZLT}
# → Used to set: NZ_LOCAL (tz-aware), DateTime_UTC (tz-naive)

# PHASE 2: SITE METADATA (Line 310)
site_data = create_site_metadata(NZ_LOCAL)
# → Input: NZ_LOCAL (pd.Timestamp, tz-aware Pacific/Auckland)
# → Returns: DataFrame with 15 sites × ~27 columns
# → Includes: site characteristics, temporal features, holiday flags

# PHASE 3: HISTORICAL DATA (Line 732)
hist_df = fetch_enterococci_data_for_sites(
    site_codes=site_codes,        # Dict: {SITE_NAME: SQ_CODE}
    from_date="2025-10-01",       # Current season start
    to_date="2025-11-28",         # Today
    logger=None,
    save_path=None,
    event_fn=lambda e, **kw: ev_hill(e, **kw),
    timeout_s=10,
    max_retries=2
)
# → Returns: DataFrame[SITE_NAME, DateTime, Enterococci]
# → Typical: 50-150 rows (15 sites × ~5-10 samples each)

# PHASE 4: SEASONAL FEATURES (Line 758)
nowcast = add_seasonal_features(nowcast, hist_df, NZ_LOCAL, logger=None)
# → Input: nowcast (15 sites with weather/tide), hist_df (Hilltop data)
# → Adds: Site_Season_Average, Site_Historical_Exceedance_Rate
# → Returns: Updated nowcast DataFrame

# PHASE 5: SCHEMA CONFORMANCE (Line 773)
INFERENCE_DATA = conform_features_to_training(INFERENCE_DATA, reference_df, logger=None)
# → Input: INFERENCE_DATA (engineered features), reference_df (training schema)
# → Drops extra columns, adds missing columns as NaN
# → Reorders to match training exactly
# → Returns: Conforming DataFrame

# PHASE 6: CATEGORICAL ENCODING (Lines 776-786)
cat_indices, cat_levels = get_cat_info_from_model_txt(MODEL_TXT_PATH)
# → Parses: probabilistic_framework.txt
# → Returns: (cat_indices: list[int], cat_levels: list[list[str]])

booster = lgb.Booster(model_file=MODEL_TXT_PATH)
feature_names = booster.feature_name()

INFERENCE_DATA = apply_model_categoricals(
    df=INFERENCE_DATA,
    feature_names=feature_names,
    cat_indices=cat_indices,
    cat_levels=cat_levels,
    logger=None
)
# → Sets pandas CategoricalDtype on columns using model metadata
# → Returns: DataFrame with categorical dtypes applied

# PHASE 7: SHAP PREPARATION (Lines 821-836)
shap_input = prepare_shap_input(
    df=INFERENCE_DATA,
    feature_names=feature_names,
    cat_indices=cat_indices,
    cat_levels=cat_levels,
    reference_df=reference_df,
    logger=logger
)
# → Replaces DateTime column with 0 (SHAP TreeExplainer incompatibility)
# → Re-applies categoricals, validates dtypes
# → Returns: SHAP-ready DataFrame

# SHAP COMPUTATION (Line 836, inside loop for each quantile)
for q in [0.20, 0.35, 0.50, ..., 0.975]:
    rf_model = model_data.quantile_ensemble.models[q]
    shap_df_q, base_val = compute_shap_df(rf_model, shap_input, logger=None)
    # → Computes SHAP values for this quantile model
    # → Returns: (shap_df: DataFrame, base_value: float)
```

---

## Phase 1: Timezone & Datetime Setup

### `get_named_timezones()`

**Location:** [helpers/inference_data_preparation.py:17-34](helpers/inference_data_preparation.py#L17-L34)

**Purpose:** Returns dictionary of named pytz timezone objects for NZ and UTC conversions

**Signature:**
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

**Usage in inference.py:** Not directly called. Used internally by `get_current_datetimes()`.

**Implementation Details:**
- `NZST`: Fixed offset timezone (`Etc/GMT-12`), always UTC+12, never observes daylight saving
- `NZLT`: Location-based timezone (`Pacific/Auckland`), UTC+12 in winter (Apr-Sep), UTC+13 in summer (Oct-Mar)
- `UTC`: Coordinated Universal Time, UTC+0

**Why Two NZ Timezones?**
- `NZST` ensures consistency with training data (fixed offset)
- `NZLT` aligns with actual NZ civil time for seasonal logic and holiday checks

---

### `get_current_datetimes()`

**Location:** [helpers/inference_data_preparation.py:36-56](helpers/inference_data_preparation.py#L36-L56)

**Purpose:** Get current time in multiple timezones, floored to hour, returned as tz-naive datetimes

**Signature:**
```python
def get_current_datetimes() -> dict
```

**Returns:**
```python
{
    "DateTime_UTC": datetime,   # e.g., 2025-11-28 01:00:00 (tz-naive)
    "DateTime_NZST": datetime,  # e.g., 2025-11-28 13:00:00 (tz-naive, fixed UTC+12)
    "DateTime_NZLT": datetime,  # e.g., 2025-11-28 14:00:00 (tz-naive, DST-aware)
}
```

**Called in inference.py:** [inference.py:68](../inference.py#L68)
```python
timestamps = get_current_datetimes()
DateTime_NZST = timestamps["DateTime_NZST"]   # Fixed +12 (no DST)
DateTime_NZLT = timestamps["DateTime_NZLT"]   # True local (Pacific/Auckland)
DateTime_UTC  = timestamps["DateTime_UTC"]
```

**Flooring Behavior:**
```python
# Implementation (lines 49-51)
nz_fixed = datetime.now(tzs["NZST"]).replace(minute=0, second=0, microsecond=0)
nz_local = datetime.now(tzs["NZLT"]).replace(minute=0, second=0, microsecond=0)
utc_now  = datetime.now(tzs["UTC"]).replace(minute=0, second=0, microsecond=0)

# Example:
# If current time is 2025-11-28 13:47:23.456789 NZLT
# → Returns 2025-11-28 13:00:00.000000 (floored to hour)
```

**Why Tz-Naive?**
1. LightGBM models don't support `datetime64[ns, tz]` dtypes
2. Azure SQL Server stores `datetime` without timezone (tz-naive)
3. Simplifies pandas DataFrame operations (no dtype conflicts)

**Critical Note:** All returned datetimes are **tz-naive** despite being computed from tz-aware sources. The timezone context is discarded after computing the correct local time.

---

### `get_current_season()`

**Location:** [helpers/inference_data_preparation.py:58-79](helpers/inference_data_preparation.py#L58-L79)

**Purpose:** Determine bathing season label based on date (Oct-Sep format "YYYY-YYYY")

**Signature:**
```python
def get_current_season(today: pd.Timestamp) -> str
```

**Parameters:**
- `today` (pd.Timestamp): Date to check (tz-naive or tz-aware, doesn't matter)

**Returns:**
- `str`: Season label in format "YYYY-YYYY" (e.g., "2025-2026")

**Algorithm:**
```python
# Lines 73-79
if today.month >= 10:  # Oct, Nov, Dec
    season_start = today.year
    season_end = today.year + 1
else:  # Jan-Sep
    season_start = today.year - 1
    season_end = today.year
return f"{season_start}-{season_end}"
```

**Examples:**
```python
get_current_season(pd.Timestamp("2025-11-28"))  # → "2025-2026"
get_current_season(pd.Timestamp("2025-01-15"))  # → "2024-2025"
get_current_season(pd.Timestamp("2025-10-01"))  # → "2025-2026" (season starts)
get_current_season(pd.Timestamp("2025-09-30"))  # → "2024-2025" (last day prev season)
```

**Used By:**
- `create_site_metadata()` (line 144) → Sets 'season' column
- `add_seasonal_features()` (via `assign_bathing_season()`) → Filters Hilltop data to current season

**Called in inference.py:** Not directly. Used internally by helper functions.

---

## Phase 2: Site Metadata Creation

### `create_site_metadata()`

**Location:** [helpers/inference_data_preparation.py:81-159](helpers/inference_data_preparation.py#L81-L159)

**Purpose:** Create 15-row DataFrame with site characteristics and temporal features for the target prediction hour

**Signature:**
```python
def create_site_metadata(datetime_ts: pd.Timestamp) -> pd.DataFrame
```

**Parameters:**
- `datetime_ts` (pd.Timestamp): Reference datetime for temporal features (typically `NZ_LOCAL`, tz-aware)

**Returns:**
- `pd.DataFrame`: 15 rows × 27 columns

**Called in inference.py:** [inference.py:310](../inference.py#L310)
```python
site_data = create_site_metadata(NZ_LOCAL)
# NZ_LOCAL is tz-aware Pacific/Auckland timestamp
# Result: 15 sites with current temporal features
```

**Output Schema:**

```python
# Site identifiers
SITE_NAME (str): Full site name (e.g., "Akaroa at main beach")
Harbour (str): "Akaroa" or "Lyttelton"

# Site characteristics (STATIC - never change)
Shallowness (str): "shallow" / "medium" / "deep"
Soil_type (str): "sandy" / "muddy" / "stony"
Catchment_slope (int): Slope rating 0-10
Landcover_catchment (str): "urban" / "grass" / "forest"
watercraft_use (int): 0=none, 1=medium, 2=high
sewage_discharge_beach (int): 0=no, 1=yes (nearby discharge)
high_intensity_agri_beach (int): 0=no, 1=yes (agricultural runoff)
beach_orientation_angle (int): Degrees 0-360 (for wind-shore calc)
Latitude (float): Decimal degrees south (negative)
Longitude (float): Decimal degrees east (positive)

# Temporal features (DYNAMIC - computed from datetime_ts)
DateTime (datetime64[ns]): Reference datetime (tz-naive, line 141)
season (str): Current season "YYYY-YYYY" (line 144)
YEAR (int): Year (line 145)
MONTH (int): Month 1-12 (line 146)
WEEK (int): ISO week 1-53 (line 147)
DAY_OF_WEEK (int): Day 0=Monday, 6=Sunday (line 148)
WEEKEND (int): 1 if Saturday/Sunday, else 0 (line 149)
TIME_OF_DAY (int): Hour 0-23 (line 150)
HOLIDAY_FLAG (int): 1 if NZ public holiday, else 0 (lines 153-157)
```

**15 Sites (Order Matters):**

1. Akaroa at main beach (Akaroa)
2. Cass Bay at boat ramp (Lyttelton)
3. Charteris Bay Paradise Beach (Lyttelton)
4. Church Bay Beach (Lyttelton)
5. Corsair Bay Beach (Lyttelton)
6. Diamond Harbour Beach (Lyttelton)
7. Duvauchelle Bay by camping ground (Akaroa)
8. French Farm Bay (Akaroa)
9. Glen Bay boat ramp (Akaroa)
10. Governors Bay Sandy Beach (Lyttelton)
11. Purau Bay Beach (Lyttelton)
12. Rapaki Bay Beach (Lyttelton)
13. Takamatua Bay Boat ramp (Akaroa)
14. Tikao Bay mid beach (Akaroa)
15. Wainui Beach Wainui Beach (Akaroa)

**Holiday Flag Logic (Lines 153-157):**
```python
# Uses HOLIDAY_DATE_DATA from src.config.constants
# Format: {'Holiday Name': ('DD-MM-YYYY', 'DD-MM-YYYY')}

for start, end in HOLIDAY_DATE_DATA.values():
    start_date = pd.to_datetime(start, format='%d-%m-%Y')
    end_date = pd.to_datetime(end, format='%d-%m-%Y')
    if start_date <= datetime_ts <= end_date:
        HOLIDAY_FLAG = 1
```

**Critical Implementation Details:**

1. **Tz-Naive Conversion (Line 141):**
   ```python
   site_data['DateTime'] = pd.to_datetime(site_data['DateTime']).dt.tz_localize(None)
   ```
   Even though `datetime_ts` is tz-aware, the output DateTime column is forced tz-naive.

2. **Broadcast DateTime (Line 139):**
   ```python
   'DateTime': [datetime_ts]*15
   ```
   All 15 sites receive the same target prediction hour.

3. **beach_orientation_angle (Line 128-129):**
   Critical for wind-shore classification in Phase 3. Examples:
   - Akaroa: 170° (south-facing)
   - Cass Bay: 15° (north-facing)
   - Charteris Bay: 90° (east-facing)

**Usage Pattern:**
```python
# In inference.py (Line 310)
site_data = create_site_metadata(NZ_LOCAL)

# Later merged with weather data (Line 681)
site_data_filtered = site_data[site_data["Harbour"].isin(harbours_to_write)].copy()
nowcast = pd.merge(site_data_filtered, df_weather, on='Harbour', how='left')
```

---

## Phase 3: Historical Enterococci Fetch

### `fetch_enterococci_data_for_sites()`

**Location:** [helpers/inference_data_preparation.py:205-382](helpers/inference_data_preparation.py#L205-L382)

**Purpose:** Fetch historical Enterococci measurements from ECAN Hilltop webservice for all sites in the current bathing season

**Signature:**
```python
def fetch_enterococci_data_for_sites(
    site_codes: dict,           # {SITE_NAME: SQ_CODE}
    from_date: str,             # "YYYY-MM-DD"
    to_date: str,               # "YYYY-MM-DD"
    logger=None,
    save_path=None,
    *,
    event_fn=None,              # Structured event logging callback
    timeout_s=20,               # Per-request timeout
    max_retries=3,              # Attempts per site
    backoff_base=2.0            # Exponential backoff base
) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:732-741](../inference.py#L732-L741)
```python
# Calculate current season start (lines 721-728)
if NZ_LOCAL.month >= 10:
    season_start_year = NZ_LOCAL.year
else:
    season_start_year = NZ_LOCAL.year - 1
from_date = f"{season_start_year}-10-01"
to_date = NZ_LOCAL.strftime("%Y-%m-%d")

# Fetch (lines 732-741)
hist_df = fetch_enterococci_data_for_sites(
    site_codes=site_codes,        # From src.config.constants.SITE_CODES
    from_date=from_date,          # E.g., "2025-10-01"
    to_date=to_date,              # E.g., "2025-11-28"
    logger=None,
    save_path=None,
    event_fn=lambda e, **kw: ev_hill(e, **kw),
    timeout_s=10,
    max_retries=2
)
```

**Parameters:**
- `site_codes` (dict): Mapping `{SITE_NAME: SQ_CODE}` from [src/config/constants.py](../src/config/constants.py)
  - Example: `{"Akaroa at main beach": "SQ32610", "Cass Bay at boat ramp": "SQ30640"}`
- `from_date`, `to_date` (str): Date range in "YYYY-MM-DD" format
- `event_fn` (callable): Event logging function `(event, level='INFO', **fields)`
- `timeout_s` (int): HTTP request timeout (inference uses 10 seconds)
- `max_retries` (int): Retry attempts per site (inference uses 2)
- `backoff_base` (float): Exponential backoff base (default 2.0)

**Returns:**
- `pd.DataFrame`: Columns `[SITE_NAME, DateTime, Enterococci]`
- Typical: 50-150 rows (15 sites × ~5-10 samples each over 2 months)

**API Details:**
```python
# Endpoint (line 234)
base_url = "http://wateruse.ecan.govt.nz/WQSurfaceWater.hts"

# Request parameters (lines 249-255)
params = {
    'Service': 'Hilltop',
    'Request': 'GetData',
    'Site': site_code,         # e.g., "SQ32610"
    'Measurement': 'Enterococci',
    'TimeInterval': f'{from_date}/{to_date}'
}

# Example URL:
# http://wateruse.ecan.govt.nz/WQSurfaceWater.hts?Service=Hilltop&Request=GetData&Site=SQ32610&Measurement=Enterococci&TimeInterval=2025-10-01/2025-11-28
```

**XML Response Format:**
```xml
<Hilltop>
  <Measurement SiteName="SQ32610 Akaroa at main beach">
    <Data>
      <E>
        <T>2025-10-15T09:30:00</T>
        <Value>52</Value>
      </E>
      <E>
        <T>2025-10-22T10:15:00</T>
        <Value>>280</Value>            <!-- Censored value -->
      </E>
      <E>
        <T>2025-11-05T09:00:00</T>
        <Value><10</Value>             <!-- Below detection limit -->
      </E>
    </Data>
  </Measurement>
</Hilltop>
```

**Censor Value Handling (Lines 287-290):**
```python
# Uses _to_float_with_censor() helper
v_node = entry.find("Value")
val_raw = v_node.text if v_node is not None else None
val_clean = _to_float_with_censor(val_raw)

# Examples:
"<10"   → 5.0    (0.5 × 10)
">280"  → 308.0  (1.1 × 280)
"52"    → 52.0
```

**Retry Logic (Lines 265-352):**
```python
for attempt in range(max_retries):
    try:
        # HTTP request with timeout (line 276)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            xml_bytes = resp.read()
        # Parse XML and extract values
        break
    except (urllib.error.URLError, TimeoutError) as e:
        if attempt < max_retries - 1:
            # Exponential backoff (line 329)
            sleep_s = (backoff_base ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(sleep_s)  # 2s, 4s for max_retries=2
            continue
        else:
            # Final attempt failed (lines 337-349)
            _log_event("SOURCE_DOWN", level="ERROR",
                source="hilltop", site_name=name, site_code=code,
                failure_kind="exception", error=str(e)[:300])
```

**Politeness Delay (Line 357):**
```python
time.sleep(0.25)  # 250ms between sites
```

**Event Emissions:**
```python
# Per-site success (lines 294-304)
_log_event("SOURCE_OK", source="hilltop",
    site_name=name, site_code=code, rows=len(records), elapsed_ms=...)

# Per-site empty response (lines 306-316)
_log_event("SOURCE_DEGRADED", source="hilltop",
    site_name=name, site_code=code, failure_kind="empty_payload")

# All sites failed (lines 373-378)
_log_event("SOURCE_DEGRADED", source="hilltop",
    failure_kind="empty_payload_all_sites", sites=15)
```

**Output DataFrame:**
```python
# Example with 3 samples
     SITE_NAME                     DateTime             Enterococci
0    Akaroa at main beach          2025-10-15 09:30:00  52.0
1    Akaroa at main beach          2025-10-22 10:15:00  308.0  # Censored ">280"
2    Akaroa at main beach          2025-11-05 09:00:00  5.0    # Censored "<10"
3    Cass Bay at boat ramp         2025-10-18 10:00:00  125.0
4    Tikao Bay mid beach           2025-10-21 09:45:00  18.0
```

**Performance:**
- Typical duration: 10-20 seconds for 15 sites (sequential fetches)
- 0.25s delay between sites = 3.75s minimum overhead
- Actual HTTP requests: ~10-15 seconds total

**Failure Modes:**
1. **Individual site timeout** → Logs warning, continues with other sites
2. **XML parse error** → Skips site, logs warning
3. **No data for site in date range** → 0 rows for that site (normal for some beaches)
4. **All sites fail** → Returns empty DataFrame, emits SOURCE_DEGRADED event

---

### `_to_float_with_censor()`

**Location:** [helpers/inference_data_preparation.py:162-201](helpers/inference_data_preparation.py#L162-L201)

**Purpose:** Convert Hilltop string values (including censor markers) to floats with appropriate handling

**Signature:**
```python
def _to_float_with_censor(text: str | None) -> float | None
```

**Parameters:**
- `text` (str | None): Raw value from Hilltop XML (e.g., "52", "<10", ">280", "≤50")

**Returns:**
- `float | None`: Numeric value with censor rules applied, or None if unparseable

**Censor Rules (Lines 179-195):**
```python
# Below detection limit (lines 183-184)
"<10"   → 5.0    (0.5 × 10)
"<5"    → 2.5    (0.5 × 5)

# Above detection limit (lines 186-186)
">280"  → 308.0  (1.1 × 280)
">500"  → 550.0  (1.1 × 500)

# Unicode comparators (lines 189-195)
"≤50"   → 50.0   (threshold itself)
"≥100"  → 110.0  (1.1 × 100)

# Plain numbers (lines 198-199)
"125"   → 125.0
"52.5"  → 52.5

# Invalid (lines 170-174)
"nan"   → None
None    → None
""      → None
```

**Algorithm:**
```python
# Lines 170-201
1. Strip whitespace, handle None/empty (lines 170-174)
2. Remove thousands separators (line 177)
3. Try regex for <, >, <=, >= patterns (lines 180-186)
4. Try regex for unicode ≤, ≥ patterns (lines 189-195)
5. Try plain float conversion (lines 198-199)
6. Return None if all fail (lines 200-201)
```

**Why These Rules?**
- Lab detection limits: Can't measure below ~5-10 CFU/100mL or above ~500 CFU/100mL
- `<n` → Use half the limit (conservative lower bound for low contamination)
- `>n` → Use 1.1× limit (conservative upper bound for high contamination)
- Prevents losing samples due to censoring (maintains sample size for statistics)

**Used By:**
- `fetch_enterococci_data_for_sites()` (line 289) → Converts all Hilltop values

---

## Phase 4: Seasonal Feature Engineering

### `add_seasonal_features()`

**Location:** [helpers/inference_data_preparation.py:553-626](helpers/inference_data_preparation.py#L553-L626)

**Purpose:** Compute site-specific seasonal features from historical Enterococci measurements

**Signature:**
```python
def add_seasonal_features(
    nowcast_df: pd.DataFrame,      # Current sites (15 rows)
    hist_df: pd.DataFrame,         # Historical Enterococci
    reference_time: pd.Timestamp,  # Current datetime (tz-aware)
    logger=None
) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:758](../inference.py#L758)
```python
nowcast = add_seasonal_features(nowcast, hist_df, NZ_LOCAL, logger=None)
# Input: nowcast (15 sites with weather/tide features)
#        hist_df (from fetch_enterococci_data_for_sites)
#        NZ_LOCAL (tz-aware Pacific/Auckland timestamp)
# Output: nowcast with 2 new columns added
```

**Parameters:**
- `nowcast_df` (pd.DataFrame): Current sites with weather/tide features (15 rows)
  - Must have `SITE_NAME` column
- `hist_df` (pd.DataFrame): Historical Enterococci from `fetch_enterococci_data_for_sites()`
  - Columns: `[SITE_NAME, DateTime, Enterococci]`
- `reference_time` (pd.Timestamp): Current datetime (tz-aware, for season check)
- `logger` (Logger, optional): For debug messages

**Returns:**
- `pd.DataFrame`: nowcast_df with 2 new columns: `Site_Season_Average`, `Site_Historical_Exceedance_Rate`

**Features Computed:**

#### 1. Site_Season_Average (Lines 617-618)

**Definition:** Rolling mean of last 5 Enterococci samples for this site in current season

**Implementation:**
```python
# Lines 605-624
for i, row in nowcast_df.iterrows():
    site = row['SITE_NAME']
    site_hist = hist_df[
        (hist_df['SITE_NAME'] == site) &
        (hist_df['Season'] == this_season) &
        (hist_df['DateTime'] < reference_time)
    ].sort_values('DateTime')

    if site_hist.empty:
        site_season_avg = np.nan
    else:
        # Rolling window with minimum 1 sample (line 617)
        means = site_hist['Enterococci'].rolling(window=5, min_periods=1).mean()
        site_season_avg = means.iloc[-1] if len(means) else np.nan
```

**Example:**
```python
# Site: Akaroa at main beach
# Historical samples (all from current season):
#   2025-10-15: 52.0
#   2025-10-22: 308.0  # Censored ">280"
#   2025-11-01: 45.0
#   2025-11-08: 67.0
#   2025-11-15: 89.0

# Rolling window (min_periods=1):
# Sample 1: mean([52]) = 52.0
# Sample 2: mean([52, 308]) = 180.0
# Sample 3: mean([52, 308, 45]) = 135.0
# Sample 4: mean([52, 308, 45, 67]) = 118.0
# Sample 5: mean([52, 308, 45, 67, 89]) = 112.2 ← FINAL VALUE

Site_Season_Average = 112.2
```

#### 2. Site_Historical_Exceedance_Rate (Lines 619-621)

**Definition:** Proportion of ALL historical samples > 280 CFU/100mL for this site in current season

**Implementation:**
```python
# Lines 619-621
n_prev   = len(site_hist)
n_exceed = (site_hist['Enterococci'] > 280).sum()
site_exc_rate = (n_exceed / n_prev) if n_prev > 0 else np.nan
```

**Alert Threshold:** 280 CFU/100mL (NZ bathing water quality standard)

**Example:**
```python
# Site: Akaroa at main beach
# All season samples (10 total):
#   52, 308, 45, 67, 89, 125, 15, 340, 22, 95

exceedances = [308, 340]  # 2 samples > 280
Site_Historical_Exceedance_Rate = 2 / 10 = 0.20  # 20%
```

**Off-Season Handling (Lines 585-588):**
```python
if not is_in_bathing_season(reference_time):
    nowcast_df['Site_Season_Average'] = np.nan
    nowcast_df['Site_Historical_Exceedance_Rate'] = np.nan
    # Off-season: don't use stale data from previous season
    return nowcast_df
```

**Empty Historical Data (Lines 591-594):**
```python
if hist_df is None or hist_df.empty or 'DateTime' not in hist_df.columns:
    # Hilltop returned nothing (new season / outage)
    nowcast_df['Site_Season_Average'] = np.nan
    nowcast_df['Site_Historical_Exceedance_Rate'] = np.nan
    return nowcast_df
```

**Season Filtering (Lines 597-603):**
```python
this_season = assign_bathing_season(reference_time)  # E.g., "2025-2026"
hist_df = hist_df.copy()
hist_df['DateTime'] = pd.to_datetime(hist_df['DateTime'], errors='coerce')
hist_df['Season'] = hist_df['DateTime'].apply(assign_bathing_season)

# Only use samples from current season (line 607-610)
site_hist = hist_df[
    (hist_df['SITE_NAME'] == site) &
    (hist_df['Season'] == this_season) &
    (hist_df['DateTime'] < reference_time)
]
```

**Why These Features?**
- **Site_Season_Average:** Captures recent bacterial trend at site
  - Some beaches consistently cleaner/dirtier than others
  - Rolling window (last 5 samples) adapts to changing conditions
  - Model learns site-specific contamination patterns
- **Site_Historical_Exceedance_Rate:** Identifies "problem beaches"
  - High exceedance rate → frequent contamination events
  - Model learns which sites are high-risk

**Feature Importance:** Both features are in the top 10 most important predictors (from SHAP analysis).

---

### `is_in_bathing_season()`

**Location:** [helpers/inference_data_preparation.py:541-543](helpers/inference_data_preparation.py#L541-L543)

**Purpose:** Check if date falls within NZ bathing season (Oct-Mar)

**Signature:**
```python
def is_in_bathing_season(date: pd.Timestamp) -> bool
```

**Implementation:**
```python
return date.month in [10, 11, 12, 1, 2, 3]
```

**Examples:**
```python
is_in_bathing_season(pd.Timestamp("2025-11-28"))  # → True (Nov)
is_in_bathing_season(pd.Timestamp("2025-01-15"))  # → True (Jan)
is_in_bathing_season(pd.Timestamp("2025-06-15"))  # → False (Jun, winter)
```

**Used By:**
- `add_seasonal_features()` (line 585) → Guards against off-season data

---

### `assign_bathing_season()`

**Location:** [helpers/inference_data_preparation.py:545-551](helpers/inference_data_preparation.py#L545-L551)

**Purpose:** Get bathing season label from datetime (wrapper around `get_current_season()`)

**Signature:**
```python
def assign_bathing_season(dt: pd.Timestamp) -> str
```

**Implementation:**
```python
# Lines 547-551
year = dt.year
if dt.month >= 10:
    return f"{year}-{year + 1}"
else:
    return f"{year - 1}-{year}"
```

**Used By:**
- `add_seasonal_features()` (lines 597, 603) → Filters historical data by season

---

## Phase 5: Schema Conformance

### `conform_features_to_training()`

**Location:** [helpers/inference_data_preparation.py:386-415](helpers/inference_data_preparation.py#L386-L415)

**Purpose:** Align inference DataFrame to exact training schema (columns, order, dtypes)

**Signature:**
```python
def conform_features_to_training(
    df: pd.DataFrame,           # Inference data
    reference_df: pd.DataFrame, # Training schema snapshot
    logger=None
) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:773](../inference.py#L773)
```python
INFERENCE_DATA = conform_features_to_training(INFERENCE_DATA, reference_df, logger=None)
# reference_df loaded from REFERENCE_SCHEMA_PATH (line 309)
# INFERENCE_DATA contains all engineered features at this point
```

**Parameters:**
- `df` (pd.DataFrame): Inference data with engineered features
- `reference_df` (pd.DataFrame): Training schema (from `reference_schema.pkl`)
  - Contains column names, dtypes, order from training time
- `logger` (Logger, optional): For warnings about dropped/added columns

**Returns:**
- `pd.DataFrame`: Conforming DataFrame (exact match to training schema)

**Algorithm (Lines 401-415):**
```python
# Step 1: Extract expected feature columns (line 401)
expected_features = [col for col in reference_df.columns if col != 'Enterococci']

# Step 2: Identify extra columns (lines 402-403)
extra_cols = set(df.columns) - set(expected_features)
missing_cols = set(expected_features) - set(df.columns)

# Step 3: Drop extra columns (lines 405-407)
if extra_cols and logger:
    logger.info(f"Removing extra columns from inference data: {extra_cols}")
df = df.drop(columns=list(extra_cols), errors='ignore')

# Step 4: Add missing columns with NaN (lines 409-412)
if missing_cols and logger:
    logger.info(f"Adding missing columns (NaN) to inference data: {missing_cols}")
for col in missing_cols:
    df[col] = np.nan

# Step 5: Reorder to match training (line 414)
df = df[expected_features]

return df
```

**Why Necessary?**
- LightGBM expects features in exact same order as training
- New features added after training → must be dropped (model doesn't know about them)
- Old features removed in new code → must be added back as NaN (model trained on them)
- Prevents "feature mismatch" errors during prediction

**Example:**
```python
# Training schema (reference_df):
columns = ['DateTime', 'SITE_NAME', 'Rainfall', '3H', '6H', 'wind_speed_3h', ...]

# Inference data (df):
columns = ['DateTime', 'SITE_NAME', 'Rainfall', '3H', '6H', 'wind_speed_3h',
           'temp_debug_col', ...]  # Extra column from new code
# Missing: 'old_deprecated_feature'  # Removed in refactor

# After conformance:
columns = ['DateTime', 'SITE_NAME', 'Rainfall', '3H', '6H', 'wind_speed_3h',
           'old_deprecated_feature', ...]  # old_deprecated_feature added with NaN
# temp_debug_col dropped
```

**Critical Note:** This function MUST be called BEFORE categorical encoding, so categorical columns are present when `apply_model_categoricals()` is called.

---

## Phase 6: Categorical Encoding

### `get_cat_info_from_model_txt()`

**Location:** [helpers/inference_data_preparation.py:421-438](helpers/inference_data_preparation.py#L421-L438)

**Purpose:** Parse LightGBM model .txt file to extract categorical feature metadata

**Signature:**
```python
def get_cat_info_from_model_txt(model_txt_path: str) -> tuple
```

**Called in inference.py:** [inference.py:776](../inference.py#L776)
```python
cat_indices, cat_levels = get_cat_info_from_model_txt(MODEL_TXT_PATH)
# MODEL_TXT_PATH = paths.MODEL_TXT_PATH = "models/probabilistic_framework.txt"
```

**Parameters:**
- `model_txt_path` (str): Path to `probabilistic_framework.txt`

**Returns:**
- `tuple`: `(cat_indices: list[int], cat_levels: list[list[str]])`
  - `cat_indices`: List of feature indices (0-based) that are categorical
  - `cat_levels`: List of category lists (one per categorical feature)

**Model .txt Format (Lines 425-437):**
```
[categorical_feature:2,5,7,9,10,11,12,13]
pandas_categorical:[['Akaroa', 'Lyttelton'], ['Alongshore', 'Offshore', 'Onshore'], ...]
```

**Parsing Logic:**
```python
# Lines 423-437
with open(model_txt_path, "r") as f:
    lines = f.readlines()

# Find categorical_feature line (line 425)
cat_idx_line = [l for l in lines if l.startswith("[categorical_feature:")]

# Find pandas_categorical line (line 426)
cat_levels_line = [l for l in lines if l.strip().startswith("pandas_categorical:")]

# Extract indices (lines 430-435)
cat_indices = [
    int(i) for i in cat_idx_line[0]
        .replace("[categorical_feature:", "")
        .replace("]", "")
        .strip().split(",") if i != ""
]

# Extract category levels (line 437)
cat_levels = ast.literal_eval(cat_levels_line[0].split(":", 1)[1].strip())
```

**Output Example:**
```python
cat_indices = [2, 5, 7, 9, 10, 11, 12, 13]
# Feature indices 2, 5, 7, etc. are categorical

cat_levels = [
    ['Akaroa', 'Lyttelton'],                          # Index 2: Harbour
    ['Alongshore', 'Offshore', 'Onshore'],            # Index 5: wind_shore_3h
    ['deep', 'medium', 'shallow'],                    # Index 7: Shallowness
    ['muddy', 'sandy', 'stony'],                      # Index 9: Soil_type
    ...
]
```

**Why Parse .txt?**
- LightGBM saves categorical metadata to .txt dump (not .joblib)
- Need exact category order (codes 0, 1, 2 must match training)
- Ensures inference uses same categorical encoding as training

---

### `apply_model_categoricals()`

**Location:** [helpers/inference_data_preparation.py:440-457](helpers/inference_data_preparation.py#L440-L457)

**Purpose:** Set pandas CategoricalDtype on DataFrame columns using model metadata

**Signature:**
```python
def apply_model_categoricals(
    df: pd.DataFrame,
    feature_names: list,       # Feature column names
    cat_indices: list,         # From get_cat_info_from_model_txt()
    cat_levels: dict,          # From get_cat_info_from_model_txt()
    logger=None
) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:780-786](../inference.py#L780-L786)
```python
INFERENCE_DATA = apply_model_categoricals(
    df=INFERENCE_DATA,
    feature_names=feature_names,
    cat_indices=cat_indices,
    cat_levels=cat_levels,
    logger=None
)
# feature_names from booster.feature_name() (line 778)
# cat_indices, cat_levels from get_cat_info_from_model_txt() (line 776)
```

**Parameters:**
- `df` (pd.DataFrame): Inference data (after schema conformance)
- `feature_names` (list): List of feature column names (from LightGBM booster)
- `cat_indices` (list): Categorical feature indices
- `cat_levels` (list): Category lists for each categorical feature
- `logger` (Logger, optional): For debug messages

**Returns:**
- `pd.DataFrame`: DataFrame with CategoricalDtype set on categorical columns

**Algorithm (Lines 442-457):**
```python
for idx, cats in zip(cat_indices, cat_levels):
    col = feature_names[idx]  # Get column name from index
    if col in df.columns:
        prev_dtype = df[col].dtype
        # Set categorical dtype (line 446)
        df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))

        # Validation (lines 449-453)
        obs_cats = set(df[col].dropna().unique())
        model_cats = set(cats)
        if not obs_cats.issubset(model_cats):
            if logger:
                logger.warning(f"{col}: Inference data has unseen categories: {obs_cats - model_cats}")
```

**Example:**
```python
# Before
df['Harbour'].dtype  # → object (string)
df['Harbour'].unique()  # → ['Akaroa', 'Lyttelton']

# After apply_model_categoricals
df['Harbour'].dtype  # → category
df['Harbour'].cat.categories  # → Index(['Akaroa', 'Lyttelton'])
df['Harbour'].cat.codes  # → [0, 1, 1, 1, ...]  # Numeric codes
```

**Critical Behavior:**
- New categories (not in training) → automatically coerced to NaN (line 446)
- Order matters: 'Akaroa'=0, 'Lyttelton'=1 (must match training)

**Edge Cases:**
```python
# Inference value not in training categories
df['Harbour'] = ['Akaroa', 'Lyttelton', 'Unknown']
# After apply_model_categoricals:
df['Harbour']  # → ['Akaroa', 'Lyttelton', NaN]  # 'Unknown' → NaN
# Warning logged: "Inference data has unseen categories: {'Unknown'}"
```

---

## Phase 7: SHAP Preparation & Computation

### `prepare_shap_input()`

**Location:** [helpers/inference_data_preparation.py:461-491](helpers/inference_data_preparation.py#L461-L491)

**Purpose:** Prepare SHAP-compatible input (handle DateTime, validate dtypes)

**Signature:**
```python
def prepare_shap_input(
    df: pd.DataFrame,
    feature_names: list,
    cat_indices: list,
    cat_levels: dict,
    reference_df: pd.DataFrame,
    logger=None
) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:821-828](../inference.py#L821-L828)
```python
shap_input = prepare_shap_input(
    df=INFERENCE_DATA,
    feature_names=feature_names,
    cat_indices=cat_indices,
    cat_levels=cat_levels,
    reference_df=reference_df,
    logger=logger
)
# INFERENCE_DATA has already had DateTime replaced with 0 (lines 789-791)
# This function ensures SHAP TreeExplainer compatibility
```

**Parameters:**
- `df` (pd.DataFrame): Inference data (after categorical encoding)
- `feature_names` (list): Feature column names
- `cat_indices` (list): Categorical metadata
- `cat_levels` (list): Category mapping
- `reference_df` (pd.DataFrame): Training schema
- `logger` (Logger, optional)

**Returns:**
- `pd.DataFrame`: SHAP-ready input (DateTime handled, dtypes validated)

**Algorithm (Lines 462-491):**
```python
# Step 1: Copy and select features (line 462)
shap_input = df[feature_names].copy()

# Step 2: Re-apply categorical dtypes (lines 464-469)
for idx, cats in zip(cat_indices, cat_levels):
    col = feature_names[idx]
    if col in shap_input.columns:
        shap_input[col] = shap_input[col].astype(
            pd.CategoricalDtype(categories=cats, ordered=False)
        )

# Step 3: Handle DateTime and align other dtypes (lines 471-483)
for col in shap_input.columns:
    if col == "DateTime":
        shap_input[col] = 0
        shap_input[col] = shap_input[col].astype("int64")
        continue

    if not pd.api.types.is_categorical_dtype(shap_input[col]):
        dtype = reference_df[col].dtype
        try:
            shap_input[col] = shap_input[col].astype(dtype)
        except Exception as e:
            if logger:
                logger.warning(f"Could not cast {col} to {dtype}: {e}")

# Step 4: Validate no object dtypes remain (lines 485-489)
bad_dtypes = [c for c in shap_input.columns if shap_input[c].dtype == object]
if bad_dtypes:
    if logger:
        logger.warning(f"SHAP input still has object dtypes: {bad_dtypes}")
    raise ValueError(f"SHAP input bad dtypes: {bad_dtypes}")

return shap_input
```

**Why Replace DateTime with 0? (Lines 472-474)**
- `shap.TreeExplainer` doesn't support `datetime64[ns]` dtype
- DateTime not a meaningful feature (model uses YEAR, MONTH, WEEK instead)
- Placeholder value (0) doesn't affect SHAP calculation
- LightGBM treats it as constant → no contribution to splits

**Critical Note:** In `inference.py`, DateTime is already replaced with 0 at lines 789-791 BEFORE this function is called. This function is defensive and ensures the replacement happened.

---

### `compute_shap_df()`

**Location:** [helpers/inference_data_preparation.py:495-537](helpers/inference_data_preparation.py#L495-L537)

**Purpose:** Compute SHAP values using TreeExplainer on LightGBM model

**Signature:**
```python
def compute_shap_df(
    model,              # LightGBM model (single quantile)
    shap_input: pd.DataFrame,
    logger=None
) -> tuple
```

**Called in inference.py:** [inference.py:836](../inference.py#L836) (inside loop for each quantile)
```python
# Lines 834-838
for q in quantiles:  # [0.20, 0.35, 0.50, ..., 0.975]
    rf_model = model_data.quantile_ensemble.models[q]
    shap_df_q, base_val = compute_shap_df(rf_model, shap_input, logger=None)
    shap_dfs.append(shap_df_q)
    base_values.append(base_val)
```

**Parameters:**
- `model`: Trained LightGBM model (e.g., `quantile_ensemble.models[0.50]`)
- `shap_input` (pd.DataFrame): SHAP-ready input from `prepare_shap_input()`
- `logger` (Logger, optional)

**Returns:**
- `tuple`: `(shap_df, base_value)`
  - `shap_df` (pd.DataFrame): SHAP values (shape: n_rows × n_features)
  - `base_value` (float): Expected prediction without features

**Algorithm (Lines 517-537):**
```python
import shap

# 1. Create TreeExplainer (line 517)
explainer = shap.TreeExplainer(model)

# 2. Compute SHAP values (line 518)
shap_values = explainer.shap_values(shap_input)  # Shape: (n_rows, n_features)

# 3. Handle list output (binary/multiclass models) (lines 519-520)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# 4. Convert to DataFrame (line 522)
shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{col}" for col in shap_input.columns])

# 5. Extract base value (lines 524-533)
base_value = explainer.expected_value
try:
    # Coerce list/ndarray → float (mean if vector)
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).mean())
    else:
        base_value = float(base_value)
except Exception:
    pass  # Leave as-is if conversion fails

return shap_df, base_value
```

**SHAP Values Interpretation:**
```python
# Example for single site
base_value = 67.2  # Expected prediction without features

shap_values = {
    'SHAP_Rainfall': 3.2,              # Rainfall increased prediction by 3.2
    'SHAP_3H': 25.3,                   # Recent rainfall (3H) increased by 25.3
    'SHAP_Site_Season_Average': 15.8,  # Historical trend increased by 15.8
    'SHAP_wind_shore_3h': -12.4,       # Wind-shore decreased by 12.4
    'SHAP_Shallowness': 2.1,           # Shallowness increased by 2.1
    ...
}

# Prediction = base_value + sum(shap_values)
prediction = 67.2 + 3.2 + 25.3 + 15.8 - 12.4 + 2.1 + ... ≈ 95.3
```

**Aggregation Across Quantiles (Lines 841-845 in inference.py):**
```python
# Stack SHAP values from all 12 quantile models
shap_stack = np.stack([df.values for df in shap_dfs], axis=2)  # (15, 60, 12)
median_shap_values = np.median(shap_stack, axis=2)             # (15, 60)

# Create final SHAP DataFrame
shap_df = pd.DataFrame(median_shap_values, columns=shap_dfs[0].columns)
shap_df["SHAP_base_value"] = np.median(base_values)
```

**Performance:**
- TreeExplainer: ~0.5-1s per quantile model
- 12 quantiles × 15 sites: 5-15s total
- Memory: ~100MB peak (15 sites × 60 features × 12 quantiles)

**Notes:**
- SHAP values are additive: `prediction = base_value + sum(shap_values)`
- Negative SHAP → feature decreased prediction
- Positive SHAP → feature increased prediction
- Magnitude → feature importance for this specific prediction

---

## Helper Functions

### `_make_run_id()` (in inference.py)

**Location:** [inference.py:38-64](../inference.py#L38-L64)

**Purpose:** Generate unique run identifier based on scheduled timestamp and environment

**Note:** This is NOT in `inference_data_preparation.py`, but is a related helper function in the main inference script.

**Signature:**
```python
def _make_run_id(sched_utc: dt.datetime) -> str
```

**Returns:**
- `str`: Format "YYYYMMDDThhmmZ-<8-character-hex>"
- Example: "20251128T0105Z-a3f2b9c1"

**Used in inference.py:** [inference.py:160](../inference.py#L160)
```python
RUN_ID = _make_run_id(scheduled_at_utc)
```

---

## Probability Calculations

### `prob_exceed_vectorised()`

**Location:** [helpers/inference_data_preparation.py:633-691](helpers/inference_data_preparation.py#L633-L691)

**Purpose:** Calculate P(Y > threshold) from quantile predictions using piecewise-linear CDF inversion

**Signature:**
```python
def prob_exceed_vectorised(
    Q: np.ndarray,          # Quantile predictions (n_obs, n_quantiles)
    p: np.ndarray,          # Quantile levels (n_quantiles,)
    y_threshold: float,     # Exceedance threshold (e.g., 280)
    debug: bool = False
) -> np.ndarray
```

**Parameters:**
- `Q` (np.ndarray): Quantile predictions, shape (n_obs, n_quantiles)
  - Example: `[[50, 75, 100, 150], [60, 90, 120, 180]]` (2 obs, 4 quantiles)
  - **Must be sorted ASCENDING** in quantile level (columns)
- `p` (np.ndarray): Quantile levels, shape (n_quantiles,)
  - Example: `[0.25, 0.50, 0.75, 0.95]`
  - **Must be sorted ASCENDING**
- `y_threshold` (float): Exceedance threshold (e.g., 280 CFU/100mL)
- `debug` (bool): If True, print debug information

**Returns:**
- `np.ndarray`: Exceedance probabilities, shape (n_obs,), values in [0, 1]

**Algorithm (Lines 657-679):**
```python
# Step 1: Locate segment k where Q[k] ≤ y_threshold < Q[k+1] (lines 658-659)
idx = (Q < y_threshold).sum(axis=1) - 1
idx = np.clip(idx, 0, n_q - 2)

# Step 2: Extract segment boundaries (lines 661-665)
row = np.arange(n_obs)
Q_lo = Q[row, idx]
Q_hi = Q[row, idx + 1]
p_lo = p[idx]
p_hi = p[idx + 1]

# Step 3: Linear interpolation of CDF at y_threshold (lines 668-669)
Fy = p_lo + (y_threshold - Q_lo) / (Q_hi - Q_lo) * (p_hi - p_lo)

# Step 4: Handle edge cases (lines 672-679)
below_min = y_threshold < Q[:, 0]
above_max = y_threshold > Q[:, -1]
Fy[below_min] = 0.0  # All mass above threshold
Fy[above_max] = 1.0  # All mass below threshold

degenerate = (Q_hi == Q_lo) | np.isnan(Fy)
Fy = np.where(degenerate, p_hi, Fy)

# Step 5: Return exceedance probability (line 691)
return 1.0 - Fy
```

**Example:**
```python
Q = np.array([[100, 150, 200, 250]])  # 1 obs, 4 quantiles
p = np.array([0.25, 0.50, 0.75, 0.95])
y_threshold = 220

# Find segment: 200 ≤ 220 < 250 (k=2, k+1=3)
# Interpolate CDF:
F(220) = 0.75 + (220-200)/(250-200) × (0.95-0.75)
       = 0.75 + (20/50) × 0.20
       = 0.75 + 0.08
       = 0.83

P(Y > 220) = 1 - 0.83 = 0.17  # 17% chance of exceeding 220
```

**Usage (Dashboard):**
```python
# This function is NOT called in inference.py
# Used in dashboard/analytics for converting quantile predictions to probabilities

quantile_cols = ['q_0.2', 'q_0.35', 'q_0.5', ..., 'q_0.975']
Q = pred_df[quantile_cols].values  # Shape: (15, 12)
p = np.array([0.20, 0.35, 0.50, ..., 0.975])

# P(Enterococci > 280)
prob_exceed_280 = prob_exceed_vectorised(Q, p, y_threshold=280)

# Add to DataFrame
pred_df['prob_exceed_280'] = prob_exceed_280
```

**Why Piecewise Linear?**
- Simple, interpretable CDF interpolation
- Computationally efficient (vectorized NumPy operations)
- Reasonable assumption: CDF smooth between quantiles
- Alternative: fit parametric distribution (e.g., lognormal) - more complex

**Performance:**
- Vectorized operations: <1ms for 15 observations × 12 quantiles

---

## Common Errors & Solutions

### Error 1: Timezone TypeError

**Error:**
```python
TypeError: Cannot compare tz-naive and tz-aware datetimes
```

**Cause:** Mixing tz-aware and tz-naive datetimes in comparisons

**Solution:**
```python
# Ensure all datetimes tz-naive (inference.py lines 81-82)
if getattr(DateTime_UTC, 'tzinfo', None) is not None:
    DateTime_UTC = pd.Timestamp(DateTime_UTC).tz_localize(None)
```

**Where it occurs:** When merging DataFrames with DateTime columns, or in Hilltop filtering

---

### Error 2: Categorical Mismatch

**Error:**
```python
ValueError: Unknown category 'Unknown' in column 'Harbour'
```

**Cause:** Inference data contains category not seen during training

**Automatic Handling:** `apply_model_categoricals()` (line 446) automatically coerces unknown categories to NaN

**Warning Logged:**
```python
logger.warning(f"{col}: Inference data has unseen categories: {obs_cats - model_cats}")
```

**Example:**
```python
df['Harbour'] = ['Akaroa', 'Lyttelton', 'Unknown']
# After apply_model_categoricals:
df['Harbour']  # → ['Akaroa', 'Lyttelton', NaN]
```

---

### Error 3: Empty Hilltop Response

**Warning:**
```python
fetch_enterococci_data_for_sites returned 0 rows
```

**Cause:**
- No samples collected in date range (normal for some sites/weeks)
- Hilltop server timeout/unreachable
- Network issues

**Handling:**
```python
# inference.py lines 751-756
if hist_df.empty:
    logger.warning("Hilltop fetch returned empty dataframe.")
    ev_hill("SOURCE_DEGRADED", source="hilltop",
            failure_kind="empty_payload")
# Seasonal features → NaN (handled by add_seasonal_features)
```

**Impact:** `Site_Season_Average` and `Site_Historical_Exceedance_Rate` will be NaN for all sites

---

### Error 4: SHAP TreeExplainer DateTime Error

**Error:**
```python
TypeError: TreeExplainer does not support datetime64
```

**Cause:** DateTime column not replaced with integer placeholder

**Solution (inference.py lines 789-791):**
```python
if "DateTime" in INFERENCE_DATA.columns:
    INFERENCE_DATA["DateTime"] = 0
    INFERENCE_DATA["DateTime"] = INFERENCE_DATA["DateTime"].astype("int64")
```

**Also handled in:** `prepare_shap_input()` (lines 472-474) as defensive measure

---

### Error 5: Schema Column Mismatch

**Error:**
```python
KeyError: "Required key column 'DateTime' missing from output_df columns"
```

**Cause:** Schema conformance dropped a required column, or column name changed

**Prevention:**
```python
# conform_features_to_training ensures all training columns present
# Even if missing in inference data, added as NaN (lines 409-412)
```

**Where to check:** Logs from `conform_features_to_training()` (lines 405-410)

---

### Error 6: NaN/Inf in Float Columns

**Error (Database Write):**
```python
pyodbc.DataError: Invalid parameter type
```

**Cause:** NaN or Inf values not sanitized before SQL insertion

**Solution (inference.py lines 1141-1210):**
```python
def _sql_param(v):
    # Handles NaN, Inf, None, pandas NA, etc.
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v

# Applied to all records before DB write (lines 1261-1265)
param_rows = []
for rec in records:
    clean = {col: _sql_param(val) for col, val in rec.items()}
    row = {ph: clean[col] for ph, col in zip(ph_names, cols)}
    param_rows.append(row)
```

---

## Function Call Graph

```
inference.py execution flow:

PHASE 1: INITIALIZATION (Lines 1-311)
  ├─ get_current_datetimes() [Line 68]
  │   └─ get_named_timezones()
  └─ create_site_metadata() [Line 310]
      └─ get_current_season()

PHASE 2: DATA ACQUISITION (Lines 312-668)
  # Weather/tide fetches (not in this module)

PHASE 3: FEATURE ENGINEERING (Lines 710-795)
  ├─ fetch_enterococci_data_for_sites() [Line 732]
  │   └─ _to_float_with_censor()
  ├─ add_seasonal_features() [Line 758]
  │   ├─ is_in_bathing_season()
  │   └─ assign_bathing_season() → get_current_season()
  ├─ conform_features_to_training() [Line 773]
  ├─ get_cat_info_from_model_txt() [Line 776]
  └─ apply_model_categoricals() [Line 780]

PHASE 4: PREDICTION & SHAP (Lines 796-880)
  ├─ prepare_shap_input() [Line 821]
  │   └─ apply_model_categoricals()
  └─ compute_shap_df() [Line 836, × 12 quantiles]

PHASE 5: DATABASE WRITE (Lines 882-1378)
  # Database operations (not in this module)
```

---

## Performance Characteristics

| Function | Typical Time | Bottleneck | Line Called |
|----------|--------------|-----------|-------------|
| `get_current_datetimes()` | <1ms | Negligible | 68 |
| `create_site_metadata()` | ~5ms | DataFrame creation | 310 |
| `fetch_enterococci_data_for_sites()` | 10-20s | HTTP requests (15 sites sequential) | 732 |
| `add_seasonal_features()` | ~50ms | Pandas groupby operations | 758 |
| `conform_features_to_training()` | ~20ms | Column operations | 773 |
| `get_cat_info_from_model_txt()` | ~5ms | File read + parsing | 776 |
| `apply_model_categoricals()` | ~10ms | Categorical dtype conversion | 780 |
| `prepare_shap_input()` | ~10ms | DataFrame copy + dtype ops | 821 |
| `compute_shap_df()` | 0.5-1s | TreeExplainer computation | 836 (×12) |

**Total Phase Breakdown:**
- Phase 1 (Timestamps + Site Metadata): ~10ms
- Phase 3 (Hilltop Fetch): 10-20s (dominates execution time)
- Phase 4 (Seasonal Features): ~50ms
- Phase 5 (Schema Conformance): ~20ms
- Phase 6 (Categorical Encoding): ~15ms
- Phase 7 (SHAP): 6-12s (12 quantile models)

**Total Data Preparation Time:** ~18-32 seconds (excluding weather/tide fetches)

---

**Related Documentation:**
- [02-RUNTIME-GUIDE.md](../docs/02-RUNTIME-GUIDE.md) - Complete inference.py execution flow
- [03-API-INTEGRATION.md](../docs/03-API-INTEGRATION.md) - Weather and tide API integration
- [WeatherAPI_Functions_GUIDE.md](WeatherAPI_Functions_GUIDE.md) - Weather API function reference
- [04-DATABASE-SCHEMA.md](../docs/04-DATABASE-SCHEMA.md) - Database schema and UPSERT operations
