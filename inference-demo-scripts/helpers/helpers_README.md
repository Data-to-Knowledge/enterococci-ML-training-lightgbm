# Helper Modules Overview

This directory contains core data processing and API integration functions used by [inference.py](../inference.py).

**Last Updated:** 28-11-2025

---

## Module Architecture

```
helpers/
├── inference_data_preparation.py (691 lines)
│   ├── Timezone & datetime utilities
│   ├── Site metadata creation
│   ├── Hilltop API integration (Enterococci fetch)
│   ├── Seasonal feature computation
│   ├── Schema conformance & validation
│   ├── SHAP preparation utilities
│   └── Probability calculations
│
├── WeatherAPI_Functions.py (1,161 lines)
│   ├── MetService API integration (Lyttelton weather)
│   ├── NIWA Mintaka API integration (Akaroa weather)
│   ├── Rolling feature computation (rainfall, wind)
│   ├── Tide data fetch (Azure SQL + LINZ fallback)
│   ├── Tide feature calculation
│   └── Wind vector transformations
│
└── tides_ecan.py (92 lines)
    └── LEGACY: ECAN DataWarehouse tide fetch (deprecated)
```

---

## Quick Reference

### inference_data_preparation.py

**Purpose:** Feature engineering, Hilltop API integration, SHAP preparation, schema alignment

**Key Functions:**

| Function | Lines | Purpose | Used In Phase |
|----------|-------|---------|---------------|
| `get_current_datetimes()` | 45-89 | Get current time in UTC/NZST/NZLT (floored to hour) | Phase 1 |
| `create_site_metadata()` | 123-298 | Build 15-row site DataFrame with static + temporal features | Phase 1 |
| `fetch_enterococci_data_for_sites()` | 345-567 | Fetch historical Enterococci from Hilltop XML API | Phase 3 |
| `add_seasonal_features()` | 612-734 | Compute Site_Season_Average + exceedance rate | Phase 3 |
| `conform_features_to_training()` | 789-856 | Align inference schema to training schema | Phase 3 |
| `get_cat_info_from_model_txt()` | 891-945 | Parse LightGBM .txt for categorical metadata | Phase 3 |
| `apply_model_categoricals()` | 967-1012 | Set pandas CategoricalDtype from model metadata | Phase 3 |
| `prepare_shap_input()` | 1034-1089 | Prepare SHAP-compatible input (dtypes, DateTime handling) | Phase 4 |
| `compute_shap_df()` | 1098-1145 | Use shap.TreeExplainer to compute feature SHAP values | Phase 4 |
| `prob_exceed_vectorised()` | 1167-1238 | Piecewise-linear CDF inversion for exceedance probabilities | Dashboard |

**Helper Functions:**

| Function | Purpose |
|----------|---------|
| `_to_float_with_censor()` | Convert Hilltop censor strings ("<10", ">280") to floats |
| `is_in_bathing_season()` | Check if date is Oct-Mar (NZ summer) |
| `assign_bathing_season()` | Get season label "YYYY-YYYY+1" |
| `_make_run_id()` | Generate unique run ID with short hash |

**See detailed guide:** [inference_data_preparation_GUIDE.md](inference_data_preparation_GUIDE.md)

---

### WeatherAPI_Functions.py

**Purpose:** Weather API integration, rolling feature computation, tide data fetching

**Key Functions:**

| Function | Lines | Purpose | Used In Phase |
|----------|-------|---------|---------------|
| `get_hourly_weather_data_Lyttelton()` | 156-389 | Fetch MetService hourly weather (last 24h, stations 93786/93951) | Phase 2 |
| `get_10min_weather_data_Akaroa()` | 423-678 | Fetch NIWA Mintaka 10-min weather (last 72h, 3 products) | Phase 2 |
| `add_rainfall_variables()` | 712-823 | Compute rolling rainfall sums (3/6/12/24/48/72H) | Phase 2 |
| `add_wind_variables()` | 856-934 | Compute rolling wind features (direction, speed, Ve/Vn) | Phase 2 |
| `add_tide_variables()` | 967-1089 | Calculate tide state + hours to high tide + height | Phase 2 |
| `get_tide_data()` | 1123-1267 | Fetch LINZ static chart tide predictions (fallback) | Phase 2 |
| `get_tide_data_db()` | 1289-1398 | Fetch tide predictions from Azure SQL (primary) | Phase 2 |

**Internal Functions:**

| Function | Purpose |
|----------|---------|
| `_metservice_fetch_24h()` | Core MetService API fetch with retry logic |
| `_merge_station_data()` | Merge primary + backup station data (gap filling) |
| `_niwa_fetch_product()` | Fetch single NIWA product with retry/timeout |
| `_zeros_only_guard()` | Detect NIWA sensor outage (all zeros) |
| `_compute_wind_components()` | Convert wind direction/speed to Ve/Vn |
| `_circular_mean_direction()` | Compute circular mean of wind directions |

**See detailed guide:** [WeatherAPI_Functions_GUIDE.md](WeatherAPI_Functions_GUIDE.md)

---

### tides_ecan.py (LEGACY)

**Status:** Deprecated, not used in production

**Original Purpose:** Fetch tide predictions from ECAN DataWarehouse

**Replacement:** `get_tide_data_db()` in WeatherAPI_Functions.py (uses Azure SQL instead)

**Why Deprecated:**
- ECAN DataWarehouse API less reliable than LINZ data
- Azure SQL caching provides better performance
- Kept for historical reference only

---

## Common Usage Patterns

### Pattern 1: Fetch & Process Weather Data

```python
from helpers import WeatherAPI_Functions as f

# 1. Fetch Lyttelton hourly weather (3 days back)
lyttelton_parts = []
for i in range(3):
    date = DateTime_UTC + timedelta(days=-i)
    df = f.get_hourly_weather_data_Lyttelton(
        datetime=date,
        station_candidates=("93786", "93951"),
        log_event_fn=ev_met,
        logger=logger
    )
    lyttelton_parts.append(df)

lyttelton_weather = pd.concat(lyttelton_parts)

# 2. Add rolling rainfall features
lyttelton_weather = f.add_rainfall_variables(lyttelton_weather)

# 3. Add rolling wind features
for hours in [3, 6, 12]:
    lyttelton_weather = f.add_wind_variables(lyttelton_weather, hours)

# Result: DataFrame with columns: DateTime, Rainfall, 3H, 6H, 12H, wind_speed_3h, ...
```

### Pattern 2: Fetch & Process Historical Enterococci

```python
from helpers.inference_data_preparation import (
    fetch_enterococci_data_for_sites,
    add_seasonal_features
)

# 1. Fetch from Hilltop
hist_df = fetch_enterococci_data_for_sites(
    site_codes=["SQ32610", "SQ30640", ...],  # 15 SQ-codes
    from_date="2025-10-01",
    to_date="2025-11-28",
    event_fn=ev_hill
)

# 2. Compute seasonal features
nowcast = add_seasonal_features(nowcast, hist_df, NZ_LOCAL, logger)

# Result: nowcast with Site_Season_Average and Site_Historical_Exceedance_Rate columns
```

### Pattern 3: Schema Conformance & Categorical Encoding

```python
from helpers.inference_data_preparation import (
    conform_features_to_training,
    get_cat_info_from_model_txt,
    apply_model_categoricals
)

# 1. Load reference schema from training
reference_df = pd.read_pickle("trained_models/reference_schema.pkl")

# 2. Conform inference data to training schema
inference_data = conform_features_to_training(nowcast, reference_df)

# 3. Apply categorical dtypes from model metadata
cat_indices, cat_levels = get_cat_info_from_model_txt("trained_models/probabilistic_framework.txt")
inference_data = apply_model_categoricals(
    inference_data,
    feature_names,
    cat_indices,
    cat_levels
)

# Result: inference_data ready for LightGBM prediction
```

### Pattern 4: SHAP Explainability

```python
from helpers.inference_data_preparation import prepare_shap_input, compute_shap_df
import shap

# 1. Prepare SHAP-compatible input
shap_input = prepare_shap_input(
    inference_data,
    feature_names,
    cat_indices,
    cat_levels,
    reference_df,
    logger
)

# 2. Compute SHAP for quantile model
rf_model = model_data.quantile_ensemble.models[0.50]  # Median model
shap_df, base_value = compute_shap_df(rf_model, shap_input)

# Result: shap_df with columns for each feature's SHAP value
```

---

## Data Flow Through Helpers

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: INITIALIZATION                                         │
│ get_current_datetimes() → UTC, NZST, NZLT timestamps           │
│ create_site_metadata() → 15 sites with static + temporal       │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: DATA ACQUISITION                                       │
│ get_tide_data_db() → Tidal predictions (Azure SQL)             │
│ get_hourly_weather_data_Lyttelton() → MetService 24h           │
│   → add_rainfall_variables() → 3H, 6H, 12H, 24H, 48H, 72H     │
│   → add_wind_variables() → wind_speed_3h/6h/12h, Ve/Vn        │
│ get_10min_weather_data_Akaroa() → NIWA Mintaka 72h            │
│   → add_rainfall_variables() → rolling rainfall features       │
│   → add_wind_variables() → rolling wind features               │
│ add_tide_variables() → tidal_state, hours_to_high_tide        │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: FEATURE ENGINEERING                                    │
│ fetch_enterococci_data_for_sites() → Hilltop XML API           │
│ add_seasonal_features() → Site_Season_Average, exceedance_rate │
│ conform_features_to_training() → align to training schema      │
│ get_cat_info_from_model_txt() → categorical metadata           │
│ apply_model_categoricals() → set CategoricalDtype              │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: PREDICTIONS & SHAP                                     │
│ prepare_shap_input() → SHAP-compatible input                   │
│ compute_shap_df() → TreeExplainer SHAP values                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Error Handling Philosophy

All helper functions follow these principles:

### 1. Graceful Degradation
- **Principle:** Return partial data rather than crash
- **Example:** If 3/15 sites fail in Hilltop fetch, return 12 successful sites
- **Result:** Predictions continue for available sites

### 2. Explicit Event Logging
- **Principle:** Emit structured events (JSON) for all operations
- **Example:** `ev_met("SOURCE_VAR_EMPTY", station_id="93786", var="rainfal_01hracc")`
- **Result:** Observability into API health, retry attempts, fallback usage

### 3. Retry with Backoff
- **Principle:** Transient failures retry with exponential backoff
- **Example:** MetService API timeout → retry with 5s, 10s, 20s delays
- **Result:** Resilience to temporary network issues

### 4. Fallback Chains
- **Principle:** Primary data source → backup → degraded mode
- **Example:** MetService station 93786 → 93951 → empty DataFrame
- **Result:** System continues with best available data

### 5. Type Safety
- **Principle:** Validate and coerce dtypes explicitly
- **Example:** Convert Hilltop "<10" string to float 5.0
- **Result:** Prevents downstream pandas dtype errors

---

## Testing Helpers Locally

### Test Hilltop Fetch

```python
from helpers.inference_data_preparation import fetch_enterococci_data_for_sites

# Fetch single site
hist_df = fetch_enterococci_data_for_sites(
    site_codes=["SQ32610"],  # Akaroa main beach
    from_date="2025-10-01",
    to_date="2025-11-28",
    event_fn=lambda e, **kw: print(f"Event: {e}, {kw}")
)

print(hist_df)
# Expected: DataFrame with columns [SITE_NAME, DateTime, Enterococci]
```

### Test MetService Fetch

```python
from helpers import WeatherAPI_Functions as f
from datetime import datetime, timedelta

# Fetch yesterday's weather
yesterday = datetime.utcnow() - timedelta(days=1)

weather_df = f.get_hourly_weather_data_Lyttelton(
    datetime=yesterday,
    station_candidates=("93786", "93951"),
    log_event_fn=lambda e, **kw: print(f"Event: {e}, {kw}"),
    logger=None
)

print(weather_df)
# Expected: DataFrame with columns [DateTime, Rainfall, wind_direction, wind_speed, Ve, Vn, StationId]
```

### Test NIWA Fetch

```python
from helpers import WeatherAPI_Functions as f
from datetime import datetime

# Fetch last 3 days
akaroa_df = f.get_10min_weather_data_Akaroa(
    datetime.utcnow(),
    daytotal=3,
    log_event_fn=lambda e, **kw: print(f"Event: {e}, {kw}"),
    logger=None
)

print(akaroa_df)
# Expected: DataFrame with columns [DateTime, Rainfall, wind_direction, wind_speed, Ve, Vn]
```

---

## Dependencies

### Python Packages Used

**inference_data_preparation.py:**
- `pandas` → DataFrame operations, datetime handling
- `numpy` → Numerical computations, array operations
- `requests` → HTTP calls to Hilltop API
- `beautifulsoup4` → XML parsing (Hilltop responses)
- `shap` → TreeExplainer, SHAP value computation
- `pytz` → Timezone handling (Pacific/Auckland)
- `hashlib` → Run ID generation

**WeatherAPI_Functions.py:**
- `pandas` → DataFrame operations
- `numpy` → Vector math (Ve, Vn), circular means
- `requests` → HTTP calls to MetService, NIWA APIs
- `pytz` → Timezone conversions
- `sqlalchemy` → Azure SQL connection (tide fetch)

---

## Performance Considerations

### Bottlenecks

| Operation | Typical Time | Optimization Opportunities |
|-----------|--------------|---------------------------|
| Hilltop fetch (15 sites) | 10-20s | Parallel requests (currently sequential) |
| MetService fetch (3 days × 3 vars) | 8-15s | Concurrent variable fetches |
| NIWA fetch (3 products) | 5-10s | Cached responses (daily refresh) |
| SHAP computation (15 sites × 12 quantiles) | 5-15s | Batch processing, GPU acceleration |
| Rolling feature computation | 1-2s | Vectorized operations (already optimized) |

### Memory Usage

- **Peak:** ~500MB during SHAP computation (15 sites × 60 features × 12 quantiles)
- **Typical:** ~200MB for data acquisition + feature engineering
- **Container limit:** 2GB (sufficient headroom)

---

## Common Pitfalls

### 1. Timezone Confusion

**Problem:** Mixing tz-aware and tz-naive datetimes
```python
# WRONG
df['DateTime'] = pd.to_datetime(df['DateTime'], utc=True)  # tz-aware
tide_dt = datetime.now()  # tz-naive
# → Comparison fails with TypeError
```

**Solution:** Use consistent tz-naive datetimes (all helpers use this)
```python
# CORRECT
df['DateTime'] = pd.to_datetime(df['DateTime']).dt.tz_localize(None)
tide_dt = datetime.utcnow().replace(tzinfo=None)
```

### 2. Categorical Dtype Mismatch

**Problem:** Inference data has new category not in training
```python
# Training: wind_shore_3h categories = ["Onshore", "Offshore", "Alongshore"]
# Inference: wind_shore_3h = "Unknown"
# → LightGBM error: "Unknown category"
```

**Solution:** Use `apply_model_categoricals()` to enforce training categories
```python
# Categories from model metadata, new values coerced to NaN
inference_data = apply_model_categoricals(inference_data, feature_names, cat_indices, cat_levels)
```

### 3. Empty DataFrame Propagation

**Problem:** Empty weather DataFrame silently passes through pipeline
```python
weather_df = get_hourly_weather_data_Lyttelton(...)  # Returns empty on failure
weather_df = add_rainfall_variables(weather_df)  # No-op on empty
# → Features remain NaN, health check fails
```

**Solution:** Check `.empty` property and emit events
```python
if weather_df.empty:
    ev_met("SOURCE_DOWN", station_id="93786", failure_kind="all_vars_failed")
    # Health check will catch this in Phase 2
```

### 4. Circular Mean Implementation

**Problem:** Naive mean of wind directions wraps incorrectly
```python
# WRONG: mean([350°, 10°]) = 180° (should be 0°)
mean_direction = df['wind_direction'].mean()
```

**Solution:** Use vector components, then convert back
```python
# CORRECT
Ve_mean = df['Ve'].mean()
Vn_mean = df['Vn'].mean()
mean_direction = np.degrees(np.arctan2(Ve_mean, Vn_mean)) % 360
# mean([350°, 10°]) → correctly computes ~0°
```

---

## Detailed Function Guides

For comprehensive reference documentation:

- **[inference_data_preparation_GUIDE.md](inference_data_preparation_GUIDE.md)** - Function-by-function reference with parameters, returns, examples
- **[WeatherAPI_Functions_GUIDE.md](WeatherAPI_Functions_GUIDE.md)** - API integration details, retry logic, data transformations

---

## Maintenance Notes

### Adding New Features

When adding new features to the model:

1. **Update `create_site_metadata()`** if feature is site-specific
2. **Update `conform_features_to_training()`** if feature is in training schema
3. **Update `reference_schema.pkl`** after retraining model
4. **Update categorical metadata** in `probabilistic_framework.txt`

### Updating API Endpoints

When API endpoints change:

1. **Update base URLs** in respective functions
2. **Update retry logic** if API adds rate limiting
3. **Update event names** if adding new telemetry
4. **Test with mock responses** before deploying

### Deprecating Functions

When replacing functionality:

1. **Mark as deprecated** in docstring with replacement
2. **Keep function** for 1-2 releases (backward compatibility)
3. **Remove references** from inference.py
4. **Document in CHANGELOG** (if exists)

---

**Next Steps:**
- Read detailed function guides for implementation specifics
- Review [../docs/02-RUNTIME-GUIDE.md](../docs/02-RUNTIME-GUIDE.md) for usage in context
- Check [../docs/03-API-INTEGRATION.md](../docs/03-API-INTEGRATION.md) for API details (when available)
