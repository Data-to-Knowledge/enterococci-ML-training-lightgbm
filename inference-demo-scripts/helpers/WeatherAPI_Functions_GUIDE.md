# WeatherAPI_Functions.py -- Function Reference

**File:** `helpers/WeatherAPI_Functions.py`

Weather API integration (MetService, NIWA), tide data fetching from LINZ, and rolling feature computation for the inference pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Execution Flow](#execution-flow)
3. [Tide Data (LINZ)](#tide-data-linz)
4. [MetService Weather (Lyttelton)](#metservice-weather-lyttelton)
5. [NIWA Weather (Akaroa)](#niwa-weather-akaroa)
6. [Rolling Feature Computation](#rolling-feature-computation)
7. [Tide Feature Computation](#tide-feature-computation)
8. [Wind Vector Utilities](#wind-vector-utilities)
9. [Internal Helpers](#internal-helpers)
10. [Common Errors and Solutions](#common-errors-and-solutions)

---

## Overview

This module provides all weather and tide data integration for the hourly inference pipeline. Functions are called in a specific sequence to fetch, process, and engineer features from:

1. **Tide data:** LINZ static tide chart API
2. **Lyttelton weather:** MetService hourly observations (two stations with fallback)
3. **Akaroa weather:** NIWA Mintaka 10-minute observations (three products)
4. **Rolling features:** Rainfall sums and wind statistics over multiple time windows
5. **Tide features:** Tidal state, hours to high tide, height, and interaction term

All API credentials are loaded from environment variables -- see `inference-env.env`.

---

## Execution Flow

The functions in this module are called in the following sequence during `inference-demo.py` execution:

```
PHASE 1: TIDE DATA
  get_tide_data()                    --> tide predictions for Akaroa and Lyttelton (LINZ)

PHASE 2: WEATHER
  get_hourly_weather_data_Lyttelton()  --> hourly obs, last 3 days (MetService)
  get_10min_weather_data_Akaroa()      --> 10-min obs, last 3 days (NIWA Mintaka)

PHASE 3: ROLLING FEATURES
  add_rainfall_variables()           --> 3H, 6H, 12H, 24H, 48H, 72H, intensity, duration
  add_wind_variables()               --> wind_direction_{n}h, wind_speed_{n}h (x3 windows)

PHASE 4: TIDE FEATURES (per harbour)
  add_tide_variables()               --> tidal_state, hours_to_high_tide, height, product
```

---

## Tide Data (LINZ)

### `get_tide_data()`

Fetch tide predictions from the LINZ static tide chart CSV API. This is the sole tide data source in the demo (the production pipeline has a database-backed version; that has been removed here).

```python
def get_tide_data(
    site_list: list,   # e.g. ['Akaroa', 'Lyttelton']
    year: bool = False # If True, fetch a specific year; False uses current year
) -> pd.DataFrame
```

**Returns:** DataFrame with columns `[DateTime, Tidal_height, Tidal_state, Harbour]`

**Called in inference-demo.py:**
```python
tides_df = f.get_tide_data(site_list=['Akaroa', 'Lyttelton'], year=False)
tides_df["DateTime"] = pd.to_datetime(tides_df["DateTime"]).dt.tz_localize(None)
```

**LINZ API endpoint:**
```
https://static.charts.linz.govt.nz/tide-tables/maj-ports/csv/{site} {year}.csv
```

**Example URLs:**
```
https://static.charts.linz.govt.nz/tide-tables/maj-ports/csv/Akaroa 2026.csv
https://static.charts.linz.govt.nz/tide-tables/maj-ports/csv/Lyttelton 2026.csv
```

No authentication required. The function fetches the current year and next year for both harbours (four requests total).

**CSV format:**
```
3 header rows skipped
Columns: Day, DayName, Month, Year, Time1, TideHeight1, Time2, TideHeight2, Time3, TideHeight3, Time4, TideHeight4
Example: 1,Monday,Jan,2026,02:15,1.85,08:42,0.32,14:58,1.91,21:03,0.28
```

Each row contains up to four tide events per day (high and low tides alternate). The function unpivots these into one row per tide event, then:

1. Parses date and time components into a `DateTime` column
2. Localises to `Pacific/Auckland` (DST-aware) to handle NZ daylight saving
3. Resolves ambiguous times during DST transitions (April/October) using neighbouring tides
4. Determines tidal state (high or low) by comparing each tide to its neighbours
5. Returns the final DataFrame in tz-naive form

**Output example:**
```python
     DateTime             Tidal_height  Tidal_state  Harbour
0    2026-03-12 02:15:00  1.85          high         Lyttelton
1    2026-03-12 08:42:00  0.32          low          Lyttelton
2    2026-03-12 03:22:00  1.62          high         Akaroa
```

**Failure modes:**
1. Network timeout -- requests will raise and propagate up to the caller
2. No data for date range -- empty DataFrame passed to `add_tide_variables()`, which returns `("unknown", NaN, NaN, NaN)`

**Performance:** Typically 2--5 seconds (four HTTP requests + CSV parsing).

---

## MetService Weather (Lyttelton)

### `get_hourly_weather_data_Lyttelton()`

Fetch MetService hourly weather observations for Lyttelton with primary/backup station fallback.

```python
def get_hourly_weather_data_Lyttelton(
    datetime: dt.datetime,              # End of 24h fetch window (UTC)
    station_candidates=("93786", "93951"),
    log_event_fn=None,                  # Optional event logging callback (not used in demo)
    logger=None,                        # Python logger
    max_retries: int = 2,
    base_backoff_sec: float = 5.0,
    total_deadline_s: float = None      # Default 20s from env MET_TDEADLINE_S
) -> pd.DataFrame
```

**Returns:** DataFrame with columns `[DateTime, Rainfall, wind_direction, wind_speed, Ve, Vn, StationId]`

**Called in inference-demo.py:**
```python
for i in range(3):   # last 3 days
    single_date = DateTime_UTC + dt.timedelta(days=-i)
    df_part = f.get_hourly_weather_data_Lyttelton(
        datetime=single_date,
        station_candidates=tuple(station_candidates),
        logger=logger
    )
```

**Output schema:**

| Column | Type | Description |
|---|---|---|
| `DateTime` | datetime64 | Timestamp (UTC, tz-naive) |
| `Rainfall` | float | Hourly rainfall accumulation (mm) |
| `wind_direction` | float | Hourly average wind direction (degrees 0--360) |
| `wind_speed` | float | Hourly average wind speed (m/s) |
| `Ve` | float | East-west wind component (m/s) |
| `Vn` | float | North-south wind component (m/s) |
| `StationId` | str | Station that provided data ("93786" or "93951") |

**MetService API endpoint:**
```
https://api.metservice.com/observations/nz/1-minute/weatherStation/{station_id}/range/{start}/{end}?vars={var}
```

**Three variables fetched per station:**
- `rainfal_01hracc` -- hourly rainfall accumulation (mm)
- `winddir_01hravg` -- hourly average wind direction (degrees)
- `windspd_01hravg` -- hourly average wind speed (m/s)

Authentication is via the `apikey` header using `METSERVICE_KEY` from environment.

**Station fallback logic:**

1. Try the primary station (first in `station_candidates`, default `93786`)
2. If the primary returns data with gaps (NaN rainfall or missing wind), try the backup station (`93951`)
3. Merge primary and backup using the `MET_FALLBACK_MODE` strategy (default `"pair_wind"`)

**Merge mode `"pair_wind"` (default):**
- Rainfall: fill missing hours from backup independently
- Wind: fill missing direction/speed only if **both** components are missing in primary (wind direction and speed must come from the same station to give a correct vector)

**Example:**
```python
# Primary (93786):
#   Row 02:00 -- Rainfall=NaN, wind OK
#   Row 03:00 -- Rainfall=0.5, wind both NaN

# Backup (93951):
#   Row 02:00 -- Rainfall=1.2, wind OK
#   Row 03:00 -- Rainfall=0.3, wind OK

# After merge:
#   Row 02:00 -- Rainfall=1.2 (from backup), wind from primary
#   Row 03:00 -- Rainfall=0.5 (from primary), wind pair from backup
```

**Failure modes:**
1. API key missing or invalid -- returns empty DataFrame immediately
2. Primary station timeout -- tries backup automatically
3. Both stations timeout -- returns empty DataFrame; caller fills with NaN row
4. Variable not available -- column becomes NaN

**Performance:** Typically 8--15 seconds for 3 variables × 24 hours. With backup station: add 5--10 seconds.

---

### `_metservice_fetch_24h()` (internal)

Core MetService API fetch with retry logic for a single station. Called by `get_hourly_weather_data_Lyttelton()` for each station attempt.

The API returns 1-minute resolution observations; this function keeps only the on-the-hour rows (minute == 0) to produce hourly data. After fetching all three variables, it merges them on DateTime and computes Ve/Vn via `wind_vector_components()`.

**Retry behaviour:** Exponential backoff with jitter -- `delay = min(base * 2^(attempt-1), 15) * random(0.85, 1.15)`.

---

### `_merge_met_station_frames()` (internal)

Merges primary and backup MetService station DataFrames. Called by `get_hourly_weather_data_Lyttelton()` when the primary has gaps.

Supports three modes via the `MET_FALLBACK_MODE` environment variable:
- `"pair_wind"` (default) -- fill rainfall and wind pair independently
- `"hour"` -- replace entire hour from backup if any variable is missing
- `"off"` -- no merging, return primary unchanged

After merging wind direction/speed, Ve/Vn are recomputed from the merged values to ensure consistency.

---

## NIWA Weather (Akaroa)

### `get_10min_weather_data_Akaroa()`

Fetch NIWA Mintaka 10-minute weather observations for Akaroa. Returns raw 10-minute data; the caller aggregates to hourly.

```python
def get_10min_weather_data_Akaroa(
    datetime: dt.datetime,   # Target datetime (UTC)
    daytotal: int,           # Days to fetch backward (inference uses 3)
    log_event_fn=None,       # Optional event logging (not used in demo)
    logger=None,
    max_retries: int = 2,
    base_backoff_sec: float = 5.0,
    total_deadline_s: float = None  # Default 20s from env NIWA_TDEADLINE_S
) -> pd.DataFrame
```

**Returns:** DataFrame with columns `[DateTime, Rainfall, wind_direction, wind_speed, Ve, Vn]` at 10-minute resolution.

**Called in inference-demo.py:**
```python
df_akaroa_10min_weather = f.get_10min_weather_data_Akaroa(DateTime_UTC, 3, logger=logger)

# Aggregate 10-min → hourly
df_akaroa_weather = df_akaroa_10min_weather.groupby('DateTime', as_index=False).agg(
    Rainfall=('Rainfall', 'sum'),   # sum rainfall per hour
    Ve=('Ve', 'mean'),              # mean wind components
    Vn=('Vn', 'mean')
)
```

**NIWA Mintaka API endpoint:**
```
https://mintaka.niwa.co.nz/rest/api/V1.1/products/{product_id}/data?startDate={start}&endDate={end}&mode=LatestCommon
```

**Three products (Akaroa station):**

| Product ID | Variable | Unit |
|---|---|---|
| 55993775 | Rainfall | mm per 10 minutes |
| 55995169 | wind_direction | degrees (0--360) |
| 55995333 | wind_speed | m/s |

Authentication is HTTP Basic Auth using `NIWA_USERNAME` and `NIWA_KEY` from environment.

**`mode=LatestCommon`:** Returns the most recent complete dataset that is consistent across all products. Data is typically updated by ~9am NZST daily.

**After fetching all three products**, the function merges them on DateTime and computes Ve/Vn via `wind_vector_components()`.

**Failure modes:**
1. Missing credentials -- returns empty DataFrame immediately (guard at top of function)
2. HTTP 401 -- invalid credentials; returns empty DataFrame after retries
3. All products fail or timeout -- returns empty DataFrame
4. Zero-only guard -- handled by the caller in `inference-demo.py` (see below)

**Zero-only guard (in inference-demo.py):** After fetching, the caller checks whether all values in the last 12 hours are exactly zero. All-zero data is treated as a sensor outage (the station is recording but not measuring). If triggered, the data is replaced with NaN, which causes the health gate to exclude Akaroa sites.

**Performance:** Typically 5--10 seconds for 3 products × ~432 rows (3 days × 6 readings per hour).

---

## Rolling Feature Computation

### `add_rainfall_variables()`

Compute rolling rainfall accumulations and intensity statistics over multiple time windows.

```python
def add_rainfall_variables(df: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- `df` -- weather DataFrame with `Rainfall` and `DateTime` columns, sorted ascending by DateTime

**Called in inference-demo.py:**
```python
df_akaroa_weather    = f.add_rainfall_variables(df_akaroa_weather)
df_lyttelton_weather = f.add_rainfall_variables(df_lyttelton_weather)
```

**Features added:**

| Column | Description |
|---|---|
| `3H` | Cumulative rainfall in last 3 hours (mm) |
| `6H` | Cumulative rainfall in last 6 hours (mm) |
| `12H` | Cumulative rainfall in last 12 hours (mm) |
| `24H` | Cumulative rainfall in last 24 hours (mm) |
| `48H` | Cumulative rainfall in last 48 hours (mm) |
| `72H` | Cumulative rainfall in last 72 hours (mm) |
| `rain_intensity_48h` | Maximum hourly rainfall in last 48 hours (mm) |
| `rain_duration_48h` | Number of hours with rainfall > 0 in last 48 hours |

All rolling windows use `min_periods` equal to their window size, so the first few rows (where fewer than the full window is available) will be NaN.

**Why these features?**
- **3H, 6H, 12H:** Short-term rainfall driving immediate bacterial wash-off
- **24H, 48H, 72H:** Medium-term rainfall capturing soil saturation and catchment runoff
- **Intensity:** Distinguishes heavy downpours from light drizzle (same total rain, very different runoff)
- **Duration:** Captures persistent vs sporadic rainfall patterns

`3H` and `12H` are among the top five most important features in the model.

---

### `add_wind_variables()`

Compute rolling wind direction and speed statistics over a time window.

```python
def add_wind_variables(df: pd.DataFrame, hours: int) -> pd.DataFrame
```

**Parameters:**
- `df` -- weather DataFrame with `Ve`, `Vn`, `DateTime` columns, sorted ascending
- `hours` -- rolling window size (inference uses 3, 6, and 12)

**Called in inference-demo.py:**
```python
for i in [3, 6, 12]:
    df_akaroa_weather    = f.add_wind_variables(df_akaroa_weather, i)
    df_lyttelton_weather = f.add_wind_variables(df_lyttelton_weather, i)
```

**Features added per call:**

| Column | Description |
|---|---|
| `wind_direction_{n}h` | Mean wind direction over last n hours (degrees 0--360) |
| `wind_speed_{n}h` | Mean wind speed over last n hours (m/s) |

**Circular mean via vector components:**

Wind direction cannot be naively averaged -- the mean of 350° and 10° is ~0°, not 180°. This function computes the rolling mean of Ve and Vn separately, then converts back to direction and speed:

```python
Ve_mean = Ve.rolling(window=hours).mean()
Vn_mean = Vn.rolling(window=hours).mean()

wind_direction_nh = (360 + arctan2(Ve_mean, Vn_mean) * 180/pi) % 360
wind_speed_nh     = hypot(Ve_mean, Vn_mean)
```

**Why multiple windows?**
- **3h:** Recent wind -- current conditions and immediate effect on bacteria
- **6h:** Sustained wind patterns
- **12h:** Prevailing direction -- offshore wind over 12h can disperse bacteria

The wind direction and speed features are later combined with beach orientation to produce wind-shore interaction features (`wind_shore_3h`, etc.) in the `FeatureEngineer` class.

---

## Tide Feature Computation

### `add_tide_variables()`

Calculate tide features for a specific datetime and harbour from the LINZ tide predictions.

```python
def add_tide_variables(
    input_datetime: dt.datetime,  # Target datetime (tz-naive NZ local)
    harbour: str,                 # "Akaroa" or "Lyttelton"
    df: pd.DataFrame              # Tide predictions from get_tide_data()
) -> tuple
```

**Returns:** `(tidal_state, hours_to_high_tide, high_tide_height, high_tide_height_X_hours_to_high_tide)`

| Value | Type | Description |
|---|---|---|
| `tidal_state` | str | "ebbing" (past high tide) or "incoming" (before high tide) |
| `hours_to_high_tide` | float | Hours from input_datetime to closest high tide (positive = past, negative = future) |
| `high_tide_height` | float | Height of closest high tide (metres) |
| `high_tide_height_X_hours_to_high_tide` | float | Interaction term (product of the above two) |

Returns `("unknown", NaN, NaN, NaN)` if the tide DataFrame is empty or the datetime is invalid.

**Called in inference-demo.py:**
```python
ak_vals = f.add_tide_variables(TARGET_LOCAL_H, "Akaroa", tides_df)
akaroa_at_hour.loc[:, ["tidal_state", "hours_to_high_tide",
                        "high_tide_height", "high_tide_height_X_hours_to_high_tide"]] = [list(ak_vals)]
```

**Algorithm:**

1. Strip timezone from `input_datetime` if tz-aware
2. Filter tide DataFrame to the target harbour (case-insensitive)
3. Search for high tides within a ±36 hour window around the input datetime
4. Pick the closest high tide within that window; if none, fall back to the closest tide of any state; if still none, use the globally nearest tide for that harbour
5. Compute `hours_to_high_tide = (input_datetime - closest_tide_time).total_seconds() / 3600`
6. Set `tidal_state` to "ebbing" if `hours_to_high_tide > 0` (high tide was in the past), else "incoming"

**Example:**
```python
# Tide predictions (Lyttelton):
#   02:15 -- high 1.85m
#   08:42 -- low  0.32m
#   14:58 -- high 1.91m  <-- closest high to 13:00
#   21:03 -- low  0.28m

input_datetime = 2026-03-12 13:00:00

hours_to_high = (13:00 - 14:58).total_seconds() / 3600 = -1.97  # negative = future tide
tidal_state = "incoming"
high_tide_height = 1.91
product = -1.97 × 1.91 = -3.76

# Returns: ("incoming", -1.97, 1.91, -3.76)
```

**Why these features?**
- `hours_to_high_tide` captures tidal phase -- bacteria concentrate differently on the flood vs ebb
- `high_tide_height` distinguishes spring tides (>2m, stronger currents, more mixing) from neap tides (<1.5m, stagnant water)
- The interaction term lets the model learn nonlinear joint effects

`hours_to_high_tide` is in the top 10 most important features.

---

## Wind Vector Utilities

### `wind_vector_components()`

Convert wind direction and speed to Cartesian vector components.

```python
def wind_vector_components(
    direction: float,   # Wind direction (degrees, 0-360, from which wind blows)
    speed: float        # Wind speed (m/s)
) -> tuple[float, float]
```

**Returns:** `(Ve, Vn)` where Ve is the eastward component and Vn is the northward component.

```python
Ve = speed * sin(direction * pi/180)
Vn = speed * cos(direction * pi/180)
```

**Convention:** `direction` is the direction *from which* the wind is blowing (meteorological convention). So a north wind (direction=0°) blows *toward* the south, giving Ve=0, Vn=+5.

**Examples:**
```python
wind_vector_components(0,   5)  # North wind  --> Ve=0.0,  Vn=5.0
wind_vector_components(90,  5)  # East wind   --> Ve=5.0,  Vn=0.0
wind_vector_components(180, 5)  # South wind  --> Ve=0.0,  Vn=-5.0
wind_vector_components(270, 5)  # West wind   --> Ve=-5.0, Vn=0.0
```

**Why vector components?**
1. You cannot average directions directly (350° + 10° should give ~0°, not 180°)
2. Rolling means of Ve and Vn are mathematically correct; converting back gives circular mean direction
3. LightGBM handles numeric Ve/Vn more robustly than circular direction values

Used internally by `_metservice_fetch_24h()`, `get_10min_weather_data_Akaroa()`, and `_merge_met_station_frames()`.

---

## Internal Helpers

### `_emit()` (internal)

Thin wrapper for the optional event logging callback.

```python
def _emit(log_event_fn, event: str, **fields)
```

Calls `log_event_fn(event, **fields)` if `log_event_fn` is not None. In the demo, all callers pass `log_event_fn=None`, so this is a no-op. It exists to avoid repeating `if log_event_fn:` checks throughout the module.

---

### `_get_tide_state()` (internal)

Determines tidal state (high/low) for a list of tide heights by comparing each value to its neighbour. Used by `get_tide_data()` during LINZ CSV processing.

---

### `TRANSFORM_CATEGORICAL_FEATURE()`

Sets `pd.CategoricalDtype` on a fixed list of known categorical columns. **Not called by `inference-demo.py`** -- categorical encoding is handled instead via `apply_model_categoricals()` in `inference_data_preparation.py`, which reads the exact levels from the model file. This function is retained for reference.

---

## Common Errors and Solutions

### MetService key missing or invalid

```
SOURCE_DOWN: missing_api_key
```

The function returns an empty DataFrame immediately if `METSERVICE_KEY` is not set. The caller in `inference-demo.py` then fills with a NaN row, which causes Lyttelton to fail the health gate.

**Fix:** Set `METSERVICE_KEY` in `inference-env.env`.

---

### NIWA 401 Unauthorized

```
HTTP 401 from mintaka.niwa.co.nz
```

**Cause:** `NIWA_USERNAME` or `NIWA_KEY` in `inference-env.env` are invalid or expired.

**Result:** All three product fetches fail. The function returns an empty DataFrame. Akaroa sites are excluded by the health gate.

**Fix:** Check credentials with your team. Obtain current credentials from Nina or the project's password manager.

---

### Akaroa zero-only guard triggered

```
WARNING: Akaroa weather failed zeros-only guard; using NaNs.
```

**Cause:** All values (Rainfall, Ve, Vn) in the last 12 hours are exactly zero -- likely a sensor outage rather than genuine calm conditions.

**Result:** Akaroa weather data is replaced with NaN. Akaroa sites fail the health gate and are excluded from predictions.

**Fix:** No action needed. The guard is conservative by design. If you believe the zeros are genuine (e.g. truly calm weather with no rain), you can adjust the `NIWA_ZERO_GUARD_H` environment variable.

---

### Tides return "unknown" for all sites

```
tidal_state = "unknown", hours_to_high_tide = NaN
```

**Cause:** `get_tide_data()` returned an empty DataFrame (network failure), or the tide DataFrame does not contain data for the target hour within the ±36 hour window.

**Fix:** Check the log for errors during the LINZ fetch. If the LINZ API is temporarily unavailable, predictions will run with NaN tide features.

---

### Wind pair incomplete after merge

```
wind_direction NaN for some hours despite wind_speed present
```

**Cause:** MetService returned wind speed but not direction for those hours, and the backup station also had gaps. The `"pair_wind"` merge mode only fills wind if **both** components are available from the backup station.

**Result:** Those hours have NaN wind features. The rolling wind averages (`wind_speed_3h` etc.) may also be NaN if the gap overlaps the rolling window. This can trigger the health gate.

---

## Performance Summary

| Function | Typical time | Notes |
|---|---|---|
| `get_tide_data()` | 2--5 seconds | 4 HTTP requests (2 sites × 2 years) |
| `get_hourly_weather_data_Lyttelton()` | 8--15 seconds | 3 variables × 24h; ×3 calls for 3 days |
| `get_10min_weather_data_Akaroa()` | 5--10 seconds | 3 products × ~432 rows |
| `add_rainfall_variables()` | <100ms | Pandas rolling ops |
| `add_wind_variables()` | <50ms per window | Called 3 times (3h, 6h, 12h) |
| `add_tide_variables()` | <5ms | DataFrame filtering only |

---

## Related Documentation

- [inference-demo-documentation.md](../inference-demo-documentation.md) -- Full inference demo overview and setup
- [inference_data_preparation_GUIDE.md](inference_data_preparation_GUIDE.md) -- Feature engineering and SHAP function reference
