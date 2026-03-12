# WeatherAPI_Functions.py Function Reference

**File:** [helpers/WeatherAPI_Functions.py](helpers/WeatherAPI_Functions.py)
**Lines:** 1,162 total
**Purpose:** Weather API integration (MetService, NIWA), tide data fetching, rolling feature computation for inference pipeline

**Last Updated:** 2025-11-30

---

## Table of Contents

1. [Overview](#overview)
2. [Execution Flow in inference.py](#execution-flow-in-inferencepy)
3. [Phase 1: Tide Data Acquisition](#phase-1-tide-data-acquisition)
4. [Phase 2: MetService Weather (Lyttelton)](#phase-2-metservice-weather-lyttelton)
5. [Phase 3: NIWA Weather (Akaroa)](#phase-3-niwa-weather-akaroa)
6. [Phase 4: Rolling Feature Computation](#phase-4-rolling-feature-computation)
7. [Phase 5: Tide Feature Computation](#phase-5-tide-feature-computation)
8. [Wind Vector Utilities](#wind-vector-utilities)
9. [Internal Helper Functions](#internal-helper-functions)
10. [Common Errors & Solutions](#common-errors--solutions)

---

## Overview

This module provides all weather and tide data integration for the hourly inference pipeline. Functions are called in a specific sequence to fetch, process, and engineer features from:

1. **Tide Data:** Azure SQL Server (primary) or LINZ static API (fallback)
2. **Lyttelton Weather:** MetService hourly observations (2 stations with fallback)
3. **Akaroa Weather:** NIWA Mintaka 10-minute observations (3 products)
4. **Rolling Features:** Rainfall sums and wind statistics over time windows
5. **Tide Features:** Tidal state, hours to high tide, height, interaction terms

**Critical Design Principle:** All functions return tz-naive datetimes for compatibility with LightGBM models and Azure SQL Server storage.

---

## Execution Flow in inference.py

The functions in this module are called in the following sequence during `inference.py` execution:

```python
# PHASE 1: TIDE DATA ACQUISITION (Lines 314-420)
# Attempt 1: Fetch from Azure SQL (Lines 339-388)
tides_df = f.get_tide_data_db(
    site_list=['Akaroa', 'Lyttelton'],
    start_dt=tide_start,  # NZ_LOCAL - 3 days (tz-naive)
    end_dt=tide_end,      # NZ_LOCAL + 14 days (tz-naive)
    schema="dbo",
    table="linz_tides",
    attach_fixed_nzst=False
)
# → Returns: DataFrame[DateTime, Tidal_height, Tidal_state, Harbour]

# Attempt 2: Fallback to LINZ static API if DB empty (Lines 364-419)
if tides_df.empty:
    tides_df = f.get_tide_data(
        site_list=['Akaroa', 'Lyttelton'],
        year=False
    )
    # → Returns: Same schema as get_tide_data_db()

# PHASE 2: METSERVICE WEATHER - LYTTELTON (Lines 436-487)
lyttelton_weather_parts = []

for i in range(3):  # Fetch last 3 days (72 hours)
    single_date = DateTime_UTC + dt.timedelta(days=-i)

    df_part = f.get_hourly_weather_data_Lyttelton(
        datetime=single_date,
        station_candidates=tuple(station_candidates),  # ("93786", "93951")
        log_event_fn=log_event,
        logger=logger
    )
    # → Returns: DataFrame[DateTime, Rainfall, wind_direction, wind_speed, Ve, Vn, StationId]
    # → Typical: 24 rows per day

    if df_part is not None:
        lyttelton_weather_parts.append(df_part)

# Concatenate all 3 days (Line 453)
df_lyttelton_weather = pd.concat(lyttelton_weather_parts, ignore_index=True)
# → Result: ~72 rows (3 days × 24 hours)

# PHASE 3: NIWA WEATHER - AKAROA (Lines 489-536)
df_akaroa_10min_weather = f.get_10min_weather_data_Akaroa(
    DateTime_UTC,
    3,  # daytotal: 3 days backward
    log_event_fn=log_event,
    logger=logger
)
# → Returns: DataFrame[DateTime, Rainfall, wind_direction, wind_speed, Ve, Vn]
# → Typical: ~432 rows (3 days × 24 hours × 6 per hour)

# Aggregate 10-min → hourly (Lines 530-534)
df_akaroa_weather = df_akaroa_10min_weather.groupby('DateTime', as_index=False).agg(
    Rainfall=('Rainfall', 'sum'),   # Sum rainfall per hour
    Ve=('Ve', 'mean'),              # Mean wind components
    Vn=('Vn', 'mean')
)
# → Result: ~72 rows (3 days × 24 hours)

# PHASE 4: ROLLING FEATURE COMPUTATION (Lines 571-595)
# Add rainfall features (3H, 6H, 12H, 24H, 48H, 72H, intensity, duration)
df_akaroa_weather = f.add_rainfall_variables(df_akaroa_weather)      # Line 591
df_lyttelton_weather = f.add_rainfall_variables(df_lyttelton_weather)  # Line 592

# Add wind features for multiple time windows (Lines 593-595)
for i in [3, 6, 12]:
    df_akaroa_weather = f.add_wind_variables(df_akaroa_weather, i)
    df_lyttelton_weather = f.add_wind_variables(df_lyttelton_weather, i)

# PHASE 5: TIDE FEATURE COMPUTATION (Lines 650-659)
# Add tide features for Akaroa (Lines 651-654)
_ak_cols = ["tidal_state", "hours_to_high_tide", "high_tide_height",
            "high_tide_height_X_hours_to_high_tide"]
ak_vals = f.add_tide_variables(TARGET_LOCAL_H, "Akaroa", tides_df)
akaroa_at_hour.loc[:, _ak_cols] = [list(ak_vals)]

# Add tide features for Lyttelton (Lines 656-659)
_lt_cols = ["tidal_state", "hours_to_high_tide", "high_tide_height",
            "high_tide_height_X_hours_to_high_tide"]
lt_vals = f.add_tide_variables(TARGET_LOCAL_H, "Lyttelton", tides_df)
lyttelton_at_hour.loc[:, _lt_cols] = [list(lt_vals)]
```

---

## Phase 1: Tide Data Acquisition

### `get_tide_data_db()`

**Location:** [helpers/WeatherAPI_Functions.py:1087-1161](helpers/WeatherAPI_Functions.py#L1087-L1161)

**Purpose:** Fetch tide predictions from Azure SQL Server (primary data source for tides)

**Signature:**
```python
def get_tide_data_db(
    site_list: list,                # ['Akaroa', 'Lyttelton']
    start_dt: dt.datetime = None,   # Start of window (tz-naive)
    end_dt: dt.datetime = None,     # End of window (tz-naive)
    schema: str = None,             # Default: "dbo"
    table: str = None,              # Default: "linz_tides"
    attach_fixed_nzst: bool = False # Attach +12:00 offset (default False)
) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:341-348](../inference.py#L341-L348)
```python
# Calculate time window (lines 325-328)
BACK_DAYS = int(os.getenv("TIDES_BACK_DAYS", 3))   # Default 3
FWD_DAYS = int(os.getenv("TIDES_FWD_DAYS", 14))    # Default 14
tide_start = now_local - dt.timedelta(days=BACK_DAYS)
tide_end = now_local + dt.timedelta(days=FWD_DAYS)

# Fetch from database (lines 341-348)
tides_df = f.get_tide_data_db(
    site_list=site_list,           # ['Akaroa', 'Lyttelton']
    start_dt=tide_start,           # now - 3 days (tz-naive)
    end_dt=tide_end,               # now + 14 days (tz-naive)
    schema=os.getenv("TIDE_SCHEMA", "dbo"),
    table=os.getenv("TIDE_TABLE", "linz_tides"),
    attach_fixed_nzst=False
)
```

**Parameters:**
- `site_list` (list): Harbour names `['Akaroa', 'Lyttelton']`
- `start_dt` (datetime, optional): Start of fetch window (tz-naive NZLT)
  - Default: `now() - 180 days` (line 1123)
- `end_dt` (datetime, optional): End of fetch window (tz-naive NZLT)
  - Default: `now() + 365 days` (line 1125)
- `schema` (str): Database schema (default from env `TIDE_SCHEMA` or "dbo")
- `table` (str): Table name (default from env `TIDE_TABLE` or "linz_tides")
- `attach_fixed_nzst` (bool): Re-attach fixed +12:00 offset (default False)

**Returns:**
- `pd.DataFrame`: Columns `[DateTime, Tidal_height, Tidal_state, Harbour]`

**Output Schema:**
```python
Columns:
  - DateTime (datetime64[ns]): Timestamp of tide event (tz-naive NZLT)
  - Tidal_height (float): Tide height in meters
  - Tidal_state (str): "high" or "low"
  - Harbour (str): "Akaroa" or "Lyttelton"

Rows: Variable (typically ~144 rows for 17-day window)
```

**SQL Query (Lines 1137-1147):**
```python
# Validated identifiers (line 1119)
_validate_ident(schema); _validate_ident(table)

# Build query with bracket escaping (lines 1133-1134)
def _br(idn: str) -> str:
    return "[" + idn.replace("]", "]]") + "]"

# Query (lines 1137-1147)
q = text(f"""
    SELECT
        {_br('harbour')}         AS "Harbour",
        {_br('datetime_nzst')}   AS "DateTime",
        {_br('tidal_height_m')}  AS "Tidal_height",
        {_br('tidal_state')}     AS "Tidal_state"
    FROM {_br(schema)}.{_br(table)}
    WHERE {_br('harbour')} IN :sites
      AND {_br('datetime_nzst')} BETWEEN :start_dt AND :end_dt
    ORDER BY {_br('harbour')}, {_br('datetime_nzst')}
""").bindparams(bindparam("sites", expanding=True))
```

**Database Connection (Lines 1037-1079):**
```python
# Build engine from environment variables
def _db_engine_from_env():
    db_url = os.getenv("SQL_URL") or os.getenv("DATABASE_URL")

    # Sanitize: ignore if not using odbc_connect= (lines 1045-1046)
    if db_url and "odbc_connect=" not in db_url:
        db_url = None

    if not db_url:
        # Build from components (lines 1049-1077)
        host = os.getenv("SQL_HOST")
        db = os.getenv("SQL_DB")
        user = os.getenv("SQL_USER")
        pw = os.getenv("SQL_PASSWORD")
        driver = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")

        odbc_str = (
            f"Driver={{{driver}}};"
            f"Server=tcp:{host},1433;"
            f"Database={db};"
            f"Uid={user};"
            f"Pwd={pw};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
        )

        db_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc_str)

    return create_engine(db_url, pool_pre_ping=True, future=True, fast_executemany=True)
```

**Example Output:**
```python
     DateTime             Tidal_height  Tidal_state  Harbour
0    2025-11-25 02:15:00  1.85          high         Lyttelton
1    2025-11-25 08:42:00  0.32          low          Lyttelton
2    2025-11-25 14:58:00  1.91          high         Lyttelton
3    2025-11-25 21:03:00  0.28          low          Lyttelton
...
100  2025-11-25 03:22:00  1.62          high         Akaroa
101  2025-11-25 09:35:00  0.45          low          Akaroa
...
```

**Data Source:**
- **Primary:** Azure SQL table `dbo.linz_tides`
- **Populated by:** Separate ETL job (fetches from LINZ Data Service)
- **Coverage:** ~10 years historical + forecast

**Performance:**
- Typical: 0.5-2 seconds for 17-day window (144 rows)
- Database indexed on (harbour, datetime_nzst)

**Failure Modes:**
1. **Connection timeout** → Return empty DataFrame (triggers fallback)
2. **Table not found** → SQLAlchemy error, return empty DataFrame
3. **No data in range** → Return empty DataFrame

---

### `get_tide_data()` (LINZ Static API Fallback)

**Location:** [helpers/WeatherAPI_Functions.py:898-1031](helpers/WeatherAPI_Functions.py#L898-L1031)

**Purpose:** Fetch tide predictions from LINZ static CSV API (fallback when database unavailable)

**Signature:**
```python
def get_tide_data(
    site_list: list,        # ['Akaroa', 'Lyttelton']
    year: bool = False      # If True, fetch full year; else ~17 days
) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:366](../inference.py#L366) (fallback path)
```python
# Fallback if database returns empty (lines 364-381)
if tides_df.empty:
    logger.warning("Tide DB returned empty DataFrame, falling back to LINZ")

    tides_df = f.get_tide_data(site_list=site_list, year=False)
    tides_df["DateTime"] = pd.to_datetime(tides_df["DateTime"]).dt.tz_localize(None)

    # Emit fallback event (lines 377-381)
    ev_tides("SOURCE_OK",
            source="linz.tides",
            endpoint_host="static.charts.linz.govt.nz",
            rows=int(len(tides_df)),
            elapsed_ms=elapsed_ms_fb)
```

**Parameters:**
- `site_list` (list): Harbour names `['Akaroa', 'Lyttelton']`
- `year` (bool): Fetch full year (True) or 17-day window (False, default in inference.py)

**Returns:**
- `pd.DataFrame`: Same schema as `get_tide_data_db()`

**LINZ Static CSV API Details:**

**Endpoint (Line 922):**
```
https://static.charts.linz.govt.nz/tide-tables/maj-ports/csv/{harbour} {year}.csv
```

**Example Request:**
```
https://static.charts.linz.govt.nz/tide-tables/maj-ports/csv/Akaroa 2025.csv
https://static.charts.linz.govt.nz/tide-tables/maj-ports/csv/Lyttelton 2025.csv
```

**CSV Format:**
```csv
Skiprows: 3 (header rows)
Columns: Day, DayName, Month, Year, Time1, TideHeight1, Time2, TideHeight2, Time3, TideHeight3, Time4, TideHeight4

Example:
1,Monday,Jan,2025,02:15,1.85,08:42,0.32,14:58,1.91,21:03,0.28
```

**Fetch Logic (Lines 918-937):**
```python
all_data = []

for site in site_list:  # ['Akaroa', 'Lyttelton']
    # Fetch current year + next year (lines 921-922)
    for yr in [year, year + 1]:
        url = f"https://static.charts.linz.govt.nz/tide-tables/maj-ports/csv/{site} {yr}.csv"

        with requests.get(url, timeout=15) as response:
            # Handle encoding (lines 925-927)
            try:
                csv_str = response.content.decode('utf-8')
            except UnicodeDecodeError:
                csv_str = response.content.decode('latin1')

            # Parse CSV (lines 928-935)
            df = pd.read_csv(
                io.StringIO(csv_str), skiprows=3,
                header=None, names=[
                    "Day", "DayName", "Month", "Year", "Time1", "TideHeight1",
                    "Time2", "TideHeight2", "Time3", "TideHeight3", "Time4",
                    "TideHeight4"
                ]
            )
            df["Harbour"] = site
            all_data.append(df)
```

**Data Transformation (Lines 940-1031):**
```python
# Combine all sites/years (line 941)
all_data = pd.concat(all_data, ignore_index=True)

# Unpivot tide columns (lines 944-958)
all_data = pd.concat([
    all_data[["Harbour", "Day", "Month", "Year", "Time1", "TideHeight1"]].rename(
        columns={"Time1": "Time", "TideHeight1": "Tidal_height"}
    ),
    all_data[["Harbour", "Day", "Month", "Year", "Time2", "TideHeight2"]].rename(
        columns={"Time2": "Time", "TideHeight2": "Tidal_height"}
    ),
    # ... Time3, Time4 similarly
], ignore_index=True)

# Remove NaN tides (line 961)
all_data = all_data.loc[all_data["Tidal_height"].notna()]

# Parse time components (lines 962-963)
all_data['hour'] = all_data['Time'].apply(lambda x: x.split(":")[0])
all_data['minute'] = all_data['Time'].apply(lambda x: x.split(":")[1])

# Create DateTime column (lines 967-969)
all_data['DateTime'] = pd.to_datetime(
    all_data[['Year', 'Month', 'Day', 'hour', 'minute']]
)

# Localize to Pacific/Auckland (DST-aware) (lines 981-985)
all_data["DateTimeTz"] = all_data["DateTime"].dt.tz_localize(
    tz='Pacific/Auckland',
    ambiguous='NaT'  # Mark ambiguous times as NaT for manual handling
)

# Handle DST ambiguity (lines 994-1017)
# During DST transitions (Apr/Oct), some times are ambiguous
# Use neighboring tide times to determine correct timezone
ambig_dts = all_data.loc[all_data["DateTimeTz"].isna()].copy()
# ... resolve using neighboring tide event differences ...

# Convert to fixed NZST (+12:00, no DST) (lines 1020-1022)
all_data["DateTimeNZST"] = all_data["DateTimeTz"].apply(
    lambda x: x.astimezone(tz(td(hours=12)))
)

# Determine tidal state (high/low) (lines 1024-1026)
all_data["Tidal_state"] = all_data.groupby("Harbour")['Tidal_height'].transform(
    lambda x: _get_tide_state(x)
)

# Return (lines 1028-1030)
return all_data[["DateTime", "Tidal_height", "Tidal_state", "Harbour"]]
```

**Performance:**
- Typical: 2-5 seconds for 2 harbours × 2 years
- Slower than database query (HTTP + CSV parsing)

**Notes:**
- **Fallback only:** Prefer `get_tide_data_db()` (faster, pre-processed)
- **Public API:** No authentication required
- **DST handling:** Complex logic to resolve ambiguous times during transitions

---

## Phase 2: MetService Weather (Lyttelton)

### `get_hourly_weather_data_Lyttelton()`

**Location:** [helpers/WeatherAPI_Functions.py:407-540](helpers/WeatherAPI_Functions.py#L407-L540)

**Purpose:** Fetch MetService hourly weather for Lyttelton with primary/backup station fallback

**Signature:**
```python
def get_hourly_weather_data_Lyttelton(
    datetime: dt.datetime,           # End of 24h window (UTC)
    station_candidates=("93786", "93951"),
    log_event_fn=None,               # Event logging callback
    logger=None,                     # Python logger
    max_retries: int = 2,            # Retries per variable
    base_backoff_sec: float = 5.0,   # Exponential backoff base
    total_deadline_s: float = None   # Total time budget per station
) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:437-452](../inference.py#L437-L452)
```python
# Fetch last 3 days (lines 438-452)
lyttelton_weather_parts = []

for i in range(3):  # Last 3 days
    single_date = DateTime_UTC + dt.timedelta(days=-i)

    try:
        df_part = f.get_hourly_weather_data_Lyttelton(
            datetime=single_date,
            station_candidates=tuple(station_candidates),  # ("93786", "93951")
            log_event_fn=log_event,
            logger=logger
        )
        if df_part is not None:
            lyttelton_weather_parts.append(df_part)
    except Exception as e:
        logger.warning(f"MetService fetch raised on {single_date.date()}: {e}")

# Concatenate all days (line 453)
df_lyttelton_weather = (
    pd.concat(lyttelton_weather_parts, ignore_index=True)
    if lyttelton_weather_parts else
    pd.DataFrame(columns=["DateTime","Rainfall","wind_direction","wind_speed","Ve","Vn","StationId"])
)
# Result: ~72 rows (3 days × 24 hours)
```

**Parameters:**
- `datetime` (datetime): End of 24-hour fetch window (UTC, tz-naive)
  - Window: `datetime - 1 day + 1 minute` to `datetime`
- `station_candidates` (tuple): Station IDs in priority order
  - Default: `("93786", "93951")` (Lyttelton Port, LPC Wharf 3)
  - Inference uses: tuple from env `METSERVICE_STATIONS` (line 144)
- `log_event_fn` (callable, optional): Event logging function
- `logger` (Logger, optional): Python logger
- `max_retries` (int): Retry attempts per variable (default 2)
- `base_backoff_sec` (float): Backoff base seconds (default 5.0)
- `total_deadline_s` (float): Time budget per station (default 20.0 from env)

**Returns:**
- `pd.DataFrame`: Columns `[DateTime, Rainfall, wind_direction, wind_speed, Ve, Vn, StationId]`

**Output Schema:**
```python
Columns:
  - DateTime (datetime64[ns]): Timestamp in UTC, tz-naive
  - Rainfall (float): Hourly rainfall accumulation (mm)
  - wind_direction (float): Hourly average wind direction (degrees, 0-360)
  - wind_speed (float): Hourly average wind speed (m/s)
  - Ve (float): East-west wind component (m/s)
  - Vn (float): North-south wind component (m/s)
  - StationId (str): Station ID that provided data ("93786" or "93951")

Rows: ~24 (one per hour in 24h window)
```

**MetService API Details:**

**Endpoint (Lines 160-162):**
```
https://api.metservice.com/observations/nz/1-minute/weatherStation/{station_id}/range/{start}/{end}?vars={var}
```

**Variables Fetched (Lines 126-130):**
```python
variable_dict = {
    'rainfal_01hracc': 'Rainfall',
    'winddir_01hravg': 'wind_direction',
    'windspd_01hravg': 'wind_speed'
}
```

**Authentication (Lines 164-167):**
```python
headers = {
    'apikey': MET_KEY,  # From environment METSERVICE_KEY
    'User-Agent': 'ecan-enterococci/1.0 (+azure-container-app)'
}
```

**Example Request:**
```
https://api.metservice.com/observations/nz/1-minute/weatherStation/93786/range/2025-11-27T01:00:00Z/2025-11-28T01:00:00Z?vars=rainfal_01hracc

Headers:
  apikey: {METSERVICE_KEY}
  User-Agent: ecan-enterococci/1.0 (+azure-container-app)
```

**Fallback Logic (Lines 451-540):**

```python
# Step 1: Fetch primary station (93786) (lines 456-471)
for idx, sid in enumerate(station_candidates):
    df = _metservice_fetch_24h(datetime, sid, ...)
    if not df.empty:
        df_primary = df
        primary_id = sid
        break  # Success → stop trying other stations

# Step 2: Evaluate data quality (lines 482-486)
p = df_primary.copy()
prim_wind_ok = (~p['wind_direction'].isna()) & (~p['wind_speed'].isna())
primary_has_gaps = p['Rainfall'].isna().any() or (~prim_wind_ok).any()

# Step 3: If gaps detected, try backup station (lines 488-522)
fallback_mode = os.getenv("MET_FALLBACK_MODE", "pair_wind").strip().lower()

if primary_has_gaps and fallback_mode != "off":
    # Try first different station as backup
    for sid in station_candidates:
        if sid == primary_id:
            continue
        df = _metservice_fetch_24h(datetime, sid, ...)
        if not df.empty:
            df_backup = df
            backup_id = sid
            break

    # Step 4: Merge primary + backup (lines 532-535)
    merged = _merge_met_station_frames(
        df_primary, df_backup,
        log_event_fn=log_event_fn,
        primary_id=primary_id,
        backup_id=backup_id,
        fallback_mode=fallback_mode
    )
    return merged
else:
    # No gaps or fallback disabled → return primary only
    return df_primary
```

**Merge Modes (via `MET_FALLBACK_MODE` environment variable):**

1. **"pair_wind" (default)** - Line 275 in `_merge_met_station_frames`:
   - Rainfall: Fill per-hour independently from backup
   - Wind: Fill BOTH direction AND speed together (paired)
   - Used in inference.py (line 431)

2. **"hour"** - Lines 330-349:
   - If ANY variable missing in primary for a hour → use ALL from backup

3. **"off"** - Line 288:
   - No merging, return primary unchanged

**Performance:**
- Typical: 8-15 seconds for 3 variables × 24 hours from primary station
- With fallback: +5-10 seconds (backup station fetch)
- With retries: +10-20 seconds (exponential backoff delays)

**Failure Modes:**
1. **Primary station timeout** → Try backup station automatically
2. **Both stations timeout** → Return empty DataFrame
3. **API key invalid** → HTTP 401, return empty DataFrame
4. **Variable not available** → Column becomes NaN

---

### `_metservice_fetch_24h()` (Internal)

**Location:** [helpers/WeatherAPI_Functions.py:79-265](helpers/WeatherAPI_Functions.py#L79-L265)

**Purpose:** Core MetService API fetch with retry logic for single station

**Signature:**
```python
def _metservice_fetch_24h(
    datetime: dt.datetime,
    station_id: str,
    *,
    log_event_fn=None,
    emit_var_ok: bool = False,
    max_retries: int = 2,
    base_backoff_sec: float = 5.0,
    total_deadline_s: float = None
) -> pd.DataFrame
```

**Parameters:**
- Same as `get_hourly_weather_data_Lyttelton()` but for single station

**Returns:**
- `pd.DataFrame`: Weather data for 24h window (or empty if all variables failed)

**Algorithm:**

```python
# Step 1: Calculate 24h window (lines 133-137)
date_start = datetime + dt.timedelta(days=-1, minutes=1)
# Window: datetime - 23h59m to datetime

# Step 2: Fetch each variable with retries (lines 151-233)
for key, var_name in variable_dict.items():
    url = (
        f"https://api.metservice.com/observations/nz/1-minute/weatherStation/"
        f"{station_id}/range/{date_start}T{time_start}Z/{date}T{time_str}Z?vars={key}"
    )

    attempt = 0
    while attempt < max_retries:
        # Check deadline (lines 173-179)
        if (time.monotonic() - t_station_start) > total_deadline_s:
            break  # Give up on this variable

        attempt += 1
        try:
            # HTTP request (line 184)
            resp = requests.get(url, headers=headers, timeout=15)
            response_json = j.loads(resp.text)

            # Extract results (line 200)
            results = response_json["results"] or []

            if results:
                # Parse to DataFrame (lines 225-232)
                data = pd.json_normalize(results)
                data['obs_timestamp'] = pd.to_datetime(data['obs_timestamp'], utc=True)

                # Filter to hourly rows only (line 229)
                data.drop(data[data['obs_timestamp'].dt.minute != 0].index, inplace=True)

                data.rename(columns={'obs_timestamp': 'DateTime', key: var_name}, inplace=True)
                l.append(data)
                break

        except Exception as e:
            if attempt < max_retries:
                # Exponential backoff with jitter (lines 121-123, 220)
                delay = min(base_backoff_sec * (2 ** (attempt - 1)), 15) * random.uniform(0.85, 1.15)
                time.sleep(delay)
                continue
            break

# Step 3: Merge all variables on DateTime (lines 243-252)
if l:
    df = pd.concat(l, ignore_index=True)
    for col in ['Rainfall','wind_direction','wind_speed']:
        if col not in df.columns:
            df[col] = np.nan

    df = df.groupby('DateTime', as_index=False).agg(
        Rainfall=('Rainfall','max'),
        wind_direction=('wind_direction','max'),
        wind_speed=('wind_speed','max')
    )

    # Step 4: Compute Ve/Vn (lines 254-256)
    df['tuple'] = df.apply(lambda r: wind_vector_components(r['wind_direction'], r['wind_speed']), axis=1)
    df[['Ve','Vn']] = df['tuple'].apply(pd.Series)
    df.drop(['tuple'], axis=1, inplace=True)

    df['StationId'] = str(station_id)
    return df
else:
    return pd.DataFrame(columns=['DateTime','Rainfall','wind_direction','wind_speed','Ve','Vn'])
```

**Retry Backoff Formula (Line 122):**
```python
delay = min(base_backoff_sec * (2 ** (attempt - 1)), 15) * random.uniform(0.85, 1.15)
# Example with base=5.0:
# Attempt 1: 5 * (2^0) * jitter = ~5s
# Attempt 2: 5 * (2^1) * jitter = ~10s
# Attempt 3+: capped at 15s
```

**Hourly Filtering (Line 229):**
```python
# API returns 1-minute resolution data
# Keep only rows where minute=0 (on the hour)
data.drop(data[data['obs_timestamp'].dt.minute != 0].index, inplace=True)

# Typical: 1440 rows (24h × 60 min) → 24 rows (hourly)
```

**Used By:**
- `get_hourly_weather_data_Lyttelton()` → Calls for primary and backup stations

---

### `_merge_met_station_frames()` (Internal)

**Location:** [helpers/WeatherAPI_Functions.py:269-404](helpers/WeatherAPI_Functions.py#L269-L404)

**Purpose:** Merge primary and backup station data, filling gaps intelligently

**Signature:**
```python
def _merge_met_station_frames(
    primary: pd.DataFrame,
    backup: pd.DataFrame | None,
    *,
    log_event_fn=None,
    primary_id: str | None = None,
    backup_id: str | None = None,
    fallback_mode: str = "pair_wind"  # "pair_wind", "hour", or "off"
) -> pd.DataFrame
```

**Parameters:**
- `primary` (DataFrame): Data from primary station (93786)
- `backup` (DataFrame | None): Data from backup station (93951)
- `fallback_mode` (str): Merge strategy
  - `"pair_wind"` (default): Fill wind as pair (both dir+speed together)
  - `"hour"`: Fill entire hour if any variable missing
  - `"off"`: No merging

**Returns:**
- `pd.DataFrame`: Merged weather data

**Merge Strategy ("pair_wind" mode - Lines 352-362):**

```python
# Step 1: Normalize datetimes to hourly (lines 308-314)
p['DateTime'] = pd.to_datetime(p['DateTime'], utc=True).dt.floor('h')
b['DateTime'] = pd.to_datetime(b['DateTime'], utc=True).dt.floor('h')

# Step 2: Build union hourly index (lines 317-320)
idx = pd.Index(sorted(set(p['DateTime']) | set(b['DateTime'])), name='DateTime')
p2 = p.set_index('DateTime')[['Rainfall','wind_direction','wind_speed']].reindex(idx)
b2 = b.set_index('DateTime')[['Rainfall','wind_direction','wind_speed']].reindex(idx)

# Step 3: Fill Rainfall independently per-hour (line 355)
Rainfall = p2['Rainfall'].where(~p2['Rainfall'].isna(), b2['Rainfall'])

# Step 4: Fill wind ONLY if BOTH components missing in primary (lines 357-360)
prim_wind_ok = (~p2['wind_direction'].isna()) & (~p2['wind_speed'].isna())
back_wind_ok = (~b2['wind_direction'].isna()) & (~b2['wind_speed'].isna())

wd = p2['wind_direction'].where(prim_wind_ok, np.where(back_wind_ok, b2['wind_direction'], np.nan))
ws = p2['wind_speed'].where(prim_wind_ok, np.where(back_wind_ok, b2['wind_speed'], np.nan))

# Step 5: Recompute Ve/Vn from final wind direction/speed (lines 367-371)
out = pd.DataFrame({'DateTime': idx, 'Rainfall': Rainfall,
                    'wind_direction': wd, 'wind_speed': ws})
tuples = out.apply(lambda r: wind_vector_components(r['wind_direction'], r['wind_speed']), axis=1)
out[['Ve','Vn']] = tuples.apply(pd.Series)
```

**Why "pair_wind" Mode?**
- Wind direction and speed are correlated (must come from same source)
- Mixing direction from one station + speed from another → incorrect wind vector
- Only use backup if BOTH components missing in primary

**Example:**
```python
# Primary station (93786):
     DateTime             Rainfall  wind_direction  wind_speed
0    2025-11-27 01:00:00  0.0       135.0           3.2
1    2025-11-27 02:00:00  NaN       180.0           5.1     # Rainfall missing
2    2025-11-27 03:00:00  0.5       NaN             NaN     # Wind pair missing

# Backup station (93951):
     DateTime             Rainfall  wind_direction  wind_speed
0    2025-11-27 01:00:00  0.0       140.0           3.5
1    2025-11-27 02:00:00  1.2       175.0           5.3
2    2025-11-27 03:00:00  0.3       225.0           2.8

# After merge (mode="pair_wind"):
     DateTime             Rainfall  wind_direction  wind_speed  StationId
0    2025-11-27 01:00:00  0.0       135.0           3.2         93786  # Primary
1    2025-11-27 02:00:00  1.2       180.0           5.1         93786  # Rainfall from backup
2    2025-11-27 03:00:00  0.5       225.0           2.8         93951  # Wind pair from backup
```

**Event Emissions (Lines 386-400):**
```python
# If rainfall hours were filled from backup
_emit(log_event_fn, "SOURCE_VAR_FALLBACK_USED",
      source="metservice",
      var="Rainfall",
      from_station=primary_id, to_station=backup_id,
      hours=[...], count=int(len(patched_rain_hours)))

# If wind pairs were filled from backup
_emit(log_event_fn, "SOURCE_WINDPAIR_FALLBACK_USED",
      source="metservice",
      from_station=primary_id, to_station=backup_id,
      hours=[...], count=int(len(patched_wind_hours)))
```

**Used By:**
- `get_hourly_weather_data_Lyttelton()` → Merges primary + backup

---

## Phase 3: NIWA Weather (Akaroa)

### `get_10min_weather_data_Akaroa()`

**Location:** [helpers/WeatherAPI_Functions.py:543-716](helpers/WeatherAPI_Functions.py#L543-L716)

**Purpose:** Fetch NIWA Mintaka 10-minute weather data for Akaroa station

**Signature:**
```python
def get_10min_weather_data_Akaroa(
    datetime: dt.datetime,      # Target datetime (UTC)
    daytotal: int,              # Days to fetch backward
    *,
    log_event_fn=None,
    logger=None,
    max_retries: int = 2,
    base_backoff_sec: float = 5.0,
    total_deadline_s: float = None
) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:492-496](../inference.py#L492-L496)
```python
df_akaroa_10min_weather = f.get_10min_weather_data_Akaroa(
    DateTime_UTC,  # Target hour (UTC, tz-naive)
    3,             # daytotal: fetch 3 days backward
    log_event_fn=log_event,
    logger=logger
)
# Returns: ~432 rows (3 days × 24 hours × 6 per hour)

# Aggregate to hourly (lines 530-534 in inference.py)
df_akaroa_weather = df_akaroa_10min_weather.groupby('DateTime', as_index=False).agg(
    Rainfall=('Rainfall', 'sum'),   # Sum 10-min rainfall → hourly total
    Ve=('Ve', 'mean'),              # Mean east-west wind component
    Vn=('Vn', 'mean')               # Mean north-south wind component
)
```

**Parameters:**
- `datetime` (datetime): Target datetime (UTC, tz-naive)
  - Fetch window: `datetime - daytotal days` to `datetime`
- `daytotal` (int): Number of days to fetch backward (inference uses 3)
- `log_event_fn` (callable, optional): Event logging callback
- `logger` (Logger, optional): Python logger
- `max_retries` (int): Retry attempts per product (default 2)
- `base_backoff_sec` (float): Backoff base seconds (default 5.0)
- `total_deadline_s` (float): Time budget for all products (default 20.0 from env)

**Returns:**
- `pd.DataFrame`: Columns `[DateTime, Rainfall, wind_direction, wind_speed, Ve, Vn]`

**Output Schema:**
```python
Columns:
  - DateTime (datetime64[ns]): Timestamp in UTC, tz-naive, 10-minute resolution
  - Rainfall (float): 10-minute rainfall accumulation (mm)
  - wind_direction (float): 10-minute wind direction (degrees, 0-360)
  - wind_speed (float): 10-minute wind speed (m/s)
  - Ve (float): East-west wind component (m/s)
  - Vn (float): North-south wind component (m/s)

Rows: ~432 (3 days × 24 hours × 6 per hour)
```

**NIWA Mintaka API Details:**

**Endpoint (Lines 623-625):**
```
https://mintaka.niwa.co.nz/rest/api/V1.1/products/{product_id}/data?startDate={start}&endDate={end}&mode=LatestCommon
```

**Products (Akaroa Station) (Lines 597-601):**
```python
product_dict = {
    55993775: 'Rainfall',        # mm per 10 minutes
    55995169: 'wind_direction',  # degrees (0-360)
    55995333: 'wind_speed'       # m/s
}
```

**Authentication (Line 643):**
```python
auth = HTTPBasicAuth(username, password)
# username from env NIWA_USERNAME
# password from env NIWA_KEY
```

**Example Request:**
```
GET https://mintaka.niwa.co.nz/rest/api/V1.1/products/55993775/data?startDate=2025-11-25T01:00:00.000Z&endDate=2025-11-28T01:00:00.000Z&mode=LatestCommon

Authorization: Basic base64(NIWA_USERNAME:NIWA_KEY)
```

**Response Format (JSON):**
```json
{
  "data": [
    {
      "tuples": [
        {
          "validityTime": "2025-11-25T01:00:00Z",
          "value": 0.0
        },
        {
          "validityTime": "2025-11-25T01:10:00Z",
          "value": 0.2
        },
        ...
      ]
    }
  ]
}
```

**Fetch Logic (Lines 614-716):**

```python
# Step 1: Calculate time window (lines 603-607)
start_date = datetime + dt.timedelta(days=-daytotal, minutes=1)
# Window: datetime - daytotal days to datetime

# Step 2: Fetch each product with retries (lines 614-691)
l = []
for product_id, var_name in product_dict.items():
    url = (
        f"https://mintaka.niwa.co.nz/rest/api/V1.1/products/{product_id}/data"
        f"?startDate={start_date}T{start_time}.000Z&endDate={date}T{time_str}.000Z&mode=LatestCommon"
    )

    attempt = 0
    while attempt < max_retries:
        # Check deadline (lines 632-638)
        if (time.monotonic() - t_station_start) > total_deadline_s:
            break

        attempt += 1
        try:
            # HTTP request (line 643)
            resp = requests.get(url, auth=HTTPBasicAuth(username, password), timeout=15)
            result = j.loads(resp.text)

            # Extract tuples (line 648)
            tuples = result.get("data", [{}])[0].get("tuples", []) or []

            if tuples:
                # Parse to DataFrame (lines 672-676)
                data = pd.json_normalize(tuples)
                data['validityTime'] = pd.to_datetime(data['validityTime'], utc=True)
                data.rename(columns={'validityTime': 'DateTime', 'value': var_name}, inplace=True)
                l.append(data)
                break

        except Exception as e:
            if attempt < max_retries:
                # Exponential backoff (lines 592-594, 689)
                delay = min(base_backoff_sec * (2 ** (attempt - 1)), 15) * random.uniform(0.85, 1.15)
                time.sleep(delay)
                continue
            break

# Step 3: Merge all products on DateTime (lines 700-705)
if l:
    df = pd.concat(l, ignore_index=True)
    for col in ['Rainfall', 'wind_direction', 'wind_speed']:
        if col not in df.columns:
            df[col] = np.nan

    df = df.groupby('DateTime', as_index=False).agg('max')

    # Step 4: Compute Ve/Vn (lines 706-708)
    df['tuple'] = df.apply(lambda r: wind_vector_components(r['wind_direction'], r['wind_speed']), axis=1)
    df[['Ve', 'Vn']] = df['tuple'].apply(pd.Series)
    df.drop(['tuple'], axis=1, inplace=True)

    return df
else:
    return pd.DataFrame(columns=['DateTime', 'Rainfall', 'wind_direction', 'wind_speed', 'Ve', 'Vn'])
```

**LatestCommon Mode:**
- Returns most recent complete dataset
- Data updated daily (~9am NZST)
- Ensures consistency (all products from same data version)

**Performance:**
- Typical: 5-10 seconds for 3 products × ~432 rows (3 days)
- Sequential product fetches (not parallelized)

**Failure Modes:**
1. **Product timeout** → Continue with other products (partial data)
2. **All products fail** → Return empty DataFrame
3. **Invalid credentials** → HTTP 401 for all products
4. **Zero-only guard** → Handled in inference.py (lines 538-568)

---

## Phase 4: Rolling Feature Computation

### `add_rainfall_variables()`

**Location:** [helpers/WeatherAPI_Functions.py:720-747](helpers/WeatherAPI_Functions.py#L720-L747)

**Purpose:** Compute rolling rainfall sums over multiple time windows

**Signature:**
```python
def add_rainfall_variables(df: pd.DataFrame) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:591-592](../inference.py#L591-L592)
```python
# Add rainfall features to both harbours (lines 591-592)
df_akaroa_weather = f.add_rainfall_variables(df_akaroa_weather)
df_lyttelton_weather = f.add_rainfall_variables(df_lyttelton_weather)

# Result: 8 new columns added to each DataFrame
```

**Parameters:**
- `df` (DataFrame): Weather data with `Rainfall` and `DateTime` columns
  - Must be sorted by DateTime (ascending)

**Returns:**
- `pd.DataFrame`: Input DataFrame with 8 new rainfall columns

**Features Created (Lines 732-743):**
```python
# Rolling sums (lines 732-737)
df['3H'] = np.round(df.Rainfall.rolling(min_periods=3, window=3).sum(), 1)
df['6H'] = np.round(df.Rainfall.rolling(min_periods=6, window=6).sum(), 1)
df['12H'] = np.round(df.Rainfall.rolling(min_periods=12, window=12).sum(), 1)
df['24H'] = np.round(df.Rainfall.rolling(min_periods=24, window=24).sum(), 1)
df['48H'] = np.round(df.Rainfall.rolling(min_periods=48, window=48).sum(), 1)
df['72H'] = np.round(df.Rainfall.rolling(min_periods=72, window=72).sum(), 1)

# Rain intensity (line 739)
df['rain_intensity_48h'] = np.round(df.Rainfall.rolling(min_periods=48, window=48).max(), 1)

# Rain duration (lines 741-743)
df = df.assign(rain_yn = lambda x: (x['Rainfall']>0))
df['rain_yn'] = df['rain_yn'].astype(int)
df['rain_duration_48h'] = np.round(df.rain_yn.rolling(min_periods=48, window=48).sum(), 1)

df = df.drop(['rain_yn'], axis=1)
```

**Output Columns:**
```python
Columns added:
  - 3H: Sum of rainfall in last 3 hours (mm)
  - 6H: Sum of rainfall in last 6 hours (mm)
  - 12H: Sum of rainfall in last 12 hours (mm)
  - 24H: Sum of rainfall in last 24 hours (mm)
  - 48H: Sum of rainfall in last 48 hours (mm)
  - 72H: Sum of rainfall in last 72 hours (mm)
  - rain_intensity_48h: MAX hourly rainfall in last 48 hours (mm/hour)
  - rain_duration_48h: Number of hours with rain > 0 in last 48 hours
```

**Why These Features?**
- **3H, 6H, 12H:** Short-term rainfall (immediate bacterial wash-off)
- **24H, 48H, 72H:** Medium-term rainfall (soil saturation, catchment runoff)
- **Intensity:** Captures rainfall rate (heavy vs light rain)
- **Duration:** Captures rainfall persistence (continuous vs sporadic)

**Feature Importance (from model):**
- `3H`: Top 3 most important feature
- `12H`: Top 5 most important feature
- `rain_intensity_48h`: Moderate importance

**Notes:**
- `min_periods`: Allows rolling sum even if fewer than window hours available
- First few rows have partial sums (e.g., 3H at row 0 = sum of 1 hour only)
- Rolling windows are backward-looking (current + past hours)

---

### `add_wind_variables()`

**Location:** [helpers/WeatherAPI_Functions.py:750-773](helpers/WeatherAPI_Functions.py#L750-L773)

**Purpose:** Compute rolling wind features (circular mean direction, mean speed) over time window

**Signature:**
```python
def add_wind_variables(df: pd.DataFrame, hours: int) -> pd.DataFrame
```

**Called in inference.py:** [inference.py:593-595](../inference.py#L593-L595)
```python
# Add wind features for 3, 6, and 12 hour windows (lines 593-595)
for i in [3, 6, 12]:
    df_akaroa_weather = f.add_wind_variables(df_akaroa_weather, i)
    df_lyttelton_weather = f.add_wind_variables(df_lyttelton_weather, i)

# Result: 6 new columns per harbour (wind_direction_3h, wind_speed_3h, ...)
```

**Parameters:**
- `df` (DataFrame): Weather data with `Ve`, `Vn`, `DateTime` columns
  - Must be sorted by DateTime (ascending)
- `hours` (int): Rolling window size (inference uses 3, 6, 12)

**Returns:**
- `pd.DataFrame`: Input DataFrame with 2 new wind columns

**Algorithm (Lines 765-772):**
```python
# Step 1: Rolling mean of wind components (lines 765-766)
df['Ve_mean'] = df.Ve.rolling(min_periods=hours, window=hours).mean()
df['Vn_mean'] = df.Vn.rolling(min_periods=hours, window=hours).mean()

# Step 2: Convert back to direction/speed (lines 768-769)
df[f'wind_direction_{hours}h'] = (360 + (np.arctan2(df['Ve_mean'], df['Vn_mean']) * 180/np.pi)) % 360
df[f'wind_speed_{hours}h'] = np.hypot(df['Ve_mean'], df['Vn_mean'])

# Step 3: Drop intermediate columns (line 771)
df.drop(['Ve_mean', 'Vn_mean'], axis=1, inplace=True)
```

**Output Columns:**
```python
Columns added:
  - wind_direction_{hours}h: Circular mean wind direction (degrees, 0-360)
  - wind_speed_{hours}h: Mean wind speed (m/s)
```

**Why Circular Mean?**
```python
# WRONG: Naive mean wraps incorrectly
# Example: mean([350°, 10°]) = 180° (should be ~0°)
naive_mean = (350 + 10) / 2  # → 180° (incorrect!)

# CORRECT: Vector mean (via components)
Ve_mean = mean([sin(350°), sin(10°)])  # → 0.09
Vn_mean = mean([cos(350°), cos(10°)])  # → 0.99
correct_mean = arctan2(0.09, 0.99) % 360  # → ~5° (correct!)
```

**Why Multiple Windows?**
- **3h:** Recent wind (current conditions)
- **6h:** Medium-term wind (sustained patterns)
- **12h:** Long-term wind (prevailing conditions)
- Model learns different temporal effects (e.g., 12h offshore wind → bacterial dispersion)

---

## Phase 5: Tide Feature Computation

### `add_tide_variables()`

**Location:** [helpers/WeatherAPI_Functions.py:813-869](helpers/WeatherAPI_Functions.py#L813-L869)

**Purpose:** Calculate tide features for specific datetime and harbour

**Signature:**
```python
def add_tide_variables(
    input_datetime: dt.datetime,   # Target datetime (tz-naive NZLT)
    harbour: str,                  # "Akaroa" or "Lyttelton"
    df: pd.DataFrame               # Tide predictions from get_tide_data_db()
) -> tuple
```

**Called in inference.py:** [inference.py:653, 658](../inference.py#L653-L659)
```python
# Add tide features for Akaroa (lines 651-654)
_ak_cols = ["tidal_state", "hours_to_high_tide", "high_tide_height",
            "high_tide_height_X_hours_to_high_tide"]
ak_vals = f.add_tide_variables(TARGET_LOCAL_H, "Akaroa", tides_df)
akaroa_at_hour.loc[:, _ak_cols] = [list(ak_vals)]

# Add tide features for Lyttelton (lines 656-659)
_lt_cols = ["tidal_state", "hours_to_high_tide", "high_tide_height",
            "high_tide_height_X_hours_to_high_tide"]
lt_vals = f.add_tide_variables(TARGET_LOCAL_H, "Lyttelton", tides_df)
lyttelton_at_hour.loc[:, _lt_cols] = [list(lt_vals)]
```

**Parameters:**
- `input_datetime` (datetime): Target datetime (tz-naive NZLT)
  - Inference uses: `TARGET_LOCAL_H` (NZ_LOCAL floored to hour)
- `harbour` (str): Harbour name ("Akaroa" or "Lyttelton")
- `df` (DataFrame): Tide predictions from `get_tide_data_db()` or `get_tide_data()`

**Returns:**
- `tuple`: `(tidal_state, hours_to_high_tide, high_tide_height, product)`
  - `tidal_state` (str): "ebbing" or "incoming"
  - `hours_to_high_tide` (float): Hours from input_datetime to closest tide
  - `high_tide_height` (float): Height of closest tide (meters)
  - `product` (float): `hours_to_high_tide × high_tide_height` (interaction term)

**Algorithm (Lines 818-869):**

```python
# Step 1: Validate inputs (lines 818-819)
if df is None or len(df) == 0 or input_datetime is None:
    return ("unknown", np.nan, np.nan, np.nan)

# Step 2: Ensure tz-naive (lines 822-823)
if hasattr(input_datetime, "tzinfo") and input_datetime.tzinfo is not None:
    input_datetime = pd.Timestamp(input_datetime).tz_localize(None)

# Step 3: Filter to harbour (lines 825-833)
tdf = df.copy()
tdf["Harbour"] = tdf["Harbour"].astype(str).str.strip()
target_h = str(harbour).strip()
tdf = tdf[tdf["Harbour"].str.casefold() == target_h.casefold()]

if tdf.empty:
    return ("unknown", np.nan, np.nan, np.nan)

# Step 4: Normalize state labels (lines 836-838)
state_norm = tdf["Tidal_state"].astype(str).str.strip().str.casefold()
tdf = tdf.assign(Tidal_state=state_norm)
HIGH_STATES = {"high", "h", "high tide"}

# Step 5: Find closest tide within ±36h window (lines 844-860)
win_lo = input_datetime - td(hours=36)
win_hi = input_datetime + td(hours=36)

# Prefer highs within window
sel_high = tdf[(tdf["Tidal_state"].isin(HIGH_STATES)) &
               (tdf["DateTime"] >= win_lo) & (tdf["DateTime"] <= win_hi)]

if not sel_high.empty:
    # Pick closest high tide
    idx = (sel_high["DateTime"] - input_datetime).abs().idxmin()
else:
    # Fallback: any tide within window
    sel_any = tdf[(tdf["DateTime"] >= win_lo) & (tdf["DateTime"] <= win_hi)]
    if not sel_any.empty:
        idx = (sel_any["DateTime"] - input_datetime).abs().idxmin()
    else:
        # Final fallback: nearest anywhere
        idx = (tdf["DateTime"] - input_datetime).abs().idxmin()

# Step 6: Extract features (lines 862-869)
closest_time = tdf.loc[idx, "DateTime"]
closest_height = float(tdf.loc[idx, "Tidal_height"]) if pd.notna(tdf.loc[idx, "Tidal_height"]) else np.nan

hours_to_high = round((input_datetime - closest_time).total_seconds() / 3600, 1)
tidal_state = "ebbing" if (pd.notna(hours_to_high) and hours_to_high > 0) else "incoming"

prod = hours_to_high * closest_height if (pd.notna(hours_to_high) and pd.notna(closest_height)) else np.nan

return (tidal_state, hours_to_high, closest_height, prod)
```

**Example:**
```python
# Tide predictions
     DateTime             Tidal_height  Tidal_state  Harbour
0    2025-11-28 02:15:00  1.85          high         Lyttelton
1    2025-11-28 08:42:00  0.32          low          Lyttelton
2    2025-11-28 14:58:00  1.91          high         Lyttelton  ← Closest to 13:00
3    2025-11-28 21:03:00  0.28          low          Lyttelton

# Input
input_datetime = pd.Timestamp("2025-11-28 13:00:00")  # tz-naive NZLT
harbour = "Lyttelton"

# Calculate
closest_time = 2025-11-28 14:58:00
hours_to_high = (13:00 - 14:58) / 3600 = -1.97 hours (negative = tide is in future)
closest_height = 1.91 meters
tidal_state = "incoming" (hours_to_high < 0)
product = -1.97 × 1.91 = -3.76

# Output
("incoming", -1.97, 1.91, -3.76)
```

**Why These Features?**
- **hours_to_high_tide:** Captures tidal phase
  - Positive (past tide) = ebbing
  - Negative (future tide) = incoming
- **high_tide_height:** Tidal range (spring vs neap tides)
  - Spring tides (height >2m) → stronger currents, more mixing
  - Neap tides (height <1.5m) → weaker currents, stagnant water
- **product:** Interaction term (model learns nonlinear effects)

**Feature Importance (from model):**
- `hours_to_high_tide`: Top 10 most important feature
- `high_tide_height`: Moderate importance
- `product`: Captures interaction effects

---

## Wind Vector Utilities

### `wind_vector_components()`

**Location:** [helpers/WeatherAPI_Functions.py:52-68](helpers/WeatherAPI_Functions.py#L52-L68)

**Purpose:** Convert wind direction/speed to Ve/Vn vector components

**Signature:**
```python
def wind_vector_components(
    direction: float | np.ndarray,  # Wind direction (degrees, 0-360)
    speed: float | np.ndarray       # Wind speed (m/s)
) -> tuple
```

**Parameters:**
- `direction` (float | array): Wind direction in degrees (0-360)
  - **Convention:** Direction **from which** wind is blowing
  - Example: 0° = North wind (blowing from north toward south)
- `speed` (float | array): Wind speed magnitude (m/s)

**Returns:**
- `tuple`: `(Ve, Vn)`
  - `Ve` (float | array): East-west component (positive = toward east)
  - `Vn` (float | array): North-south component (positive = toward north)

**Algorithm (Lines 65-66):**
```python
Ve = speed * np.sin(direction * np.pi/180)
Vn = speed * np.cos(direction * np.pi/180)

return Ve, Vn
```

**Examples:**
```python
# North wind (0°, blowing from north)
Ve, Vn = wind_vector_components(0, 5)
# → Ve=0.0, Vn=5.0 (positive north component)

# East wind (90°, blowing from east)
Ve, Vn = wind_vector_components(90, 5)
# → Ve=5.0, Vn=0.0 (positive east component)

# South wind (180°, blowing from south)
Ve, Vn = wind_vector_components(180, 5)
# → Ve=0.0, Vn=-5.0 (negative north component = southward)

# West wind (270°, blowing from west)
Ve, Vn = wind_vector_components(270, 5)
# → Ve=-5.0, Vn=0.0 (negative east component = westward)
```

**Why Vector Components?**
1. **Circular mean:** Can't average directions directly (350° + 10° ≠ 180°)
2. **Wind aggregation:** 10-min to hourly (mean of components, not directions)
3. **Model input:** LightGBM handles numeric Ve/Vn better than circular direction

**Used By:**
- `_metservice_fetch_24h()` (line 254) → Computes Ve/Vn from API response
- `get_10min_weather_data_Akaroa()` (line 706) → Computes Ve/Vn from API response
- `_merge_met_station_frames()` (line 368) → Recomputes after merging stations

---

## Internal Helper Functions

### `_emit()` (Internal)

**Location:** [helpers/WeatherAPI_Functions.py:71-77](helpers/WeatherAPI_Functions.py#L71-L77)

**Purpose:** Thin wrapper for event logging (allows None-safe calls)

**Signature:**
```python
def _emit(log_event_fn, event: str, **fields)
```

**Algorithm (Lines 76-77):**
```python
if log_event_fn:
    log_event_fn(event, **fields)
```

**Why Needed?**
- Functions accept optional `log_event_fn` parameter
- `_emit()` prevents `if log_event_fn: log_event_fn(...)` repetition
- Allows library use without logging (log_event_fn=None)

---

### `_validate_ident()` (Internal)

**Location:** [helpers/WeatherAPI_Functions.py:1082-1085](helpers/WeatherAPI_Functions.py#L1082-L1085)

**Purpose:** Validate SQL identifier (table/schema name) to prevent SQL injection

**Signature:**
```python
def _validate_ident(name: str) -> str
```

**Algorithm (Lines 1035, 1083-1085):**
```python
_VALID_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _validate_ident(name: str) -> str:
    if not _VALID_IDENT.match(name):
        raise ValueError(f"Invalid identifier: {name!r}")
    return name
```

**Used By:**
- `get_tide_data_db()` (line 1119) → Validates schema and table names

---

### `_get_tide_state()` (Internal, Deprecated)

**Location:** [helpers/WeatherAPI_Functions.py:776-809](helpers/WeatherAPI_Functions.py#L776-L809)

**Purpose:** Determine tide state from tide heights (legacy function)

**Status:** Not used in current inference pipeline (replaced by LINZ API tide state)

**Note:** Kept for backward compatibility, used by `get_tide_data()` fallback

---

## Common Errors & Solutions

### Error 1: MetService API Key Invalid

**Error:**
```python
SOURCE_DOWN: missing_api_key or 401 Unauthorized
```

**Cause:** `METSERVICE_KEY` environment variable not set or invalid

**Solution (inference.py lines 114-118):**
```python
if not MET_KEY:
    _emit(log_event_fn, "SOURCE_DOWN",
          source="metservice", failure_kind="missing_api_key")
    return pd.DataFrame(...)
```

**Check:** Verify `METSERVICE_KEY` in environment

---

### Error 2: NIWA Zero-Only Guard Triggered

**Error:**
```python
SOURCE_WINDOW_DEGRADED: zeros_only_guard
```

**Cause:** Last 12 hours of NIWA data are all zeros (sensor outage)

**Handling (inference.py lines 538-568):**
```python
def _zeros_only_guard(df_hourly, now_utc, lookback_h=12):
    window = df_hourly[df_hourly["DateTime"] > now_utc - timedelta(hours=lookback_h)]

    for col in ("Rainfall", "Ve", "Vn"):
        if (window[col].fillna(0).abs() > 0).any():
            return False  # At least one non-zero value
    return True  # All zeros → suspicious

if _zeros_only_guard(df_akaroa_weather, DateTime_UTC, 12):
    # Force NaNs to trigger health check exclusion
    df_akaroa_weather = pd.DataFrame([{
        "DateTime": DateTime_UTC,
        "Rainfall": np.nan,
        "Ve": np.nan,
        "Vn": np.nan
    }])
```

**Impact:** Akaroa excluded from predictions (health check fails)

---

### Error 3: Empty Tide DataFrame

**Error:**
```python
Tide DB returned empty DataFrame
```

**Cause:** Database connection timeout or no data in date range

**Solution (inference.py lines 364-381):**
```python
if tides_df.empty:
    logger.warning("Tide DB returned empty, falling back to LINZ")

    tides_df = f.get_tide_data(site_list=['Akaroa', 'Lyttelton'], year=False)
    # Fallback to LINZ static API automatically
```

**Impact:** Minimal (LINZ fallback provides same data)

---

### Error 4: Wind Pair Incomplete

**Error:**
```python
Wind direction present but wind_speed NaN
```

**Cause:** MetService API returned incomplete wind data for an hour

**Solution:** `_merge_met_station_frames()` (lines 327-328, 357-358) fills from backup
```python
prim_wind_ok = (~p['wind_direction'].isna()) & (~p['wind_speed'].isna())

# Only use backup if BOTH components missing
wd = p['wind_direction'].where(prim_wind_ok,
                                np.where(back_wind_ok, b['wind_direction'], np.nan))
```

**Impact:** Gaps filled from backup station (93951)

---

## Performance Summary

| Function | Typical Time | Bottleneck | Line Called in inference.py |
|----------|--------------|-----------|----------------------------|
| `get_tide_data_db()` | 0.5-2s | Database query | 341 |
| `get_tide_data()` | 2-5s | HTTP + CSV parse | 366 (fallback) |
| `get_hourly_weather_data_Lyttelton()` | 8-15s | HTTP (3 vars × 24h) | 442 (×3 days) |
| `get_10min_weather_data_Akaroa()` | 5-10s | HTTP (3 products) | 492 |
| `add_rainfall_variables()` | 20-50ms | Pandas rolling ops | 591-592 |
| `add_wind_variables()` | 10-30ms | Pandas rolling ops | 593-595 (×3 windows) |
| `add_tide_variables()` | <1ms | DataFrame filtering | 653, 658 |
| `wind_vector_components()` | <1ms | NumPy trig | Many (internal) |

---

**Related Documentation:**
- [02-RUNTIME-GUIDE.md](../docs/02-RUNTIME-GUIDE.md) - Complete inference.py execution flow
- [03-API-INTEGRATION.md](../docs/03-API-INTEGRATION.md) - Detailed API integration patterns
- [inference_data_preparation_GUIDE.md](inference_data_preparation_GUIDE.md) - Helper functions reference
- [04-DATABASE-SCHEMA.md](../docs/04-DATABASE-SCHEMA.md) - Database schema details
