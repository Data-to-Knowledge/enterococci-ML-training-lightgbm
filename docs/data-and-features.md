# Data Pipeline and Feature Engineering

This document explains what data the models use, how it is cleaned, and what features are engineered before training.

## Raw Data

The training data lives in `data/processed/training_data.csv`. Each row represents a single water quality sample at a specific site and time. The key columns fall into several categories:

### Target Variable

| Column | Description |
|--------|-------------|
| `Enterococci` | Bacterial concentration in MPN/100mL. This is what we are predicting. |

### Identifiers

| Column | Description |
|--------|-------------|
| `DateTime` | When the sample was taken |
| `SITE_NAME` | Which beach the sample is from |

### Rainfall (accumulated over different windows)

| Column | Description |
|--------|-------------|
| `3H`, `6H`, `12H`, `24H`, `48H`, `72H` | Accumulated rainfall (mm) over the preceding 3, 6, 12, 24, 48, and 72 hours |
| `rain_intensity_48h` | Rainfall intensity over the past 48 hours |
| `rain_duration_48h` | Duration of rainfall in the past 48 hours |

Rainfall is one of the strongest predictors of Enterococci. Heavy rain washes contaminants from land into waterways and out to coastal sites.

### Wind

| Column | Description |
|--------|-------------|
| `wind_speed_3h`, `wind_speed_6h`, `wind_speed_12h` | Average wind speed over the preceding 3, 6, and 12 hours |
| `wind_direction_3h`, `wind_direction_6h`, `wind_direction_12h` | Wind direction (degrees) over the same windows |
| `beach_orientation_angle` | The angle the beach faces, used to calculate whether wind is onshore or offshore |

### Tidal Conditions

| Column | Description |
|--------|-------------|
| `tidal_state` | Categorical: `"incoming"` (before high tide) or `"ebbing"` (after high tide) |
| `hours_to_high_tide` | Signed hours to the closest high tide -- positive means the high tide has passed, negative means it is still approaching |
| `high_tide_height` | Height of the nearest high tide (metres) |

### Site Characteristics (static per site)

| Column | Description |
|--------|-------------|
| `Harbour` | Harbour name the site belongs to: `"Lyttelton"` or `"Akaroa"` -- used as a grouping variable for weather features and site metadata |
| `Latitude`, `Longitude` | Geographic coordinates |
| `Shallowness` | How shallow the water is at the monitoring point |
| `Soil_type` | Dominant soil type in the catchment |
| `Catchment_slope` | Average slope of the catchment draining to the site |
| `Landcover_catchment` | Dominant land cover (e.g. pastoral, urban, forest) |
| `watercraft_use` | Whether watercraft are common near the site |
| `sewage_discharge_beach` | Proximity to sewage discharge points |
| `high_intensity_agri_beach` | Whether high-intensity agriculture is nearby |

### Temporal

| Column | Description |
|--------|-------------|
| `Season` | Summer, Autumn, Winter, Spring |

## Preprocessing

Preprocessing is handled by the `Preprocessor` class in `src/data/preprocessing.py`. The steps are:

### 1. Column Selection

Only the relevant subset of columns is kept (roughly 35 columns from what may be a wider raw dataset). This ensures the model only sees features that are available in production.

### 2. DateTime Parsing

Dates are parsed and standardised. The code handles multiple date formats and converts everything to a consistent `dd/mm/yyyy HH:MM` format with `dayfirst=True`.

### 3. Deduplication

If a site has multiple samples at the same datetime, only the first is kept. Data is sorted by site and then by date.

### 4. Target Capping

Enterococci values are capped at **10,000 MPN/100mL**. Extreme outliers beyond this are almost certainly measurement artefacts and would distort model training.

### 5. Categorical Type Conversion

Columns like `SITE_NAME`, `Season`, `tidal_state`, `Harbour`, `Soil_type`, and others are converted to the `category` dtype. LightGBM can handle categorical features natively, which is more efficient than one-hot encoding.

### 6. Missing Value Handling

- **Numeric columns:** filled with the column mean (or median, configurable)
- **Daily measurements:** forward-filled within each day per site
- **Remaining gaps:** filled with the row mean (the average across all sites for that timestamp)

## Feature Engineering

Feature engineering is handled by the `FeatureEngineer` class in `src/data/feature_engineering.py`. Three types of features are created:

### 1. Temporal Features

Extracted from the `DateTime` column:

| Feature | Description |
|---------|-------------|
| `MONTH` | Month of year (1 to 12) |
| `WEEK` | ISO week number |
| `DAY_OF_WEEK` | Day of week (0 = Monday, 6 = Sunday) |
| `WEEKEND` | Binary flag: 1 if Saturday or Sunday |
| `TIME_OF_DAY` | Hour of day (0 to 23) |
| `HOLIDAY_FLAG` | Binary flag for New Zealand public holidays |

The holidays covered include New Year's Day, Waitangi Day, Good Friday, Easter Monday, Anzac Day, Queen's Birthday, Labour Day, Christmas Day, and Boxing Day, with approximate date ranges for each year.

Note: `YEAR` and `Season` are created temporarily but dropped after they are used to compute lagged features (to avoid data leakage or overfitting to specific years).

### 2. Wind-Shore Features

These classify wind direction relative to the beach orientation. For each time window (3h, 6h, 12h):

| Feature | Description |
|---------|-------------|
| `wind_shore_3h` | Wind classification for the 3-hour window |
| `wind_shore_6h` | Wind classification for the 6-hour window |
| `wind_shore_12h` | Wind classification for the 12-hour window |

The classification works as follows:

1. Calculate the "from" direction of the beach (the direction the beach faces + 180 degrees)
2. Compute the angular difference between the wind direction and the beach "from" direction
3. Classify:
   - **Onshore** — wind blowing from the sea towards the beach (angular difference < 45 degrees)
   - **Offshore** — wind blowing from land out to sea (angular difference > 135 degrees)
   - **Alongshore** — wind blowing roughly parallel to the beach (everything else)

Onshore winds push surface water (and contaminants) towards the beach, while offshore winds push them away. This is a meaningful predictor for water quality.

### 3. Lagged Enterococci Features

These capture the recent history of contamination at each site, grouped by season:

| Feature | Description |
|---------|-------------|
| `Site_Season_Average` | Rolling mean of the last 5 Enterococci samples for this site and season |
| `Site_Historical_Exceedance_Rate` | The proportion of all past samples (for this site and season) that exceeded 280 MPN/100mL |

Both features are **shifted by one observation** to prevent data leakage. The model never sees the current sample's value when computing these features.

These features are important because contamination at a site tends to be persistent. If a site has been elevated recently, it is more likely to still be elevated.

## Data Flow Summary

```
Raw CSV (data/processed/training_data.csv)
    │
    v
Preprocessor.clean_data()
    │  Select columns, parse dates, deduplicate, sort
    v
Preprocessor.set_max_target_value()
    │  Cap Enterococci at 10,000
    v
FeatureEngineer.engineer_features()
    │  Add temporal, wind-shore, and lagged features
    v
Preprocessor.transform_categorical_variable_type()
    │  Convert specified columns to category dtype
    v
Create clean snapshot (filter to dates <= 2024-12-31)
    │
    v
Model-ready DataFrame
```

The final DataFrame has roughly 40+ columns and is ready for model training, evaluation, or prediction.
