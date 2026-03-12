import pandas as pd
import numpy as np
import pytz
import os
import shap
import ast
from src_inference.config.constants import HOLIDAY_DATE_DATA
from datetime import datetime
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import random


def get_named_timezones():
    """
    Returns a dictionary of named timezones, which are used for generating the current
    date and time in different timezones.

    The timezones returned are:
        - "NZST": New Zealand Standard Time (UTC+12)
        - "NZLT": New Zealand Local Time (UTC+12/13, depending on daylight saving time)
        - "UTC": Coordinated Universal Time (UTC+0)

    Returns:
        dict: A dictionary of named timezones, mapped to their respective `pytz` objects.
    """
    return {
        "NZST": pytz.timezone('Etc/GMT-12'),
        "NZLT": pytz.timezone('Pacific/Auckland'),
        "UTC": pytz.timezone('UTC')
    }

def get_current_datetimes():
    """
    Returns a dictionary of the current date and time in different timezones.

    The timezones returned are:
        - "DateTime_NZST": New Zealand Standard Time (UTC+12), fixed and not daylight saving time (DST) aware
        - "DateTime_NZLT": New Zealand Local Time (UTC+12/13), DST-aware
        - "DateTime_UTC": Coordinated Universal Time (UTC+0)

    Returns:
        dict: A dictionary of the current date and time in different timezones, mapped to their respective `datetime` objects.
    """
    tzs = get_named_timezones()
    nz_fixed = datetime.now(tzs["NZST"]).replace(minute=0, second=0, microsecond=0)   # fixed +12
    nz_local = datetime.now(tzs["NZLT"]).replace(minute=0, second=0, microsecond=0)   # DST-aware
    utc_now  = datetime.now(tzs["UTC"]).replace(minute=0, second=0, microsecond=0)
    return {
        "DateTime_NZST": nz_fixed,
        "DateTime_NZLT": nz_local,   # prefer this for all "civil/local" timestamps
        "DateTime_UTC":  utc_now,
    }

def get_current_season(today: pd.Timestamp) -> str:
    """
    Determine the current season based on the given date.

    The season is defined as a period spanning from October of the current year
    to September of the following year. For example, if the input date is in
    November 2023, the returned season will be "2023-2024".

    Args:
        today (pd.Timestamp): The date for which the current season is to be determined.

    Returns:
        str: The current season in the format "YYYY-YYYY".
    """

    if today.month >= 10:
        season1 = today.year
        season2 = today.year + 1
    else:
        season1 = today.year - 1
        season2 = today.year
    return f"{season1}-{season2}"

def create_site_metadata(datetime_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Create site metadata including characteristics and temporal features.

    This function generates a pandas DataFrame containing metadata for various
    beach sites, including site characteristics, geographical coordinates,
    and temporal features based on the provided datetime. It also flags
    whether the date falls within a holiday period.

    Args:
        datetime_ts (pd.Timestamp): The reference datetime used to populate
        the DataFrame's temporal features.

    Returns:
        pd.DataFrame: A DataFrame containing the metadata for each site with
        additional temporal features and holiday flags.
    """

    site_data = pd.DataFrame({
        'SITE_NAME': [
            'Akaroa at main beach', 'Cass Bay at boat ramp', 'Charteris Bay Paradise Beach',
            'Church Bay Beach', 'Corsair Bay Beach', 'Diamond Harbour Beach',
            'Duvauchelle Bay by camping ground', 'French Farm Bay', 'Glen Bay boat ramp',
            'Governors Bay Sandy Beach', 'Purau Bay Beach', 'Rapaki Bay Beach',
            'Takamatua Bay Boat ramp', 'Tikao Bay mid beach', 'Wainui Beach Wainui Beach'
        ],
        'Harbour': [
            'Akaroa', 'Lyttelton', 'Lyttelton', 'Lyttelton', 'Lyttelton', 'Lyttelton',
            'Akaroa', 'Akaroa', 'Akaroa', 'Lyttelton', 'Lyttelton', 'Lyttelton',
            'Akaroa', 'Akaroa', 'Akaroa'
        ],
        'Shallowness': [
            'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'shallow',
            'medium', 'deep', 'shallow', 'medium', 'medium', 'medium', 'medium', 'deep'
        ],
        'Soil_type': [
            'sandy', 'stony', 'muddy', 'sandy', 'sandy', 'stony', 'muddy', 'muddy',
            'stony', 'muddy', 'muddy', 'muddy', 'muddy', 'sandy', 'stony'
        ],
        'Catchment_slope': [3, 6, 6, 7, 7, 7, 3, 5, 5, 6, 3, 6, 4, 7, 3],
        'Landcover_catchment': [
            'urban', 'urban', 'forest', 'forest', 'urban', 'urban', 'grass', 'grass',
            'grass', 'grass', 'grass', 'grass', 'grass', 'grass', 'grass'
        ],
        'watercraft_use': [1, 1, 1, 1, 1, 2, 1, 2, 1, 0, 1, 1, 1, 1, 1],
        'sewage_discharge_beach': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        'high_intensity_agri_beach': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        'beach_orientation_angle': [
            170, 15, 90, 130, 32, 140, 40, 230, 170, 8, 170, 20, 110, 320, 321
        ],
        'Latitude': [
            -43.8057, -43.6055, -43.6419, -43.633, -43.6079, -43.6255, -43.7544,
            -43.776, -43.8151, -43.6198, -43.6382, -43.6074, -43.785, -43.7958, -43.8129
        ],
        'Longitude': [
            172.9663, 172.6939, 172.7121, 172.7207, 172.6999, 172.7372, 172.9415,
            172.9104, 172.9511, 172.6579, 172.7528, 172.684, 172.9676, 172.9166, 172.9079
        ],
        'DateTime': [datetime_ts]*15
    })
    site_data['DateTime'] = pd.to_datetime(site_data['DateTime']).dt.tz_localize(None)
    
    # Add temporal features
    site_data['season'] = get_current_season(datetime_ts)
    site_data['YEAR'] = site_data['DateTime'].dt.year
    site_data['MONTH'] = site_data['DateTime'].dt.month
    site_data['WEEK'] = site_data['DateTime'].dt.isocalendar().week
    site_data['DAY_OF_WEEK'] = site_data['DateTime'].dt.dayofweek
    site_data['WEEKEND'] = site_data['DateTime'].dt.dayofweek.isin([5, 6]).astype(int)
    site_data['TIME_OF_DAY'] = site_data['DateTime'].dt.hour

    # Add holiday flag
    site_data['HOLIDAY_FLAG'] = 0
    for start, end in HOLIDAY_DATE_DATA.values():
        start_date = pd.to_datetime(start, format='%d-%m-%Y')
        end_date = pd.to_datetime(end, format='%d-%m-%Y')
        site_data.loc[(site_data['DateTime'] >= start_date) & (site_data['DateTime'] <= end_date), 'HOLIDAY_FLAG'] = 1

    return site_data


def _to_float_with_censor(text: str | None) -> float | None:
    """
    Convert Hilltop string values into numeric with censor rules:
      - '<n' -> 0.5 * n
      - '>n' -> 1.1 * n
    Handles commas, spaces, and rare unicode comparators.
    Returns None if conversion is impossible.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s or s.lower() in ("nan", "null", "none"):
        return None

    # remove thousands separators
    s_nocomma = s.replace(",", "")

    # <n / >n / <=n / >=n
    m = re.match(r'^([<>]=?)\s*(\d+(?:\.\d+)?)$', s_nocomma)
    if m:
        op, num = m.group(1), float(m.group(2))
        if op.startswith("<"):
            return 0.5 * num
        else:  # '>' or '>='
            return 1.1 * num

    # unicode comparators ≤ ≥
    m2 = re.match(r'^([≤≥])\s*(\d+(?:\.\d+)?)$', s_nocomma)
    if m2:
        op, num = m2.group(1), float(m2.group(2))
        if op == "≤":          # treat as threshold itself
            return num
        else:                  # ≥ behaves like '>'
            return 1.1 * num

    # plain number
    try:
        return float(s_nocomma)
    except Exception:
        return None


# --- Fetch Enterococci Helper ---
def fetch_enterococci_data_for_sites(
    site_codes,
    from_date,
    to_date,
    logger=None,
    save_path=None,
    *,
    event_fn=None,            # pass inference.log_event here; optional
    timeout_s=20,
    max_retries=3,
    backoff_base=2.0,
):
    """
    Fetch Enterococci data for the given sites and time window.

    Args:
        site_codes (dict): Site name -> Hilltop code.
        from_date (str): 'YYYY-MM-DD'
        to_date   (str): 'YYYY-MM-DD'
        logger (Logger, optional)
        save_path (str, optional): If set, saves CSV of raw results.
        event_fn (callable, optional): function(event, level='INFO', **fields)
        timeout_s (int): per-request timeout seconds
        max_retries (int): attempts per site
        backoff_base (float): exponential backoff base

    Returns:
        pd.DataFrame with ['SITE_NAME','DateTime','Enterococci']
    """
    base_url = "http://wateruse.ecan.govt.nz/WQSurfaceWater.hts"
    endpoint_host = "wateruse.ecan.govt.nz"
    measurement = "Enterococci"
    all_data = []

    def _log_event(evt, level="INFO", **kw):
        if event_fn:
            try:
                event_fn(evt, level=level, **kw)
            except Exception:
                # never let event logging break the fetch
                if logger:
                    logger.exception("Failed to emit hilltop event")

    for name, code in site_codes.items():
        params = {
            "Service": "Hilltop",
            "Request": "GetData",
            "Site": code,
            "Measurement": measurement,
            "TimeInterval": f"{from_date}/{to_date}",
        }
        url = base_url + "?" + urllib.parse.urlencode(params)
        if logger:
            logger.info(f"  Site: {name} ({code}) - {url}")

        attempt = 0
        site_records = []
        http_status = None
        t0_run = time.perf_counter()

        while attempt < max_retries:
            attempt += 1
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "ECAN-Entero-Inference/1.0 (+Azure Container Apps)",
                        "Accept": "application/xml",
                    },
                )
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    http_status = getattr(resp, "status", None) or resp.getcode()
                    xml_bytes = resp.read()
                elapsed_ms = int((time.perf_counter() - t0) * 1000)

                # Parse XML
                root = ET.fromstring(xml_bytes)
                records = []
                for entry in root.findall(".//E"):
                    t_node = entry.find("T")
                    v_node = entry.find("Value")
                    dt_text = t_node.text if t_node is not None else None
                    val_raw = v_node.text if v_node is not None else None
                    val_clean = _to_float_with_censor(val_raw)
                    records.append({"SITE_NAME": name, "DateTime": dt_text, "Enterococci": val_clean})

                if records:
                    site_records = records
                    _log_event(
                        "SOURCE_OK",
                        source="hilltop",
                        endpoint_host=endpoint_host,
                        site_name=name,
                        site_code=code,
                        rows=len(records),
                        http_status=http_status,
                        elapsed_ms=elapsed_ms,
                        retry_count=attempt - 1,
                    )
                else:
                    _log_event(
                        "SOURCE_DEGRADED",
                        source="hilltop",
                        endpoint_host=endpoint_host,
                        site_name=name,
                        site_code=code,
                        failure_kind="empty_payload",
                        http_status=http_status,
                        elapsed_ms=elapsed_ms,
                        retry_count=attempt - 1,
                    )
                # success path (even if empty) -> stop retrying
                break

            except Exception as e:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                if attempt < max_retries:
                    if logger:
                        logger.warning(
                            f"    Hilltop fetch error for {name} ({code}) attempt {attempt}/{max_retries}: {e} "
                            f"(elapsed {elapsed_ms} ms). Retrying..."
                        )
                    # jittered exponential backoff
                    sleep_s = (backoff_base ** (attempt - 1)) + random.uniform(0, 0.5)
                    time.sleep(sleep_s)
                    continue
                else:
                    if logger:
                        logger.warning(
                            f"    Hilltop fetch failed for {name} after {max_retries} attempts: {e}"
                        )
                    _log_event(
                        "SOURCE_DOWN",
                        level="ERROR",
                        source="hilltop",
                        endpoint_host=endpoint_host,
                        site_name=name,
                        site_code=code,
                        failure_kind="exception",
                        error=str(e)[:300],
                        http_status=http_status,
                        elapsed_ms=elapsed_ms,
                        retry_count=attempt,
                    )
                    # give up on this site
                    site_records = []
                    break

        # accumulate any records (may be empty for this site)
        all_data.extend(site_records)
        # small politeness delay between sites
        time.sleep(0.25)

    # build DataFrame
    df = pd.DataFrame(all_data)
    if not df.empty:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce").dt.tz_localize(None)

        if save_path:
            df.to_csv(save_path, index=False)
            if logger:
                logger.info(f"Raw Enterococci values saved to {save_path}")

        if logger:
            logger.info(f"Returned dataframe shape: {df.shape}")
    else:
        # Nothing came back for the entire window
        _log_event(
            "SOURCE_DEGRADED",
            source="hilltop",
            endpoint_host=endpoint_host,
            failure_kind="empty_payload_all_sites",
            sites=len(site_codes),
            window=f"{from_date}/{to_date}",
        )

    return df



def conform_features_to_training(df: pd.DataFrame, reference_df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Align inference DataFrame to match the training DataFrame structure:
    - Removes extra columns.
    - Adds missing columns as NaN.
    - Reorders columns to match training.

    Args:
        df (pd.DataFrame): Inference data to align.
        reference_df (pd.DataFrame): DataFrame used during training (includes expected feature columns).
        logger (logging.Logger, optional): Logger to record operations.

    Returns:
        pd.DataFrame: Aligned DataFrame with the same columns and order as the training data.
    """
    expected_features = [col for col in reference_df.columns if col != 'Enterococci']
    extra_cols = set(df.columns) - set(expected_features)
    missing_cols = set(expected_features) - set(df.columns)

    if extra_cols and logger:
        logger.info(f"Removing extra columns from inference data: {extra_cols}")
    df = df.drop(columns=list(extra_cols), errors='ignore')

    if missing_cols and logger:
        logger.info(f"Adding missing columns (NaN) to inference data: {missing_cols}")
    for col in missing_cols:
        df[col] = np.nan

    df = df[expected_features]  # Reorder to match training
    return df


#### The following two functions: get_cat_info_from_model_txt and apply_model_categoricals, are key for solving one of the biggest headaches in real-world inferenc with scikit-learn and LightGBM models: categorical features ######
#### MAKING SURE your inference-time categorical columns exactly match the encoding, order, and levels used during training #### 

def get_cat_info_from_model_txt(model_txt_path):
    """Extract categorical indices and category levels from model text file."""
    with open(model_txt_path, "r") as f:
        lines = f.readlines()
    cat_idx_line = [l for l in lines if l.startswith("[categorical_feature:")]
    cat_levels_line = [l for l in lines if l.strip().startswith("pandas_categorical:")]
    if not cat_idx_line or not cat_levels_line:
        raise ValueError("Categorical info not found in model file.")
    # Extract indices (as ints)
    cat_indices = [
        int(i) for i in cat_idx_line[0]
            .replace("[categorical_feature:", "")
            .replace("]", "")
            .strip().split(",") if i != ""
    ]
    # Extract list of list-of-levels (outer list is one per categorical col, inner list is string categories)
    cat_levels = ast.literal_eval(cat_levels_line[0].split(":", 1)[1].strip())
    return cat_indices, cat_levels

def apply_model_categoricals(df, feature_names, cat_indices, cat_levels, logger=None):
    """Assign category dtype using model-derived levels."""
    for idx, cats in zip(cat_indices, cat_levels):
        col = feature_names[idx]
        if col in df.columns:
            prev_dtype = df[col].dtype
            df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))
            if logger:
                logger.info(f"Set {col} as category with {len(cats)} levels from model file. Prev dtype: {prev_dtype}")
            obs_cats = set(df[col].dropna().unique())
            model_cats = set(cats)
            if not obs_cats.issubset(model_cats):
                if logger:
                    logger.warning(f"{col}: Inference data has unseen categories: {obs_cats - model_cats}")
        else:
            if logger:
                logger.warning(f"{col} from model categoricals not found in inference data.")
    return df



def prepare_shap_input(df, feature_names, cat_indices, cat_levels, reference_df, logger=None):
    shap_input = df[feature_names].copy()

    for idx, cats in zip(cat_indices, cat_levels):
        col = feature_names[idx]
        if col in shap_input.columns:
            shap_input[col] = shap_input[col].astype(
                pd.CategoricalDtype(categories=cats, ordered=False)
            )

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

    bad_dtypes = [c for c in shap_input.columns if shap_input[c].dtype == object]
    if bad_dtypes:
        if logger:
            logger.warning(f"SHAP input still has object dtypes: {bad_dtypes}")
        raise ValueError(f"SHAP input bad dtypes: {bad_dtypes}")

    return shap_input



def compute_shap_df(model, shap_input, logger=None):
    """
    Compute SHAP values for interpretability using TreeExplainer.
    
    Parameters
    ----------
    model : object
        Trained LightGBM model
    shap_input : pd.DataFrame
        Preprocessed DataFrame with exactly the same
        categorical levels and dtypes as the model expects
    logger : Optional[logging.Logger]
        Logger to write SHAP computation details to
        (default: None)

    Returns
    -------
    shap_df : pd.DataFrame
        DataFrame of SHAP values (one column per feature)
    base_value : float
        Model's expected value (i.e., the value when all SHAP values are zero)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_input)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{col}" for col in shap_input.columns])

    base_value = explainer.expected_value
    try:
        # Coerce list/ndarray → float (mean if vector)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.asarray(base_value).mean())
        else:
            base_value = float(base_value)
    except Exception:
        # last-ditch: leave as-is
        pass

    if logger:
        logger.info(f"SHAP values computed. SHAP DataFrame shape: {shap_df.shape}. Base value: {base_value}")
    return shap_df, base_value



def is_in_bathing_season(date: pd.Timestamp):
    """Return True if date is in NZ bathing season (Oct–Mar), else False."""
    return date.month in [10, 11, 12, 1, 2, 3]

def assign_bathing_season(dt):
    """Return season label as 'YYYY-YYYY+1'."""
    year = dt.year
    if dt.month >= 10:
        return f"{year}-{year + 1}"
    else:
        return f"{year - 1}-{year}"

def add_seasonal_features(nowcast_df, hist_df, reference_time, logger=None):
    # Default: blanks
    """
    Computes two seasonal features for each site in nowcast_df

    1. `Site_Season_Average`: the rolling mean of the last 5 Enterococci samples (min 1) for the current season

    2. `Site_Historical_Exceedance_Rate`: the proportion of past samples >280 for the current season

    If the current date is OUT of the bathing season, these features are left blank.

    If Hilltop returned nothing for this season (new season / outage), these features are left blank.

    Parameters
    ----------
    nowcast_df : pd.DataFrame
        Nowcast data with 'SITE_NAME' and 'DateTime' columns
    hist_df : pd.DataFrame or None
        Historical Enterococci data with 'SITE_NAME' and 'DateTime' columns
    reference_time : pd.Timestamp
        Current time for which to add seasonal features
    logger : logging.Logger or None

    Returns
    -------
    pd.DataFrame
        Nowcast data with added seasonal features
    """
    nowcast_df['Site_Season_Average'] = np.nan
    nowcast_df['Site_Historical_Exceedance_Rate'] = np.nan

    # Off-season → keep blanks
    if not is_in_bathing_season(reference_time):
        if logger:
            logger.info(f"{reference_time.date()} is OUT of bathing season; features left blank.")
        return nowcast_df

    # If Hilltop returned nothing (new season / outage), keep blanks and exit
    if hist_df is None or hist_df.empty or 'DateTime' not in hist_df.columns:
        if logger:
            logger.info("Hilltop history is empty for this season; seasonal features left blank.")
        return nowcast_df

    # From here on we definitely have rows + DateTime
    this_season = assign_bathing_season(reference_time)
    if logger:
        logger.info(f"{reference_time.date()} in bathing season ({this_season}); computing seasonal features.")

    hist_df = hist_df.copy()
    hist_df['DateTime'] = pd.to_datetime(hist_df['DateTime'], errors='coerce')
    hist_df['Season'] = hist_df['DateTime'].apply(assign_bathing_season)

    for i, row in nowcast_df.iterrows():
        site = row['SITE_NAME']
        site_hist = hist_df[
            (hist_df['SITE_NAME'] == site) &
            (hist_df['Season'] == this_season) &
            (hist_df['DateTime'] < reference_time)
        ].sort_values('DateTime')

        if site_hist.empty:
            site_season_avg = np.nan
            site_exc_rate   = np.nan
        else:
            means = site_hist['Enterococci'].rolling(window=5, min_periods=1).mean()
            site_season_avg = means.iloc[-1] if len(means) else np.nan
            n_prev   = len(site_hist)
            n_exceed = (site_hist['Enterococci'] > 280).sum()
            site_exc_rate = (n_exceed / n_prev) if n_prev > 0 else np.nan

        nowcast_df.at[i, 'Site_Season_Average'] = site_season_avg
        nowcast_df.at[i, 'Site_Historical_Exceedance_Rate'] = site_exc_rate

    return nowcast_df


## our model gives us a spread of likely outcomes at different percentiles (quantiles), and not a single number. For the point forecast we currently use the median of the 12 quantiles and this falls somewhere around the 78th percentile, which is a good risk-averse setup. ##
## So, making use of the full distribution, we can convert these quantiles to construct an estimated CDF (cumulative distribution function) and then answer the question of 'what % of outcomes will exceed the threshold of 280?' ##


def prob_exceed_vectorised(Q: np.ndarray, p: np.ndarray, y_threshold: float, debug: bool = False) -> np.ndarray:
    """
    Vectorised piece-wise linear CDF inversion to compute P(Y > y_threshold)
    from a set of predicted quantiles.

    Parameters
    ----------
    Q : ndarray, shape (n_obs, n_q)
        Quantile predictions per row, ASCENDING in τ.
    p : ndarray, shape (n_q,)
        Quantile levels (0 < τ < 1), ASCENDING (e.g., 0.05, 0.2, ...).
    y_threshold : float
        The exceedance threshold (e.g. 280).
    debug : bool
        If True, prints small samples of intermediate arrays.

    Returns
    -------
    ndarray, shape (n_obs,)
        P(Y > y_threshold) for each observation.
    """
    Q = Q.astype(float, copy=False)
    n_obs, n_q = Q.shape

    # Locate segment k s.t. Q_k <= y < Q_{k+1}; clip to valid range
    idx = (Q < y_threshold).sum(axis=1) - 1
    idx = np.clip(idx, 0, n_q - 2)

    row = np.arange(n_obs)
    Q_lo = Q[row, idx]
    Q_hi = Q[row, idx + 1]
    p_lo = p[idx]
    p_hi = p[idx + 1]

    # Linear interpolation of CDF at y_threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        Fy = p_lo + (y_threshold - Q_lo) / (Q_hi - Q_lo) * (p_hi - p_lo)

    # Strict bounds: let equality be handled by interpolation
    below_min = y_threshold < Q[:, 0]
    above_max = y_threshold > Q[:, -1]
    Fy[below_min] = 0.0
    Fy[above_max] = 1.0

    # Degenerate segments (Q_hi == Q_lo) or NaNs from division → fall back to upper τ
    degenerate = (Q_hi == Q_lo) | np.isnan(Fy)
    Fy = np.where(degenerate, p_hi, Fy)

    if debug:
        print("idx sample:", idx[:5])
        print("Q_lo sample:", Q_lo[:5])
        print("Q_hi sample:", Q_hi[:5])
        print("p_lo sample:", p_lo[:5])
        print("p_hi sample:", p_hi[:5])
        print("Fy sample:", Fy[:5])
        print("P(Y>y) sample:", (1.0 - Fy)[:5])

    # P(Y > y) = 1 - F(y)
    return 1.0 - Fy
            
########### end #############

# def align_remaining_dtypes(df, reference_df, exclude=["DateTime"], logger=None):
#     for col in df.columns:
#         if col in exclude:
#             continue
#         if not pd.api.types.is_categorical_dtype(df[col]):
#             dtype = reference_df[col].dtype
#             try:
#                 df[col] = df[col].astype(dtype)
#             except Exception as e:
#                 if logger:
#                     logger.warning(f"Could not cast {col} to {dtype}: {e}")
#     return df


# def update_predictions_csv(new_df, storage_path, key_cols=["DateTime", "SITE_NAME"]):
#     """
#     Appends current predictions to an hourly CSV log, replacing any existing
#     records for the same DateTime + SITE_NAME keys.

#     Ensures clean merging and avoids legacy column pollution.
#     """

#     # Ensure directory exists
#     os.makedirs(os.path.dirname(storage_path), exist_ok=True)

#     if os.path.exists(storage_path):
#         # Read existing
#         old_df = pd.read_csv(storage_path)

#         # Ensure merge keys are strings for consistency
#         for col in key_cols:
#             old_df[col] = old_df[col].astype(str)
#             new_df[col] = new_df[col].astype(str)

#         # Remove any old rows that match the new DateTime+SITE_NAME
#         deduped_old = old_df.merge(new_df[key_cols], on=key_cols, how='left', indicator=True)
#         deduped_old = deduped_old[deduped_old['_merge'] == 'left_only'].drop(columns=['_merge'])

#         # Combine and sort
#         combined_df = pd.concat([deduped_old, new_df], ignore_index=True)
#         combined_df = combined_df.sort_values(key_cols).reset_index(drop=True)

#     else:
#         combined_df = new_df

#     # Save back
#     combined_df.to_csv(storage_path, index=False)