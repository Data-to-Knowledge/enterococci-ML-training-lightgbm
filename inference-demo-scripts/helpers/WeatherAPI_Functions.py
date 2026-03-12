# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:32:35 2024

@author: AtmanD and NinaS
"""

#%% packages
# Try .env for local dev; explicit path so it works from any CWD
try:
    from dotenv import load_dotenv, find_dotenv
    for env_name in (".env.local", ".env"):
        p = find_dotenv(env_name, usecwd=True)
        if p:
            load_dotenv(p, override=False)
except Exception:
    pass

import requests
import json as j
from requests.auth import HTTPBasicAuth
import pandas as pd
import datetime as dt
import numpy as np
import io
from datetime import timedelta as td
from datetime import timezone as tz
import time
import os
import random
import urllib.parse
import re


#%% set up API credentials (from environment variables)

MET_KEY = os.getenv('METSERVICE_KEY')
username = os.getenv('NIWA_USERNAME')
password = os.getenv('NIWA_KEY')

#%% functions

def wind_vector_components(direction, speed):
    """Get the wind vector components from wind speed and direction.

    Parameters
    ----------
    direction : wind direction (0-360 degrees), specified as the direction from which the wind is blowing.
    speed : wind speed (magnitude, usually in knots)
   
    Returns
    -------
    Ve, Vn: tuple of wind components in the x (east-west) and y (north-south) directions.

    """
    Ve = speed * np.sin(direction * np.pi/180)
    Vn = speed * np.cos(direction * np.pi/180)

    return Ve, Vn


def _emit(log_event_fn, event, **fields):
    """
    Thin shim so callers can pass a componentised emitter (e.g., ev_met/ev_niwa).
    Keeps keys flat/small for KQL.
    """
    if log_event_fn:
        log_event_fn(event, **fields)

def _metservice_fetch_24h(datetime, station_id, *,
                          log_event_fn=None, emit_var_ok=False,
                          max_retries=2, base_backoff_sec=5,
                          total_deadline_s=None):

    """
    Fetch MetService API data for a given station_id and datetime.

    Parameters
    ----------
    datetime : datetime object, representing the end of the 24-hour window.
    station_id : str, the MetService station ID.

    Returns
    -------
    pd.DataFrame, containing the fetched data. The DataFrame has columns 'DateTime', 'Rainfall', 'wind_direction', 'wind_speed', 'Ve', 'Vn', and 'StationId'. The 'DateTime' column is in UTC time.

    Notes
    ------
    * The function will perform exponential backoff with jitter if max_retries > 0.
    * The function will log events if log_event_fn is not None.
    * The function will emit a 'SOURCE_DOWN' event if the given station_id is invalid or the given datetime is not a valid date.
    * The function will emit a 'SOURCE_VAR_EMPTY' event if a variable is empty.
    * The function will emit a 'SOURCE_VAR_OK' event if a variable is not empty.
    * The function will emit a 'SOURCE_RETRY' event if an exception occurs during the request.
    * The function will emit a 'SOURCE_OK' event if the request is successful.
    * The function will emit a 'SOURCE_DOWN' event if the station time budget is exhausted.
    * The function will emit a 'SOURCE_RETRY' event if the station time budget is exceeded during the request.

    Hard-stop if the total time spent for this station (all vars + retries)
    exceeds `total_deadline_s` (if provided).
    """

    host = "api.metservice.com"

    if not MET_KEY:
        _emit(log_event_fn, "SOURCE_DOWN",
              source="metservice", endpoint_host=host, station_id=station_id,
              failure_kind="missing_api_key")
        return pd.DataFrame(columns=['DateTime','Rainfall','wind_direction','wind_speed','Ve','Vn'])

    # Optional: exponential backoff with jitter
    def _sleep_backoff(base, attempt):
        delay = min(base * (2 ** (attempt - 1)), 15) * random.uniform(0.85, 1.15)
        time.sleep(delay)

    l = []
    variable_dict = {
        'rainfal_01hracc': 'Rainfall',
        'winddir_01hravg': 'wind_direction',
        'windspd_01hravg': 'wind_speed'
    }

    # window: last ~24h (23h59m back)
    date_start = datetime + dt.timedelta(days=-1, minutes=1)
    time_start = dt.datetime.strftime(date_start, '%H:%M:%S')
    date_start = date_start.date()
    date = datetime.date()
    time_str = dt.datetime.strftime(datetime, '%H:%M:%S')

    dropped_non_hour_rows = 0
    successes = 0
    retries_total = 0

    t_station_start = time.monotonic()
    # allow env override for ops tuning (e.g., MET_TDEADLINE_S=20)
    if total_deadline_s is None:
        try:
            total_deadline_s = float(os.getenv("MET_TDEADLINE_S", "20"))
        except Exception:
            total_deadline_s = 20.0

    for key, var_name in variable_dict.items():
        # Hard stop if the station time budget is exhausted
        if (time.monotonic() - t_station_start) > total_deadline_s:
            _emit(log_event_fn, "SOURCE_DOWN",
                  source="metservice", endpoint_host=host, station_id=station_id,
                  failure_kind="station_deadline_exceeded",
                  deadline_s=total_deadline_s, vars_collected=len(l))
            break

        url = (
            f"https://{host}/observations/nz/1-minute/weatherStation/"
            f"{station_id}/range/{date_start}T{time_start}Z/{date}T{time_str}Z?vars={key}"
        )
        headers = {
            'apikey': MET_KEY,
            'User-Agent': 'ecan-enterococci/1.0 (+azure-container-app)'
        }
        results = []
        attempt = 0

        while attempt < max_retries:
            # Hard stop check before each attempt
            if (time.monotonic() - t_station_start) > total_deadline_s:
                _emit(log_event_fn, "SOURCE_RETRY",
                      source="metservice", endpoint_host=host, station_id=station_id, var=key,
                      failure_kind="station_deadline_exceeded",
                      attempt=attempt, max_retries=max_retries, deadline_s=total_deadline_s)
                attempt = max_retries  # force exit this var
                break

            attempt += 1
            t0 = time.monotonic()
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                status = resp.status_code

                response_json = j.loads(resp.text)
                if "results" not in response_json:
                    _emit(log_event_fn, "SOURCE_RETRY",
                          source="metservice", endpoint_host=host, station_id=station_id, var=key,
                          failure_kind="missing_results_key", http_status=status,
                          attempt=attempt, max_retries=max_retries, retry_in_sec=base_backoff_sec)
                    if attempt < max_retries:
                        retries_total += 1
                        _sleep_backoff(base_backoff_sec, attempt)
                        continue
                    break

                results = response_json["results"] or []
                if not results:
                    _emit(log_event_fn, "SOURCE_VAR_EMPTY",
                          source="metservice", endpoint_host=host, station_id=station_id, var=key,
                          http_status=status, attempt=attempt, elapsed_ms=elapsed_ms)
                else:
                    if emit_var_ok:
                        _emit(log_event_fn, "SOURCE_VAR_OK",
                              source="metservice", endpoint_host=host, station_id=station_id, var=key,
                              http_status=status, rows=len(results), elapsed_ms=elapsed_ms, attempt=attempt)
                break  # done with this var (even if empty)
            except Exception as e:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                _emit(log_event_fn, "SOURCE_RETRY",
                      source="metservice", endpoint_host=host, station_id=station_id, var=key,
                      failure_kind="exception", error=str(e)[:300],
                      attempt=attempt, max_retries=max_retries, retry_in_sec=base_backoff_sec,
                      elapsed_ms=elapsed_ms)
                if attempt < max_retries:
                    retries_total += 1
                    _sleep_backoff(base_backoff_sec, attempt)
                    continue
                break

        # json → dataframe
        if results:
            data = pd.json_normalize(results)
            data['obs_timestamp'] = pd.to_datetime(data['obs_timestamp'], utc=True)
            before = len(data)
            data.drop(data[data['obs_timestamp'].dt.minute != 0].index, inplace=True)
            dropped_non_hour_rows += (before - len(data))
            data.rename(columns={'obs_timestamp': 'DateTime', key: var_name}, inplace=True)
            l.append(data)
            successes += 1

    if not l:
        _emit(log_event_fn, "SOURCE_DOWN",
              source="metservice", endpoint_host=host, station_id=station_id,
              failure_kind="all_vars_empty_or_failed_or_deadline",
              window_hours=24, retries=retries_total, deadline_s=total_deadline_s)
        return pd.DataFrame(columns=['DateTime','Rainfall','wind_direction','wind_speed','Ve','Vn'])

    # combine (guard missing cols)
    df = pd.concat(l, ignore_index=True)
    for col in ['Rainfall','wind_direction','wind_speed']:
        if col not in df.columns:
            df[col] = np.nan

    df = df.groupby('DateTime', as_index=False).agg(
        Rainfall=('Rainfall','max'),
        wind_direction=('wind_direction','max'),
        wind_speed=('wind_speed','max')
    )

    df['tuple'] = df.apply(lambda r: wind_vector_components(r['wind_direction'], r['wind_speed']), axis=1)
    df[['Ve','Vn']] = df['tuple'].apply(pd.Series)
    df.drop(['tuple'], axis=1, inplace=True)
    df.sort_values(by='DateTime', ascending=False, inplace=True)
    df['StationId'] = str(station_id)

    _emit(log_event_fn, "SOURCE_OK",
          source="metservice", endpoint_host=host, station_id=station_id, window_hours=24,
          rows=len(df), vars_success=successes, retries=retries_total,
          dropped_non_hour_rows=dropped_non_hour_rows, deadline_s=total_deadline_s)

    return df


# --- Helper: merge primary + backup station hour-by-hour ----------------------
def _merge_met_station_frames(primary: pd.DataFrame,
                              backup: pd.DataFrame | None,
                              *,
                              log_event_fn=None,
                              primary_id: str | None = None,
                              backup_id: str | None = None,
                              fallback_mode: str = "pair_wind") -> pd.DataFrame:
    """
    Merge MetService primary + backup by hour.
    - Rainfall filled independently per-hour from backup if primary is NaN.
    - Wind treated as a pair: if either dir/speed missing on primary for an hour,
      and backup has BOTH, take BOTH from backup for that hour.
    - Recompute Ve/Vn from the final wind dir/speed after merge (single source of truth).
    - fallback_mode:
        * "pair_wind" (default): logic above.
        * "hour": if ANY of (Rainfall, wind_dir, wind_speed) missing on primary at hour,
                  use ALL THREE from backup for that hour (if available).
        * "off": return primary unchanged.
    """
    if fallback_mode == "off" or backup is None or backup.empty:
        # Keep original behaviour
        out = primary.copy()
        # Ensure expected columns exist
        for c in ['Rainfall','wind_direction','wind_speed','Ve','Vn','StationId']:
            if c not in out.columns:
                out[c] = np.nan
        # Recompute Ve/Vn from dir/speed (idempotent; preserves consistency)
        try:
            tuples = out.apply(lambda r: wind_vector_components(r.get('wind_direction', np.nan),
                                                               r.get('wind_speed', np.nan)), axis=1)
            out[['Ve','Vn']] = tuples.apply(pd.Series)
        except Exception:
            pass
        return out

    p = primary.copy()
    b = backup.copy()

    # Normalize datetimes to hourly UTC, ensure columns exist
    p['DateTime'] = pd.to_datetime(p['DateTime'], utc=True, errors='coerce')
    b['DateTime'] = pd.to_datetime(b['DateTime'], utc=True, errors='coerce')
    p['DateTime'] = p['DateTime'].dt.floor('h')
    b['DateTime'] = b['DateTime'].dt.floor('h')
    for c in ['Rainfall','wind_direction','wind_speed']:
        if c not in p.columns: p[c] = np.nan
        if c not in b.columns: b[c] = np.nan

    # Build union hourly index over the window we actually fetched
    idx = pd.Index(sorted(set(p['DateTime'].dropna().tolist()) |
                          set(b['DateTime'].dropna().tolist())), name='DateTime')
    p2 = p.set_index('DateTime')[['Rainfall','wind_direction','wind_speed']].reindex(idx)
    b2 = b.set_index('DateTime')[['Rainfall','wind_direction','wind_speed']].reindex(idx)

    # Provenance (optional; helpful for debugging — can be dropped later)
    src_rain = pd.Series(data=np.where(~p2['Rainfall'].isna(), 'primary', 
                                       np.where(~b2['Rainfall'].isna(), 'backup', 'none')),
                         index=idx, name='src_rain')
    # Wind pair decision (primary complete? else backup complete? else none)
    prim_wind_ok = (~p2['wind_direction'].isna()) & (~p2['wind_speed'].isna())
    back_wind_ok = (~b2['wind_direction'].isna()) & (~b2['wind_speed'].isna())

    if fallback_mode == "hour":
        # If ANY var missing on primary at an hour → use backup for ALL THREE if backup has them
        prim_any_missing = (
            p2['Rainfall'].isna() | (~prim_wind_ok)
        )
        take_backup_all = prim_any_missing & (
            (~b2['Rainfall'].isna()) & back_wind_ok
        )
        # Start with primary
        Rainfall = p2['Rainfall'].copy()
        wd = p2['wind_direction'].copy()
        ws = p2['wind_speed'].copy()
        # Overwrite hours where we take backup-all
        Rainfall.loc[take_backup_all] = b2['Rainfall'].loc[take_backup_all]
        wd.loc[take_backup_all] = b2['wind_direction'].loc[take_backup_all]
        ws.loc[take_backup_all] = b2['wind_speed'].loc[take_backup_all]
        src_wind = pd.Series(np.where(prim_wind_ok, 'primary',
                               np.where(back_wind_ok, 'backup', 'none')), index=idx, name='src_wind')
        # Force src_wind to 'backup' at hours we took all from backup
        src_wind.loc[take_backup_all] = 'backup'
        patched_rain_hours = idx[ (~p2['Rainfall'].isna()) & (src_rain == 'primary') == False & (Rainfall.notna()) ]
        patched_wind_hours = idx[ (~prim_wind_ok) & back_wind_ok ]
    else:
        # pair_wind default
        # Rainfall: independent fill per-hour
        Rainfall = p2['Rainfall'].where(~p2['Rainfall'].isna(), b2['Rainfall'])
        # Wind: pairwise – use primary if complete else backup if complete else NaN
        wd = p2['wind_direction'].where(prim_wind_ok, np.where(back_wind_ok, b2['wind_direction'], np.nan))
        ws = p2['wind_speed'].where(prim_wind_ok, np.where(back_wind_ok, b2['wind_speed'], np.nan))
        src_wind = pd.Series(np.where(prim_wind_ok, 'primary',
                               np.where(back_wind_ok, 'backup', 'none')), index=idx, name='src_wind')
        patched_rain_hours = idx[ p2['Rainfall'].isna() & Rainfall.notna() ]
        patched_wind_hours = idx[ (~prim_wind_ok) & back_wind_ok ]

    # Recompute Ve/Vn from final wind dir/speed
    out = pd.DataFrame({'DateTime': idx, 'Rainfall': Rainfall, 
                        'wind_direction': wd, 'wind_speed': ws})
    try:
        tuples = out.apply(lambda r: wind_vector_components(r['wind_direction'], r['wind_speed']), axis=1)
        out[['Ve','Vn']] = tuples.apply(pd.Series)
    except Exception:
        out['Ve'] = np.nan; out['Vn'] = np.nan

    # Keep a single StationId column (primary by default). We can't mix per-hour in this column
    # without changing downstream contracts, so we summarize usage instead.
    out['StationId'] = primary_id if primary_id is not None else (primary.get('StationId').iloc[0] if 'StationId' in primary.columns and not primary.empty else None)

    # Optional provenance columns (drop later if undesired)
    out['src_rain'] = src_rain.values
    out['src_wind'] = src_wind.values
    if primary_id or backup_id:
        used = [s for s in {primary_id, backup_id} if s]
        out['stations_used'] = ",".join(used)

    # Telemetry
    try:
        if len(patched_rain_hours) > 0:
            _emit(log_event_fn, "SOURCE_VAR_FALLBACK_USED",
                  source="metservice",
                  endpoint_host="api.metservice.com",
                  var="Rainfall",
                  from_station=primary_id, to_station=backup_id,
                  hours=[ts.strftime("%Y-%m-%dT%H:%MZ") for ts in patched_rain_hours.tolist()],
                  count=int(len(patched_rain_hours)))
        if len(patched_wind_hours) > 0:
            _emit(log_event_fn, "SOURCE_WINDPAIR_FALLBACK_USED",
                  source="metservice",
                  endpoint_host="api.metservice.com",
                  from_station=primary_id, to_station=backup_id,
                  hours=[ts.strftime("%Y-%m-%dT%H:%MZ") for ts in patched_wind_hours.tolist()],
                  count=int(len(patched_wind_hours)))
    except Exception:
        pass

    return out.reset_index(drop=True)


def get_hourly_weather_data_Lyttelton(datetime,
                                      station_candidates=("93786", "93951"),
                                      log_event_fn=None, logger=None,
                                      max_retries=2, base_backoff_sec=5,
                                      total_deadline_s=None):
    """
    Fetch hourly weather data for the given datetime from MetService API for stations at Lyttelton.

    Behaviour:
      - Fetch primary station first
      - If the 24h windows have ANY gaps (Rainfall NaNs OR the wind pair incomplete)
        and there is time budget, fetch the first available backup station and
        patch per-hour:
            * Rainfall filled per-hour if primary is NaN.
            * Wind filled as a pair (direction+speed) when primary pair missing.
        Then recompute Ve/Vn from the final wind direction/speed.
      - If no gaps, we do NOT hit the backup.

    Config:
      - MET_FALLBACK_MODE = "pair_wind" (default) | "hour" | "off"
      - MET_BACKUP_TDEADLINE_S (optional) defaults to MET_TDEADLINE_S if not set.
    """
    emit_var_ok = os.getenv("EMIT_VAR_OK", "0") == "1"
    host = "api.metservice.com"
    fallback_mode = os.getenv("MET_FALLBACK_MODE", "pair_wind").strip().lower()

    # Allow env overrides
    def _to_float_env(name, default):
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    if total_deadline_s is None:
        total_deadline_s = _to_float_env("MET_TDEADLINE_S", 20.0)
    try:
        max_retries = int(os.getenv("MET_RETRIES", str(max_retries)))
    except Exception:
        pass
    try:
        base_backoff_sec = float(os.getenv("MET_BACKOFF_S", str(base_backoff_sec)))
    except Exception:
        pass

    # Primary first
    tried = []
    primary_id = None
    df_primary = pd.DataFrame(columns=['DateTime','Rainfall','wind_direction','wind_speed','Ve','Vn','StationId'])

    for idx, sid in enumerate(station_candidates):
        sid = str(sid)
        tried.append(sid)
        df = _metservice_fetch_24h(
            datetime, sid,
            log_event_fn=log_event_fn,
            emit_var_ok=emit_var_ok,
            max_retries=max_retries,
            base_backoff_sec=base_backoff_sec,
            total_deadline_s=total_deadline_s
        )
        if not df.empty and len(df) > 0:
            df['StationId'] = sid
            df_primary = df
            primary_id = sid
            break  # success → treat as primary and stop

    if df_primary.empty:
        # All stations failed as primary; preserve current
        _emit(log_event_fn, "SOURCE_DOWN",
              source="metservice", endpoint_host=host,
              failure_kind="all_stations_failed_or_deadline",
              stations_tried=",".join(tried),
              window_hours=24, deadline_s=total_deadline_s)
        return df_primary

    # Evaluate gaps in primary (24h window)
    p = df_primary.copy()
    p['DateTime'] = pd.to_datetime(p['DateTime'], utc=True, errors='coerce').dt.floor('h')
    prim_wind_ok = (~p['wind_direction'].isna()) & (~p['wind_speed'].isna())
    primary_has_gaps = p['Rainfall'].isna().any() or (~prim_wind_ok).any()

    if not primary_has_gaps or fallback_mode == "off":
        _emit(log_event_fn, "SOURCE_BACKUP_SKIPPED_NO_GAPS",
              source="metservice", endpoint_host=host,
              station_id=primary_id, fallback_mode=fallback_mode)
        return df_primary

    # Try fetching a single backup (the first candidate different to primary)
    backup_id = None
    df_backup = pd.DataFrame(columns=['DateTime','Rainfall','wind_direction','wind_speed','Ve','Vn','StationId'])

    # Separate deadline for backup (optional override)
    backup_deadline = _to_float_env("MET_BACKUP_TDEADLINE_S", total_deadline_s)

    for sid in station_candidates:
        sid = str(sid)
        if sid == primary_id:
            continue
        # Attempt backup fetch
        df = _metservice_fetch_24h(
            datetime, sid,
            log_event_fn=log_event_fn,
            emit_var_ok=emit_var_ok,
            max_retries=max_retries,
            base_backoff_sec=base_backoff_sec,
            total_deadline_s=backup_deadline
        )
        if not df.empty and len(df) > 0:
            df['StationId'] = sid
            df_backup = df
            backup_id = sid
            _emit(log_event_fn, "SOURCE_BACKUP_FETCHED",
                  source="metservice", endpoint_host=host,
                  primary_station=primary_id, backup_station=backup_id)
            break

    if df_backup.empty:
        _emit(log_event_fn, "SOURCE_BACKUP_EMPTY",
              source="metservice", endpoint_host=host,
              primary_station=primary_id,
              stations_considered=",".join([s for s in station_candidates if str(s) != str(primary_id)]))
        # Return primary unchanged (status quo)
        return df_primary

    # Merge + patch per agreed rules and recompute Ve/Vn
    merged = _merge_met_station_frames(df_primary, df_backup,
                                       log_event_fn=log_event_fn,
                                       primary_id=primary_id, backup_id=backup_id,
                                       fallback_mode=fallback_mode)

    # Keep the same column order contract as before
    cols = ['DateTime','Rainfall','wind_direction','wind_speed','Ve','Vn','StationId']
    extra = [c for c in merged.columns if c not in cols]  # provenance columns
    return merged[cols + extra]


def get_10min_weather_data_Akaroa(datetime, daytotal,
                                  log_event_fn=None, logger=None,
                                  max_retries=2, base_backoff_sec=5,
                                  total_deadline_s=None):
    """
    Fetches Akaroa 10-min weather data from the given date and number of days before.

    Parameters
    ----------
    datetime : datetime object, representing the date to fetch data for.
    daytotal : int, the number of days before the given date to fetch data for.

    log_event_fn : function, which takes the event name, source, endpoint_host, failure_kind, product_id, var, http_status, attempt, max_retries, retry_in_sec, elapsed_ms, and rows as arguments.
    logger : logger object, used to log messages.
    max_retries : int, the maximum number of retries for each station before giving up.
    base_backoff_sec : float, the base backoff time in seconds.
    total_deadline_s : float, the total deadline in seconds.

    Returns
    -------
    pd.DataFrame: A DataFrame with the fetched data, or an empty DataFrame if all stations failed or deadline was reached.
    """

    emit_var_ok = os.getenv("EMIT_VAR_OK", "0") == "1"
    host = "mintaka.niwa.co.nz"

    # Early guard: missing NIWA credentials
    if not username or not password:
        _emit(log_event_fn, "SOURCE_DOWN",
              source="niwa.mintaka", endpoint_host=host,
              failure_kind="missing_credentials")
        return pd.DataFrame(columns=['DateTime', 'Rainfall', 'wind_direction', 'wind_speed', 'Ve', 'Vn'])

    # Env overrides (ops tuning)
    if total_deadline_s is None:
        try:
            total_deadline_s = float(os.getenv("NIWA_TDEADLINE_S", "20"))
        except Exception:
            total_deadline_s = 20.0
    try:
        max_retries = int(os.getenv("NIWA_RETRIES", str(max_retries)))
    except Exception:
        pass
    try:
        base_backoff_sec = float(os.getenv("NIWA_BACKOFF_S", str(base_backoff_sec)))
    except Exception:
        pass

    # Backoff helper
    def _sleep_backoff(base, attempt):
        delay = min(base * (2 ** (attempt - 1)), 15) * random.uniform(0.85, 1.15)
        time.sleep(delay)

    l = []
    product_dict = {
        55993775: 'Rainfall',
        55995169: 'wind_direction',
        55995333: 'wind_speed'
    }

    start_date = datetime + dt.timedelta(days=-daytotal, minutes=1)
    start_time = dt.datetime.strftime(start_date, '%H:%M:%S')
    start_date = start_date.date()
    time_str = dt.datetime.strftime(datetime, '%H:%M:%S')
    date = datetime.date()

    retries_total = 0
    successes = 0

    t_station_start = time.monotonic()

    for product_id, var_name in product_dict.items():
        # Hard stop if budget exhausted
        if (time.monotonic() - t_station_start) > total_deadline_s:
            _emit(log_event_fn, "SOURCE_DOWN",
                  source="niwa.mintaka", endpoint_host=host,
                  failure_kind="station_deadline_exceeded",
                  deadline_s=total_deadline_s, vars_collected=len(l))
            break

        url = (
            f"https://{host}/rest/api/V1.1/products/{product_id}/data"
            f"?startDate={start_date}T{start_time}.000Z&endDate={date}T{time_str}.000Z&mode=LatestCommon"
        )

        result = {"data": [{"tuples": []}]}
        attempt = 0
        while attempt < max_retries:
            # Budget check before each attempt
            if (time.monotonic() - t_station_start) > total_deadline_s:
                _emit(log_event_fn, "SOURCE_RETRY",
                      source="niwa.mintaka", endpoint_host=host, product_id=product_id, var=var_name,
                      failure_kind="station_deadline_exceeded",
                      attempt=attempt, max_retries=max_retries, deadline_s=total_deadline_s)
                attempt = max_retries  # exit this var
                break

            attempt += 1
            t0 = time.monotonic()
            try:
                resp = requests.get(url, auth=HTTPBasicAuth(username, password), timeout=15)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                status = resp.status_code
                result = j.loads(resp.text)

                tuples = result.get("data", [{}])[0].get("tuples", []) or []
                if not tuples:
                    _emit(log_event_fn, "SOURCE_RETRY",
                          source="niwa.mintaka", endpoint_host=host,
                          product_id=product_id, var=var_name,
                          failure_kind="empty_tuples", http_status=status,
                          attempt=attempt, max_retries=max_retries, retry_in_sec=base_backoff_sec)
                    if attempt < max_retries:
                        retries_total += 1
                        _sleep_backoff(base_backoff_sec, attempt)
                        continue
                    # record empty and move on
                    _emit(log_event_fn, "SOURCE_VAR_EMPTY",
                          source="niwa.mintaka", endpoint_host=host,
                          product_id=product_id, var=var_name,
                          http_status=status, attempt=attempt, elapsed_ms=elapsed_ms)
                    break
                else:
                    if emit_var_ok:
                        _emit(log_event_fn, "SOURCE_VAR_OK",
                              source="niwa.mintaka", endpoint_host=host,
                              product_id=product_id, var=var_name,
                              http_status=status, rows=len(tuples), elapsed_ms=elapsed_ms, attempt=attempt)
                    # success for this var
                    data = pd.json_normalize(tuples)
                    data['validityTime'] = pd.to_datetime(data['validityTime'], utc=True)
                    data.rename(columns={'validityTime': 'DateTime', 'value': var_name}, inplace=True)
                    l.append(data)
                    successes += 1
                    break

            except Exception as e:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                _emit(log_event_fn, "SOURCE_RETRY",
                      source="niwa.mintaka", endpoint_host=host,
                      product_id=product_id, var=var_name,
                      failure_kind="exception", error=str(e)[:300],
                      attempt=attempt, max_retries=max_retries, retry_in_sec=base_backoff_sec,
                      elapsed_ms=elapsed_ms)
                if attempt < max_retries:
                    retries_total += 1
                    _sleep_backoff(base_backoff_sec, attempt)
                    continue
                break

    if not l:
        _emit(log_event_fn, "SOURCE_DOWN",
              source="niwa.mintaka", endpoint_host=host,
              failure_kind="all_vars_empty_or_failed_or_deadline",
              window_hours=daytotal * 24, retries=retries_total, deadline_s=total_deadline_s)
        return pd.DataFrame(columns=['DateTime', 'Rainfall', 'wind_direction', 'wind_speed', 'Ve', 'Vn'])

    df = pd.concat(l, ignore_index=True)
    for col in ['Rainfall', 'wind_direction', 'wind_speed']:
        if col not in df.columns:
            df[col] = np.nan

    df = df.groupby('DateTime', as_index=False).agg('max')
    df['tuple'] = df.apply(lambda r: wind_vector_components(r['wind_direction'], r['wind_speed']), axis=1)
    df[['Ve', 'Vn']] = df['tuple'].apply(pd.Series)
    df.drop(['tuple'], axis=1, inplace=True)
    df.sort_values(by='DateTime', ascending=False, inplace=True)

    _emit(log_event_fn, "SOURCE_OK",
          source="niwa.mintaka", endpoint_host=host,
          window_hours=daytotal * 24, rows=len(df), vars_success=successes,
          retries=retries_total, deadline_s=total_deadline_s)

    return df



def add_rainfall_variables(df):
    """Get accumulated rainfall, rain intensity, and rain duration.

    Parameters
    ----------
    df : dataframe with hourly accumulated rainfall (column 'Rainfall') per hour (row)

    Returns
    -------
    df: dataframe with accumulated rainfall in previous 3, 6, 12, 24, 48, and 72 hours, max rain intensity in previous 48 hours, and sum of hours in which it rained in previous 48 hours.

    """
    df['3H'] = np.round(df.Rainfall.rolling(min_periods=3, window=3).sum(), 1)
    df['6H'] = np.round(df.Rainfall.rolling(min_periods=6, window=6).sum(), 1)
    df['12H'] = np.round(df.Rainfall.rolling(min_periods=12, window=12).sum(), 1)
    df['24H'] = np.round(df.Rainfall.rolling(min_periods=24, window=24).sum(), 1)
    df['48H'] = np.round(df.Rainfall.rolling(min_periods=48, window=48).sum(), 1)
    df['72H'] = np.round(df.Rainfall.rolling(min_periods=72, window=72).sum(), 1)

    df['rain_intensity_48h'] = np.round(df.Rainfall.rolling(min_periods=48, window=48).max(), 1)

    df = df.assign(rain_yn = lambda x: (x['Rainfall']>0))
    df['rain_yn'] = df['rain_yn'].astype(int)
    df['rain_duration_48h'] = np.round(df.rain_yn.rolling(min_periods=48, window=48).sum(), 1)

    df = df.drop(['rain_yn'], axis=1)

    return df


def add_wind_variables(df, hours):
    
    """Get average wind direction and speed.

    Parameters
    ----------
    df : dataframe with hourly average wind components Ve and Vn
    hours : number of hours (rows) you want to take the average over

    Returns
    -------
    df: dataframe with average wind direction and speed in previous hours

    """

    df['Ve_mean'] = df.Ve.rolling(min_periods=hours, window=hours).mean()
    df['Vn_mean'] = df.Vn.rolling(min_periods=hours, window=hours).mean()

    df[f'wind_direction_{hours}h'] = (360 + (np.arctan2(df['Ve_mean'], df['Vn_mean']) * 180/np.pi)) % 360
    df[f'wind_speed_{hours}h'] = np.hypot(df['Ve_mean'], df['Vn_mean'])

    df.drop(['Ve_mean', 'Vn_mean'], axis=1, inplace=True)

    return df

# tides
def _get_tide_state(tide_heights):
    
    """Determine tidal state of list of tidal heights.

    Parameters
    ----------
    tide_heights : list with tide heights, sorted by time

    Returns
    -------
    df: dataframe with average wind direction and speed in previous hours

    """

    states = []
    # check either side of tide for each site to determine if a value is high or low tide
    for i in tide_heights.index:    
        if i-1 in tide_heights.index:
            if tide_heights[i-1] > tide_heights[i]:
                states.append("low")
            elif tide_heights[i-1] < tide_heights[i]:
                states.append("high")
            else:
                print(f'error at index {i}')
        elif i+1 in tide_heights.index:
            if tide_heights[i+1] > tide_heights[i]:
                states.append("low")
            elif tide_heights[i+1] < tide_heights[i]:
                states.append("high")
            else:
                print(f'error at index {i}')
        else:
            print(f'error at index {i}')
    return states
    

### function that calculates tidal state, hours to high tide, and high tide time:
def add_tide_variables(input_datetime, harbour, df):
    """
    Return (tidal_state, hours_to_high_tide, high_tide_height, high_tide_height_X_hours_to_high_tide)
    Never raises: returns ('unknown', NaN, NaN, NaN) on insufficient data.
    """
    if df is None or len(df) == 0 or input_datetime is None or pd.isna(input_datetime):
        return ("unknown", np.nan, np.nan, np.nan)

    # Ensure naive timestamps for comparisons
    if hasattr(input_datetime, "tzinfo") and input_datetime.tzinfo is not None:
        input_datetime = pd.Timestamp(input_datetime).tz_localize(None)

    tdf = df.copy()

    # Normalize Harbour and Tidal_state for robust matching
    tdf["Harbour"] = tdf["Harbour"].astype(str).str.strip()
    target_h = str(harbour).strip()
    tdf = tdf[tdf["Harbour"].str.casefold() == target_h.casefold()]

    if tdf.empty:
        return ("unknown", np.nan, np.nan, np.nan)

    # Normalize state labels and accept common variants
    state_norm = tdf["Tidal_state"].astype(str).str.strip().str.casefold()
    tdf = tdf.assign(Tidal_state=state_norm)
    HIGH_STATES = {"high", "h", "high tide"}  # accept common encodings

    # Ensure DateTime is naive
    if pd.api.types.is_datetime64tz_dtype(tdf["DateTime"].dtype):
        tdf["DateTime"] = pd.to_datetime(tdf["DateTime"]).dt.tz_localize(None)

    # Prefer highs within ±36h; if none, widen fallback to any point ±36h; if still none, nearest anywhere
    win_lo = input_datetime - td(hours=36)
    win_hi = input_datetime + td(hours=36)

    sel_high = tdf[(tdf["Tidal_state"].isin(HIGH_STATES)) &
                   (tdf["DateTime"] >= win_lo) & (tdf["DateTime"] <= win_hi)]

    if not sel_high.empty:
        # pick closest high tide within window
        idx = (sel_high["DateTime"] - input_datetime).abs().idxmin()
    else:
        sel_any = tdf[(tdf["DateTime"] >= win_lo) & (tdf["DateTime"] <= win_hi)]
        if not sel_any.empty:
            idx = (sel_any["DateTime"] - input_datetime).abs().idxmin()
        else:
            # final fallback: nearest anywhere for this harbour
            idx = (tdf["DateTime"] - input_datetime).abs().idxmin()

    closest_time = tdf.loc[idx, "DateTime"]
    closest_height = float(tdf.loc[idx, "Tidal_height"]) if pd.notna(tdf.loc[idx, "Tidal_height"]) else np.nan

    hours_to_high = round((input_datetime - closest_time).total_seconds() / 3600, 1)
    tidal_state = "ebbing" if (pd.notna(hours_to_high) and hours_to_high > 0) else "incoming"

    prod = hours_to_high * closest_height if (pd.notna(hours_to_high) and pd.notna(closest_height)) else np.nan
    return (tidal_state, hours_to_high, closest_height, prod)


def TRANSFORM_CATEGORICAL_FEATURE(df):
    """
    Ensures categorical features are correctly classified, for the model to use in learning/nowcasting.
    Only columns that were categorical (object or category dtype) in training are cast to category.
    Numeric features remain numeric.

    Parameters
    ----------
    df : DataFrame including all features to be used by the model

    Returns
    -------
    DataFrame with categorical features correctly classified
    """
    # Only columns that were truly categorical/object in training:
    CAT_FEATURES = [
        'SITE_NAME', 'Harbour', 'Shallowness', 'Soil_type', 
        'Landcover_catchment', 'tidal_state',
        'wind_shore_3h', 'wind_shore_6h', 'wind_shore_12h'
    ]
    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype('category')
    return df


def get_tide_data(site_list, year = False):
    
    """Get tidal data from LINZ

    Parameters
    ----------
    site_list : list of sites (using the names as available on LINZ)
    year : optional, defaults to current year

    Returns
    -------
    all_data: dataframe with all tidal data for the specified sites and year + next year, corrected for NZST

    """
    # gets data from https://www.linz.govt.nz/products-services/tides-and-tidal-streams/tide-predictions

    # if a year has not been specified, use the current year
    if not year:
        year = dt.datetime.now().year

    all_data = []
    for site in site_list:
        # get data for a site and the current/specified year
        for yr in [year, year + 1]:
            url = f"https://static.charts.linz.govt.nz/tide-tables/maj-ports/csv/{site} {yr}.csv"
            with requests.get(url, timeout=15) as response:
                try:
                    csv_str = response.content.decode('utf-8')
                except UnicodeDecodeError:
                    csv_str = response.content.decode('latin1')
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


    # combine datasets for all years and sites
    all_data = pd.concat(all_data, ignore_index = True)

    # combined all tide times top to bottom
    all_data = pd.concat([
        all_data[["Harbour", "Day", "Month", "Year", "Time1", "TideHeight1"]].rename(
            columns = {"Time1": "Time", "TideHeight1": "Tidal_height"}
            ),
        all_data[["Harbour", "Day", "Month", "Year", "Time2", "TideHeight2"]].rename(
            columns = {"Time2": "Time", "TideHeight2": "Tidal_height"}
            ),
        all_data[["Harbour", "Day", "Month", "Year", "Time3", "TideHeight3"]].rename(
            columns = {"Time3": "Time", "TideHeight3": "Tidal_height"}
            ),
        all_data[["Harbour", "Day", "Month", "Year", "Time4", "TideHeight4"]].rename(
            columns = {"Time4": "Time", "TideHeight4": "Tidal_height"}
            )],
        ignore_index = True
    )

    # remove Nan tide height rows
    all_data = all_data.loc[all_data["Tidal_height"].notna()]
    all_data['hour'] = all_data['Time'].apply(lambda x: x.split(":")[0])
    all_data['minute'] = all_data['Time'].apply(lambda x: x.split(":")[1])

    # add a datetime column
    # localize to NZST
    all_data['DateTime'] = pd.to_datetime(
        all_data[['Year', 'Month', 'Day', 'hour', 'minute']]
        )

    # sort values by harbour and datetime
    all_data.sort_values(
        by = ['Harbour', 'DateTime'], ignore_index = True, inplace = True
        )    
    
    #all_data.to_csv(r'C:\Users\BenHi\OneDrive - Environment Canterbury\Downloads\timezone_testUNCONVERTED.csv')

    # correct the datetimes to be NZST
    # get the timezone information for each timestamp (adjusts with daylight
    # savings) set any ambigious times to NaN
    all_data["DateTimeTz"] = all_data["DateTime"].dt.tz_localize(
        tz = 'Pacific/Auckland',
        #ambiguous = False
        ambiguous = 'NaT'
        ).copy()

    # deal with ambigious times on a case by case basis. After converting to the
    # correct timezone, the time to the previous observation and the time
    # difference to the next observation should be in 1 hours of each other.
    # Since tide peaks occur approx every 6 hours, the correct tz will show a 6
    # hour difference to the prev and next timestamp. The incorrect tz will show
    # a 5 and 7 hour difference to the timestamps either side.

    # identify ambigious times
    ambig_dts = all_data.loc[all_data["DateTimeTz"].isna()].copy()
    # ambigious data as daylight time
    dl_data = ambig_dts['DateTime'].dt.tz_localize(
        tz = 'Pacific/Auckland', ambiguous = True
        )
    # ambigious data as standard time
    st_data = ambig_dts['DateTime'].dt.tz_localize(
        tz = 'Pacific/Auckland', ambiguous = False
            )
    for i in ambig_dts.index:
        # get the time differences to the next and prev times for daylight time
        dl_diff_next = dl_data[i] - all_data['DateTimeTz'][i-1]
        dl_diff_prev = all_data['DateTimeTz'][i+1] - dl_data[i]

        # should be within one hour of each other
        dl_diff = abs(dl_diff_next - dl_diff_prev)

        if dl_diff < td(hours = 1):
            # set to daylight time
            all_data.loc[i, "DateTimeTz"] = dl_data[i]
        else:
            # set to standard time
            all_data.loc[i, "DateTimeTz"] = st_data[i]

    # convert to NZST (skip NaT values)
    all_data["DateTimeNZST"] = all_data["DateTimeTz"].apply(
        lambda x: x.astimezone(tz(td(hours=12))) if pd.notna(x) else pd.NaT
        )
    # get high/low tide from tide heights
    all_data["Tidal_state"] = all_data.groupby("Harbour")['Tidal_height'].transform(
        lambda x: _get_tide_state(x)
        )
    # format dataframe
    all_data = all_data[["DateTimeNZST", "Tidal_height", "Tidal_state", "Harbour"]]
    all_data.rename(columns = {"DateTimeNZST": "DateTime"}, inplace = True)

    return all_data

