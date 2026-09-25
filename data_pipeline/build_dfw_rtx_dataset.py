#!/usr/bin/env python3
"""Build a model-ready DFW flight + weather dataset from official BTS and NOAA data.

Default scope:
- DFW-origin flights
- 2016 through 2025
- BTS Reporting Carrier On-Time Performance
- NOAA Global Hourly station 72259003927 (KDFW)
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

BTS_BASE = "https://transtats.bts.gov/PREZIP"
NOAA_BASE = "https://www.ncei.noaa.gov/data/global-hourly/access"
LOCAL_TZ = "America/Chicago"
DEFAULT_STATION = "72259003927"
WEATHER_TOLERANCE_MIN = 75

HEADERS = {
    "User-Agent": "UTD-RTX-Flight-Predictive-Analytics/1.0 academic project",
    "Accept": "*/*",
}

BTS_FIELDS = {
    "Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Origin", "Dest", "DestCityName", "DestState",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes", "ArrDel15",
    "Cancelled", "CancellationCode", "Diverted",
    "CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay",
}

SAFE_FEATURES = [
    "flight_year", "flight_month", "flight_day", "day_of_week", "is_weekend",
    "scheduled_dep_hour", "scheduled_dep_minute",
    "scheduled_dep_minutes_since_midnight",
    "carrier", "flight_number", "destination", "destination_state",
    "scheduled_arrival_hhmm", "scheduled_elapsed_minutes", "distance_miles",
    "covid_period",
    "weather_temp_c", "weather_dewpoint_c", "weather_wind_dir_deg",
    "weather_wind_speed_mps", "weather_wind_gust_mps",
    "weather_visibility_m", "weather_ceiling_m",
    "weather_sea_level_pressure_hpa", "weather_precip_mm",
    "weather_precip_period_h", "weather_present_code",
]

TARGETS = [
    "target_departure_delayed_15min",
    "target_departure_delay_minutes",
    "target_cancelled",
    "target_diverted",
]


def log(message: str) -> None:
    print(message, flush=True)


def get(url: str, *, stream: bool = False, attempts: int = 5) -> requests.Response:
    last = None
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=180, stream=stream)
            if r.status_code == 404:
                return r
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last = exc
            if attempt == attempts:
                raise
            wait = min(30, 2 ** attempt)
            log(f"Request failed: {exc}. Retrying in {wait}s")
            time.sleep(wait)
    raise last


def download(url: str, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    r = get(url, stream=True)
    if r.status_code == 404:
        return False
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    temp.replace(path)
    return True


def bts_urls(year: int, month: int) -> list[str]:
    names = [
        f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip",
        f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.zip",
    ]
    return [f"{BTS_BASE}/{quote(name, safe='()_-.')}" for name in names]


def download_bts_month(year: int, month: int, raw_dir: Path) -> tuple[Path, str]:
    dest = raw_dir / f"bts_{year}_{month:02d}.zip"
    marker = raw_dir / f"bts_{year}_{month:02d}.url.txt"
    if dest.exists() and dest.stat().st_size > 1000:
        src = marker.read_text().strip() if marker.exists() else "cached"
        return dest, src

    for url in bts_urls(year, month):
        log(f"Downloading BTS {year}-{month:02d}")
        if download(url, dest):
            marker.write_text(url)
            return dest, url
    raise RuntimeError(f"BTS archive not found for {year}-{month:02d}")


def build_bts_month(
    year: int,
    month: int,
    origin: str,
    raw_dir: Path,
    cache_dir: Path,
) -> Path:
    out = cache_dir / f"{origin}_{year}_{month:02d}.parquet"
    if out.exists() and out.stat().st_size > 500:
        log(f"Reusing {out.name}")
        return out

    zpath, source_url = download_bts_month(year, month, raw_dir)
    pieces: list[pd.DataFrame] = []

    with zipfile.ZipFile(zpath) as zf:
        csv_members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV found in {zpath.name}")
        member = max(csv_members, key=lambda m: m.file_size)
        with zf.open(member) as file_obj:
            chunks = pd.read_csv(
                file_obj,
                usecols=lambda c: c in BTS_FIELDS,
                chunksize=175_000,
                low_memory=False,
            )
            for chunk in chunks:
                if "Origin" not in chunk.columns:
                    raise RuntimeError("BTS data is missing Origin")
                keep = chunk["Origin"].astype("string").str.strip().eq(origin)
                if keep.any():
                    pieces.append(chunk.loc[keep].copy())

    if not pieces:
        raise RuntimeError(f"No {origin} departures found for {year}-{month:02d}")

    df = pd.concat(pieces, ignore_index=True)
    df["bts_source_url"] = source_url
    df.to_parquet(out, index=False, compression="zstd")
    log(f"Saved {len(df):,} DFW rows for {year}-{month:02d}")

    zpath.unlink(missing_ok=True)
    (raw_dir / f"bts_{year}_{month:02d}.url.txt").unlink(missing_ok=True)
    return out


def split_first(series: pd.Series) -> pd.Series:
    return series.astype("string").str.split(",", n=1).str[0]


def parse_scaled(series: pd.Series, missing: set[str], scale: float = 1.0) -> pd.Series:
    token = split_first(series).str.strip()
    token = token.mask(token.isin(missing))
    return pd.to_numeric(token, errors="coerce") / scale


def parse_wind(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.astype("string").str.split(",", expand=True)
    direction = pd.to_numeric(parts[0], errors="coerce")
    direction = direction.mask(direction.eq(999))
    if parts.shape[1] > 3:
        speed = pd.to_numeric(parts[3], errors="coerce").mask(lambda s: s.eq(9999)) / 10.0
    else:
        speed = pd.Series(np.nan, index=series.index)
    return direction, speed


def parse_precip(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    options = []
    for col in ("AA1", "AA2", "AA3", "AA4"):
        if col not in df.columns:
            continue
        parts = df[col].astype("string").str.split(",", expand=True)
        if parts.shape[1] < 2:
            continue
        hours = pd.to_numeric(parts[0], errors="coerce").replace(99, np.nan)
        depth = pd.to_numeric(parts[1], errors="coerce").replace(9999, np.nan) / 10.0
        options.append((hours, depth))

    if not options:
        empty = pd.Series(np.nan, index=df.index)
        return empty.copy(), empty.copy()

    h = pd.concat([x[0] for x in options], axis=1)
    d = pd.concat([x[1] for x in options], axis=1)
    h.columns = range(h.shape[1])
    d.columns = range(d.shape[1])

    valid = h.notna() & d.notna()
    ranked = h.where(valid)
    chosen_col = ranked.idxmin(axis=1, skipna=True)

    out_h = pd.Series(np.nan, index=df.index, dtype="float64")
    out_d = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in h.columns:
        mask = chosen_col.eq(col)
        out_h.loc[mask] = h.loc[mask, col]
        out_d.loc[mask] = d.loc[mask, col]
    return out_d, out_h


def download_noaa_year(year: int, station: str, raw_dir: Path) -> Path:
    out = raw_dir / f"noaa_{station}_{year}.csv"
    if out.exists() and out.stat().st_size > 500:
        return out
    url = f"{NOAA_BASE}/{year}/{station}.csv"
    log(f"Downloading NOAA {year}: {url}")
    if not download(url, out):
        raise RuntimeError(f"NOAA station file not found: {url}")
    return out


def build_noaa_year(year: int, station: str, raw_dir: Path, cache_dir: Path) -> Path:
    out = cache_dir / f"noaa_{station}_{year}.parquet"
    if out.exists() and out.stat().st_size > 500:
        log(f"Reusing {out.name}")
        return out

    csv_path = download_noaa_year(year, station, raw_dir)
    df = pd.read_csv(csv_path, low_memory=False)
    if "DATE" not in df.columns:
        raise RuntimeError(f"NOAA {year} data has no DATE column")

    wx = pd.DataFrame(index=df.index)
    wx["weather_timestamp_utc"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
    wx["weather_station"] = station
    wx["weather_station_name"] = df["NAME"].astype("string") if "NAME" in df.columns else pd.NA
    wx["weather_report_type"] = df["REPORT_TYPE"].astype("string") if "REPORT_TYPE" in df.columns else pd.NA

    if "TMP" in df.columns:
        wx["weather_temp_c"] = parse_scaled(df["TMP"], {"+9999", "-9999", "9999"}, 10.0)
    if "DEW" in df.columns:
        wx["weather_dewpoint_c"] = parse_scaled(df["DEW"], {"+9999", "-9999", "9999"}, 10.0)
    if "WND" in df.columns:
        direction, speed = parse_wind(df["WND"])
        wx["weather_wind_dir_deg"] = direction
        wx["weather_wind_speed_mps"] = speed
    if "OC1" in df.columns:
        wx["weather_wind_gust_mps"] = parse_scaled(df["OC1"], {"9999", "+9999"}, 10.0)
    if "VIS" in df.columns:
        wx["weather_visibility_m"] = parse_scaled(df["VIS"], {"999999"}, 1.0)
    if "CIG" in df.columns:
        wx["weather_ceiling_m"] = parse_scaled(df["CIG"], {"99999"}, 1.0)
    if "SLP" in df.columns:
        wx["weather_sea_level_pressure_hpa"] = parse_scaled(df["SLP"], {"99999"}, 10.0)

    precip, period = parse_precip(df)
    wx["weather_precip_mm"] = precip
    wx["weather_precip_period_h"] = period

    present = pd.Series(np.nan, index=df.index)
    if "MW1" in df.columns:
        present = pd.to_numeric(split_first(df["MW1"]), errors="coerce")
    if "AW1" in df.columns:
        automated = pd.to_numeric(split_first(df["AW1"]), errors="coerce")
        present = present.fillna(automated)
    wx["weather_present_code"] = present

    wanted = [
        "weather_timestamp_utc", "weather_station", "weather_station_name",
        "weather_report_type", "weather_temp_c", "weather_dewpoint_c",
        "weather_wind_dir_deg", "weather_wind_speed_mps",
        "weather_wind_gust_mps", "weather_visibility_m", "weather_ceiling_m",
        "weather_sea_level_pressure_hpa", "weather_precip_mm",
        "weather_precip_period_h", "weather_present_code",
    ]
    for col in wanted:
        if col not in wx.columns:
            wx[col] = np.nan

    wx = wx[wanted].dropna(subset=["weather_timestamp_utc"])
    wx = wx.sort_values("weather_timestamp_utc").drop_duplicates("weather_timestamp_utc", keep="last")
    wx.to_parquet(out, index=False, compression="zstd")
    csv_path.unlink(missing_ok=True)
    log(f"Saved {len(wx):,} NOAA observations for {year}")
    return out


def num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def clean_flights(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    date = pd.to_datetime(df["FlightDate"], errors="coerce")
    crs_dep = num(df, "CRSDepTime").round().astype("Int64")

    dep_hour = (crs_dep // 100).astype("Int64")
    dep_minute = (crs_dep % 100).astype("Int64")
    total_min = (dep_hour * 60 + dep_minute).astype("Int64")
    total_min = total_min.where(total_min.between(0, 1440))

    scheduled_naive = date.dt.normalize() + pd.to_timedelta(total_min.fillna(0), unit="m")
    scheduled_naive = scheduled_naive.where(total_min.notna())

    # DFW is in Central Time. ambiguous=False chooses standard time during the
    # duplicated fall-back hour; nonexistent spring-forward times shift forward.
    scheduled_local = scheduled_naive.dt.tz_localize(
        LOCAL_TZ, ambiguous=False, nonexistent="shift_forward"
    )

    out["flight_date"] = date.dt.date
    out["scheduled_departure_local"] = scheduled_local
    out["scheduled_departure_utc"] = scheduled_local.dt.tz_convert("UTC")
    out["flight_year"] = date.dt.year.astype("Int16")
    out["flight_month"] = date.dt.month.astype("Int8")
    out["flight_day"] = date.dt.day.astype("Int8")
    out["day_of_week"] = num(df, "DayOfWeek").astype("Int8")
    out["is_weekend"] = out["day_of_week"].isin([6, 7]).astype("Int8")
    out["scheduled_dep_hour"] = dep_hour.astype("Int8")
    out["scheduled_dep_minute"] = dep_minute.astype("Int8")
    out["scheduled_dep_minutes_since_midnight"] = total_min.astype("Int16")

    out["carrier"] = df["Reporting_Airline"].astype("string")
    out["flight_number"] = num(df, "Flight_Number_Reporting_Airline").astype("Int32")
    out["origin"] = origin
    out["destination"] = df["Dest"].astype("string")
    out["destination_city"] = df["DestCityName"].astype("string") if "DestCityName" in df.columns else pd.NA
    out["destination_state"] = df["DestState"].astype("string") if "DestState" in df.columns else pd.NA
    out["scheduled_arrival_hhmm"] = num(df, "CRSArrTime").astype("Int32")
    out["scheduled_elapsed_minutes"] = num(df, "CRSElapsedTime")
    out["distance_miles"] = num(df, "Distance")
    out["covid_period"] = out["flight_year"].isin([2020, 2021]).astype("Int8")

    out["target_departure_delayed_15min"] = num(df, "DepDel15")
    out["target_departure_delay_minutes"] = num(df, "DepDelayMinutes")
    out["target_cancelled"] = num(df, "Cancelled").fillna(0).astype("Int8")
    out["target_diverted"] = num(df, "Diverted").fillna(0).astype("Int8")

    # Post-event columns are deliberately prefixed outcome_ to prevent leakage.
    outcome_map = {
        "DepTime": "outcome_actual_departure_hhmm",
        "DepDelay": "outcome_departure_delay_signed_minutes",
        "ArrTime": "outcome_actual_arrival_hhmm",
        "ArrDelay": "outcome_arrival_delay_signed_minutes",
        "ArrDelayMinutes": "outcome_arrival_delay_minutes",
        "ArrDel15": "outcome_arrival_delayed_15min",
        "CancellationCode": "outcome_cancellation_code",
        "ActualElapsedTime": "outcome_actual_elapsed_minutes",
        "AirTime": "outcome_air_time_minutes",
        "CarrierDelay": "outcome_carrier_delay_minutes",
        "WeatherDelay": "outcome_bts_weather_delay_minutes",
        "NASDelay": "outcome_nas_delay_minutes",
        "SecurityDelay": "outcome_security_delay_minutes",
        "LateAircraftDelay": "outcome_late_aircraft_delay_minutes",
    }
    for src, dst in outcome_map.items():
        if src in df.columns:
            out[dst] = df[src]

    out["flight_id"] = (
        pd.Series(out["flight_date"].astype("string"), index=out.index)
        + "_" + out["carrier"].fillna("NA")
        + "_" + out["flight_number"].astype("string").fillna("NA")
        + "_" + out["destination"].fillna("NA")
        + "_" + out["scheduled_dep_minutes_since_midnight"].astype("string").fillna("NA")
    )
    return out


def merge_weather(flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    left = flights.sort_values("scheduled_departure_utc").reset_index(drop=True)
    right = weather.sort_values("weather_timestamp_utc").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="scheduled_departure_utc",
        right_on="weather_timestamp_utc",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=WEATHER_TOLERANCE_MIN),
    )
    merged["weather_age_minutes"] = (
        (merged["scheduled_departure_utc"] - merged["weather_timestamp_utc"])
        .abs()
        .dt.total_seconds()
        / 60.0
    )
    merged["weather_match_ok"] = merged["weather_timestamp_utc"].notna().astype("Int8")
    return merged


def assign_split(year: pd.Series, end_year: int) -> pd.Series:
    validation_year = end_year - 1
    conditions = [
        year <= validation_year - 1,
        year == validation_year,
        year == end_year,
    ]
    return pd.Series(
        np.select(conditions, ["train", "validation", "test"], default="other"),
        index=year.index,
        dtype="string",
    )


def write_dictionary(path: Path) -> None:
    rows = [
        ["flight_id", "Derived", "Unique-ish flight record identifier", "metadata"],
        ["scheduled_departure_local", "BTS-derived", "Scheduled DFW departure in America/Chicago", "join key"],
        ["carrier", "BTS", "Reporting airline code", "safe feature"],
        ["destination", "BTS", "Destination IATA airport code", "safe feature"],
        ["distance_miles", "BTS", "Flight distance", "safe feature"],
        ["weather_temp_c", "NOAA TMP", "Air temperature in Celsius", "safe feature"],
        ["weather_dewpoint_c", "NOAA DEW", "Dew point in Celsius", "safe feature"],
        ["weather_wind_dir_deg", "NOAA WND", "Wind direction in degrees", "safe feature"],
        ["weather_wind_speed_mps", "NOAA WND", "Wind speed in m/s", "safe feature"],
        ["weather_wind_gust_mps", "NOAA OC1", "Wind gust in m/s when present", "safe feature"],
        ["weather_visibility_m", "NOAA VIS", "Horizontal visibility in meters", "safe feature"],
        ["weather_ceiling_m", "NOAA CIG", "Cloud ceiling in meters", "safe feature"],
        ["weather_sea_level_pressure_hpa", "NOAA SLP", "Sea-level pressure in hPa", "safe feature"],
        ["weather_precip_mm", "NOAA AA1-AA4", "Precipitation depth in mm", "safe feature"],
        ["weather_present_code", "NOAA MW1/AW1", "Present-weather code", "safe feature"],
        ["target_departure_delayed_15min", "BTS DepDel15", "Departure delayed at least 15 minutes", "TARGET"],
        ["target_departure_delay_minutes", "BTS DepDelayMinutes", "Nonnegative departure delay minutes", "TARGET"],
        ["target_cancelled", "BTS Cancelled", "Flight cancellation indicator", "TARGET"],
        ["target_diverted", "BTS Diverted", "Flight diversion indicator", "TARGET"],
        ["outcome_*", "BTS", "Post-event actual times and delay-cause fields", "LEAKAGE - historical analysis only"],
        ["dataset_split", "Derived", "Chronological train/validation/test split", "ML split"],
    ]
    pd.DataFrame(rows, columns=["column", "source", "description", "usage"]).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--origin", default="DFW")
    parser.add_argument("--station", default=DEFAULT_STATION)
    parser.add_argument("--output-dir", default="rtx_dfw_data")
    parser.add_argument("--parquet-only", action="store_true")
    args = parser.parse_args()

    if args.end_year < args.start_year:
        parser.error("--end-year must be >= --start-year")

    root = Path(args.output_dir)
    raw = root / "cache" / "raw"
    bts_cache = root / "cache" / "bts_monthly"
    noaa_cache = root / "cache" / "noaa_yearly"
    final = root / "final"
    for directory in (raw, bts_cache, noaa_cache, final):
        directory.mkdir(parents=True, exist_ok=True)

    log(f"Building {args.origin} dataset for {args.start_year}-{args.end_year}")

    bts_files = []
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            bts_files.append(build_bts_month(year, month, args.origin, raw, bts_cache))

    noaa_files = []
    for year in range(args.start_year, args.end_year + 1):
        noaa_files.append(build_noaa_year(year, args.station, raw, noaa_cache))

    log("Combining DFW flight records")
    flights_raw = pd.concat((pd.read_parquet(p) for p in bts_files), ignore_index=True)
    flights = clean_flights(flights_raw, args.origin)
    del flights_raw

    log("Combining DFW weather records")
    weather = pd.concat((pd.read_parquet(p) for p in noaa_files), ignore_index=True)
    weather = weather.sort_values("weather_timestamp_utc").reset_index(drop=True)
    weather_path = final / f"dfw_weather_{args.start_year}_{args.end_year}.parquet"
    weather.to_parquet(weather_path, index=False, compression="zstd")

    log("Matching nearest weather observation to each scheduled departure")
    merged = merge_weather(flights, weather)
    merged["dataset_split"] = assign_split(merged["flight_year"], args.end_year)

    base = f"rtx_dfw_flights_weather_{args.start_year}_{args.end_year}"
    complete_path = final / f"{base}.parquet"
    merged.to_parquet(complete_path, index=False, compression="zstd")

    if not args.parquet_only:
        merged.to_csv(final / f"{base}.csv.gz", index=False, compression="gzip")

    safe = [c for c in SAFE_FEATURES if c in merged.columns]
    model_cols = [
        "flight_id", "flight_date", "scheduled_departure_local", "dataset_split",
        *safe, *TARGETS,
    ]
    model_cols = [c for c in model_cols if c in merged.columns]
    model_path = final / f"{base}_model_input.parquet"
    merged[model_cols].to_parquet(model_path, index=False, compression="zstd")

    write_dictionary(final / "data_dictionary.csv")
    (final / "safe_feature_columns.txt").write_text("\n".join(safe) + "\n")
    (final / "target_columns.txt").write_text("\n".join(TARGETS) + "\n")

    summary = {
        "origin": args.origin,
        "years": [args.start_year, args.end_year],
        "noaa_station": args.station,
        "rows": int(len(merged)),
        "date_min": str(merged["flight_date"].min()),
        "date_max": str(merged["flight_date"].max()),
        "unique_destinations": int(merged["destination"].nunique()),
        "unique_carriers": int(merged["carrier"].nunique()),
        "weather_match_rate": float(merged["weather_match_ok"].mean()),
        "delay_15_rate_nonmissing": float(merged["target_departure_delayed_15min"].mean(skipna=True)),
        "cancellation_rate": float(merged["target_cancelled"].mean()),
        "diversion_rate": float(merged["target_diverted"].mean()),
        "split_counts": {
            str(k): int(v)
            for k, v in merged["dataset_split"].value_counts(dropna=False).to_dict().items()
        },
        "safe_features": safe,
        "targets": TARGETS,
        "warning": "Never use outcome_* columns for pre-departure prediction; they contain post-flight information.",
    }
    (final / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    log("DONE")
    log(json.dumps(summary, indent=2))
    log(f"Complete dataset: {complete_path}")
    log(f"Model-ready dataset: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
