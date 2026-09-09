# data_loaders.py
# Load NASA POWER CSVs and UP SW telemetry into canonical schema (data_schema.COL).
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_schema import ALIASES, COL, DIR_NASA, DIR_UP_SW

IST = "Asia/Kolkata"
ROOT = Path(__file__).resolve().parent


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns via ALIASES; leave unknown columns as-is."""
    rename = {}
    for c in df.columns:
        key = str(c).strip()
        if key in ALIASES:
            rename[c] = ALIASES[key]
        else:
            # case-insensitive fallback
            low = key.lower()
            for a, canon in ALIASES.items():
                if a.lower() == low:
                    rename[c] = canon
                    break
    return df.rename(columns=rename)


def _to_ist_index(series: pd.Series) -> pd.DatetimeIndex:
    ts = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(IST, ambiguous="NaT", nonexistent="NaT")
    else:
        ts = ts.dt.tz_convert(IST)
    return pd.DatetimeIndex(ts)


# ---------------------------------------------------------------------------
# NASA POWER
# ---------------------------------------------------------------------------
def parse_nasa_csv(path: Path) -> pd.DataFrame:
    """
    NASA POWER CSV has metadata lines before the table.
    Header row contains YEAR,MO,DY,HR and parameter names.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\bYEAR\b", line) and re.search(r"\bHR\b", line):
            header_idx = i
            break
    if header_idx is None:
        # try pandas default
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, skiprows=header_idx)

    df = _normalize_columns(df)

    # Build timestamp from YEAR MO DY HR if present
    cols_upper = {c.upper(): c for c in df.columns}
    if all(k in cols_upper for k in ("YEAR", "MO", "DY", "HR")):
        y = df[cols_upper["YEAR"]].astype(int)
        m = df[cols_upper["MO"]].astype(int)
        d = df[cols_upper["DY"]].astype(int)
        h = df[cols_upper["HR"]].astype(int)
        # NASA hourly is often UTC end-of-hour; treat as UTC then convert
        ts = pd.to_datetime(
            dict(year=y, month=m, day=d, hour=h),
            errors="coerce",
            utc=True,
        ).dt.tz_convert(IST)
        df.index = ts
    elif "time" in df.columns:
        df.index = _to_ist_index(df["time"])
    else:
        raise ValueError(f"Cannot build time index from {path.name}")

    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Unit fixes
    # NASA PS is kPa → hPa-ish (×10); Open-Meteo surface_pressure is hPa
    if COL["pressure"] in df.columns:
        ps = pd.to_numeric(df[COL["pressure"]], errors="coerce")
        # if values look like kPa (~90–105), convert to hPa
        med = ps.median(skipna=True)
        if np.isfinite(med) and med < 200:
            df[COL["pressure"]] = ps * 10.0

    # Fill fill-values (-999 etc.)
    for c in df.columns:
        if c in ("YEAR", "MO", "DY", "HR") or str(c).startswith("-"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.mask(s < -900)
        df[c] = s

    keep = [c for c in df.columns if c in set(ALIASES.values()) or c in COL.values()]
    # always keep canonical weather cols if present
    out = df[[c for c in keep if c in df.columns]].copy()
    return out


def load_all_nasa(root: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Scan data/nasa/up/<district>/*.csv → {site_key: DataFrame}.
    site_key taken from filename stem prefix before lat.
    """
    base = (root or ROOT) / DIR_NASA
    if not base.exists():
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for path in sorted(base.rglob("*.csv")):
        try:
            df = parse_nasa_csv(path)
            # key: parent district + file stem
            key = f"{path.parent.name}/{path.stem}"
            out[key] = df
        except Exception as exc:
            print(f"[nasa] FAIL {path}: {exc}")
    return out


def nearest_nasa_frame(
    lat: float,
    lon: float,
    nasa_map: Dict[str, pd.DataFrame],
    max_km: float = 80.0,
) -> Optional[Tuple[str, pd.DataFrame]]:
    """Pick NASA series whose filename encodes lat_lon closest to target."""
    best = None
    best_d = 1e18
    for key, df in nasa_map.items():
        # expect ..._24.7695_82.3303 in stem
        m = re.search(r"_(-?\d+\.\d+)_(-?\d+\.\d+)", key)
        if not m:
            continue
        slat, slon = float(m.group(1)), float(m.group(2))
        # rough km
        d = ((slat - lat) ** 2 + (slon - lon) ** 2) ** 0.5 * 111.0
        if d < best_d:
            best_d = d
            best = (key, df)
    if best is None or best_d > max_km:
        return None
    return best


# ---------------------------------------------------------------------------
# UP SW telemetry
# ---------------------------------------------------------------------------
UP_SW_FILE_HINTS = {
    "solar": ("solar", "rediation", "radiation"),
    "temp": ("temprature", "temperature", "temp"),
    "humid": ("humid",),
    "rain": ("rainfall", "rain"),
    "wind_speed": ("wind_speed", "wind speed"),
    "wind_dir": ("wind_direction", "wind direction"),
    "pressure": ("pressure",),
}


def _match_up_sw_file(path: Path) -> Optional[str]:
    name = path.name.lower()
    for kind, hints in UP_SW_FILE_HINTS.items():
        if any(h in name for h in hints):
            return kind
    return None


def load_up_sw_variable(path: Path, value_aliases: List[str]) -> pd.DataFrame:
    """Load one UP SW CSV → index=time, columns include Station, lat, lon, value canonical."""
    df = pd.read_csv(path, low_memory=False)
    df = _normalize_columns(df)

    # time
    time_col = None
    for c in df.columns:
        if "acquisition" in str(c).lower() or str(c).lower() in ("time", "datetime", "date"):
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"No time column in {path.name}")

    df["time"] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].dt.tz_localize(IST, ambiguous="NaT", nonexistent="NaT")
    df = df.dropna(subset=["time"])

    # station / coords
    station_col = next((c for c in df.columns if str(c).lower() == "station"), None)
    lat_col = next((c for c in df.columns if str(c).lower() == "latitude"), None)
    lon_col = next((c for c in df.columns if str(c).lower() == "longitude"), None)
    dist_col = next((c for c in df.columns if str(c).lower() == "district"), None)

    # value column: already renamed to canonical if alias matched
    val_col = None
    for cand in value_aliases:
        if cand in df.columns:
            val_col = cand
            break
    if val_col is None:
        # last numeric column often is the measurement
        for c in reversed(list(df.columns)):
            if c in ("time", station_col, lat_col, lon_col, dist_col):
                continue
            if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == object:
                val_col = c
                break
    if val_col is None:
        raise ValueError(f"No value column in {path.name}")

    out = pd.DataFrame({
        "time": df["time"],
        "station": df[station_col].astype(str) if station_col else "unknown",
        "district": df[dist_col].astype(str).str.upper() if dist_col else "",
        "lat": pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan,
        "lon": pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan,
        "value": pd.to_numeric(df[val_col], errors="coerce"),
    }).dropna(subset=["time", "value"])
    return out


def load_all_up_sw(root: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Load every CSV in data/up_sw/ into long tables per variable kind.
    Returns dict kind → long DataFrame (time, station, district, lat, lon, value).
    """
    base = (root or ROOT) / DIR_UP_SW
    if not base.exists():
        return {}

    value_map = {
        "solar": [COL["ghi"], "Solar Radiation (Watt/m2)"],
        "temp": [COL["temp"]],
        "humid": [COL["rh"]],
        "rain": [COL["rain"], COL["precip"]],
        "wind_speed": [COL["wind"]],
        "wind_dir": ["wind_direction_deg"],
        "pressure": [COL["pressure"]],
    }

    out: Dict[str, pd.DataFrame] = {}
    for path in sorted(base.glob("*.csv")):
        kind = _match_up_sw_file(path)
        if not kind:
            print(f"[up_sw] skip unrecognized: {path.name}")
            continue
        try:
            long_df = load_up_sw_variable(path, value_map.get(kind, ["value"]))
            # unit: wind km/h → m/s
            if kind == "wind_speed":
                long_df["value"] = long_df["value"] / 3.6
            out[kind] = long_df
            print(f"[up_sw] {kind}: {len(long_df):,} rows from {path.name}")
        except Exception as exc:
            print(f"[up_sw] FAIL {path.name}: {exc}")
    return out


def up_sw_station_series(
    up_sw: Dict[str, pd.DataFrame],
    kind: str,
    lat: float,
    lon: float,
    max_km: float = 50.0,
) -> Optional[pd.Series]:
    """Nearest-station hourly series for one variable, indexed by time."""
    if kind not in up_sw:
        return None
    df = up_sw[kind]
    if df.empty or df["lat"].notna().sum() == 0:
        return None
    # unique stations with coords
    st = (
        df.dropna(subset=["lat", "lon"])
        .groupby("station", as_index=False)
        .agg(lat=("lat", "first"), lon=("lon", "first"))
    )
    st["dist"] = ((st["lat"] - lat) ** 2 + (st["lon"] - lon) ** 2) ** 0.5 * 111.0
    st = st[st["dist"] <= max_km].sort_values("dist")
    if st.empty:
        return None
    station = st.iloc[0]["station"]
    sub = df[df["station"] == station].copy()
    sub = sub.set_index("time").sort_index()
    s = sub["value"].groupby(level=0).mean()
    s.name = kind
    return s


def merge_up_sw_into_weather(
    weather: pd.DataFrame,
    up_sw: Dict[str, pd.DataFrame],
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """
    Prefer UP SW measured GHI / temp / RH / wind / rain / pressure when a nearby
    station exists. Overwrites canonical columns on matching timestamps.
    """
    df = weather.copy()
    mapping = {
        "solar": COL["ghi"],
        "temp": COL["temp"],
        "humid": COL["rh"],
        "rain": COL["rain"],
        "wind_speed": COL["wind"],
        "pressure": COL["pressure"],
    }
    for kind, col in mapping.items():
        series = up_sw_station_series(up_sw, kind, lat, lon)
        if series is None or series.empty:
            continue
        # align to weather index
        aligned = series.reindex(df.index).dropna()
        if aligned.empty:
            continue
        if col not in df.columns:
            df[col] = np.nan
        df.loc[aligned.index, col] = aligned.values
    return df
