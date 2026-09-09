# train_up_v1.py
# ---------------------------------------------------------------
# SolarFlow — Uttar Pradesh focused trainer (v1)
# Target: robust UP-only model using LOCAL Mendeley measured solar
# + Open-Meteo weather for UP cities. Saves models/up/modelfile.pkl
# ---------------------------------------------------------------
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
import pvlib
import requests
from requests.adapters import HTTPAdapter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from up_geography import UP_STATE_NAME, UP_TRAINING_LOCATIONS
from data_schema import FEATURES, TARGET, DIR_MENDELEY, DIR_NASA, DIR_UP_SW, DIR_MODELS, DIR_REPORTS
from data_loaders import load_all_nasa, load_all_up_sw, nearest_nasa_frame, merge_up_sw_into_weather

ROOT = Path(__file__).resolve().parent
MENDELEY_DIR = ROOT / DIR_MENDELEY
CACHE_DIR = ROOT / "data" / "cache" / "open_meteo_up"
MODEL_DIR = ROOT / DIR_MODELS
REPORT_DIR = ROOT / DIR_REPORTS
NASA_DIR = ROOT / DIR_NASA
UP_SW_DIR = ROOT / DIR_UP_SW

IST = "Asia/Kolkata"
REQUEST_GAP_SECONDS = max(45.0, float(os.getenv("SOLARFLOW_REQUEST_GAP", "60")))
RETRY_BASE_SECONDS = max(60.0, float(os.getenv("SOLARFLOW_RETRY_BASE", "120")))
MAX_RETRIES = max(3, int(os.getenv("SOLARFLOW_MAX_RETRIES", "6")))
BATCH_SIZE = max(2, int(os.getenv("SOLARFLOW_BATCH_SIZE", "4")))
FORCE_REFRESH = os.getenv("SOLARFLOW_FORCE_REFRESH", "0") == "1"
# PRODUCTION: app.py forecasts use Open-Meteo → training PRIMARY must be Open-Meteo
# so feature distributions match (avoid train/serve skew).
# NASA = fallback only. UP-SW = optional soft GHI calibration (not full replace).
# SOLARFLOW_SKIP_OPEN_METEO=1 is offline experiment only — do NOT use for production models.
SKIP_OPEN_METEO = os.getenv("SOLARFLOW_SKIP_OPEN_METEO", "0") == "1"
UP_SW_GHI_CALIBRATE = os.getenv("SOLARFLOW_UP_SW_GHI_CALIBRATE", "1") == "1"

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "rain", "showers", "cloud_cover",
    "windspeed_10m", "windgusts_10m", "surface_pressure",
    "visibility", "vapour_pressure_deficit",
]
SOLAR_VARS = [
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "sunshine_duration", "weather_code", "is_day",
]
AQ_VARS = ["pm10", "pm2_5", "dust", "european_aqi"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("solarflow-up-trainer")


@dataclass(frozen=True)
class Location:
    city: str
    lat: float
    lon: float


UP_LOCATIONS = [Location(c, lat, lon) for c, lat, lon in UP_TRAINING_LOCATIONS]

session = requests.Session()
session.headers.update({"User-Agent": "SolarFlow-UP-Trainer/1.0", "Accept": "application/json"})
session.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0))
last_request = 0.0


def throttle(force_gap: float | None = None) -> None:
    global last_request
    gap = REQUEST_GAP_SECONDS if force_gap is None else force_gap
    now = time.monotonic()
    wait = gap - (now - last_request)
    if wait > 0:
        log.info("Rate-limit sleep %.0fs", wait)
        time.sleep(wait)
    last_request = time.monotonic()


def robust_get_json(url: str, params: Dict[str, str]) -> dict | list:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        throttle()
        try:
            r = session.get(url, params=params, timeout=(30, 300))
        except requests.RequestException as exc:
            last_err = exc
            delay = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            log.warning("Network error %d/%d: %s; sleep %.0fs", attempt, MAX_RETRIES, exc, delay)
            time.sleep(delay)
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except Exception as exc:
                last_err = exc
                time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
                continue
        if r.status_code == 429 or 500 <= r.status_code <= 599:
            delay = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            last_err = RuntimeError(f"HTTP {r.status_code}")
            log.warning("HTTP %s; sleep %.0fs", r.status_code, delay)
            time.sleep(delay)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    raise RuntimeError(f"Failed after retries: {last_err}")


def cache_path(kind: str, locations: Sequence[Location], start: str, end: str) -> Path:
    payload = kind + "|" + start + "|" + end + "|" + "|".join(f"{x.lat:.5f},{x.lon:.5f}" for x in locations)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
    return CACHE_DIR / f"{kind}_{start}_{end}_{digest}.json"


def batched(items: Sequence[Location], n: int) -> Iterable[List[Location]]:
    for i in range(0, len(items), n):
        yield list(items[i : i + n])


def fetch_open_meteo(kind: str, endpoint: str, locations: Sequence[Location],
                     start: str, end: str, variables: Sequence[str]) -> Dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}

    def fetch_group(group: List[Location], depth: int = 0) -> None:
        cp = cache_path(kind, group, start, end)
        raw = None
        if cp.exists() and not FORCE_REFRESH:
            try:
                raw = json.loads(cp.read_text(encoding="utf-8"))
                log.info("CACHE %s | %d locs | %s", kind, len(group), cp.name)
            except Exception:
                raw = None
        if raw is None:
            params = {
                "latitude": ",".join(f"{x.lat:.5f}" for x in group),
                "longitude": ",".join(f"{x.lon:.5f}" for x in group),
                "start_date": start,
                "end_date": end,
                "hourly": ",".join(variables),
                "timezone": IST,
            }
            if kind != "aq":
                params["models"] = "best_match"
                params["cell_selection"] = "land"
            log.info("REQUEST %s | %d loc(s) | %s → %s", kind, len(group), start, end)
            try:
                raw = robust_get_json(endpoint, params)
            except Exception as exc:
                if len(group) > 1 and depth < 5:
                    mid = len(group) // 2
                    fetch_group(group[:mid], depth + 1)
                    fetch_group(group[mid:], depth + 1)
                    return
                raise RuntimeError(f"Open-Meteo {kind} failed: {exc}")
            cp.write_text(json.dumps(raw, separators=(",", ":")), encoding="utf-8")

        payloads = raw if isinstance(raw, list) else [raw]
        remaining = group.copy()
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            lat = float(payload.get("latitude", np.nan))
            lon = float(payload.get("longitude", np.nan))
            if not np.isfinite(lat) or not np.isfinite(lon):
                continue
            match = min(remaining, key=lambda x: abs(x.lat - lat) + abs(x.lon - lon), default=None)
            if match is None:
                continue
            hourly = payload.get("hourly") or {}
            if "time" not in hourly:
                raise RuntimeError(f"Missing hourly time for {match.city}")
            df = pd.DataFrame(hourly)
            idx = pd.to_datetime(df.pop("time"), errors="coerce")
            df.index = idx.dt.tz_localize(IST, ambiguous="NaT", nonexistent="NaT")
            df = df[~df.index.isna()].sort_index()
            df["elevation_m"] = float(payload.get("elevation", 0.0) or 0.0)
            out[match.city] = df
            remaining.remove(match)
        if remaining:
            raise RuntimeError("Incomplete batch: " + ", ".join(x.city for x in remaining))

    for group in batched(list(locations), BATCH_SIZE):
        fetch_group(group)
    return out


# --------------- Mendeley (same parsers as pan-India trainer) ---------------
def parse_monthly_mendeley(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None, engine="openpyxl")
    if raw.shape[0] < 3:
        raise ValueError("Sheet1 too short")
    headers = [str(x).strip() if x is not None else "" for x in raw.iloc[1].tolist()]
    solar_idx = None
    for i, h in enumerate(headers):
        if h.lower() == "solar" or "all_ind_solar|p" in h.lower():
            solar_idx = i
            break
    if solar_idx is None:
        raise ValueError("No Solar column in Sheet1")
    ts = pd.to_datetime(raw.iloc[2:, 0], errors="coerce")
    val = pd.to_numeric(raw.iloc[2:, solar_idx], errors="coerce")
    df = pd.DataFrame({"time": ts, "solar_mw": val}).dropna(subset=["time"])
    df["solar_mw"] = df["solar_mw"].replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    df = df.dropna(subset=["solar_mw"])
    df["time"] = df["time"].dt.tz_localize(IST, ambiguous="NaT", nonexistent="NaT")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df[["solar_mw"]]


def parse_consolidated_mendeley(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Report", usecols=["Timestamp", "Solar (MW)"], engine="openpyxl")
    ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    solar = pd.to_numeric(df["Solar (MW)"], errors="coerce")
    out = pd.DataFrame({"time": ts, "solar_mw": solar}).dropna(subset=["time"])
    out["solar_mw"] = out["solar_mw"].replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    out = out.dropna(subset=["solar_mw"])
    out["time"] = out["time"].dt.tz_localize(IST, ambiguous="NaT", nonexistent="NaT")
    out = out.dropna(subset=["time"]).set_index("time").sort_index()
    return out[["solar_mw"]]


def load_all_mendeley() -> pd.DataFrame:
    if not MENDELEY_DIR.exists():
        raise FileNotFoundError(
            f"Missing Mendeley folder: {MENDELEY_DIR}\n"
            "Put your Grid-India / Mendeley Excel files there, then re-run."
        )
    files = sorted(MENDELEY_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Excel files in {MENDELEY_DIR}")
    log.info("MENDELEY FILES: %d", len(files))
    pieces, failures = [], []
    for p in files:
        try:
            if "January 2024- June 2025" in p.name:
                df = parse_consolidated_mendeley(p)
            else:
                df = parse_monthly_mendeley(p)
            pieces.append(df)
            log.info("  OK %s rows=%s", p.name[:40], f"{len(df):,}")
        except Exception as exc:
            failures.append((p.name, str(exc)))
            log.error("  FAIL %s %s", p.name, exc)
    if not pieces:
        raise RuntimeError("No Mendeley workbook parsed successfully.")
    master = pd.concat(pieces).sort_index()
    master = master.groupby(level=0)["solar_mw"].mean().to_frame()
    master["solar_mw"] = master["solar_mw"].clip(lower=0.0)
    master = master.resample("1h").mean()
    if master["solar_mw"].notna().sum() < 500:
        raise RuntimeError("Too few measured hourly rows after parsing.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if failures:
        (REPORT_DIR / "mendeley_parse_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    log.info("MEASURED: %s rows | %s → %s", f"{len(master):,}", master.index.min(), master.index.max())
    return master.dropna()


def build_features(raw: pd.DataFrame, location: Location) -> pd.DataFrame:
    df = raw.copy()
    expected = WEATHER_VARS + SOLAR_VARS + AQ_VARS
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    numeric = [c for c in expected if c != "weather_code"] + ["elevation_m"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    numeric = [c for c in numeric if c != "elevation_m"]
    df[numeric] = (
        df[numeric].replace([np.inf, -np.inf], np.nan)
        .interpolate(method="time", limit=6).ffill().bfill()
    )
    for c in [
        "precipitation", "rain", "showers", "cloud_cover", "windspeed_10m", "windgusts_10m",
        "shortwave_radiation", "direct_radiation", "diffuse_radiation",
        "direct_normal_irradiance", "sunshine_duration", "pm10", "pm2_5", "dust", "european_aqi",
    ]:
        df[c] = df[c].clip(lower=0.0)

    sol = pvlib.solarposition.get_solarposition(df.index, location.lat, location.lon)
    df["solar_zenith"] = sol["apparent_zenith"].clip(0, 180)
    df["solar_azimuth"] = sol["azimuth"].mod(360)
    tilt = float(np.clip(abs(location.lat) * 0.87, 5, 40))
    azimuth = 180.0
    dni = df["direct_normal_irradiance"].fillna(df["direct_radiation"]).clip(lower=0)
    ghi = df["shortwave_radiation"].clip(lower=0)
    dhi = df["diffuse_radiation"].clip(lower=0)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azimuth,
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=df["solar_zenith"], solar_azimuth=df["solar_azimuth"],
        albedo=0.20, dni_extra=pvlib.irradiance.get_extra_radiation(df.index),
        model="perez",
    )
    df["poa_global"] = pd.to_numeric(poa["poa_global"], errors="coerce").clip(lower=0)
    df["poa_direct"] = pd.to_numeric(poa["poa_direct"], errors="coerce").clip(lower=0)
    df["poa_diffuse"] = pd.to_numeric(poa["poa_diffuse"], errors="coerce").clip(lower=0)
    try:
        df["aoi"] = pvlib.irradiance.aoi(tilt, azimuth, df["solar_zenith"], df["solar_azimuth"]).clip(0, 180)
    except Exception:
        df["aoi"] = 90.0
    df["cell_temp"] = pvlib.temperature.faiman(df["poa_global"], df["temperature_2m"], df["windspeed_10m"])

    soiling = []
    ratio = 1.0
    for precip, pm10 in zip(df["precipitation"].fillna(0), df["pm10"].fillna(100)):
        if float(precip) >= 2.5:
            ratio = 1.0
        else:
            pm10v = max(0.0, float(pm10))
            ratio = max(0.65, ratio - (((pm10v / 500.0) * 0.0035) / 24.0))
        soiling.append(ratio)
    df["soiling_proxy"] = np.asarray(soiling)

    h = df.index.hour.to_numpy(float)
    d = df.index.dayofyear.to_numpy(float)
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["doy_sin"] = np.sin(2 * np.pi * d / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * d / 365.25)
    derived_day = (df["solar_zenith"] < 90).astype(float)
    df["is_day"] = pd.to_numeric(df["is_day"], errors="coerce").fillna(derived_day)
    df["elevation_m"] = pd.to_numeric(df["elevation_m"], errors="coerce").fillna(0)

    night = df["solar_zenith"] >= 90
    for c in [
        "poa_global", "poa_direct", "poa_diffuse", "direct_radiation",
        "diffuse_radiation", "direct_normal_irradiance", "sunshine_duration",
    ]:
        df.loc[night, c] = 0.0
    return df.replace([np.inf, -np.inf], np.nan)


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.20):
    times = np.array(sorted(df.index.unique()))
    if len(times) < 200:
        raise RuntimeError("Too few unique hours for validation.")
    cut = max(1, min(len(times) - 1, int(len(times) * (1 - test_fraction))))
    cut_time = times[cut]
    return df.loc[df.index < cut_time], df.loc[df.index >= cut_time]


def fit_and_save(df: pd.DataFrame, scale_mw: float) -> dict:
    train, test = chronological_split(df)
    X_train = train[FEATURES].astype(float).fillna(0.0)
    X_test = test[FEATURES].astype(float).fillna(0.0)
    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)

    # Multi-source + day weights (production-robust, low bias toward offline sensors)
    # - Open-Meteo primary rows: weight 1.0  (matches live app weather)
    # - NASA fallback rows:     weight 0.40 (extra climate coverage, cannot dominate)
    # - Daytime slightly up-weighted; night down-weighted (generation is day-driven)
    day_w = np.where(train["is_day"].to_numpy() > 0, 1.25, 0.55)
    if "source_weight" in train.columns:
        src_w = pd.to_numeric(train["source_weight"], errors="coerce").fillna(1.0).to_numpy()
    else:
        src_w = np.ones(len(train), dtype=float)
    counts = train.groupby(level=0)["city"].transform("count").to_numpy()
    weights = (day_w * src_w) / np.maximum(counts, 1)
    weights = np.clip(weights, 1e-3, None)

    model = HistGradientBoostingRegressor(
        learning_rate=0.035,
        max_iter=600,
        max_leaf_nodes=47,
        min_samples_leaf=60,
        l2_regularization=1.2,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=40,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=weights)

    pred = np.clip(model.predict(X_test), 0, 5.5)
    # Night must be zero in eval
    if "is_day" in test.columns:
        pred = np.where(test["is_day"].to_numpy() <= 0, 0.0, pred)

    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae_5kw_equivalent": float(mean_absolute_error(y_test, pred)),
        "rmse_5kw_equivalent": float(np.sqrt(mean_squared_error(y_test, pred))),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        # Rough "skill" style score for dashboards (not marketing accuracy)
        "daytime_mae": float(
            mean_absolute_error(
                y_test[test["is_day"] > 0],
                pred[test["is_day"].to_numpy() > 0],
            )
        ) if (test["is_day"] > 0).any() else None,
    }

    target_dir = MODEL_DIR / "up"
    target_dir.mkdir(parents=True, exist_ok=True)
    package = {
        "schema_version": 3,
        "target_type": "5kw_equivalent",
        "model": model,
        "features": FEATURES,
        "zone": "up",
        "scope": UP_STATE_NAME,
        "target": TARGET,
        "normalization_reference_mw": float(scale_mw),
        "metrics": metrics,
        "synthetic_target_used": False,
        "target_source": "local Mendeley/Grid-India measured grid-level solar generation",
        "locations": sorted(train["city"].unique().tolist()),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(package, target_dir / "modelfile.pkl")
    manifest = {k: v for k, v in package.items() if k != "model"}
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(
        "TRAINED UP | R² %.4f | MAE %.4f | RMSE %.4f | rows %s/%s",
        metrics["r2"], metrics["mae_5kw_equivalent"], metrics["rmse_5kw_equivalent"],
        f"{len(train):,}", f"{len(test):,}",
    )
    return metrics


def _frame_from_nasa(nasa_df: pd.DataFrame, loc: Location) -> pd.DataFrame:
    """Normalize NASA frame to expected weather columns."""
    df = nasa_df.copy()
    for col in WEATHER_VARS + SOLAR_VARS + AQ_VARS + ["elevation_m"]:
        if col not in df.columns:
            df[col] = np.nan
    # derive is_day later in build_features from zenith if missing
    if "is_day" not in df.columns or df["is_day"].isna().all():
        df["is_day"] = np.nan
    if "elevation_m" not in df.columns or df["elevation_m"].isna().all():
        df["elevation_m"] = 0.0
    # clip index to reasonable range
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def main() -> int:
    print("=" * 72)
    print(" SOLARFLOW UTTAR PRADESH TRAINER v1")
    print("=" * 72)
    print(f"Scope           : {UP_STATE_NAME} only")
    print(f"UP cities       : {len(UP_LOCATIONS)}")
    print(f"Mendeley folder : {MENDELEY_DIR}")
    print(f"NASA folder     : {NASA_DIR}")
    print(f"UP SW folder    : {UP_SW_DIR}")
    print(f"Model output    : {MODEL_DIR / 'up' / 'modelfile.pkl'}")
    print(f"FEATURES count  : {len(FEATURES)} (from data_schema.py)")
    print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    measured = load_all_mendeley()
    start = os.getenv("SOLARFLOW_START_DATE", measured.index.min().date().isoformat())
    end = os.getenv("SOLARFLOW_END_DATE", measured.index.max().date().isoformat())
    start = max(start, measured.index.min().date().isoformat())
    end = min(end, measured.index.max().date().isoformat())
    if start >= end:
        raise RuntimeError(f"Invalid window: {start} → {end}")

    # ---- Weather sources (train/serve aligned) ----
    # PRIMARY  = Open-Meteo  (same provider as app.py future forecasts)
    # FALLBACK = NASA        (only if Open-Meteo missing for that city)
    # OPTIONAL = UP-SW GHI soft calibration (30% blend; other features stay OM)
    nasa_map = load_all_nasa(ROOT)
    up_sw = load_all_up_sw(ROOT)
    log.info("LOCAL NASA sites loaded: %d", len(nasa_map))
    log.info("LOCAL UP-SW variables: %s", list(up_sw.keys()))
    log.info(
        "Policy: Open-Meteo PRIMARY (matches production). NASA=fallback. UP-SW GHI calibrate=%s",
        UP_SW_GHI_CALIBRATE,
    )

    weather: Dict[str, pd.DataFrame] = {}
    solar: Dict[str, pd.DataFrame] = {}
    aq: Dict[str, pd.DataFrame] = {}

    if not SKIP_OPEN_METEO:
        try:
            weather = fetch_open_meteo("weather", OPEN_METEO_ARCHIVE, UP_LOCATIONS, start, end, WEATHER_VARS)
            solar = fetch_open_meteo("solar", OPEN_METEO_ARCHIVE, UP_LOCATIONS, start, end, SOLAR_VARS)
            aq = fetch_open_meteo("aq", OPEN_METEO_AQ, UP_LOCATIONS, start, end, AQ_VARS)
        except Exception as exc:
            log.warning("Open-Meteo failed (%s) — will try NASA fallback per city", exc)
            weather, solar, aq = {}, {}, {}
    else:
        log.warning(
            "SOLARFLOW_SKIP_OPEN_METEO=1 → offline mode. "
            "Model may NOT match production Open-Meteo forecasts. Experiments only."
        )

    location_data = {}
    source_log = {}
    for loc in UP_LOCATIONS:
        sources = []
        base = None

        # 1) PRIMARY: Open-Meteo (production distribution)
        if loc.city in weather and loc.city in solar:
            om = weather[loc.city].join(solar[loc.city], how="outer", rsuffix="_solar")
            if loc.city in aq:
                om = om.join(aq[loc.city], how="outer", rsuffix="_aq")
            elevation = 0.0
            for src in (weather.get(loc.city), solar.get(loc.city), aq.get(loc.city)):
                if src is not None and "elevation_m" in src.columns and src["elevation_m"].notna().any():
                    elevation = float(src["elevation_m"].dropna().iloc[0])
                    break
            om["elevation_m"] = elevation
            base = om
            sources.append("open_meteo_primary")

        # 2) FALLBACK: NASA only if Open-Meteo missing
        if base is None:
            hit = nearest_nasa_frame(loc.lat, loc.lon, nasa_map, max_km=90.0)
            if hit is not None:
                key, nasa_df = hit
                base = _frame_from_nasa(nasa_df, loc)
                sources.append(f"nasa_fallback:{key}")

        if base is None:
            log.warning("No weather source for %s — skip", loc.city)
            continue

        # 3) Soft GHI calibration with UP-SW (keep other cols = Open-Meteo)
        if up_sw and UP_SW_GHI_CALIBRATE and "open_meteo_primary" in sources:
            from data_loaders import up_sw_station_series
            from data_schema import COL
            ghi_s = up_sw_station_series(up_sw, "solar", loc.lat, loc.lon, max_km=50.0)
            if ghi_s is not None and not ghi_s.empty and COL["ghi"] in base.columns:
                aligned = ghi_s.reindex(base.index).dropna()
                if len(aligned) > 24:
                    om_ghi = pd.to_numeric(base.loc[aligned.index, COL["ghi"]], errors="coerce")
                    blended = 0.7 * om_ghi + 0.3 * aligned.values
                    base.loc[aligned.index, COL["ghi"]] = blended
                    sources.append("up_sw_ghi_soft_calib_30pct")
        elif up_sw and SKIP_OPEN_METEO:
            # Offline experiment only: full overlay allowed when not training for production
            base = merge_up_sw_into_weather(base, up_sw, loc.lat, loc.lon)
            sources.append("up_sw_overlay_offline_only")

        # Restrict to training window
        base = base.loc[(base.index.date >= pd.to_datetime(start).date()) &
                        (base.index.date <= pd.to_datetime(end).date())]
        if base.empty:
            log.warning("Empty frame after window for %s", loc.city)
            continue

        location_data[loc.city] = build_features(base, loc)
        source_log[loc.city] = sources
        log.info("LOC %s sources=%s rows=%s", loc.city, sources, f"{len(location_data[loc.city]):,}")

    if not location_data:
        raise RuntimeError(
            "No location weather built. Need Open-Meteo and/or NASA CSVs in data/nasa/up/."
        )

    scale_mw = float(np.nanpercentile(measured["solar_mw"].to_numpy(), 99.5))
    if not np.isfinite(scale_mw) or scale_mw <= 0:
        raise RuntimeError("Measured target normalization failed.")
    measured[TARGET] = np.clip(measured["solar_mw"] / scale_mw * 5.0, 0, 5)

    pieces = []
    for loc in UP_LOCATIONS:
        if loc.city not in location_data:
            continue
        frame = location_data[loc.city].join(measured[[TARGET]], how="inner")
        if frame.empty:
            log.warning("No overlap for %s", loc.city)
            continue
        frame["city"] = loc.city
        frame["zone"] = "up"
        # Production-aligned source weights (not a one-hot bias feature at serve time)
        srcs = source_log.get(loc.city, [])
        if any(s.startswith("open_meteo") for s in srcs):
            frame["weather_source"] = "open_meteo"
            frame["source_weight"] = 1.0
        elif any(s.startswith("nasa") for s in srcs):
            frame["weather_source"] = "nasa_fallback"
            frame["source_weight"] = 0.40
        else:
            frame["weather_source"] = "other"
            frame["source_weight"] = 0.40
        pieces.append(frame)
    if not pieces:
        raise RuntimeError("No usable UP training rows.")

    zdf = pd.concat(pieces).sort_index()
    for c in FEATURES:
        if c not in zdf:
            zdf[c] = 0.0
        zdf[c] = pd.to_numeric(zdf[c], errors="coerce")
    zdf[FEATURES] = zdf[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    zdf = zdf.dropna(subset=[TARGET])

    metrics = fit_and_save(zdf, scale_mw)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 3,
        "scope": UP_STATE_NAME,
        "measured_source": "LOCAL Mendeley/Grid-India Excel files",
        "mendeley_dir": str(MENDELEY_DIR),
        "mendeley_files": [p.name for p in sorted(MENDELEY_DIR.glob("*.xlsx"))],
        "nasa_dir": str(NASA_DIR),
        "nasa_files_loaded": len(nasa_map),
        "up_sw_dir": str(UP_SW_DIR),
        "up_sw_variables": list(up_sw.keys()),
        "location_sources": source_log,
        "features": FEATURES,
        "measured_start": str(measured.index.min()),
        "measured_end": str(measured.index.max()),
        "measured_hourly_rows": int(len(measured)),
        "normalization_reference_mw_99_5": scale_mw,
        "locations": sorted(location_data.keys()),
        "model_metrics": metrics,
        "note": (
            "Metrics are chronological hold-out on 5 kW-equivalent target. "
            "Not a guaranteed site-level 98% accuracy claim. "
            "PRODUCTION ALIGNMENT: train weather PRIMARY=Open-Meteo (same as app forecasts). "
            "NASA=fallback only. UP-SW optional soft GHI calibration (30%) near stations."
        ),
        "train_serve_policy": {
            "inference_weather": "Open-Meteo forecast API (app.py)",
            "train_weather_primary": "Open-Meteo archive",
            "train_weather_fallback": "NASA POWER local CSV",
            "up_sw_role": "soft GHI calibration only when Open-Meteo primary",
            "sample_weights": {
                "open_meteo": 1.0,
                "nasa_fallback": 0.40,
                "daytime_mult": 1.25,
                "night_mult": 0.55,
                "note": "Weights only at train time; no source_id feature at inference (avoids serve bias)",
            },
            "inference_stack": "physics (pvlib) + ML + dynamic blend in app.py/solar_engine.py",
        },
    }
    (MODEL_DIR / "up" / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print(" UP TRAINING COMPLETE")
    print(f" Model → {MODEL_DIR / 'up' / 'modelfile.pkl'}")
    print(f" R²={metrics['r2']:.4f}  MAE={metrics['mae_5kw_equivalent']:.4f}  RMSE={metrics['rmse_5kw_equivalent']:.4f}")
    print(" Train/serve match: PRIMARY weather = Open-Meteo (same as future forecasts)")
    print(" NASA = fallback only | UP-SW = soft GHI calibration (not full replace)")
    print(" Features locked in data_schema.py (same list used at inference).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log.error("Interrupted.")
        raise SystemExit(130)
    except Exception as exc:
        log.exception("UP TRAINING FAILED: %s", exc)
        print("\nTRAINING STOPPED — place Mendeley Excel files in data/mendeley/ and retry.")
        raise SystemExit(1)
