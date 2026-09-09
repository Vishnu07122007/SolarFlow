# solar_engine.py
# SolarFlow physics engine — time-based multi shading + soiling + inverter curve
# Fixed: column-name aliases, full ML feature parity with trainer, hard night zeroing

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import pvlib
from pvlib import solarposition, irradiance, temperature

IST = "Asia/Kolkata"

SHADE_FACTORS = {
    "none": 1.00,
    "light": 0.88,
    "medium": 0.65,  # buildings
    "heavy": 0.40,   # high / dense
}

# Multi-select keys → (period, level)
SHADE_OPTION_MAP = {
    "morning_light": ("morning", "light"),
    "morning_medium": ("morning", "medium"),
    "morning_heavy": ("morning", "heavy"),
    "midday_light": ("midday", "light"),
    "midday_medium": ("midday", "medium"),
    "midday_heavy": ("midday", "heavy"),
    "evening_light": ("evening", "light"),
    "evening_medium": ("evening", "medium"),
    "evening_heavy": ("evening", "heavy"),
}

# Map every upstream name → canonical names from data_schema (single source of truth).
try:
    from data_schema import ALIASES as COLUMN_ALIASES
except Exception:
    COLUMN_ALIASES = {
        "relativehumidity_2m": "relative_humidity_2m",
        "relative_humidity_2m": "relative_humidity_2m",
        "wind_gusts_10m": "windgusts_10m",
        "windgusts_10m": "windgusts_10m",
        "weathercode": "weather_code",
        "weather_code": "weather_code",
        "rain": "rain",
    }


def parse_shading_selection(selected: list | str | None) -> Dict[str, str]:
    """
    selected: list of keys like ["morning_light", "evening_heavy"]
    returns profile: {"morning":"light","midday":"none","evening":"heavy"}
    If multiple levels for same period, heaviest wins.
    """
    profile = {"morning": "none", "midday": "none", "evening": "none"}
    rank = {"none": 0, "light": 1, "medium": 2, "heavy": 3}

    if not selected:
        return profile
    if isinstance(selected, str):
        selected = [x.strip() for x in selected.split(",") if x.strip()]

    for key in selected:
        key = (key or "").strip().lower()
        if key not in SHADE_OPTION_MAP:
            continue
        period, level = SHADE_OPTION_MAP[key]
        if rank.get(level, 0) > rank.get(profile[period], 0):
            profile[period] = level
    return profile


def shading_factor_for_hour(hour: int, profile: Dict[str, str]) -> float:
    if 6 <= hour < 10:
        level = profile.get("morning", "none")
    elif 10 <= hour < 14:
        level = profile.get("midday", "none")
    elif 14 <= hour < 18:
        level = profile.get("evening", "none")
    else:
        return 1.0
    return float(SHADE_FACTORS.get(level, 1.0))


def apply_time_shading(poa: np.ndarray, index: pd.DatetimeIndex, profile: Dict[str, str]) -> np.ndarray:
    hours = index.hour
    factors = np.array([shading_factor_for_hour(int(h), profile) for h in hours], dtype=float)
    return np.asarray(poa, dtype=float) * factors


@dataclass
class SystemInputs:
    latitude: float
    longitude: float
    num_panels: int
    num_inverters: int
    panel_stc_w: float
    panel_gamma: float
    inverter_paco_w: float
    inverter_eff: float
    installation_year: int
    manual_clean_date: Optional[str] = None
    tilt: Optional[float] = None
    azimuth: Optional[float] = None
    shading_selection: list = field(default_factory=list)  # multi-select keys
    albedo: float = 0.20
    mounting: str = "rooftop"
    bifacial_gain_pct: float = 0.0
    wiring_loss_pct: float = 2.0
    mismatch_loss_pct: float = 1.5
    availability_pct: float = 98.0
    degradation_pct_per_year: float = 0.5


WEATHER_MAP = {
    0: ("Clear", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Drizzle", "🌦️"),
    55: ("Heavy drizzle", "🌧️"),
    61: ("Light rain", "🌧️"),
    63: ("Rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    71: ("Light snow", "🌨️"),
    73: ("Snow", "❄️"),
    75: ("Heavy snow", "❄️"),
    80: ("Light showers", "🌦️"),
    81: ("Showers", "🌧️"),
    82: ("Heavy showers", "🌧️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with hail", "⛈️"),
    99: ("Thunderstorm with heavy hail", "⛈️"),
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Open-Meteo legacy column names to the trainer schema."""
    rename = {}
    for col in list(df.columns):
        if col in COLUMN_ALIASES:
            canonical = COLUMN_ALIASES[col]
            if canonical != col and canonical not in df.columns:
                rename[col] = canonical
            elif canonical != col and canonical in df.columns:
                # Prefer non-null values from either name
                df[canonical] = df[canonical].fillna(df[col])
                df = df.drop(columns=[col])
    if rename:
        df = df.rename(columns=rename)
    return df


def build_hourly_weather(weather_json: dict, aq_json: dict | None = None) -> pd.DataFrame:
    wh = weather_json.get("hourly") or {}
    df = pd.DataFrame(wh)
    if df.empty or "time" not in df:
        raise ValueError("Empty weather payload")

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    if df.index.tz is None:
        df.index = df.index.tz_localize(IST)
    else:
        df.index = df.index.tz_convert(IST)

    df = _normalize_columns(df)

    # Canonical columns expected downstream (trainer + UI)
    required = [
        "temperature_2m", "relative_humidity_2m", "apparent_temperature", "dew_point_2m",
        "precipitation_probability", "precipitation", "rain", "showers", "cloud_cover",
        "visibility", "windspeed_10m", "windgusts_10m", "surface_pressure",
        "uv_index", "sunshine_duration", "shortwave_radiation", "direct_radiation",
        "diffuse_radiation", "direct_normal_irradiance", "weather_code", "is_day",
        "vapour_pressure_deficit",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = 0.0 if col not in ("is_day",) else np.nan

    # Keep legacy aliases for any remaining UI code that still reads old names
    if "relative_humidity_2m" in df.columns and "relativehumidity_2m" not in df.columns:
        df["relativehumidity_2m"] = df["relative_humidity_2m"]
    if "windgusts_10m" in df.columns and "wind_gusts_10m" not in df.columns:
        df["wind_gusts_10m"] = df["windgusts_10m"]
    if "weather_code" in df.columns and "weathercode" not in df.columns:
        df["weathercode"] = df["weather_code"]

    if aq_json:
        ah = aq_json.get("hourly") or {}
        adf = pd.DataFrame(ah)
        if not adf.empty and "time" in adf:
            adf["time"] = pd.to_datetime(adf["time"])
            adf = adf.set_index("time")
            if adf.index.tz is None:
                adf.index = adf.index.tz_localize(IST)
            else:
                adf.index = adf.index.tz_convert(IST)
            for c in ["pm10", "pm2_5", "european_aqi", "dust"]:
                if c in adf.columns:
                    df[c] = adf[c].reindex(df.index)

    for c in ["pm10", "pm2_5", "european_aqi", "dust"]:
        if c not in df.columns:
            df[c] = np.nan

    # If rain missing, approximate from precipitation
    if "rain" in df.columns:
        df["rain"] = pd.to_numeric(df["rain"], errors="coerce").fillna(df["precipitation"]).fillna(0.0)
    else:
        df["rain"] = pd.to_numeric(df.get("precipitation", 0.0), errors="coerce").fillna(0.0)

    codes = pd.to_numeric(df["weather_code"], errors="coerce").fillna(0).astype(int)
    df["weather_code"] = codes
    df["weathercode"] = codes
    df["weather_name"] = codes.map(lambda x: WEATHER_MAP.get(int(x), ("Variable", "🌤️"))[0])
    df["weather_emoji"] = codes.map(lambda x: WEATHER_MAP.get(int(x), ("Variable", "🌤️"))[1])
    return df


def _auto_tilt(lat: float) -> float:
    """
    Rule-of-thumb fixed-tilt for annual residential yield:
    tilt ≈ |latitude| (degrees). For UP (~24–31°N) this is ~24–31°.
    """
    return float(np.clip(round(abs(float(lat)), 2), 5.0, 45.0))


def _parse_clean_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _soiling_ratio_series(
    index: pd.DatetimeIndex,
    clean_date: Optional[date],
    hist_precip: Optional[pd.DataFrame],
    pm25: pd.Series,
    dust: pd.Series,
    mounting: str,
    precip: Optional[pd.Series] = None,
    humidity: Optional[pd.Series] = None,
    wind: Optional[pd.Series] = None,
    tilt_deg: Optional[float] = None,
    installation_year: Optional[int] = None,
) -> pd.Series:
    """
    Cumulative soiling transmittance (1 = clean) over the forecast horizon.

    Starting state uses:
      - last clean date (preferred), or
      - install age when clean date unknown, or
      - mild default
      - then warm-start recovery from recent historical rain

    Hourly evolution uses:
      - rain wash (thresholded recovery)
      - PM2.5 + dust deposition
      - humidity (sticky film when high + dusty)
      - wind (light self-clean when dry+windy; more deposit when dusty+gusty)
      - tilt (steeper panels retain less dust)
      - mounting (rooftop vs ground)
    """
    n = len(index)
    out = np.ones(n, dtype=float)
    if n == 0:
        return pd.Series(out, index=index)

    try:
        start_day = index[0].date() if hasattr(index[0], "date") else pd.Timestamp(index[0]).date()
    except Exception:
        start_day = date.today()

    # --- Base daily dirt rate (UP residential) ---
    mount = (mounting or "rooftop").lower()
    base_daily = 0.0038 if mount in ("rooftop", "roof", "flush") else 0.0028
    # Tilt: flat holds more dust; steep sheds more (0°→1.15×, 30°→1.0×, 60°→0.75×)
    try:
        t = float(tilt_deg) if tilt_deg is not None else 25.0
    except Exception:
        t = 25.0
    t = float(np.clip(t, 0.0, 60.0))
    tilt_factor = float(np.clip(1.15 - (t / 60.0) * 0.40, 0.70, 1.20))
    hourly_base = (base_daily * tilt_factor) / 24.0

    # --- Starting cleanliness ---
    current = 0.96  # default mild dirt
    if clean_date is not None:
        days_since = max(0, (start_day - clean_date).days)
        # ~0.28%/day after clean for first weeks, slows; floor 0.68
        # Non-linear: first 30 days faster, then slower asymptote
        if days_since <= 30:
            loss = 0.0028 * days_since
        else:
            loss = 0.0028 * 30 + 0.0012 * (days_since - 30)
        current = float(np.clip(1.0 - loss, 0.68, 1.0))
    else:
        # No clean date: use install age as proxy (older system → dirtier start)
        years = 0
        try:
            if installation_year is not None:
                years = max(0, int(start_day.year) - int(installation_year))
        except Exception:
            years = 0
        # 0 yr → 0.97, 2 yr → 0.93, 5 yr → 0.88, 10+ → ~0.82 (capped)
        current = float(np.clip(0.97 - 0.015 * min(years, 10), 0.80, 0.97))

    # --- Historical rain warm-start (last ~30 days archive if provided) ---
    if hist_precip is not None and not hist_precip.empty:
        try:
            col = "precipitation" if "precipitation" in hist_precip.columns else hist_precip.columns[0]
            recent = pd.to_numeric(hist_precip[col], errors="coerce").fillna(0.0)
            # Prefer last 7 / 3 / 1 days of cumulative rain
            r24 = float(recent.tail(24).sum()) if len(recent) else 0.0
            r72 = float(recent.tail(72).sum()) if len(recent) else 0.0
            r168 = float(recent.tail(168).sum()) if len(recent) else 0.0
            # Heavy recent rain → cleaner start
            if r24 >= 12.0:
                current = min(1.0, current + 0.10)
            elif r24 >= 6.0:
                current = min(1.0, current + 0.06)
            elif r72 >= 15.0:
                current = min(1.0, current + 0.05)
            elif r168 >= 25.0:
                current = min(1.0, current + 0.03)
            # Long dry spell in history → slightly dirtier
            if r168 < 2.0 and len(recent) >= 120:
                current = max(0.70, current - 0.03)
        except Exception:
            pass

    current = float(np.clip(current, 0.65, 1.0))

    # --- Hourly series ---
    pm = pd.to_numeric(pm25, errors="coerce").reindex(index).fillna(45.0).to_numpy()
    if dust is not None:
        du = pd.to_numeric(dust, errors="coerce").reindex(index).fillna(0.0).to_numpy()
    else:
        du = np.zeros(n)
    if precip is not None:
        pr = pd.to_numeric(precip, errors="coerce").reindex(index).fillna(0.0).to_numpy()
    else:
        pr = np.zeros(n)
    if humidity is not None:
        rh = pd.to_numeric(humidity, errors="coerce").reindex(index).fillna(55.0).to_numpy()
    else:
        rh = np.full(n, 55.0)
    if wind is not None:
        ws = pd.to_numeric(wind, errors="coerce").reindex(index).fillna(2.0).to_numpy()
    else:
        ws = np.full(n, 2.0)

    # Month factor: UP pre-monsoon dustier (Mar–Jun), monsoon washes more effectively
    try:
        months = np.array([
            (index[i].month if hasattr(index[i], "month") else pd.Timestamp(index[i]).month)
            for i in range(n)
        ], dtype=int)
    except Exception:
        months = np.full(n, start_day.month)

    for i in range(n):
        rain_mm = float(pr[i])
        m = int(months[i])
        # Monsoon months: stronger wash; pre-monsoon: weaker light rain effect
        monsoon = m in (6, 7, 8, 9)
        dry_season = m in (3, 4, 5, 11, 12, 1, 2)

        # Rain recovery first
        if rain_mm >= 8.0:
            current = min(1.0, current + (0.16 if monsoon else 0.13))
        elif rain_mm >= 4.0:
            current = min(1.0, current + (0.10 if monsoon else 0.08))
        elif rain_mm >= 1.5:
            current = min(1.0, current + (0.05 if monsoon else 0.04))
        elif rain_mm >= 0.4:
            current = min(1.0, current + (0.025 if monsoon else 0.015))

        # Deposition this hour
        pm_i = float(pm[i])
        du_i = float(du[i])
        rh_i = float(np.clip(rh[i], 0.0, 100.0))
        ws_i = float(max(0.0, ws[i]))

        pm_factor = 1.0 + min(pm_i / 110.0, 2.8)
        dust_factor = 1.0 + min(du_i / 55.0, 1.8)
        # High humidity + pollution → stickier film
        humidity_factor = 1.0 + max(0.0, (rh_i - 60.0) / 100.0) * (0.35 if pm_i > 40 else 0.15)
        # Season: pre-monsoon dustier
        season_factor = 1.20 if dry_season and m in (3, 4, 5) else (0.90 if monsoon else 1.0)

        dep = hourly_base * pm_factor * dust_factor * humidity_factor * season_factor

        # Wind: dry + windy + low PM → slight self-clean; dusty + gusty → more deposit
        if ws_i >= 6.0 and rain_mm < 0.1:
            if pm_i < 35 and du_i < 20:
                current = min(1.0, current + 0.004)  # light blow-off
            elif pm_i > 80 or du_i > 40:
                dep *= 1.15  # wind-blown dust

        current = max(0.55, current - dep)
        out[i] = float(np.clip(current, 0.55, 1.0))

    return pd.Series(out, index=index)


def inverter_efficiency_curve(load_frac: np.ndarray, rated_eff: float) -> np.ndarray:
    """Load-dependent inverter efficiency."""
    lf = np.clip(load_frac, 0, 1.2)
    curve = rated_eff * (0.92 + 0.08 * np.minimum(lf / 0.2, 1.0))
    curve = np.where(lf < 0.02, 0.0, curve)
    return np.clip(curve, 0.0, 0.99)


def enrich_frame(
    frame: pd.DataFrame,
    inputs: SystemInputs,
    hist_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Physics baseline + full feature set matching train_pan_india_zones_v3 FEATURES.
    Forces night-time AC power to zero.
    """
    df = frame.copy()
    lat, lon = inputs.latitude, inputs.longitude

    tilt_auto = inputs.tilt is None
    az_auto = inputs.azimuth is None
    tilt = float(inputs.tilt) if inputs.tilt is not None else _auto_tilt(lat)
    azimuth = float(inputs.azimuth) if inputs.azimuth is not None else 180.0

    # Solar position — use apparent_zenith to match trainer
    solpos = solarposition.get_solarposition(df.index, lat, lon)
    solar_zenith = solpos["apparent_zenith"].clip(0, 180)
    solar_azimuth = solpos["azimuth"].mod(360)
    dni_extra = irradiance.get_extra_radiation(df.index)

    ghi = pd.to_numeric(df.get("shortwave_radiation", 0), errors="coerce").fillna(0.0).clip(lower=0)
    dni = pd.to_numeric(df.get("direct_normal_irradiance", 0), errors="coerce").fillna(0.0).clip(lower=0)
    dhi = pd.to_numeric(df.get("diffuse_radiation", 0), errors="coerce").fillna(0.0).clip(lower=0)
    direct_rad = pd.to_numeric(df.get("direct_radiation", 0), errors="coerce").fillna(0.0).clip(lower=0)

    # Approximate DNI if completely missing
    cos_z = np.cos(np.deg2rad(solar_zenith.clip(0, 90)))
    if float(dni.sum()) == 0.0 and float(ghi.sum()) > 0.0:
        dni = ((ghi - dhi) / cos_z.replace(0, np.nan)).fillna(0).clip(lower=0)

    poa = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        model="perez",
        albedo=inputs.albedo,
    )
    poa_global = pd.to_numeric(poa["poa_global"], errors="coerce").fillna(0).clip(lower=0)
    poa_direct = pd.to_numeric(poa.get("poa_direct", 0), errors="coerce").fillna(0).clip(lower=0)
    poa_diffuse = pd.to_numeric(poa.get("poa_diffuse", 0), errors="coerce").fillna(0).clip(lower=0)

    # Night hard-zero on irradiance components (matches trainer)
    night_mask = solar_zenith >= 90.0
    poa_global = poa_global.where(~night_mask, 0.0)
    poa_direct = poa_direct.where(~night_mask, 0.0)
    poa_diffuse = poa_diffuse.where(~night_mask, 0.0)

    # Multi-select time shading (daytime only; already zero at night)
    shade_profile = parse_shading_selection(inputs.shading_selection)
    poa_global = pd.Series(
        apply_time_shading(poa_global.to_numpy(), df.index, shade_profile),
        index=df.index,
    ).clip(lower=0)

    # Cell temperature
    wind = pd.to_numeric(df.get("windspeed_10m", 1.0), errors="coerce").fillna(1.0)
    temp_air = pd.to_numeric(df.get("temperature_2m", 25.0), errors="coerce").fillna(25.0)
    if inputs.mounting == "ground":
        t_cell = temperature.faiman(poa_global, temp_air, wind, u0=25.0, u1=6.84)
    else:
        t_cell = temperature.faiman(poa_global, temp_air, wind, u0=27.0, u1=6.0)

    gamma = float(inputs.panel_gamma)  # fraction / °C, e.g. -0.0035
    stc = float(inputs.panel_stc_w)
    years = max(0, datetime.now().year - int(inputs.installation_year))
    deg_factor = max(0.70, 1.0 - (inputs.degradation_pct_per_year / 100.0) * years)

    # Soiling ratio (used by physics + UI) — multi-factor cumulative model
    clean_d = _parse_clean_date(inputs.manual_clean_date)
    precip = pd.to_numeric(df.get("precipitation", 0), errors="coerce").fillna(0.0)
    hum = pd.to_numeric(
        df.get("relative_humidity_2m", df.get("relativehumidity_2m", 55.0)),
        errors="coerce",
    ).fillna(55.0)
    wind_s = pd.to_numeric(df.get("windspeed_10m", df.get("wind_speed_10m", 2.0)), errors="coerce").fillna(2.0)
    tilt_for_soil = inputs.tilt
    if tilt_for_soil is None:
        try:
            tilt_for_soil = abs(float(inputs.latitude))  # rule-of-thumb when auto
        except Exception:
            tilt_for_soil = 25.0
    soiling = _soiling_ratio_series(
        df.index,
        clean_d,
        hist_df,
        df.get("pm2_5", pd.Series(np.nan, index=df.index)),
        df.get("dust", pd.Series(np.nan, index=df.index)),
        inputs.mounting,
        precip=precip,
        humidity=hum,
        wind=wind_s,
        tilt_deg=float(tilt_for_soil) if tilt_for_soil is not None else 25.0,
        installation_year=int(inputs.installation_year) if inputs.installation_year else None,
    )

    # Trainer-style soiling_proxy — same cumulative behaviour (aligned with physics)
    soiling_proxy = soiling.copy()

    temp_factor = 1.0 + gamma * (t_cell - 25.0)
    bifacial = 1.0 + (inputs.bifacial_gain_pct / 100.0)
    loss = (1.0 - inputs.wiring_loss_pct / 100.0) * (1.0 - inputs.mismatch_loss_pct / 100.0)
    avail = inputs.availability_pct / 100.0

    # DC power kW  (nameplate * POA/1000 * factors)
    system_dc_kw = stc * inputs.num_panels / 1000.0
    dc_kw = (
        (poa_global / 1000.0)
        * system_dc_kw
        * temp_factor.clip(lower=0.5)
        * soiling
        * deg_factor
        * bifacial
        * loss
        * avail
    ).clip(lower=0)

    ac_cap = (inputs.inverter_paco_w * inputs.num_inverters) / 1000.0
    load_frac = (dc_kw / max(ac_cap, 1e-6)).to_numpy()
    inv_eff = inverter_efficiency_curve(load_frac, float(inputs.inverter_eff))
    ac_kw = np.minimum(dc_kw.to_numpy() * inv_eff, ac_cap)

    # HARD night zero on physics baseline
    ac_kw = np.where(night_mask.to_numpy(), 0.0, ac_kw)

    # ---- Write all columns used by physics UI ----
    df["poa_global"] = poa_global
    df["poa_direct"] = poa_direct
    df["poa_diffuse"] = poa_diffuse
    df["cell_temp"] = t_cell
    df["soiling_ratio"] = soiling
    df["inverter_efficiency"] = inv_eff
    df["physics_baseline_kw"] = ac_kw

    # ---- Full ML feature parity with train_pan_india_zones_v3 FEATURES ----
    df["solar_zenith"] = solar_zenith
    df["solar_azimuth"] = solar_azimuth
    try:
        df["aoi"] = irradiance.aoi(tilt, azimuth, solar_zenith, solar_azimuth).clip(0, 180)
    except Exception:
        df["aoi"] = 90.0

    # is_day: prefer API, else derive from zenith
    if "is_day" in df.columns and df["is_day"].notna().any():
        df["is_day"] = pd.to_numeric(df["is_day"], errors="coerce")
        df["is_day"] = df["is_day"].fillna((solar_zenith < 90).astype(float))
    else:
        df["is_day"] = (solar_zenith < 90).astype(float)
    # Force night is_day = 0
    df.loc[night_mask, "is_day"] = 0.0

    # Night labels: never show daytime sky emojis (☀️ Clear at 3 AM confuses users).
    # Keep rain/storm meaning; rewrite clear / sunny / mainly-clear style names only.
    if "weather_name" in df.columns:
        night_labels = night_mask.to_numpy()
        names = df["weather_name"].astype(str).to_numpy()
        emojis = df["weather_emoji"].astype(str).to_numpy() if "weather_emoji" in df.columns else np.array(["🌙"] * len(df))
        for i, is_night in enumerate(night_labels):
            if not is_night:
                continue
            n = names[i].lower()
            if any(k in n for k in ("thunder", "storm")):
                names[i], emojis[i] = "Night thunderstorm", "⛈️"
            elif any(k in n for k in ("drizzle", "rain", "shower")):
                names[i], emojis[i] = "Night rain", "🌧️"
            elif "fog" in n or "mist" in n or "haze" in n:
                names[i], emojis[i] = "Night fog", "🌫️"
            elif "overcast" in n or "cloud" in n:
                names[i], emojis[i] = "Cloudy night", "☁️"
            else:
                # Clear / mainly clear / sunny / variable → night
                names[i], emojis[i] = "Clear night", "🌙"
        df["weather_name"] = names
        df["weather_emoji"] = emojis

    h = df.index.hour.to_numpy(dtype=float)
    d = df.index.dayofyear.to_numpy(dtype=float)
    df["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * d / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * d / 365.25)

    df["soiling_proxy"] = soiling_proxy
    if "elevation_m" not in df.columns:
        df["elevation_m"] = 0.0
    df["elevation_m"] = pd.to_numeric(df["elevation_m"], errors="coerce").fillna(0.0)

    # Ensure humidity / gusts exist under trainer names
    if "relative_humidity_2m" not in df.columns:
        df["relative_humidity_2m"] = pd.to_numeric(
            df.get("relativehumidity_2m", 50), errors="coerce"
        ).fillna(50)
    if "windgusts_10m" not in df.columns:
        df["windgusts_10m"] = pd.to_numeric(
            df.get("wind_gusts_10m", 0), errors="coerce"
        ).fillna(0)
    if "vapour_pressure_deficit" not in df.columns:
        df["vapour_pressure_deficit"] = 0.0

    # Zero solar radiation fields at night (feature hygiene)
    for c in [
        "poa_global", "poa_direct", "poa_diffuse",
        "shortwave_radiation", "direct_radiation",
        "diffuse_radiation", "direct_normal_irradiance", "sunshine_duration",
    ]:
        if c in df.columns:
            df.loc[night_mask, c] = 0.0

    meta = {
        "tilt": tilt,
        "tilt_auto": tilt_auto,
        "azimuth": azimuth,
        "azimuth_auto": az_auto,
        "baseline_clean_date": clean_d.isoformat() if clean_d else None,
        "shading_profile": shade_profile,
        "night_hours": int(night_mask.sum()),
        "system_dc_kw": system_dc_kw,
    }
    return df, meta
