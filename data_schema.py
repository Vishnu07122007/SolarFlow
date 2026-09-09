# data_schema.py
# ---------------------------------------------------------------------------
# SolarFlow UP — SINGLE SOURCE OF TRUTH for column / feature names.
# Every module (trainer, solar_engine, model_registry, loaders) must use these
# names only. Never invent parallel aliases in other files.
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Dict, List

# ---- Canonical weather / irradiance columns (after any source is loaded) ----
# These are the names stored in DataFrames and passed to ML / physics.
COL = {
    # time index is always tz-aware Asia/Kolkata DatetimeIndex named "time"
    "ghi": "shortwave_radiation",          # W/m²  (GHI)
    "dni": "direct_normal_irradiance",      # W/m²
    "dhi": "diffuse_radiation",             # W/m²
    "direct": "direct_radiation",           # W/m² horizontal direct (optional)
    "temp": "temperature_2m",               # °C
    "rh": "relative_humidity_2m",           # %
    "dew": "dew_point_2m",                  # °C
    "precip": "precipitation",              # mm
    "rain": "rain",                         # mm
    "showers": "showers",                   # mm
    "cloud": "cloud_cover",                 # %
    "wind": "windspeed_10m",                # m/s
    "gust": "windgusts_10m",                # m/s
    "pressure": "surface_pressure",         # hPa (normalized from any source)
    "visibility": "visibility",
    "vpd": "vapour_pressure_deficit",
    "sunshine": "sunshine_duration",
    "weather_code": "weather_code",
    "is_day": "is_day",
    "elevation": "elevation_m",
    # AQ
    "pm10": "pm10",
    "pm25": "pm2_5",
    "dust": "dust",
    "aqi": "european_aqi",
    # derived physics / ML
    "poa_global": "poa_global",
    "poa_direct": "poa_direct",
    "poa_diffuse": "poa_diffuse",
    "solar_zenith": "solar_zenith",
    "solar_azimuth": "solar_azimuth",
    "aoi": "aoi",
    "cell_temp": "cell_temp",
    "soiling_proxy": "soiling_proxy",
    "hour_sin": "hour_sin",
    "hour_cos": "hour_cos",
    "doy_sin": "doy_sin",
    "doy_cos": "doy_cos",
}

# Alias map: ANY incoming name → canonical COL value
# Used when reading Open-Meteo, NASA POWER, or UP SW telemetry.
ALIASES: Dict[str, str] = {
    # Open-Meteo legacy / variants
    "relativehumidity_2m": COL["rh"],
    "relative_humidity_2m": COL["rh"],
    "wind_gusts_10m": COL["gust"],
    "windgusts_10m": COL["gust"],
    "weathercode": COL["weather_code"],
    "weather_code": COL["weather_code"],
    "shortwave_radiation": COL["ghi"],
    "direct_normal_irradiance": COL["dni"],
    "diffuse_radiation": COL["dhi"],
    "direct_radiation": COL["direct"],
    "temperature_2m": COL["temp"],
    "precipitation": COL["precip"],
    "rain": COL["rain"],
    "cloud_cover": COL["cloud"],
    "windspeed_10m": COL["wind"],
    "surface_pressure": COL["pressure"],
    "is_day": COL["is_day"],
    # NASA POWER
    "ALLSKY_SFC_SW_DWN": COL["ghi"],
    "ALLSKY_SFC_SW_DNI": COL["dni"],
    "ALLSKY_SFC_SW_DIFF": COL["dhi"],
    "T2M": COL["temp"],
    "RH2M": COL["rh"],
    "WS10M": COL["wind"],
    "PRECTOTCORR": COL["precip"],
    "PS": COL["pressure"],  # NASA PS is kPa → convert in loader
    # UP SW telemetry (raw header fragments → canonical)
    "Solar Radiation (Watt/m2)": COL["ghi"],
    "Air Temperature Telemetry Hourly (ºC)": COL["temp"],
    "Air Temperature Telemetry Hourly (°C)": COL["temp"],
    "Telemetry Hourly Relative Humidity (%)": COL["rh"],
    "Telemetry Hourly Rainfall (mm)": COL["rain"],
    "Telemetry Hourly Wind Speed (Km/Hr)": COL["wind"],  # convert km/h → m/s in loader
    "Telemetry Hourly Wind Direction (Degree)": "wind_direction_deg",
    "Telemetry_Hourly_Atmospheric Pressure (mb)": COL["pressure"],
}

# ML feature list — MUST match solar_engine.enrich_frame output and model package
FEATURES: List[str] = [
    "poa_global", "poa_direct", "poa_diffuse",
    "solar_zenith", "solar_azimuth", "aoi", "cell_temp",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "rain", "showers", "cloud_cover",
    "windspeed_10m", "windgusts_10m", "surface_pressure", "visibility",
    "vapour_pressure_deficit", "shortwave_radiation", "direct_radiation",
    "diffuse_radiation", "direct_normal_irradiance", "sunshine_duration",
    "pm10", "pm2_5", "dust", "european_aqi", "elevation_m",
    "soiling_proxy", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_day",
]

TARGET = "target_5kw_equivalent"

# Folder layout (relative to project root)
DIR_MENDELEY = "data/mendeley"
DIR_NASA = "data/nasa/up"
DIR_UP_SW = "data/up_sw"
DIR_CACHE = "data/cache"
DIR_MODELS = "models"
DIR_REPORTS = "data/training_reports"
