# model_registry.py
# SolarFlow — Model Registry (UP-first)
# Platform currently serves Uttar Pradesh only. Models live under models/up/

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np

from up_geography import UP_STATE_NAME, is_up_state

# Preferred load order for UP inference
_UP_MODEL_CANDIDATES = ("up", "north", "global")


def _models_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def load_zone_model(zone: str) -> Optional[Dict[str, Any]]:
    """Load a trained model package for the given zone folder name."""
    path = os.path.join(_models_dir(), zone, "modelfile.pkl")
    if not os.path.exists(path):
        return None
    try:
        pkg = joblib.load(path)
        if not isinstance(pkg, dict):
            return None
        if "model" not in pkg or "features" not in pkg:
            return None
        return pkg
    except Exception:
        return None


def zone_for_state(state: str) -> str:
    """
    UP-first mapping. Non-UP states are not served by the product layer,
    but if called we still return 'up' as the only supported zone.
    """
    if is_up_state(state) or not state:
        return "up"
    return "up"  # product is UP-only; do not route to other zones


def get_model_for_state(state_name: str) -> Dict[str, Any]:
    """
    Main entry used by the application.
    Tries models/up first, then north / global as soft fallbacks while you retrain.
    """
    # Enforce product scope
    if state_name and not is_up_state(state_name):
        return {
            "zone": "up",
            "model": None,
            "features": [],
            "available": False,
            "target_type": None,
            "normalization_mw": None,
            "package": None,
            "error": "outside_up_scope",
        }

    chosen_zone = None
    pkg = None
    for zone in _UP_MODEL_CANDIDATES:
        pkg = load_zone_model(zone)
        if pkg is not None:
            chosen_zone = zone
            break

    if pkg is None:
        return {
            "zone": "up",
            "model": None,
            "features": [],
            "available": False,
            "target_type": None,
            "normalization_mw": None,
            "package": None,
        }

    return {
        "zone": chosen_zone or "up",
        "model": pkg["model"],
        "features": pkg["features"],
        "available": True,
        "target_type": pkg.get("target", "target_5kw_equivalent"),
        "normalization_mw": pkg.get("normalization_reference_mw", None),
        "package": pkg,
        "scope": UP_STATE_NAME,
    }


def predict_model(package: Dict[str, Any], frame) -> np.ndarray:
    """
    Run prediction using the loaded model.
    Returns 0→5 kW equivalent scale. Caller scales to system size.
    Night rows forced to 0 when is_day / solar_zenith / poa available.
    """
    if not package.get("available") or package.get("model") is None:
        return np.zeros(len(frame))

    features = package["features"]
    model = package["model"]
    X = frame.copy()

    aliases = {
        "relative_humidity_2m": ["relativehumidity_2m"],
        "windgusts_10m": ["wind_gusts_10m"],
        "weather_code": ["weathercode"],
    }
    for canonical, alts in aliases.items():
        if canonical not in X.columns:
            for a in alts:
                if a in X.columns:
                    X[canonical] = X[a]
                    break

    for f in features:
        if f not in X.columns:
            X[f] = 0.0

    X = (
        X[features]
        .replace([np.inf, -np.inf], np.nan)
        .bfill()
        .ffill()
        .fillna(0.0)
    )

    try:
        pred = np.asarray(model.predict(X), dtype=float)
        pred = np.clip(pred, 0.0, 5.5)
    except Exception:
        return np.zeros(len(frame))

    try:
        if "is_day" in frame.columns:
            night = frame["is_day"].to_numpy() <= 0
            pred = np.where(night, 0.0, pred)
        elif "solar_zenith" in frame.columns:
            night = frame["solar_zenith"].to_numpy() >= 90.0
            pred = np.where(night, 0.0, pred)
        if "poa_global" in frame.columns:
            pred = np.where(frame["poa_global"].to_numpy() <= 0.0, 0.0, pred)
    except Exception:
        pass

    return pred


def get_model_status() -> Dict[str, Any]:
    """Admin / debug: which model folders are present."""
    status = {}
    for zone in ["up", "global", "north", "west", "south", "east"]:
        pkg = load_zone_model(zone)
        status[zone] = {
            "available": pkg is not None,
            "features_count": len(pkg["features"]) if pkg else 0,
            "target": pkg.get("target") if pkg else None,
            "scope": "Uttar Pradesh" if zone in ("up", "north", "global") else zone,
        }
    return status
