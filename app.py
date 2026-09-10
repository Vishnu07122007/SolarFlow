# app.py
# ---------------------------------------------------------------
# SolarFlow Enterprise - Main Application
# Adaptive physics+ML blend | Google Maps location | multi shading
# ---------------------------------------------------------------

import os
import uuid
from datetime import datetime, timedelta, timezone, date
import time
import pytz
import requests
import numpy as np
import pandas as pd
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from dotenv import load_dotenv
from hardware_catalog import (
    INDIAN_CUSTOM_PANELS,
    INDIAN_CUSTOM_INVERTERS,
)

# Load .env from project directory (works on PythonAnywhere + local)
_BASE_DIR_EARLY = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE_DIR_EARLY, ".env"))

from hardware_catalog import CEC_MOD_T, CEC_INV_T, get_panel_specs, get_inverter_specs
from model_registry import get_model_for_state, predict_model, get_model_status
from up_geography import (
    UP_STATE_NAME, UP_DISTRICTS, is_up_state, is_up_district,
    normalize_state, normalize_district, pincode_looks_like_up,
    reject_non_up_message, reject_outside_up_coords_message,
    district_list_for_ui, is_inside_up, UP_BBOX,
)
from solar_engine import SystemInputs, build_hourly_weather, enrich_frame, parse_shading_selection

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.join(_BASE_DIR, "templates")
# Prefer templates/ if present; also load HTML next to app.py
_template_search = []
if os.path.isdir(_TEMPLATES_DIR):
    _template_search.append(_TEMPLATES_DIR)
_template_search.append(_BASE_DIR)

app = Flask(
    __name__,
    template_folder=_template_search[0],
    static_folder=_BASE_DIR,
)
from jinja2 import ChoiceLoader, FileSystemLoader
app.jinja_loader = ChoiceLoader([FileSystemLoader(d) for d in _template_search])

app.secret_key = os.getenv("SECRET_KEY", "solarflow_secure_key_change_this_in_production")
# Admin credentials — set only in .env (never hardcode in source)
ADMIN_EMAIL = (os.getenv("ADMIN_EMAIL") or "admin@solarflow.com").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or ""
ADMIN_NAME = (os.getenv("ADMIN_NAME") or "Platform Administrator").strip()
# When "1"/"true": on startup, sync admin email/password from .env (easy rotate).
# Set ADMIN_SYNC=0 after first deploy if you change password only in the UI.
ADMIN_SYNC = (os.getenv("ADMIN_SYNC") or "1").strip().lower() in ("1", "true", "yes", "on")

# Database: SQLite default; on Render use Postgres via DATABASE_URL.
# Free Render disk is ephemeral — SQLite is wiped on redeploy/restart.
_INSTANCE_DIR = os.path.join(_BASE_DIR, "instance")
os.makedirs(_INSTANCE_DIR, exist_ok=True)
_DEFAULT_DB = os.path.join(_INSTANCE_DIR, "solarflow_up.db")
_db_url = (os.getenv("DATABASE_URL") or "").strip()
if _db_url.startswith("postgres://"):
    # SQLAlchemy needs postgresql:// (Render may give postgres://)
    _db_url = "postgresql://" + _db_url[len("postgres://") :]
if not _db_url:
    _db_url = f"sqlite:///{_DEFAULT_DB}"
app.config["SQLALCHEMY_DATABASE_URI"] = _db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Sessions (HTTPS on Render)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
if os.getenv("SESSION_COOKIE_SECURE", "1").strip() in ("1", "true", "yes"):
    app.config["SESSION_COOKIE_SECURE"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=14)

db = SQLAlchemy(app)
IST_TZ = pytz.timezone("Asia/Kolkata")
CEA_GRID_EMISSION_FACTOR = 0.71
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
OSM_HEADERS = {"User-Agent": "SolarFlow-SolarPredictor/4.2"}


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
class User(db.Model):
    __tablename__ = "users"
    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), default="Solar Engineer")
    is_admin = db.Column(db.Boolean, default=False)
    system = db.relationship("SolarSystem", backref="user", uselist=False, cascade="all, delete-orphan")
    logs = db.relationship("ForecastLog", backref="user", uselist=False, cascade="all, delete-orphan")
    history = db.relationship("HistoricalLog", backref="user", cascade="all, delete-orphan")


class SolarSystem(db.Model):
    __tablename__ = "solar_systems"
    system_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    state = db.Column(db.String(100))
    district = db.Column(db.String(100))
    city_locality = db.Column(db.String(150))
    pincode = db.Column(db.String(10))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    brand_model = db.Column(db.String(150))
    inverter_model = db.Column(db.String(150))
    num_panels = db.Column(db.Integer)
    num_inverters = db.Column(db.Integer)
    tilt_angle = db.Column(db.Float)
    azimuth_angle = db.Column(db.Float)
    installation_year = db.Column(db.Integer)
    last_cleaned_date = db.Column(db.String(20))
    shading_selection = db.Column(db.String(250), default="")  # comma-separated keys
    daily_units = db.Column(db.Float)
    albedo = db.Column(db.Float, default=0.20)
    mounting_type = db.Column(db.String(30), default="rooftop")
    bifacial_gain_pct = db.Column(db.Float, default=0.0)
    wiring_loss_pct = db.Column(db.Float, default=2.0)
    mismatch_loss_pct = db.Column(db.Float, default=1.5)
    availability_pct = db.Column(db.Float, default=98.0)
    degradation_pct_per_year = db.Column(db.Float, default=0.5)
    calibration_slope = db.Column(db.Float, default=1.0)
    # First day this system was saved / forecasted — history never invents days before this
    system_active_from = db.Column(db.String(20), nullable=True)


class ForecastLog(db.Model):
    __tablename__ = "forecast_logs"
    log_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    predicted_gen_kwh = db.Column(db.Float, nullable=False, default=0)
    today_gen_kwh = db.Column(db.Float, default=0)
    daily_consumption_kwh = db.Column(db.Float, nullable=False, default=0)
    carbon_saved_kg = db.Column(db.Float, nullable=False, default=0)
    today_carbon_saved_kg = db.Column(db.Float, default=0)
    last_updated = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                             onupdate=lambda: datetime.now(timezone.utc))


class HistoricalLog(db.Model):
    __tablename__ = "historical_logs"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    log_date = db.Column(db.String(20), nullable=False)
    generated_kwh = db.Column(db.Float, nullable=False, default=0.0)  # predicted (or actual if calibrated)
    predicted_kwh = db.Column(db.Float, nullable=True)  # model forecast for that day
    actual_kwh = db.Column(db.Float, nullable=True)  # user-entered inverter / meter truth
    consumption_kwh = db.Column(db.Float, nullable=False, default=0.0)
    soiling_loss_pct = db.Column(db.Float, default=0)
    carbon_saved_kg = db.Column(db.Float, default=0.0)  # daily CO₂ offset snapshot
    is_calibrated = db.Column(db.Boolean, default=False)
    calibrated_at = db.Column(db.DateTime, nullable=True)


with app.app_context():
    db.create_all()
    # Lightweight SQLite column migrate (create_all does not alter existing tables)
    try:
        cols = {r[1] for r in db.session.execute(db.text("PRAGMA table_info(historical_logs)")).fetchall()}
        alters = []
        if "predicted_kwh" not in cols:
            alters.append("ALTER TABLE historical_logs ADD COLUMN predicted_kwh FLOAT")
        if "actual_kwh" not in cols:
            alters.append("ALTER TABLE historical_logs ADD COLUMN actual_kwh FLOAT")
        if "is_calibrated" not in cols:
            alters.append("ALTER TABLE historical_logs ADD COLUMN is_calibrated BOOLEAN DEFAULT 0")
        if "calibrated_at" not in cols:
            alters.append("ALTER TABLE historical_logs ADD COLUMN calibrated_at DATETIME")
        if "carbon_saved_kg" not in cols:
            alters.append("ALTER TABLE historical_logs ADD COLUMN carbon_saved_kg FLOAT DEFAULT 0")
        for sql in alters:
            db.session.execute(db.text(sql))
        if alters:
            db.session.commit()
    except Exception:
        db.session.rollback()
    try:
        sys_cols = {r[1] for r in db.session.execute(db.text("PRAGMA table_info(solar_systems)")).fetchall()}
        if "system_active_from" not in sys_cols:
            db.session.execute(db.text("ALTER TABLE solar_systems ADD COLUMN system_active_from VARCHAR(20)"))
            db.session.commit()
    except Exception:
        db.session.rollback()
    # Admin account from .env only (ADMIN_EMAIL / ADMIN_PASSWORD / ADMIN_NAME)
    try:
        if not ADMIN_PASSWORD:
            print("WARNING: ADMIN_PASSWORD not set in .env — admin account not created/updated.")
        else:
            admin = User.query.filter_by(is_admin=True).first()
            by_email = User.query.filter_by(email=ADMIN_EMAIL).first()
            if admin is None and by_email is None:
                db.session.add(User(
                    email=ADMIN_EMAIL,
                    password_hash=generate_password_hash(ADMIN_PASSWORD),
                    name=ADMIN_NAME,
                    is_admin=True,
                ))
                db.session.commit()
                print(f"Admin created from .env ({ADMIN_EMAIL})")
            elif ADMIN_SYNC:
                # Rotate credentials from .env without editing code
                target = admin or by_email
                if target:
                    # Ensure single admin identity
                    target.email = ADMIN_EMAIL
                    target.password_hash = generate_password_hash(ADMIN_PASSWORD)
                    target.name = ADMIN_NAME or target.name
                    target.is_admin = True
                    # Demote any other admin rows (safety)
                    for other in User.query.filter(User.is_admin.is_(True), User.user_id != target.user_id).all():
                        other.is_admin = False
                    db.session.commit()
                    print(f"Admin synced from .env ({ADMIN_EMAIL})")
    except Exception as e:
        db.session.rollback()
        print("Admin seed error:", e)


# ---------------------------------------------------------------
# Auth helpers — signed token + session binding + DB role truth
# ---------------------------------------------------------------
AUTH_TOKEN_MAX_AGE = int(os.getenv("AUTH_TOKEN_MAX_AGE", str(60 * 60 * 24 * 14)))  # 14 days


def _auth_serializer():
    return URLSafeTimedSerializer(app.secret_key, salt="solarflow-up-auth-v1")


def _role_for_user(u) -> str:
    return "admin" if (u and getattr(u, "is_admin", False)) else "user"


def issue_auth_session(u) -> str:
    """Create signed token and bind it to server session. Clears prior auth keys."""
    role = _role_for_user(u)
    token = _auth_serializer().dumps({"uid": u.user_id, "role": role})
    # Drop any legacy loose keys
    for k in list(session.keys()):
        if k.startswith("user_token_") or k.startswith("admin_token_"):
            session.pop(k, None)
    session.clear()  # prevent role mixing in same browser session
    session["auth_token"] = token
    session["user_id"] = u.user_id
    session["role"] = role
    session.permanent = True
    return token


def _extract_request_token():
    token = request.args.get("token") or request.form.get("token")
    if not token and request.method in ("POST", "PUT", "PATCH"):
        body = request.get_json(silent=True) or {}
        token = body.get("token")
    return token


def current_user_from_token(token=None):
    """
    Strict auth:
    1) Token must be present and cryptographically signed
    2) Token must match server session binding
    3) Session user_id / role must match token payload
    4) DB is_admin is source of truth (must match role)
    No fallback to 'any token in session'.
    """
    token = token or _extract_request_token()
    if not token:
        return None

    # Session binding — token alone is not enough
    if session.get("auth_token") != token:
        return None

    try:
        data = _auth_serializer().loads(token, max_age=AUTH_TOKEN_MAX_AGE)
    except SignatureExpired:
        return None
    except BadSignature:
        return None
    except Exception:
        return None

    uid = data.get("uid")
    role = data.get("role")
    if not uid or role not in ("admin", "user"):
        return None
    if session.get("user_id") != uid or session.get("role") != role:
        return None

    u = db.session.get(User, uid)
    if not u:
        return None

    # DB role wins — prevents privilege escalation if session/token was tampered
    expected = _role_for_user(u)
    if role != expected:
        return None
    if session.get("role") != expected:
        return None
    return u


def require_login(role=None):
    """
    role=None  -> any authenticated user
    role='admin' -> admin only
    role='user'  -> non-admin only (admins redirected to /admin for HTML)
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            token = _extract_request_token()
            u = current_user_from_token(token)
            wants_json = (
                request.path.startswith("/api/")
                or request.accept_mimetypes.best == "application/json"
                or request.is_json
            )
            if not u:
                if wants_json:
                    return jsonify({"error": "Unauthorized", "code": "auth_required"}), 401
                return redirect(url_for("login"))

            if role == "admin":
                if not u.is_admin:
                    if wants_json:
                        return jsonify({"error": "Admin only", "code": "admin_required"}), 403
                    return redirect(url_for("index", token=token) if token else url_for("login"))
            elif role == "user":
                if u.is_admin:
                    if wants_json:
                        return jsonify({"error": "User portal only", "code": "user_only"}), 403
                    return redirect(url_for("admin_portal", token=token) if token else url_for("login"))

            return fn(*args, **kwargs)
        return wrapper
    return decorator


def api_get_json(url, params=None, timeout=20):
    r = requests.get(url, params=params, timeout=timeout, headers=OSM_HEADERS)
    r.raise_for_status()
    return r.json()


def safe_int(value, default=0):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return int(round(float(value)))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def normalize_shading_selection(raw) -> list:
    """
    Accept list or comma string.
    Enforce ONLY ONE option per period (morning / midday / evening).
    If multiple for same period, keep the heaviest.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    else:
        items = []

    rank = {"light": 1, "medium": 2, "heavy": 3}
    best = {}  # period -> (rank, full_key)

    for key in items:
        k = key.lower()
        if k.startswith("morning_"):
            period, level = "morning", k.replace("morning_", "")
        elif k.startswith("midday_"):
            period, level = "midday", k.replace("midday_", "")
        elif k.startswith("evening_"):
            period, level = "evening", k.replace("evening_", "")
        else:
            continue
        if level not in rank:
            continue
        r = rank[level]
        if period not in best or r > best[period][0]:
            best[period] = (r, f"{period}_{level}")

    return [best[p][1] for p in ("morning", "midday", "evening") if p in best]


def adaptive_blend_weights(frame, model_used: bool):
    """Whole-frame average (kept for KPI). Prefer adaptive_blend_weights_day for cards."""
    if not model_used:
        return 1.0, 0.0
    cloud = float(np.nanmean(frame["cloud_cover"].to_numpy())) if "cloud_cover" in frame else 40.0
    rain = float(np.nanmean(frame["precipitation_probability"].to_numpy())) if "precipitation_probability" in frame else 20.0
    return _blend_from_cloud_rain(cloud, rain)


def _blend_from_cloud_rain(cloud: float, rain: float):
    """
    Physics-heavy on clear days (irradiance models more reliable).
    Slightly more ML when cloudy/rainy (patterns help), still physics-majority.
    """
    w_ml = 0.20
    if cloud >= 75 or rain >= 65:
        w_ml = 0.38
    elif cloud >= 55 or rain >= 45:
        w_ml = 0.30
    elif cloud <= 25 and rain <= 15:
        w_ml = 0.10
    w_ml = float(np.clip(w_ml, 0.08, 0.42))
    return 1.0 - w_ml, w_ml


# Candidate physics/ML mixes evaluated per day; best confidence wins
_BLEND_CANDIDATES = (
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
    (0.62, 0.38),
    (0.55, 0.45),
    (0.50, 0.50),
)


def _day_cloud_rain(day_frame):
    cloud = float(np.nanmean(day_frame["cloud_cover"].to_numpy())) if "cloud_cover" in day_frame else 40.0
    rain = (
        float(np.nanmean(day_frame["precipitation_probability"].to_numpy()))
        if "precipitation_probability" in day_frame
        else 20.0
    )
    return cloud, rain


def _score_blend_candidate(phys, ml, w_p, w_m, cloud, rain, prior_wm):
    """
    Confidence-style score for one blend on one day.
    Higher = tighter relative uncertainty + closer to weather prior.
    """
    phys = np.asarray(phys, dtype=float)
    ml = np.asarray(ml, dtype=float)
    pred = np.maximum(0.0, w_p * phys + w_m * ml)
    day = pred > 0.02
    if not np.any(day):
        return 55.0, pred

    method_half = 0.35 * np.abs(phys - ml)
    weather_frac = 0.03 + 0.06 * (cloud / 100.0) + 0.04 * (rain / 100.0)
    train_frac = 0.08
    half = np.sqrt(method_half ** 2 + (pred * weather_frac) ** 2 + (pred * train_frac) ** 2)
    half = np.minimum(half, pred * 0.18)
    width_rel = float(np.mean((2.0 * half[day]) / np.maximum(pred[day], 1e-3)))
    conf = 96.0 - 55.0 * width_rel

    # Light prior — allow other blends when physics and ML disagree
    conf -= 8.0 * abs(float(w_m) - float(prior_wm))

    # Prefer totals that sit between pure physics and pure ML day totals
    tp = float(np.sum(phys[day]))
    tm = float(np.sum(ml[day]))
    tpred = float(np.sum(pred[day]))
    if tp > 0.2 and tm > 0.2:
        lo, hi = (tp, tm) if tp <= tm else (tm, tp)
        if tpred < lo - 0.05 or tpred > hi + 0.05:
            conf -= 6.0
        gap = abs(tp - tm) / max(tp, tm, 1e-3)
        if gap > 0.08:
            mid = 0.5 * (tp + tm)
            conf += 4.0 * (1.0 - min(1.0, abs(tpred - mid) / max(abs(tp - tm), 1e-3)))

    if cloud <= 25 and rain <= 15 and w_m <= 0.22:
        conf += 1.0
    if (cloud >= 70 or rain >= 55) and w_m >= 0.28:
        conf += 1.0

    return float(np.clip(conf, 50.0, 96.0)), pred


def select_best_blend_day(day_frame, phys_slice, ml_slice, model_used: bool):
    """
    Try several physics/ML blends for this day.
    Always returns 4-tuple: (w_phys, w_ml, pred_series, confidence_pct).
    """
    phys = np.asarray(phys_slice, dtype=float)
    if not model_used:
        pred0 = np.maximum(0.0, phys)
        return (1.0, 0.0, pred0, 78.0)

    ml = np.asarray(ml_slice, dtype=float)
    cloud, rain = _day_cloud_rain(day_frame)
    prior_wp, prior_wm = _blend_from_cloud_rain(cloud, rain)
    prior_pred = np.maximum(0.0, prior_wp * phys + prior_wm * ml)

    candidates = list(_BLEND_CANDIDATES)
    if not any(abs(c[1] - prior_wm) < 1e-6 for c in candidates):
        candidates.append((prior_wp, prior_wm))

    best_wp, best_wm, best_pred, best_conf = prior_wp, prior_wm, prior_pred, 70.0
    best_score = -1e9
    for w_p, w_m in candidates:
        conf, pred = _score_blend_candidate(phys, ml, w_p, w_m, cloud, rain, prior_wm)
        score = conf + (0.15 if abs(w_m - prior_wm) < 0.02 else 0.0)
        if score > best_score:
            best_score = score
            best_wp, best_wm, best_pred, best_conf = float(w_p), float(w_m), pred, float(conf)
    return (best_wp, best_wm, best_pred, best_conf)


def _apply_best_blend(day_frame, phys_slice, ml_slice, model_used: bool):
    """Safe wrapper — always yields (w_p, w_m, pred, conf)."""
    res = select_best_blend_day(day_frame, phys_slice, ml_slice, model_used)
    if isinstance(res, (list, tuple)):
        if len(res) >= 4:
            return res[0], res[1], res[2], float(res[3])
        if len(res) == 3:
            return res[0], res[1], res[2], 78.0
    phys = np.asarray(phys_slice, dtype=float)
    return 1.0, 0.0, np.maximum(0.0, phys), 78.0


def adaptive_blend_weights_day(day_frame, model_used: bool):
    """Weather prior only (used when slices are not available)."""
    if not model_used:
        return 1.0, 0.0
    cloud, rain = _day_cloud_rain(day_frame)
    return _blend_from_cloud_rain(cloud, rain)


# Server forecast cache (15 min) — same idea as production SolarFlow
_FORECAST_CACHE: dict = {}
_FORECAST_CACHE_TTL = 900


def _forecast_cache_key(payload: dict) -> str:
    import hashlib
    import json
    keep = {k: payload.get(k) for k in (
        "state", "district", "city_locality", "pincode", "lat", "lon",
        "brand_model", "inverter_model", "num_panels", "num_inverters",
        "installation_year", "manual_clean_date", "tilt", "azimuth",
        "shading_selection", "albedo", "mounting_type", "bifacial_gain_pct",
        "wiring_loss_pct", "mismatch_loss_pct", "availability_pct",
        "degradation_pct_per_year",
    )}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def _cached_forecast(key: str):
    hit = _FORECAST_CACHE.get(key)
    if hit and (time.time() - hit["ts"]) < _FORECAST_CACHE_TTL:
        return hit["data"]
    return None


def _store_forecast_cache(key: str, data: dict) -> None:
    _FORECAST_CACHE[key] = {"ts": time.time(), "data": data}
    if len(_FORECAST_CACHE) > 256:
        cutoff = time.time() - _FORECAST_CACHE_TTL
        for k in [k for k, v in list(_FORECAST_CACHE.items()) if v["ts"] < cutoff]:
            _FORECAST_CACHE.pop(k, None)


def fetch_forecast_weather(lat: float, lon: float) -> dict:
    """Open-Meteo with model fallbacks (ecmwf → gfs → best_match)."""
    hourly = (
        "temperature_2m,relativehumidity_2m,relative_humidity_2m,"
        "apparent_temperature,dew_point_2m,"
        "precipitation_probability,precipitation,rain,showers,cloud_cover,visibility,"
        "windspeed_10m,wind_gusts_10m,windgusts_10m,surface_pressure,uv_index,sunshine_duration,"
        "shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,"
        "weathercode,weather_code,is_day,vapour_pressure_deficit"
    )
    base = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&forecast_days=7&timezone=Asia%2FKolkata&hourly={hourly}"
    )
    last_err = None
    for model in ("ecmwf_ifs025", "gfs_seamless", "best_match"):
        try:
            data = api_get_json(base + f"&models={model}", timeout=30)
            if data.get("hourly") and data["hourly"].get("time"):
                data["_model_used"] = model
                return data
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Open-Meteo weather failed: {last_err}")


def _training_rel_error(package) -> float:
    """Relative residual from training MAE/R² when present (real metric, not random)."""
    try:
        pkg = package or {}
        m = pkg.get("metrics") or pkg.get("model_metrics") or {}
        if not m and isinstance(pkg.get("package"), dict):
            m = (pkg.get("package") or {}).get("metrics") or {}
        mae = m.get("mae_5kw_equivalent", m.get("mae"))
        r2 = m.get("r2")
        if mae is not None:
            return float(np.clip(float(mae) / 5.0, 0.04, 0.14))
        if r2 is not None:
            return float(np.clip(0.45 * (1.0 - float(r2)), 0.04, 0.14))
    except Exception:
        pass
    return 0.08


def uncertainty_bands(pred, frame, physics, ml_scaled, model_used: bool, package=None):
    """
    Hourly p10/p90 from real signals (RSS), not a fixed random ±%:

    1) |physics − ML| / 2  — method disagreement for that hour
    2) Weather irradiance uncertainty (~5% clear → ~15% heavy cloud/rain)
    3) Training residual from model MAE / R²
    """
    pred = np.asarray(pred, dtype=float)
    n = len(pred)
    phys = np.asarray(physics, dtype=float) if physics is not None else np.zeros(n)
    cloud = frame["cloud_cover"].to_numpy(dtype=float) if "cloud_cover" in frame.columns else np.full(n, 40.0)
    rain = (
        frame["precipitation_probability"].to_numpy(dtype=float)
        if "precipitation_probability" in frame.columns
        else np.full(n, 20.0)
    )

    if model_used and ml_scaled is not None and len(ml_scaled) == n:
        # Partial credit for method gap (not full half-span — avoids huge bands)
        method_half = 0.35 * np.abs(phys - np.asarray(ml_scaled, dtype=float))
    else:
        method_half = np.zeros(n)

    weather_frac = (
        0.03
        + 0.06 * np.clip(cloud / 100.0, 0, 1)
        + 0.04 * np.clip(rain / 100.0, 0, 1)
    )
    weather_half = pred * weather_frac
    train_half = pred * (_training_rel_error(package) if model_used else 0.08)

    half = np.sqrt(method_half ** 2 + weather_half ** 2 + train_half ** 2)
    half = np.minimum(half, pred * 0.18)
    half = np.where(pred > 0.02, np.maximum(half, 0.01), 0.0)

    p10 = np.maximum(0.0, pred - half)
    p90 = pred + half
    return p10, p90


def daily_range_from_day(g, total_kwh: float, model_used: bool, package=None) -> tuple:
    """
    Daily range from that day's real physics/ML totals + weather + training error.
    Does not sum hourly extremes and does not use a fixed random ±%.
    """
    total = max(0.0, float(total_kwh))
    if total <= 0:
        return 0.0, 0.0

    cloud = float(g["cloud_cover"].mean()) if "cloud_cover" in g else 40.0
    rain = (
        float(g["precipitation_probability"].mean())
        if "precipitation_probability" in g
        else 20.0
    )

    method_half = 0.0
    if model_used and "physics_baseline_kw" in g.columns:
        phys_day = float(g["physics_baseline_kw"].sum())
        if "ml_scaled_kw" in g.columns:
            method_half = 0.35 * abs(phys_day - float(g["ml_scaled_kw"].sum()))
        else:
            method_half = 0.35 * abs(phys_day - total)

    weather_half = total * (0.03 + 0.05 * (cloud / 100.0) + 0.04 * (rain / 100.0))
    train_half = total * (_training_rel_error(package) if model_used else 0.08)
    half = float(np.sqrt(method_half ** 2 + weather_half ** 2 + train_half ** 2))
    # Keep daily band honest and tight: about ±3% to ±12%
    half = min(max(half, total * 0.03), total * 0.12)
    return round(max(0.0, total - half), 2), round(total + half, 2)


def confidence_from_bands(p10, p50, p90, frame, model_used, years_old, avg_soiling_loss):
    p10 = np.asarray(p10, dtype=float)
    p50 = np.asarray(p50, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    day = p50 > 0.05
    if not np.any(day):
        return 88
    # Full width / p50 (with tighter bands this stays ~0.12–0.35)
    width_rel = float(np.mean((p90[day] - p10[day]) / np.maximum(p50[day], 1e-3)))
    conf = 96.0 - 55.0 * width_rel
    if not model_used:
        conf -= 5.0
    cloud = float(np.nanmean(frame["cloud_cover"])) if "cloud_cover" in frame else 40.0
    rain = (
        float(np.nanmean(frame["precipitation_probability"]))
        if "precipitation_probability" in frame
        else 20.0
    )
    if cloud <= 25 and rain <= 15:
        conf += 2.0
    elif cloud >= 70 or rain >= 60:
        conf -= 5.0
    elif cloud >= 50 or rain >= 40:
        conf -= 2.0
    if avg_soiling_loss > 12:
        conf -= 4.0
    elif avg_soiling_loss > 7:
        conf -= 2.0
    if years_old >= 8:
        conf -= 2.0
    elif years_old <= 2:
        conf += 1.0
    return int(round(float(np.clip(conf, 62, 96))))


def compute_confidence(frame, model_used, years_old, avg_soiling_loss, exact_location=True):
    conf = 80.0
    if model_used:
        conf += 6.0
    if exact_location:
        conf += 3.0
    cloud = float(np.nanmean(frame["cloud_cover"])) if "cloud_cover" in frame else 40.0
    rain = float(np.nanmean(frame["precipitation_probability"])) if "precipitation_probability" in frame else 20.0
    if cloud <= 25 and rain <= 15:
        conf += 7.0
    elif cloud >= 70 or rain >= 60:
        conf -= 8.0
    elif cloud >= 50 or rain >= 40:
        conf -= 4.0
    if avg_soiling_loss > 12:
        conf -= 5.0
    elif avg_soiling_loss > 7:
        conf -= 2.0
    if years_old <= 2:
        conf += 2.0
    elif years_old >= 8:
        conf -= 2.0
    return int(round(float(np.clip(conf, 62, 94))))


# ---------------------------------------------------------------
# Location APIs (Postal free + Google optional)
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# PINCODE: Postal ONLY (state + district)
# ---------------------------------------------------------------
import time

def _postal_get_json(url, params=None, timeout=12, retries=3):
    last_err = None
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SolarFlow/4.2",
        "Accept": "application/json",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1))
    raise last_err



@app.route("/api/up_districts")
def api_up_districts():
    """List UP districts for UI filters (product is UP-only)."""
    return jsonify({
        "state": UP_STATE_NAME,
        "districts": district_list_for_ui(),
        "scope": "uttar_pradesh_only",
    })


@app.route("/api/verify_pincode")
def verify_pincode():
    pin = (request.args.get("pin") or "").strip()
    if len(pin) != 6 or not pin.isdigit():
        return jsonify({"ok": False, "error": "Enter a valid 6-digit pincode"}), 400
    # Soft pre-filter: most UP pins are 20xxxx–28xxxx (Postal remains authoritative)
    if not pincode_looks_like_up(pin):
        # Still allow API lookup for edge cases, but warn; hard reject happens on state
        pass

    # --- 1) Primary: Postal India ---
    try:
        data = _postal_get_json(f"https://api.postalpincode.in/pincode/{pin}", timeout=12, retries=3)
        if data and data[0].get("Status") == "Success":
            offices = data[0].get("PostOffice") or []
            if offices:
                first = offices[0]
                post_offices, seen = [], set()
                for po in offices:
                    name = (po.get("Name") or "").strip()
                    if not name or name.lower() in seen:
                        continue
                    seen.add(name.lower())
                    post_offices.append({
                        "name": name,
                        "display": f"{name}, {po.get('District', '')}, {po.get('State', '')}",
                    })
                state_raw = first.get("State") or ""
                district_raw = first.get("District") or ""
                if not is_up_state(state_raw):
                    return jsonify({
                        "ok": False,
                        "error": reject_non_up_message(state_raw, pin),
                        "state": state_raw,
                        "district": district_raw,
                        "scope": "uttar_pradesh_only",
                    }), 403
                return jsonify({
                    "ok": True,
                    "pincode": pin,
                    "state": UP_STATE_NAME,
                    "district": normalize_district(district_raw) or district_raw,
                    "post_offices": post_offices,
                    "source": "Postal",
                    "scope": "uttar_pradesh_only",
                })
    except Exception as e:
        postal_error = str(e)
    else:
        postal_error = "Postal returned no data"

    # --- 2) Fallback: Google Geocoding of the pincode ---
    if GOOGLE_MAPS_API_KEY:
        try:
            d = api_get_json(
                "https://maps.googleapis.com/maps/api/geocode/json",
                {
                    "address": f"{pin}, India",
                    "components": "country:IN",
                    "key": GOOGLE_MAPS_API_KEY,
                },
                timeout=10,
            )
            results = d.get("results") or []
            if results:
                comps = results[0].get("address_components") or []
                state = district = ""
                for c in comps:
                    types = c.get("types") or []
                    if "administrative_area_level_1" in types:
                        state = c.get("long_name") or ""
                    if "administrative_area_level_2" in types or "administrative_area_level_3" in types:
                        if not district:
                            district = c.get("long_name") or ""
                    if "locality" in types and not district:
                        district = c.get("long_name") or ""

                if not is_up_state(state):
                    return jsonify({
                        "ok": False,
                        "error": reject_non_up_message(state, pin),
                        "state": state or "Unknown",
                        "district": district or "Unknown",
                        "scope": "uttar_pradesh_only",
                    }), 403
                return jsonify({
                    "ok": True,
                    "pincode": pin,
                    "state": UP_STATE_NAME,
                    "district": normalize_district(district) or district or "Unknown",
                    "post_offices": [],
                    "source": "Google",
                    "warning": f"Postal unavailable ({postal_error}); used Google fallback",
                    "scope": "uttar_pradesh_only",
                })
        except Exception as ge:
            return jsonify({
                "ok": False,
                "error": f"Postal failed ({postal_error}); Google also failed ({ge})",
            }), 502

    return jsonify({
        "ok": False,
        "error": f"Postal lookup failed: {postal_error}. Add GOOGLE_MAPS_API_KEY for fallback.",
    }), 502

# ---------------------------------------------------------------
# GOOGLE: bounds for map (restrict to pincode area)
# ---------------------------------------------------------------
@app.route("/api/pincode_bounds")
def pincode_bounds():
    """
    Return a usable map window for a UP pincode.
    Google viewport boxes are often too tight/square and cut localities —
    we expand them, enforce a minimum size, and clip to UP envelope.
    """
    pin = (request.args.get("pin") or "").strip()
    if len(pin) != 6 or not pin.isdigit():
        return jsonify({"error": "invalid pin"}), 400

    def expand_box(south, west, north, east, lat, lon, pad_ratio=0.35, min_delta=0.06):
        """Pad viewport so places near the edge are not cut off; enforce min size."""
        h = max(abs(north - south), 1e-6)
        w = max(abs(east - west), 1e-6)
        # Expand by ratio
        south -= h * pad_ratio
        north += h * pad_ratio
        west -= w * pad_ratio
        east += w * pad_ratio
        # Minimum ~6–7 km half-span so small pins still usable
        if (north - south) < min_delta * 2:
            south, north = lat - min_delta, lat + min_delta
        if (east - west) < min_delta * 2:
            west, east = lon - min_delta, lon + min_delta
        # Clip lightly to UP bbox so map never opens on another state
        south = max(south, UP_BBOX["south"] - 0.05)
        north = min(north, UP_BBOX["north"] + 0.05)
        west = max(west, UP_BBOX["west"] - 0.05)
        east = min(east, UP_BBOX["east"] + 0.05)
        return {
            "south": south,
            "west": west,
            "north": north,
            "east": east,
            "center_lat": lat,
            "center_lon": lon,
        }

    def box_from_point(lat, lon, delta=0.06):
        return expand_box(lat - delta, lon - delta, lat + delta, lon + delta, lat, lon, pad_ratio=0.0)

    def finalize(out, source):
        lat, lon = out["center_lat"], out["center_lon"]
        if not is_inside_up(lat, lon):
            return None
        out["source"] = source
        out["scope"] = "uttar_pradesh_only"
        out["up_bbox"] = UP_BBOX
        return out

    # ---------- 1) Google Geocoding ----------
    if GOOGLE_MAPS_API_KEY:
        queries = [
            {"address": pin, "components": f"postal_code:{pin}|country:IN"},
            {"address": f"{pin}, Uttar Pradesh, India", "components": "country:IN"},
            {"address": f"{pin}, India", "components": "country:IN"},
            {"address": f"pincode {pin}, Uttar Pradesh, India", "components": "country:IN"},
        ]
        for q in queries:
            try:
                params = {"key": GOOGLE_MAPS_API_KEY, **q}
                d = api_get_json(
                    "https://maps.googleapis.com/maps/api/geocode/json",
                    params,
                    timeout=10,
                )
                results = d.get("results") or []
                if d.get("status") != "OK" or not results:
                    continue
                geo = results[0].get("geometry") or {}
                loc = geo.get("location") or {}
                if loc.get("lat") is None or loc.get("lng") is None:
                    continue
                lat = float(loc["lat"])
                lon = float(loc["lng"])
                if not is_inside_up(lat, lon):
                    continue
                vb = geo.get("viewport") or {}
                ne = vb.get("northeast") or {}
                sw = vb.get("southwest") or {}
                if ne.get("lat") is not None and sw.get("lat") is not None:
                    out = expand_box(
                        float(sw["lat"]), float(sw["lng"]),
                        float(ne["lat"]), float(ne["lng"]),
                        lat, lon,
                    )
                else:
                    out = box_from_point(lat, lon)
                done = finalize(out, "Google")
                if done:
                    return jsonify(done)
            except Exception:
                continue

    # ---------- 2) OSM Nominatim fallback (broader box) ----------
    try:
        d = api_get_json(
            "https://nominatim.openstreetmap.org/search",
            {
                "postalcode": pin,
                "country": "India",
                "format": "json",
                "limit": 5,
                "addressdetails": 1,
            },
            timeout=12,
        )
        if d:
            lats, lons = [], []
            for row in d:
                try:
                    la, lo = float(row["lat"]), float(row["lon"])
                except Exception:
                    continue
                if is_inside_up(la, lo):
                    lats.append(la)
                    lons.append(lo)
            if lats:
                lat = sum(lats) / len(lats)
                lon = sum(lons) / len(lons)
                # Union of hits + pad
                out = expand_box(min(lats), min(lons), max(lats), max(lons), lat, lon, pad_ratio=0.5, min_delta=0.07)
                done = finalize(out, "OSM")
                if done:
                    return jsonify(done)
    except Exception:
        pass

    return jsonify({
        "error": "bounds not found or outside Uttar Pradesh",
        "hint": "Use a valid UP pincode. Check GOOGLE_MAPS_API_KEY / Geocoding API.",
        "scope": "uttar_pradesh_only",
    }), 404
    


# ---------------------------------------------------------------
# GOOGLE: locality autocomplete (search)
# ---------------------------------------------------------------
# Locality search cache + Nominatim throttle (from production SolarFlow)
_LOCALITY_CACHE: dict = {}
_LOCALITY_CACHE_TTL = 60 * 30  # 30 minutes
_LOCALITY_LAST_CALL: dict = {"t": 0.0}
_LOCALITY_MIN_GAP = 1.05


@app.route("/api/locality_autocomplete")
def locality_autocomplete():
    """
    Google Places first, then OpenStreetMap Nominatim fallback.
    Cached 30 min; Nominatim throttled to ~1 req/s.
    """
    query = (request.args.get("q") or "").strip()
    pin = (request.args.get("pin") or "").strip()
    district = (request.args.get("district") or "").strip()
    if len(query) < 3:
        return jsonify([])

    cache_key = f"{query.lower()}|{pin}|{district.lower()}"
    hit = _LOCALITY_CACHE.get(cache_key)
    if hit and (time.time() - hit["ts"]) < _LOCALITY_CACHE_TTL:
        return jsonify(hit["data"])

    out, seen = [], set()

    if GOOGLE_MAPS_API_KEY:
        try:
            input_text = f"{query}, {pin}, India" if pin else f"{query}, {district}, India"
            d = api_get_json(
                "https://maps.googleapis.com/maps/api/place/autocomplete/json",
                {
                    "input": input_text,
                    "types": "geocode",
                    "components": "country:in",
                    "key": GOOGLE_MAPS_API_KEY,
                },
                timeout=8,
            )
            api_status = d.get("status", "OK")
            if api_status in ("OK", "ZERO_RESULTS"):
                for pred in d.get("predictions", [])[:12]:
                    main = (pred.get("structured_formatting") or {}).get("main_text") \
                           or pred.get("description", "").split(",")[0]
                    key = (main or "").lower()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append({
                        "name": main,
                        "display": pred.get("description", ""),
                        "place_id": pred.get("place_id"),
                        "source": "Google",
                    })
            # REQUEST_DENIED / OVER_QUERY_LIMIT → fall through to OSM
        except Exception:
            pass

    if not out:
        wait = _LOCALITY_MIN_GAP - (time.time() - _LOCALITY_LAST_CALL["t"])
        if wait > 0:
            time.sleep(wait)
        _LOCALITY_LAST_CALL["t"] = time.time()
        try:
            q = f"{query}, {district}, {pin}, Uttar Pradesh, India" if district else f"{query}, {pin}, Uttar Pradesh, India"
            d = api_get_json(
                "https://nominatim.openstreetmap.org/search",
                {"q": q, "countrycodes": "in", "format": "json", "limit": 10, "addressdetails": 1},
                timeout=10,
            )
            for item in d or []:
                addr = item.get("address", {}) or {}
                # Prefer UP results
                st = (addr.get("state") or "").lower()
                if st and "uttar" not in st and st not in ("up", "u.p.", "u.p"):
                    continue
                name = (
                    addr.get("suburb")
                    or addr.get("neighbourhood")
                    or addr.get("residential")
                    or addr.get("village")
                    or addr.get("hamlet")
                    or addr.get("town")
                    or addr.get("city")
                    or (item.get("display_name") or "").split(",")[0]
                )
                if not name:
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "name": name,
                    "display": item.get("display_name", ""),
                    "place_id": None,
                    "lat": item.get("lat"),
                    "lon": item.get("lon"),
                    "source": "OSM",
                })
        except Exception:
            pass

    _LOCALITY_CACHE[cache_key] = {"ts": time.time(), "data": out}
    return jsonify(out)


@app.route("/api/geocode")
def geocode_location():
    address = request.args.get("address", "")
    pin = request.args.get("pin", "")
    district = request.args.get("district", "")
    place_id = request.args.get("place_id", "")

    if GOOGLE_MAPS_API_KEY and place_id:
        try:
            d = api_get_json(
                "https://maps.googleapis.com/maps/api/place/details/json",
                {
                    "place_id": place_id,
                    "fields": "geometry,name,formatted_address",
                    "key": GOOGLE_MAPS_API_KEY,
                },
                timeout=8,
            )
            res = d.get("result") or {}
            loc = (res.get("geometry") or {}).get("location") or {}
            if loc.get("lat") is not None:
                return jsonify({
                    "lat": loc["lat"],
                    "lon": loc["lng"],
                    "name": res.get("name") or "",
                    "display": res.get("formatted_address") or "",
                    "source": "Google",
                })
        except Exception:
            pass

    if GOOGLE_MAPS_API_KEY and address:
        try:
            d = api_get_json(
                "https://maps.googleapis.com/maps/api/geocode/json",
                {
                    "address": f"{address}, {district}, {pin}, India",
                    "components": "country:IN",
                    "key": GOOGLE_MAPS_API_KEY,
                },
                timeout=8,
            )
            results = d.get("results") or []
            if results:
                loc = results[0]["geometry"]["location"]
                return jsonify({
                    "lat": loc["lat"],
                    "lon": loc["lng"],
                    "name": results[0].get("formatted_address", "").split(",")[0],
                    "display": results[0].get("formatted_address", ""),
                    "source": "Google",
                })
        except Exception:
            pass

    # OSM fallback so locality still resolves without Google billing
    try:
        q = f"{address}, {district}, {pin}, Uttar Pradesh, India".strip(", ")
        d = api_get_json(
            "https://nominatim.openstreetmap.org/search",
            {"format": "json", "countrycodes": "in", "q": q, "limit": 1},
            timeout=10,
        )
        if d:
            return jsonify({
                "lat": d[0]["lat"],
                "lon": d[0]["lon"],
                "name": (d[0].get("display_name") or "").split(",")[0],
                "display": d[0].get("display_name") or "",
                "source": "OSM",
            })
    except Exception:
        pass
    return jsonify({"error": "Failed to resolve coordinates."}), 404


# ---------------------------------------------------------------
# GOOGLE first: reverse geocode (exact name for text field)
# ---------------------------------------------------------------
@app.route("/api/reverse_geocode")
def reverse_geocode():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    if not lat or not lon:
        return jsonify({"error": "Missing coordinates"}), 400
    try:
        lat_f, lon_f = float(lat), float(lon)
    except ValueError:
        return jsonify({"error": "Invalid coordinates"}), 400

    # Hard UP gate — closes loophole where any-state coords could be reverse-geocoded
    if not is_inside_up(lat_f, lon_f):
        return jsonify({
            "error": reject_outside_up_coords_message(),
            "scope": "uttar_pradesh_only",
            "inside_up": False,
        }), 403

    if GOOGLE_MAPS_API_KEY:
        try:
            d = api_get_json(
                "https://maps.googleapis.com/maps/api/geocode/json",
                {"latlng": f"{lat_f},{lon_f}", "key": GOOGLE_MAPS_API_KEY},
                timeout=8,
            )
            results = d.get("results") or []
            if results:
                top = results[0]
                # Prefer administrative_area_level_1 check when present
                state_name = ""
                for c in top.get("address_components") or []:
                    if "administrative_area_level_1" in (c.get("types") or []):
                        state_name = c.get("long_name") or ""
                        break
                if state_name and not is_up_state(state_name):
                    return jsonify({
                        "error": reject_non_up_message(state_name),
                        "state": state_name,
                        "scope": "uttar_pradesh_only",
                        "inside_up": False,
                    }), 403
                locality = ""
                for c in top.get("address_components") or []:
                    types = c.get("types") or []
                    if any(t in types for t in (
                        "sublocality_level_1", "sublocality", "neighborhood",
                        "premise", "street_number"
                    )):
                        locality = c.get("long_name")
                        break
                if not locality:
                    for c in top.get("address_components") or []:
                        types = c.get("types") or []
                        if "locality" in types or "administrative_area_level_3" in types:
                            locality = c.get("long_name")
                            break
                if not locality:
                    locality = (top.get("formatted_address") or "").split(",")[0]
                return jsonify({
                    "locality": locality,
                    "display": top.get("formatted_address") or "",
                    "source": "Google",
                    "inside_up": True,
                    "scope": "uttar_pradesh_only",
                })
        except Exception:
            pass

    # OSM fallback when Google fails / no key
    try:
        d = api_get_json(
            "https://nominatim.openstreetmap.org/reverse",
            {"format": "json", "lat": lat_f, "lon": lon_f, "zoom": 18, "addressdetails": 1},
            timeout=10,
        )
        addr = d.get("address", {}) or {}
        locality = (
            addr.get("suburb")
            or addr.get("neighbourhood")
            or addr.get("residential")
            or addr.get("village")
            or addr.get("town")
            or addr.get("city")
            or (d.get("display_name") or "").split(",")[0]
        )
        return jsonify({
            "locality": locality or "Selected Location",
            "display": d.get("display_name", ""),
            "source": "OSM",
            "inside_up": True,
            "scope": "uttar_pradesh_only",
        })
    except Exception:
        pass

    return jsonify({
        "locality": "Selected Location",
        "display": "",
        "source": "none",
        "inside_up": True,
        "scope": "uttar_pradesh_only",
    })


@app.route("/api/validate_location")
def validate_location():
    """Client GPS / map pick: must be inside UP (and optionally report pin box separately)."""
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid coordinates"}), 400
    if not is_inside_up(lat, lon):
        return jsonify({
            "ok": False,
            "inside_up": False,
            "error": reject_outside_up_coords_message(),
            "scope": "uttar_pradesh_only",
        }), 403
    return jsonify({
        "ok": True,
        "inside_up": True,
        "lat": lat,
        "lon": lon,
        "scope": "uttar_pradesh_only",
    })

@app.route("/api/maps_config")
def maps_config():
    return jsonify({
        "google_key": GOOGLE_MAPS_API_KEY or "",
        "provider": "google" if GOOGLE_MAPS_API_KEY else "leaflet",
    })


# ---------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------
@app.route("/api/search_hardware")
def search_hardware():
    """Full panel/inverter list for styled <select> (optional filter by q)."""
    query = request.args.get("q", "").lower().strip()
    typ = request.args.get("type", "panel")

    try:
        if typ == "panel":
            m = INDIAN_CUSTOM_PANELS
            if query:
                m = m[m.index.astype(str).str.lower().str.contains(query, na=False)]
            return jsonify([
                {
                    "id": str(n),
                    "label": f"{n} ({float(r.get('STC_W', 0)):.0f}W)",
                    "power": float(r.get("STC_W", 0)),
                }
                for n, r in m.iterrows()
            ])

        m = INDIAN_CUSTOM_INVERTERS
        if query:
            m = m[m.index.astype(str).str.lower().str.contains(query, na=False)]
        return jsonify([
            {
                "id": str(n),
                "label": f"{n} ({float(r.get('Paco', 0)):.0f}W AC)",
                "power": float(r.get("Paco", 0)),
            }
            for n, r in m.iterrows()
        ])

    except Exception as e:
        print("Hardware search error:", e)
        return jsonify([])
# ---------------------------------------------------------------
# Auth
# ---------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = User.query.filter_by(email=request.form.get("email")).first()
        if u and check_password_hash(u.password_hash, request.form.get("password")):
            t = issue_auth_session(u)
            if u.is_admin:
                return redirect(url_for("admin_portal", token=t))
            return redirect(url_for("index", token=t))
        return render_template("login.html", error="Invalid email or password")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        if User.query.filter_by(email=email).first():
            return render_template("register.html", error="Email already registered")
        u = User(
            email=email,
            password_hash=generate_password_hash(request.form.get("password", "")),
            name=request.form.get("name", ""),
            is_admin=False,  # never allow self-register as admin
        )
        db.session.add(u)
        db.session.commit()
        db.session.add(SolarSystem(user_id=u.user_id))
        db.session.commit()
        t = issue_auth_session(u)
        return redirect(url_for("index", token=t))
    return render_template("register.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """
    Pilot reset: verify registered email + full name, then set a new password.
    (No email SMTP yet — name match is the identity check.)
    """
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        name = (request.form.get("name") or "").strip()
        pw = request.form.get("password") or ""
        pw2 = request.form.get("password2") or ""
        u = User.query.filter_by(email=email).first()
        if not u:
            return render_template(
                "forgot_password.html",
                error="No account found with that email.",
            )
        if u.is_admin:
            return render_template(
                "forgot_password.html",
                error="Admin password cannot be reset here. Contact the platform owner.",
            )
        # Case-insensitive name check
        if (u.name or "").strip().lower() != name.lower() or not name:
            return render_template(
                "forgot_password.html",
                error="Name does not match our records for this email.",
            )
        if len(pw) < 6:
            return render_template(
                "forgot_password.html",
                error="Password must be at least 6 characters.",
            )
        if pw != pw2:
            return render_template(
                "forgot_password.html",
                error="Passwords do not match.",
            )
        u.password_hash = generate_password_hash(pw)
        db.session.commit()
        return render_template(
            "forgot_password.html",
            success="Password updated. You can sign in with your new password.",
        )
    return render_template("forgot_password.html")


@app.route("/logout")
def logout():
    # Full session wipe — role query param cannot leave residual access
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@require_login(role="user")
def index():
    token = _extract_request_token()
    u = current_user_from_token(token)
    return render_template(
        "index.html",
        today_date=datetime.now(IST_TZ).strftime("%Y-%m-%d"),
        user=u,
        token=token,
    )


# ---------------------------------------------------------------
# History backfill — days user was offline still get archive rows
# ---------------------------------------------------------------
_ARCHIVE_HOURLY = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
    "precipitation,rain,showers,cloud_cover,surface_pressure,"
    "wind_speed_10m,wind_gusts_10m,shortwave_radiation,direct_radiation,"
    "diffuse_radiation,direct_normal_irradiance,weather_code,is_day"
)


def _system_inputs_from_db(s: "SolarSystem"):
    """Build SystemInputs from saved SolarSystem; None if incomplete."""
    if not s or s.latitude is None or s.longitude is None:
        return None
    if not s.brand_model or not s.num_panels:
        return None
    try:
        stc, gamma, _eff = get_panel_specs(s.brand_model)
        paco, inv_eff = get_inverter_specs(s.inverter_model or "")
    except Exception:
        return None
    try:
        shade = normalize_shading_selection(getattr(s, "shading_selection", None) or [])
    except Exception:
        shade = []
    year = int(s.installation_year or datetime.now().year)
    return SystemInputs(
        latitude=float(s.latitude),
        longitude=float(s.longitude),
        num_panels=int(s.num_panels or 1),
        num_inverters=int(s.num_inverters or 1),
        panel_stc_w=float(stc),
        panel_gamma=float(gamma),
        inverter_paco_w=float(paco),
        inverter_eff=float(inv_eff),
        installation_year=year,
        manual_clean_date=getattr(s, "last_cleaned_date", None) or None,
        tilt=float(s.tilt_angle) if s.tilt_angle is not None else None,
        azimuth=float(s.azimuth_angle) if s.azimuth_angle is not None else None,
        shading_selection=shade,
        albedo=float(getattr(s, "albedo", None) or 0.20),
        mounting=getattr(s, "mounting_type", None) or "rooftop",
        bifacial_gain_pct=float(getattr(s, "bifacial_gain_pct", None) or 0),
        wiring_loss_pct=float(getattr(s, "wiring_loss_pct", None) or 2),
        mismatch_loss_pct=float(getattr(s, "mismatch_loss_pct", None) or 1.5),
        availability_pct=float(getattr(s, "availability_pct", None) or 98),
        degradation_pct_per_year=float(getattr(s, "degradation_pct_per_year", None) or 0.5),
    )


def _parse_iso_date(sval):
    try:
        return datetime.strptime(str(sval)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def backfill_user_history(u, max_gap_days: int = 90) -> int:
    """
    Fill *missing* past days after the user's system became active.

    Rules:
    - Never invent history *before* system_active_from (first setup / first forecast)
    - Never invent history for accounts that have never saved a system
    - Never overwrite calibrated rows
    - Never touch today / future
    - If user is offline 10 days, fills only the gap since active_from (capped)

    Returns number of days inserted.
    """
    s = getattr(u, "system", None)
    inputs = _system_inputs_from_db(s)
    if inputs is None:
        return 0

    today = datetime.now(IST_TZ).date()
    end = today - timedelta(days=1)

    # Anchor = first day this rooftop was tracked (not "today - 21")
    anchor = _parse_iso_date(getattr(s, "system_active_from", None))
    if anchor is None:
        # Legacy users: earliest existing history row, else wait for first forecast
        earliest = (
            db.session.query(db.func.min(HistoricalLog.log_date))
            .filter(HistoricalLog.user_id == u.user_id)
            .scalar()
        )
        anchor = _parse_iso_date(earliest)
        if anchor is None:
            return 0
        # Persist so we don't expand earlier later
        try:
            s.system_active_from = anchor.isoformat()
            db.session.commit()
        except Exception:
            db.session.rollback()

    start = anchor
    if end < start:
        return 0  # system activated today — no past days yet

    # Cap one backfill window (API size / free-tier safety)
    if (end - start).days > int(max_gap_days):
        start = end - timedelta(days=int(max_gap_days))
        if start < anchor:
            start = anchor

    existing = {
        r.log_date
        for r in HistoricalLog.query.filter_by(user_id=u.user_id)
        .filter(HistoricalLog.log_date >= start.isoformat())
        .filter(HistoricalLog.log_date <= end.isoformat())
        .all()
    }
    missing = []
    d = start
    while d <= end:
        if d.isoformat() not in existing:
            missing.append(d)
        d += timedelta(days=1)
    if not missing:
        return 0

    lat, lon = float(s.latitude), float(s.longitude)
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
            f"&hourly={_ARCHIVE_HOURLY}&timezone=Asia%2FKolkata"
        )
        weather_json = api_get_json(url, timeout=60)
    except Exception as e:
        print("history backfill archive fetch failed:", e)
        return 0

    try:
        frame = build_hourly_weather(weather_json, None)
        # Archive has no precip probability — proxy from rainfall
        if "precipitation" in frame.columns and (
            "precipitation_probability" not in frame.columns
            or float(frame["precipitation_probability"].fillna(0).max() or 0) == 0
        ):
            frame["precipitation_probability"] = np.where(
                pd.to_numeric(frame["precipitation"], errors="coerce").fillna(0) > 0.1,
                70.0,
                10.0,
            )
        frame, _meta = enrich_frame(frame, inputs, None)
    except Exception as e:
        print("history backfill enrich failed:", e)
        return 0

    package = get_model_for_state(s.state or UP_STATE_NAME)
    physics = frame["physics_baseline_kw"].to_numpy().astype(float)
    model_used = bool(package.get("available"))
    system_dc_kw = float(inputs.panel_stc_w) * int(inputs.num_panels) / 1000.0
    if model_used:
        try:
            ml_raw = predict_model(package, frame)
            ml_scaled = np.asarray(ml_raw, dtype=float) * (system_dc_kw / 5.0)
        except Exception:
            ml_scaled = np.zeros_like(physics)
            model_used = False
    else:
        ml_scaled = np.zeros_like(physics)

    pred = np.zeros_like(physics)
    for d0, g in frame.groupby(frame.index.date):
        idx = g.index
        pos = frame.index.get_indexer(idx)
        _wp, _wm, day_pred, _c = _apply_best_blend(
            g, physics[pos], ml_scaled[pos], model_used
        )
        pred[pos] = np.maximum(0.0, day_pred)

    if "is_day" in frame.columns:
        night = frame["is_day"].to_numpy() <= 0
        pred = np.where(night, 0.0, pred)
    elif "solar_zenith" in frame.columns:
        night = frame["solar_zenith"].to_numpy() >= 90.0
        pred = np.where(night, 0.0, pred)
    if "poa_global" in frame.columns:
        pred = np.where(frame["poa_global"].to_numpy() <= 0.0, 0.0, pred)

    cal = float(np.clip(float(getattr(s, "calibration_slope", 1.0) or 1.0), 0.6, 1.4))
    paco_cap = float(inputs.inverter_paco_w) * int(inputs.num_inverters) / 1000.0
    if paco_cap > 0:
        pred = np.clip(pred * cal, 0.0, paco_cap)
    else:
        pred = np.maximum(pred * cal, 0.0)
    frame["pred_p50"] = pred
    frame["hourly_carbon"] = pred * CEA_GRID_EMISSION_FACTOR

    load = float(s.daily_units or 0)
    inserted = 0
    for day in missing:
        try:
            g = frame[frame.index.date == day]
            if g.empty:
                continue
            total = float(g["pred_p50"].sum())
            day_co2 = float(g["hourly_carbon"].sum()) if "hourly_carbon" in g else total * CEA_GRID_EMISSION_FACTOR
            soil_pct = 0.0
            if "soiling_ratio" in g.columns:
                sr = g["soiling_ratio"].replace(0, np.nan)
                mean_sr = float(sr.mean()) if sr.notna().any() else 1.0
                mean_sr = float(np.clip(mean_sr, 0.5, 1.0))
                soil_pct = max(0.0, (1.0 - mean_sr) * 100.0)
            dstr = day.isoformat()
            if HistoricalLog.query.filter_by(user_id=u.user_id, log_date=dstr).first():
                continue
            db.session.add(HistoricalLog(
                user_id=u.user_id,
                log_date=dstr,
                generated_kwh=round(total, 3),
                predicted_kwh=round(total, 3),
                consumption_kwh=float(load),
                soiling_loss_pct=round(soil_pct, 2),
                carbon_saved_kg=round(day_co2, 3),
                is_calibrated=False,
            ))
            inserted += 1
        except Exception as e:
            print("history backfill day failed", day, e)
            continue

    if inserted:
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print("history backfill commit failed:", e)
            return 0
    return inserted


@app.route("/history")
@require_login(role="user")
def history_page():
    token = _extract_request_token()
    u = current_user_from_token(token)
    # Backfill any past days missed while user was offline
    try:
        backfill_user_history(u, max_gap_days=90)
    except Exception as e:
        print("history backfill error:", e)
    # Month selector (YYYY-MM); default = current month IST
    today = datetime.now(IST_TZ).date()
    month = (request.args.get("month") or today.strftime("%Y-%m")).strip()
    try:
        y, m = int(month[:4]), int(month[5:7])
        month_start = date(y, m, 1)
    except Exception:
        y, m = today.year, today.month
        month = f"{y:04d}-{m:02d}"
        month_start = date(y, m, 1)
    if m == 12:
        month_end = date(y + 1, 1, 1)
    else:
        month_end = date(y, m + 1, 1)
    today_s = today.isoformat()
    start_s = month_start.isoformat()
    end_s = month_end.isoformat()

    rows = (
        HistoricalLog.query.filter_by(user_id=u.user_id)
        .filter(HistoricalLog.log_date >= start_s)
        .filter(HistoricalLog.log_date < end_s)
        .filter(HistoricalLog.log_date < today_s)  # past only
        .order_by(HistoricalLog.log_date.desc())
        .all()
    )

    # Month stats
    days_in_month = (month_end - month_start).days
    pred_sum = 0.0
    act_sum = 0.0
    cons_sum = 0.0
    co2_sum = 0.0
    n_cal = 0
    n_rows = len(rows)
    for r in rows:
        pred = r.predicted_kwh if r.predicted_kwh is not None else (r.generated_kwh or 0)
        pred_sum += float(pred or 0)
        cons_sum += float(r.consumption_kwh or 0)
        co2_sum += float(r.carbon_saved_kg or 0)
        if r.is_calibrated and r.actual_kwh is not None:
            n_cal += 1
            act_sum += float(r.actual_kwh)
        else:
            act_sum += float(pred or 0)  # fallback display total energy
    # Net using preferred actual when calibrated else predicted
    net_sum = 0.0
    for r in rows:
        e = float(r.actual_kwh) if (r.is_calibrated and r.actual_kwh is not None) else float(
            r.predicted_kwh if r.predicted_kwh is not None else (r.generated_kwh or 0)
        )
        net_sum += e - float(r.consumption_kwh or 0)

    # Bound prev/next to months that actually have past history data
    all_dates = [
        r[0] for r in db.session.query(HistoricalLog.log_date)
        .filter_by(user_id=u.user_id)
        .filter(HistoricalLog.log_date < today_s)
        .distinct()
        .all()
        if r[0]
    ]
    months_with_data = sorted({d[:7] for d in all_dates if len(d) >= 7})
    current_month = today.strftime("%Y-%m")
    prev_m = None
    next_m = None
    if months_with_data:
        # prev = latest month strictly before selected that has data
        earlier = [x for x in months_with_data if x < month]
        later = [x for x in months_with_data if x > month]
        prev_m = earlier[-1] if earlier else None
        next_m = later[0] if later else None
        # Allow next up to current month even if empty (user can land on current)
        if next_m is None and month < current_month:
            next_m = current_month
    # Always allow jumping to current month from UI

    return render_template(
        "history.html",
        history=rows,
        user=u,
        token=token,
        month=month,
        current_month=current_month,
        month_label=month_start.strftime("%B %Y"),
        prev_month=prev_m,
        next_month=next_m,
        days_in_month=days_in_month,
        stats={
            "pred_sum": round(pred_sum, 2),
            "energy_sum": round(act_sum, 2),  # actual preferred mix
            "cons_sum": round(cons_sum, 2),
            "net_sum": round(net_sum, 2),
            "co2_sum": round(co2_sum, 2),
            "trees": round(co2_sum / 21.0, 2),
            "n_cal": n_cal,
            "n_rows": n_rows,
            "cal_label": f"{n_cal}/{days_in_month}",
        },
    )


@app.route("/profile", methods=["GET", "POST"])
@require_login()  # admin or user may open own profile
def profile():
    token = _extract_request_token()
    u = current_user_from_token(token)
    if request.method == "POST":
        err = None
        # Name: both admin and user can update display name
        new_name = (request.form.get("name") or "").strip()
        if new_name:
            u.name = new_name
        elif not u.is_admin:
            err = "Name cannot be empty."

        # Password change (optional)
        pw = request.form.get("new_password") or ""
        pw2 = request.form.get("new_password2") or ""
        if pw or pw2:
            if len(pw) < 6:
                err = "Password must be at least 6 characters."
            elif pw != pw2:
                err = "New passwords do not match."
            else:
                u.password_hash = generate_password_hash(pw)

        if err:
            return render_template(
                "profile.html",
                user=u,
                error=err,
                token=token,
                admin_sync=ADMIN_SYNC,
            )

        db.session.commit()
        msg = "Profile updated successfully!"
        if pw and u.is_admin and ADMIN_SYNC:
            msg += (
                " Note: ADMIN_SYNC=1 in .env — on next server restart, "
                "admin password will be reset from ADMIN_PASSWORD in .env. "
                "Set ADMIN_SYNC=0 to keep this UI password."
            )
        return render_template(
            "profile.html",
            user=u,
            success=msg,
            token=token,
            admin_sync=ADMIN_SYNC,
        )
    return render_template(
        "profile.html",
        user=u,
        token=token,
        admin_sync=ADMIN_SYNC,
    )


@app.route("/api/user_config")
@require_login()
def user_config():
    u = current_user_from_token()
    if not u:
        return jsonify({}), 401
    # Quiet backfill when user opens home/profile (offline days still appear in history)
    if u and not u.is_admin:
        try:
            backfill_user_history(u, max_gap_days=90)
        except Exception as e:
            print("user_config backfill:", e)
    s = u.system or SolarSystem(user_id=u.user_id)
    if not u.system:
        db.session.add(s)
        db.session.commit()
    pw = iw = 0
    try:
        if s.brand_model:
            pw = get_panel_specs(s.brand_model)[0]
        if s.inverter_model:
            iw = get_inverter_specs(s.inverter_model)[0]
    except Exception:
        pass
    return jsonify({
        "state": s.state,
        "district": s.district,
        "city_locality": s.city_locality,
        "pincode": s.pincode,
        "latitude": s.latitude,
        "longitude": s.longitude,
        "brand_model": s.brand_model,
        "inverter_model": s.inverter_model,
        "num_panels": s.num_panels,
        "num_inverters": s.num_inverters,
        "tilt_angle": round(float(s.tilt_angle), 2) if s.tilt_angle is not None else None,
        "azimuth_angle": round(float(s.azimuth_angle), 1) if s.azimuth_angle is not None else None,
        "dc_capacity_kw": round((float(pw or 0) * int(s.num_panels or 0)) / 1000.0, 2) if s.num_panels else 0.0,
        "ac_capacity_kw": round((float(iw or 0) * int(s.num_inverters or 0)) / 1000.0, 2) if s.num_inverters else 0.0,
        "installation_year": s.installation_year,
        "last_cleaned_date": s.last_cleaned_date,
        "shading_selection": (s.shading_selection or "").split(",") if s.shading_selection else [],
        "daily_units": s.daily_units,
        "albedo": s.albedo,
        "mounting_type": s.mounting_type,
        "bifacial_gain_pct": s.bifacial_gain_pct,
        "wiring_loss_pct": s.wiring_loss_pct,
        "mismatch_loss_pct": s.mismatch_loss_pct,
        "availability_pct": s.availability_pct,
        "degradation_pct_per_year": s.degradation_pct_per_year,
        "panel_watts": float(pw or 0),
        "inverter_watts": float(iw or 0),
        "calibration_slope": s.calibration_slope or 1.0,
        # Used by home UI: summary mode vs full edit form
        "is_complete": bool(
            s.pincode
            and s.latitude is not None
            and s.longitude is not None
            and s.brand_model
            and s.inverter_model
            and s.num_panels
            and int(s.num_panels or 0) >= 1
            and s.num_inverters
            and int(s.num_inverters or 0) >= 1
            and s.daily_units
            and float(s.daily_units or 0) > 0
            and s.installation_year
        ),
    })


# ---------------------------------------------------------------
# FORECAST
# ---------------------------------------------------------------
@app.route("/api/forecast", methods=["POST"])
@require_login(role="user")
def forecast():
    p = request.get_json() or {}
    u = current_user_from_token()
    if not u:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        state = p.get("state")
        district = p.get("district")
        city = p.get("city_locality")
        pin = p.get("pincode")
        lat = float(p.get("lat"))
        lon = float(p.get("lon"))
        panel = p.get("brand_model")
        inv = p.get("inverter_model")

        if not all([state, district, city, pin]):
            return jsonify({"error": "Complete location mapping first."}), 400
        if not is_up_state(state):
            return jsonify({
                "error": reject_non_up_message(state, pin),
                "scope": "uttar_pradesh_only",
            }), 403
        state = UP_STATE_NAME
        district = normalize_district(district) or district
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({"error": "Invalid coordinates."}), 400
        if not is_inside_up(lat, lon):
            return jsonify({
                "error": reject_outside_up_coords_message(),
                "scope": "uttar_pradesh_only",
            }), 403
        if panel not in CEC_MOD_T.index or inv not in CEC_INV_T.index:
            return jsonify({"error": "Invalid hardware selection."}), 400

        # Server cache (15 min) for identical system + location payload
        cache_k = _forecast_cache_key(p)
        cached = _cached_forecast(cache_k)
        if cached is not None:
            return jsonify(cached)

        npan = int(p.get("num_panels", 1))
        ninv = int(p.get("num_inverters", 1))
        load = float(p.get("daily_units", 1))
        year = int(p.get("installation_year", datetime.now().year))
        if npan < 1 or ninv < 1:
            return jsonify({"error": "Panel and inverter count must be at least 1."}), 400

        tilt = float(p["tilt"]) if p.get("tilt") not in [None, ""] else None
        az = float(p["azimuth"]) if p.get("azimuth") not in [None, ""] else None

        shade_list = normalize_shading_selection(p.get("shading_selection"))

        stc, gamma, eff = get_panel_specs(panel)
        paco, inv_eff = get_inverter_specs(inv)

        inputs = SystemInputs(
            latitude=lat,
            longitude=lon,
            num_panels=npan,
            num_inverters=ninv,
            panel_stc_w=stc,
            panel_gamma=gamma,
            inverter_paco_w=paco,
            inverter_eff=inv_eff,
            installation_year=year,
            manual_clean_date=p.get("manual_clean_date") or p.get("last_cleaned_date") or None,
            tilt=tilt,
            azimuth=az,
            shading_selection=shade_list,
            albedo=float(p.get("albedo", 0.20)),
            mounting=p.get("mounting_type", "rooftop"),
            bifacial_gain_pct=float(p.get("bifacial_gain_pct", 0)),
            wiring_loss_pct=float(p.get("wiring_loss_pct", 2)),
            mismatch_loss_pct=float(p.get("mismatch_loss_pct", 1.5)),
            availability_pct=float(p.get("availability_pct", 98)),
            degradation_pct_per_year=float(p.get("degradation_pct_per_year", 0.5)),
        )

        start = datetime.now(IST_TZ).date()
        hist_start = start - timedelta(days=30)
        hist_end = start - timedelta(days=1)
        try:
            hurl = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lon}"
                f"&start_date={hist_start}&end_date={hist_end}"
                f"&hourly=precipitation&timezone=Asia%2FKolkata"
            )
            hist_json = api_get_json(hurl, timeout=30)
            hist_df = pd.DataFrame(hist_json.get("hourly", {}))
            if not hist_df.empty:
                hist_df["time"] = pd.to_datetime(hist_df["time"])
                hist_df = hist_df.set_index("time")
            else:
                hist_df = None
        except Exception:
            hist_df = None

        # Multi-model Open-Meteo fallback (from production SolarFlow)
        weather_json = fetch_forecast_weather(lat, lon)
        aurl = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}&forecast_days=7&timezone=Asia%2FKolkata"
            f"&hourly=pm10,pm2_5,european_aqi,dust"
        )
        try:
            aq_json = api_get_json(aurl)
        except Exception:
            aq_json = None

        frame = build_hourly_weather(weather_json, aq_json)
        frame, meta = enrich_frame(frame, inputs, hist_df)

        package = get_model_for_state(state)
        physics = frame["physics_baseline_kw"].to_numpy().astype(float)
        model_used = bool(package.get("available"))

        # Per-day blend (not one universal weight for the whole week)
        system_dc_kw = stc * npan / 1000.0
        if model_used:
            ml_raw = predict_model(package, frame)
            ml_scaled = ml_raw * (system_dc_kw / 5.0)
        else:
            ml_scaled = np.zeros_like(physics)

        pred = np.zeros_like(physics)
        day_blend_map = {}  # date -> (w_phys, w_ml, conf)
        for d, g in frame.groupby(frame.index.date):
            idx = g.index
            pos = frame.index.get_indexer(idx)
            # Evaluate several blends; keep highest-confidence mix for this day
            w_p, w_m, day_pred, day_conf = _apply_best_blend(
                g, physics[pos], ml_scaled[pos], model_used
            )
            day_blend_map[d] = (w_p, w_m, day_conf)
            pred[pos] = np.maximum(0.0, day_pred)

        # Week-level average (for header KPI only)
        w_phys, w_ml = adaptive_blend_weights(frame, model_used)

        # HARD night zero after blend — ML must never invent nighttime generation
        night_mask = None
        if "is_day" in frame.columns:
            night_mask = frame["is_day"].to_numpy() <= 0
        elif "solar_zenith" in frame.columns:
            night_mask = frame["solar_zenith"].to_numpy() >= 90.0
        if night_mask is not None:
            pred = np.where(night_mask, 0.0, pred)
            physics = np.where(night_mask, 0.0, physics)
            ml_scaled = np.where(night_mask, 0.0, ml_scaled)
            frame["physics_baseline_kw"] = physics

        # Extra safety: if POA is zero, power must be zero
        if "poa_global" in frame.columns:
            zero_poa = frame["poa_global"].to_numpy() <= 0.0
            pred = np.where(zero_poa, 0.0, pred)
            ml_scaled = np.where(zero_poa, 0.0, ml_scaled)

        calibration = float(getattr(u.system, "calibration_slope", 1.0) or 1.0)
        calibration = float(np.clip(calibration, 0.6, 1.4))
        pred = np.clip(pred * calibration, 0.0, (paco * ninv) / 1000.0)
        # Keep method tracks on same scale for real disagreement bands
        physics = physics * calibration
        ml_scaled = ml_scaled * calibration
        frame["physics_baseline_kw"] = physics
        frame["ml_scaled_kw"] = ml_scaled

        # Re-apply night mask after calibration
        if night_mask is not None:
            pred = np.where(night_mask, 0.0, pred)
            physics = np.where(night_mask, 0.0, physics)
            ml_scaled = np.where(night_mask, 0.0, ml_scaled)
            frame["physics_baseline_kw"] = physics
            frame["ml_scaled_kw"] = ml_scaled

        frame["pred_p50"] = pred
        # Real uncertainty bands (physics vs ML + weather + training residual)
        p10, p90 = uncertainty_bands(pred, frame, physics, ml_scaled, model_used, package=package)
        if night_mask is not None:
            p10 = np.where(night_mask, 0.0, p10)
            p90 = np.where(night_mask, 0.0, p90)
        if "poa_global" in frame.columns:
            zero_poa = frame["poa_global"].to_numpy() <= 0.0
            p10 = np.where(zero_poa, 0.0, p10)
            p90 = np.where(zero_poa, 0.0, p90)
        frame["pred_p10"] = p10
        frame["pred_p90"] = p90
        frame["hourly_carbon"] = pred * CEA_GRID_EMISSION_FACTOR

        avg_soiling_loss = safe_float((1 - frame["soiling_ratio"]).mean() * 100)
        years_old = max(0, datetime.now().year - year)
        confidence_pct = confidence_from_bands(
            p10, pred, p90, frame, model_used, years_old, avg_soiling_loss
        )

        health = 100.0
        health -= min(avg_soiling_loss * 1.1, 18)
        health -= min(years_old * 1.2, 12)
        shade_profile = meta.get("shading_profile") or {}
        for level in shade_profile.values():
            if level == "heavy":
                health -= 5
            elif level == "medium":
                health -= 3
            elif level == "light":
                health -= 1
        health = float(np.clip(health, 55, 98))
        health_score = int(round(health))

        # Per-day soiling helper (same formula everywhere)
        def _day_soiling(g, total_kwh: float):
            if "soiling_ratio" not in g.columns or len(g) == 0:
                return 0.0, 0.0
            sr = g["soiling_ratio"].replace(0, np.nan)
            mean_sr = float(sr.mean()) if sr.notna().any() else 1.0
            mean_sr = float(np.clip(mean_sr, 0.5, 1.0))
            soil_pct = max(0.0, (1.0 - mean_sr) * 100.0)
            loss_kwh = max(0.0, (total_kwh / mean_sr) - total_kwh) if mean_sr > 0 else 0.0
            return soil_pct, loss_kwh

        summaries = []
        for d, g in frame.groupby(frame.index.date):
            peak = g["pred_p50"].idxmax()
            total = float(g["pred_p50"].sum())
            rains = safe_float(g["precipitation_probability"].mean())
            aqi_val = None
            if "european_aqi" in g.columns:
                aqi_mean = g["european_aqi"].mean()
                if pd.notna(aqi_mean):
                    aqi_val = safe_int(aqi_mean)
            blend_t = day_blend_map.get(d, (w_phys, w_ml, None))
            w_p_d, w_m_d = blend_t[0], blend_t[1]
            day_conf = blend_t[2] if len(blend_t) > 2 else None
            cloud_d = safe_float(g["cloud_cover"].mean()) if "cloud_cover" in g else 40.0

            # Daily range from real day physics/ML + weather + training residual
            total_p10, total_p90 = daily_range_from_day(g, total, model_used, package=package)
            soil_pct_d, soil_loss_d = _day_soiling(g, total)
            summaries.append({
                "date": d.strftime("%Y-%m-%d"),
                "display_date": d.strftime("%d %b"),
                "day_name": d.strftime("%A"),
                "total_units_kwh": round(total, 2),
                "total_units_p10": round(total_p10, 2),
                "total_units_p90": round(total_p90, 2),
                "peak_kw": round(float(g.loc[peak, "pred_p50"]), 2),
                "peak_time": peak.strftime("%I:%M %p"),
                "temp_min": round(safe_float(g.temperature_2m.min()), 1),
                "temp_max": round(safe_float(g.temperature_2m.max()), 1),
                "condition": g["weather_name"].iloc[0] if "weather_name" in g else "Variable",
                "rain_prob": safe_int(rains),
                "cloud_avg": safe_int(cloud_d),
                "blend_physics": round(float(w_p_d), 3),
                "blend_ml": round(float(w_m_d), 3),
                "confidence_pct": int(round(day_conf)) if day_conf is not None else None,
                "humidity": safe_int(g["relative_humidity_2m"].mean() if "relative_humidity_2m" in g else g.get("relativehumidity_2m", pd.Series([50])).mean(), 50),
                "aqi": aqi_val,
                "carbon_saved_kg": round(float(g.hourly_carbon.sum()), 2),
                "soiling_loss_pct": round(soil_pct_d, 2),
                "soiling_loss_kwh": round(soil_loss_d, 2),
                "hourly": {
                    "times": g.index.strftime("%H:%M").tolist(),
                    "power": g.pred_p50.round(3).tolist(),
                    "power_p10": (g["pred_p10"].round(3).tolist() if "pred_p10" in g else g.pred_p50.round(3).tolist()),
                    "power_p90": (g["pred_p90"].round(3).tolist() if "pred_p90" in g else g.pred_p50.round(3).tolist()),
                    "cell_temp": g.cell_temp.round(1).tolist(),
                    "ambient_temp": g.temperature_2m.round(1).tolist(),
                    "rain_prob": [safe_int(x) for x in g.precipitation_probability.tolist()],
                    "rain_mm": g.precipitation.round(2).tolist(),
                    "showers_mm": g.showers.round(2).tolist() if "showers" in g else [0] * len(g),
                    "cloud": [safe_int(x) for x in g.cloud_cover.tolist()],
                    "wind": g.windspeed_10m.round(1).tolist(),
                    "gust": (g["windgusts_10m"] if "windgusts_10m" in g else g["wind_gusts_10m"] if "wind_gusts_10m" in g else pd.Series([0]*len(g))).round(1).tolist(),
                    "soiling": (100 * (1 - g.soiling_ratio)).round(2).tolist(),
                    "poa": g.poa_global.round(0).tolist(),
                    "inverter_eff": (100 * g.inverter_efficiency).round(1).tolist(),
                    "conditions": g.weather_name.tolist(),
                    "condition_emojis": g.weather_emoji.tolist(),
                    "carbon_saved": g.hourly_carbon.round(3).tolist(),
                    "humidity": [safe_int(x, 50) for x in (g["relative_humidity_2m"] if "relative_humidity_2m" in g else g.get("relativehumidity_2m", pd.Series([50]*len(g)))).tolist()],
                    "aqi": [safe_int(x) for x in g.european_aqi.fillna(0).tolist()] if "european_aqi" in g else [0] * len(g),
                },
            })

        s = u.system or SolarSystem(user_id=u.user_id)
        s.state, s.district, s.city_locality, s.pincode = state, district, city, pin
        s.latitude, s.longitude = lat, lon
        s.brand_model, s.inverter_model = panel, inv
        s.num_panels, s.num_inverters = npan, ninv
        s.tilt_angle = round(float(meta["tilt"]), 2)
        s.azimuth_angle = round(float(meta["azimuth"]), 1)
        s.installation_year = year
        s.last_cleaned_date = p.get("manual_clean_date") or None
        s.shading_selection = ",".join(shade_list)
        s.daily_units = load
        s.albedo = inputs.albedo
        s.mounting_type = inputs.mounting
        s.bifacial_gain_pct = inputs.bifacial_gain_pct
        s.wiring_loss_pct = inputs.wiring_loss_pct
        s.mismatch_loss_pct = inputs.mismatch_loss_pct
        s.availability_pct = inputs.availability_pct
        s.degradation_pct_per_year = inputs.degradation_pct_per_year
        # First successful forecast marks history start — no pre-signup invented days
        if not getattr(s, "system_active_from", None):
            s.system_active_from = datetime.now(IST_TZ).date().isoformat()
            # Drop any accidental pre-setup history rows (uncalibrated only)
            try:
                HistoricalLog.query.filter_by(user_id=u.user_id).filter(
                    HistoricalLog.log_date < s.system_active_from,
                    HistoricalLog.is_calibrated.is_(False),
                ).delete(synchronize_session=False)
            except Exception:
                pass
        if not u.system:
            db.session.add(s)

        fl = u.logs if u.logs else ForecastLog(user_id=u.user_id)
        if not getattr(fl, "log_id", None):
            db.session.add(fl)

        total7 = float(sum(x["total_units_kwh"] for x in summaries))
        fl.predicted_gen_kwh = round(total7, 2)
        fl.today_gen_kwh = summaries[0]["total_units_kwh"]
        fl.daily_consumption_kwh = load * 7
        fl.carbon_saved_kg = round(float(frame.hourly_carbon.sum()), 2)
        fl.today_carbon_saved_kg = summaries[0]["carbon_saved_kg"]
        fl.last_updated = datetime.now(timezone.utc)

        # Snapshot predicted kWh for each forecast day. Past days freeze once written.
        # History PAGE only lists days < today; storing today/future is how those
        # rows appear later as "past" without re-fetching weather.
        today_d = datetime.now(IST_TZ).date()
        for day in summaries:
            dstr = day["date"][:10]
            try:
                day_d = datetime.strptime(dstr, "%Y-%m-%d").date()
            except Exception:
                continue
            pred = float(day["total_units_kwh"])
            day_co2 = float(day.get("carbon_saved_kg") or 0.0)
            row = HistoricalLog.query.filter_by(user_id=u.user_id, log_date=dstr).first()
            if row is None:
                db.session.add(HistoricalLog(
                    user_id=u.user_id,
                    log_date=dstr,
                    generated_kwh=pred,
                    predicted_kwh=pred,
                    consumption_kwh=float(load),
                    soiling_loss_pct=round(avg_soiling_loss, 2),
                    carbon_saved_kg=round(day_co2, 3),
                    is_calibrated=False,
                ))
                continue
            if row.is_calibrated:
                continue
            # Past day already snapshotted — do not recalculate/overwrite
            if day_d < today_d and row.predicted_kwh is not None:
                continue
            # Today / future: refresh predicted snapshot on new forecast
            if day_d >= today_d:
                row.predicted_kwh = pred
                if not row.is_calibrated:
                    row.generated_kwh = pred
                row.consumption_kwh = float(load)
                row.soiling_loss_pct = round(avg_soiling_loss, 2)
                row.carbon_saved_kg = round(day_co2, 3)

        db.session.commit()

        # Lifetime CO₂ from all past archived days (for trees KPI)
        today_s = today_d.isoformat()
        past_rows = (
            HistoricalLog.query.filter_by(user_id=u.user_id)
            .filter(HistoricalLog.log_date < today_s)
            .all()
        )
        lifetime_co2 = float(sum(float(r.carbon_saved_kg or 0) for r in past_rows))
        # Include today's forecast CO₂ in "till now" total
        lifetime_co2 += float(summaries[0]["carbon_saved_kg"]) if summaries else 0.0
        lifetime_trees = lifetime_co2 / 21.0

        today = frame.iloc[0]
        payload = {
            "current_weather": {
                "temp": round(safe_float(today.temperature_2m), 1),
                "feels_like": round(safe_float(today.apparent_temperature), 1),
                "humidity": safe_int(today.get("relative_humidity_2m", today.get("relativehumidity_2m", 50)), 50),
                "rain_prob": safe_int(today.precipitation_probability),
                "rain_mm": round(safe_float(today.precipitation), 2),
                "cloud": safe_int(today.cloud_cover),
                "wind": round(safe_float(today.windspeed_10m), 1),
                "gust": round(safe_float(today.get("windgusts_10m", today.get("wind_gusts_10m", 0))), 1),
                "condition": today.weather_name if "weather_name" in frame.columns else "Variable",
                "emoji": today.weather_emoji if "weather_emoji" in frame.columns else "🌤️",
                "aqi": safe_int(today.european_aqi) if "european_aqi" in frame.columns and pd.notna(today.european_aqi) else None,
            },
            "system": {
                "zone": package.get("zone"),
                "regional_model_available": model_used,
                "tilt": round(float(meta["tilt"]), 2),
                "tilt_auto": meta["tilt_auto"],
                "azimuth": round(float(meta["azimuth"]), 1),
                "azimuth_auto": meta["azimuth_auto"],
                "baseline_clean_date": meta["baseline_clean_date"],
                "shading_profile": meta.get("shading_profile"),
                "blend_physics": round(w_phys, 2),
                "blend_ml": round(w_ml, 2),
                "calibration_slope": round(calibration, 3),
                "dc_capacity_kw": round(stc * npan / 1000, 2),
                "ac_capacity_kw": round(paco * ninv / 1000, 2),
                "panel_efficiency_pct": round(eff, 2),
                "panel_temp_coeff_pct_per_c": round(gamma * 100, 3),
                "inverter_efficiency_pct": round(inv_eff * 100, 2),
            },
            "insights": {
                "confidence_pct": confidence_pct,
                # Today only — same numbers as first day card (no week-average mix-up)
                "soiling_status": (
                    "Clean" if (summaries and summaries[0].get("soiling_loss_pct", 0) < 4)
                    else ("Mild dust" if (summaries and summaries[0].get("soiling_loss_pct", 0) < 9) else "Dusty")
                ),
                "soiling_loss_pct": round(float(summaries[0]["soiling_loss_pct"]), 2) if summaries else 0.0,
                "today_soiling_loss_kwh": round(float(summaries[0]["soiling_loss_kwh"]), 2) if summaries else 0.0,
                "today_gen_kwh": round(float(summaries[0]["total_units_kwh"]), 2) if summaries else 0.0,
                "cleaning_tip": (
                    "Panels look clean — no action needed."
                    if (summaries and summaries[0].get("soiling_loss_pct", 0) < 4)
                    else (
                        "Light cleaning in 7–10 days can recover a bit more energy."
                        if (summaries and summaries[0].get("soiling_loss_pct", 0) < 9)
                        else "Cleaning soon will recover lost generation."
                    )
                ),
            },
            "total_7d_gen_units": round(total7, 2),
            "total_7d_gen_p10": round(float(sum(x.get("total_units_p10", x["total_units_kwh"]) for x in summaries)), 2),
            "total_7d_gen_p90": round(float(sum(x.get("total_units_p90", x["total_units_kwh"]) for x in summaries)), 2),
            "today_gen_units": summaries[0]["total_units_kwh"],
            "total_7d_cons_units": round(load * 7, 2),
            "net_balance_units": round(total7 - load * 7, 2),
            "total_carbon_saved_kg": round(float(frame.hourly_carbon.sum()), 2),
            "lifetime_carbon_kg": round(lifetime_co2, 2),
            "lifetime_trees": round(lifetime_trees, 2),
            "daily_forecasts": summaries,
            "weather_model": weather_json.get("_model_used", "best_match") if isinstance(weather_json, dict) else "best_match",
            "model_note": "",
        }
        try:
            _store_forecast_cache(cache_k, payload)
        except Exception:
            pass
        return jsonify(payload)

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


def _recompute_calibration_slope(user_id: int) -> float:
    """
    Recompute calibration_slope from calibrated days only (idempotent).

    Uses median(actual / predicted). Re-saving the same day does NOT
    stack multiple EMA steps — slope is always derived fresh from the
    current set of calibrated days.
    """
    rows = (
        HistoricalLog.query.filter_by(user_id=user_id, is_calibrated=True)
        .order_by(HistoricalLog.log_date.desc())
        .limit(60)
        .all()
    )
    ratios = []
    for r in rows:
        pred = r.predicted_kwh if r.predicted_kwh is not None else None
        act = r.actual_kwh
        if pred is None or act is None:
            continue
        if float(pred) < 0.5:
            continue
        ratios.append(float(act) / float(pred))

    s = SolarSystem.query.filter_by(user_id=user_id).first()
    if not s:
        return 1.0

    if not ratios:
        s.calibration_slope = 1.0
        db.session.commit()
        return 1.0

    ratios = sorted(ratios)
    mid = ratios[len(ratios) // 2]
    mid = float(np.clip(mid, 0.7, 1.3))
    n = len(ratios)
    # Few samples: shrink toward 1.0 so one day is mild, many days = full trust
    if n == 1:
        mid = 0.55 * 1.0 + 0.45 * mid
    elif n == 2:
        mid = 0.30 * 1.0 + 0.70 * mid
    elif n == 3:
        mid = 0.15 * 1.0 + 0.85 * mid
    # n >= 4: use median as-is (still clipped)

    s.calibration_slope = round(float(np.clip(mid, 0.7, 1.3)), 4)
    db.session.commit()
    return float(s.calibration_slope)


@app.route("/api/calibrate_day", methods=["POST"])
@require_login(role="user")
def api_calibrate_day():
    """
    User enters actual daily generation (kWh) for a PAST day only.
    Validates lightly and updates calibration_slope slowly.
    """
    p = request.get_json(silent=True) or {}
    u = current_user_from_token()
    if not u:
        return jsonify({"error": "Unauthorized"}), 401

    log_date = (p.get("log_date") or "").strip()[:10]
    raw_actual = p.get("actual_kwh")
    try:
        actual = float(raw_actual)
    except (TypeError, ValueError):
        return jsonify({"error": "Enter a valid actual generation in kWh (number)."}), 400

    if not log_date or len(log_date) < 8:
        return jsonify({"error": "Missing date."}), 400
    if actual < 0:
        return jsonify({"error": "Actual generation cannot be negative."}), 400
    if actual > 500:
        return jsonify({"error": "Value looks unrealistically high (max 500 kWh)."}), 400

    try:
        d = datetime.strptime(log_date, "%Y-%m-%d").date()
    except Exception:
        return jsonify({"error": "Invalid date format (use YYYY-MM-DD)."}), 400

    today = datetime.now(IST_TZ).date()
    # History / calibration = completed days only (not today, not future)
    if d >= today:
        return jsonify({
            "error": "Calibrate only past days (yesterday or earlier). Today is still running."
        }), 400

    s = u.system
    # Soft capacity check — warn via message but only hard-reject extreme outliers
    if s and s.num_panels and s.brand_model:
        try:
            stc, _, _ = get_panel_specs(s.brand_model)
            dc_kw = stc * int(s.num_panels) / 1000.0
            max_day = max(15.0, dc_kw * 12.0)  # generous residential bound
            if actual > max_day:
                return jsonify({
                    "error": f"Actual {actual:.1f} kWh exceeds plausible max (~{max_day:.0f} kWh) for your system. Check the number."
                }), 400
        except Exception:
            pass

    row = HistoricalLog.query.filter_by(user_id=u.user_id, log_date=log_date).first()
    if row is None:
        row = HistoricalLog(
            user_id=u.user_id,
            log_date=log_date,
            generated_kwh=actual,
            predicted_kwh=None,
            consumption_kwh=float(s.daily_units) if s and s.daily_units else 0.0,
            soiling_loss_pct=0.0,
        )
        db.session.add(row)

    was_calibrated = bool(row.is_calibrated)
    row.actual_kwh = round(actual, 3)
    row.generated_kwh = round(actual, 3)
    row.is_calibrated = True
    row.calibrated_at = datetime.now(timezone.utc)
    # Keep original model prediction stable for ratio = actual / predicted
    if row.predicted_kwh is None and was_calibrated is False:
        # if no predicted snapshot, leave None (slope ignores this day)
        pass
    db.session.commit()

    new_slope = _recompute_calibration_slope(u.user_id)
    pred = row.predicted_kwh
    ratio = (actual / float(pred)) if pred and float(pred) > 0.05 else None

    return jsonify({
        "ok": True,
        "log_date": row.log_date,
        "actual_kwh": row.actual_kwh,
        "predicted_kwh": row.predicted_kwh,
        "ratio": round(ratio, 3) if ratio else None,
        "is_calibrated": True,
        "updated": was_calibrated,
        "calibration_slope": new_slope,
        "message": (
            f"{'Updated' if was_calibrated else 'Saved'} actual for {row.log_date}. "
            f"Calibration factor is {new_slope:.3f}× "
            f"(recomputed from all calibrated days — same day is not counted twice)."
        ),
    })


@app.route("/api/uncalibrate_day", methods=["POST"])
@require_login(role="user")
def api_uncalibrate_day():
    """
    Remove calibration for a single day only (keeps the history row).
    Restores displayed energy to predicted and recomputes slope.
    """
    p = request.get_json(silent=True) or {}
    u = current_user_from_token()
    if not u:
        return jsonify({"error": "Unauthorized"}), 401

    log_date = (p.get("log_date") or "").strip()[:10]
    if not log_date:
        return jsonify({"error": "Missing date."}), 400

    row = HistoricalLog.query.filter_by(user_id=u.user_id, log_date=log_date).first()
    if not row:
        return jsonify({"error": "No history row for that date."}), 404
    if not row.is_calibrated and row.actual_kwh is None:
        return jsonify({"error": "This day is not calibrated."}), 400

    row.is_calibrated = False
    row.actual_kwh = None
    row.calibrated_at = None
    if row.predicted_kwh is not None:
        row.generated_kwh = float(row.predicted_kwh)
    db.session.commit()

    new_slope = _recompute_calibration_slope(u.user_id)
    return jsonify({
        "ok": True,
        "log_date": log_date,
        "is_calibrated": False,
        "calibration_slope": new_slope,
        "message": f"Calibration removed for {log_date}. Factor is now {new_slope:.3f}×.",
    })


@app.route("/api/reset_calibration", methods=["POST"])
@require_login(role="user")
def api_reset_calibration():
    """Reset slope to 1.0 and optionally clear calibrated flags / actuals."""
    p = request.get_json(silent=True) or {}
    u = current_user_from_token()
    if not u:
        return jsonify({"error": "Unauthorized"}), 401

    clear_actuals = bool(p.get("clear_actuals", True))
    s = u.system
    if s:
        s.calibration_slope = 1.0

    if clear_actuals:
        rows = HistoricalLog.query.filter_by(user_id=u.user_id, is_calibrated=True).all()
        for r in rows:
            r.is_calibrated = False
            r.actual_kwh = None
            r.calibrated_at = None
            # Restore display energy to predicted if available
            if r.predicted_kwh is not None:
                r.generated_kwh = r.predicted_kwh
    db.session.commit()
    return jsonify({
        "ok": True,
        "calibration_slope": 1.0,
        "message": "Calibration reset. Future forecasts use factor 1.0 again.",
    })


@app.route("/api/calibration_status")
@require_login(role="user")
def api_calibration_status():
    u = current_user_from_token()
    if not u:
        return jsonify({"error": "Unauthorized"}), 401
    s = u.system
    n = HistoricalLog.query.filter_by(user_id=u.user_id, is_calibrated=True).count()
    return jsonify({
        "calibration_slope": float(s.calibration_slope or 1.0) if s else 1.0,
        "calibrated_days": n,
        "is_calibrated": n > 0 and s is not None and abs(float(s.calibration_slope or 1.0) - 1.0) > 0.01,
    })


@app.route("/api/history_export")
@require_login(role="user")
def history_export():
    """CSV export of past history + system setup (no weather recompute)."""
    u = current_user_from_token()
    if not u:
        return jsonify({"error": "Unauthorized"}), 401
    today_s = datetime.now(IST_TZ).date().isoformat()
    rows = (
        HistoricalLog.query.filter_by(user_id=u.user_id)
        .filter(HistoricalLog.log_date < today_s)
        .order_by(HistoricalLog.log_date.desc())
        .all()
    )
    s = u.system
    sys_pin = (s.pincode if s else "") or ""
    sys_state = (s.state if s else "") or ""
    sys_district = (s.district if s else "") or ""
    sys_city = (s.city_locality if s else "") or ""
    sys_lat = s.latitude if s and s.latitude is not None else ""
    sys_lon = s.longitude if s and s.longitude is not None else ""
    sys_panel = (s.brand_model if s else "") or ""
    sys_inv = (s.inverter_model if s else "") or ""
    sys_npan = s.num_panels if s and s.num_panels is not None else ""
    sys_ninv = s.num_inverters if s and s.num_inverters is not None else ""
    sys_tilt = round(float(s.tilt_angle), 2) if s and s.tilt_angle is not None else ""
    sys_az = round(float(s.azimuth_angle), 1) if s and s.azimuth_angle is not None else ""
    sys_year = s.installation_year if s and s.installation_year is not None else ""
    sys_mount = (s.mounting_type if s else "") or ""
    sys_calib = round(float(s.calibration_slope or 1.0), 4) if s else 1.0
    sys_dc_kw = ""
    sys_ac_kw = ""
    try:
        if s and s.brand_model and s.num_panels:
            pw0 = get_panel_specs(s.brand_model)[0]
            sys_dc_kw = round(pw0 * int(s.num_panels) / 1000.0, 2)
        if s and s.inverter_model and s.num_inverters:
            iw0 = get_inverter_specs(s.inverter_model)[0]
            sys_ac_kw = round(iw0 * int(s.num_inverters) / 1000.0, 2)
    except Exception:
        pass

    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "date", "predicted_kwh", "actual_kwh", "consumption_kwh",
        "is_calibrated", "soiling_loss_pct", "carbon_saved_kg", "calibrated_at",
        "pincode", "state", "district", "locality",
        "latitude", "longitude",
        "panel_model", "num_panels", "dc_capacity_kw",
        "inverter_model", "num_inverters", "ac_capacity_kw",
        "tilt_deg", "azimuth_deg", "installation_year", "mounting_type",
        "calibration_slope",
    ])
    for r in rows:
        w.writerow([
            r.log_date,
            r.predicted_kwh if r.predicted_kwh is not None else r.generated_kwh,
            r.actual_kwh if r.actual_kwh is not None else "",
            r.consumption_kwh,
            "yes" if r.is_calibrated else "no",
            r.soiling_loss_pct,
            r.carbon_saved_kg if r.carbon_saved_kg is not None else 0,
            r.calibrated_at.isoformat() if r.calibrated_at else "",
            sys_pin, sys_state, sys_district, sys_city,
            sys_lat, sys_lon,
            sys_panel, sys_npan, sys_dc_kw,
            sys_inv, sys_ninv, sys_ac_kw,
            sys_tilt, sys_az, sys_year, sys_mount,
            sys_calib,
        ])
    from flask import Response
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=solarflow_up_history.csv"
        },
    )


@app.route("/admin")
@require_login(role="admin")
def admin_portal():
    token = _extract_request_token()
    u = current_user_from_token(token)
    state = request.args.get("state", "") or UP_STATE_NAME
    district = request.args.get("district", "")
    if state and not is_up_state(state):
        state = UP_STATE_NAME
    page = request.args.get("page", 1, type=int)
    q = db.session.query(SolarSystem, ForecastLog, User).outerjoin(
        ForecastLog, SolarSystem.user_id == ForecastLog.user_id
    ).join(User, SolarSystem.user_id == User.user_id)
    if state:
        q = q.filter(SolarSystem.state == state)
    if district:
        q = q.filter(SolarSystem.district.ilike(f"%{district}%"))
    pag = q.paginate(page=page, per_page=100, error_out=False)
    fleet = []
    total_cap = 0.0
    total_ac = 0.0
    for s, f, usr in pag.items:
        pw = 0.0
        ac = 0.0
        try:
            if s.brand_model and s.num_panels:
                pw = float(get_panel_specs(s.brand_model)[0]) * int(s.num_panels or 0) / 1000.0
                total_cap += pw
        except Exception:
            pass
        try:
            if s.inverter_model and s.num_inverters:
                ac = float(get_inverter_specs(s.inverter_model)[0]) * int(s.num_inverters or 0) / 1000.0
                total_ac += ac
        except Exception:
            pass
        n_cal = HistoricalLog.query.filter_by(user_id=s.user_id, is_calibrated=True).count()
        lat = s.latitude
        lon = s.longitude
        coords = (
            f"{float(lat):.4f}, {float(lon):.4f}"
            if lat is not None and lon is not None else "—"
        )
        tilt_s = round(float(s.tilt_angle), 2) if s.tilt_angle is not None else "—"
        az_s = round(float(s.azimuth_angle), 1) if s.azimuth_angle is not None else "—"
        fleet.append({
            "user_name": usr.name or "",
            "email": usr.email or "",
            "location": f"{s.city_locality or ''} ({s.pincode or 'No PIN'}), {s.district or ''}",
            "coords": coords,
            "hardware": f"{s.num_panels or 0}× {s.brand_model or '—'}",
            "inverter": f"{s.num_inverters or 0}× {s.inverter_model or '—'}",
            "dc_kw": round(pw, 2),
            "ac_kw": round(ac, 2),
            "tilt": tilt_s,
            "azimuth": az_s,
            "load_kwh": round(float(s.daily_units or 0), 1),
            "year": s.installation_year or "—",
            "calibration_slope": round(float(s.calibration_slope or 1.0), 3),
            "calibrated_days": n_cal,
            "today_gen": round(f.today_gen_kwh if f else 0, 2),
            "gen_7d": round(f.predicted_gen_kwh if f else 0, 2),
            "today_carbon": round(f.today_carbon_saved_kg if f else 0, 2),
            "carbon_saved": round(f.carbon_saved_kg if f else 0, 2),
            "last_updated": f.last_updated.strftime("%b %d, %Y %H:%M") if f and f.last_updated else "Pending",
        })

    # Model manifest (if trained)
    model_info = {"available": False, "r2": None, "mae": None, "path": "models/up/modelfile.pkl"}
    try:
        import json
        from pathlib import Path
        man = Path(__file__).resolve().parent / "models" / "up" / "training_manifest.json"
        pkl = Path(__file__).resolve().parent / "models" / "up" / "modelfile.pkl"
        model_info["available"] = pkl.is_file()
        if man.is_file():
            m = json.loads(man.read_text(encoding="utf-8"))
            metrics = m.get("model_metrics") or m.get("metrics") or {}
            model_info["r2"] = metrics.get("r2")
            model_info["mae"] = metrics.get("mae_5kw_equivalent") or metrics.get("mae")
            model_info["note"] = (m.get("note") or "")[:180]
    except Exception:
        pass

    n_users = User.query.filter_by(is_admin=False).count()
    n_cal_users = (
        db.session.query(HistoricalLog.user_id)
        .filter(HistoricalLog.is_calibrated.is_(True))
        .distinct()
        .count()
    )
    states = [UP_STATE_NAME]
    up_districts = district_list_for_ui()
    return render_template(
        "admin.html",
        fleet=fleet,
        pagination=pag,
        unique_states=states,
        up_districts=up_districts,
        selected_state=state or UP_STATE_NAME,
        selected_district=district,
        total_users=q.count(),
        total_capacity=round(total_cap, 1),
        total_ac_capacity=round(total_ac, 1),
        total_today=sum(x["today_gen"] for x in fleet),
        total_generation=sum(x["gen_7d"] for x in fleet),
        total_today_carbon=sum(x["today_carbon"] for x in fleet),
        total_7d_carbon=sum(x["carbon_saved"] for x in fleet),
        n_registered_users=n_users,
        n_calibrated_users=n_cal_users,
        model_info=model_info,
        admin=u,
        token=token,
    )


if __name__ == "__main__":
    # use_reloader=False + threaded: avoids auto-exit / restart loops on Windows
    port = int(os.getenv("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        use_reloader=False,
        threaded=True,
    )
