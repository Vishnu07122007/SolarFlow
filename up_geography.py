# up_geography.py
# SolarFlow — Uttar Pradesh only geography helpers
# Platform is intentionally limited to UP until regional accuracy is proven.

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

# Canonical state name used everywhere in DB / API / models
UP_STATE_NAME = "Uttar Pradesh"
UP_STATE_ALIASES = {
    "uttar pradesh",
    "u.p.",
    "u.p",
    "up",
    "uttarpradesh",
}

# Official / commonly used district names (normalized lookup uses lower/strip)
UP_DISTRICTS: List[str] = [
    "Agra", "Aligarh", "Ambedkar Nagar", "Amethi", "Amroha", "Auraiya",
    "Ayodhya", "Azamgarh", "Bagpat", "Bahraich", "Ballia", "Balrampur",
    "Banda", "Barabanki", "Bareilly", "Basti", "Bhadohi", "Bijnor",
    "Budaun", "Bulandshahr", "Chandauli", "Chitrakoot", "Deoria", "Etah",
    "Etawah", "Farrukhabad", "Fatehpur", "Firozabad", "Gautam Buddha Nagar",
    "Ghaziabad", "Ghazipur", "Gonda", "Gorakhpur", "Hamirpur", "Hapur",
    "Hardoi", "Hathras", "Jalaun", "Jaunpur", "Jhansi", "Kannauj",
    "Kanpur Dehat", "Kanpur Nagar", "Kasganj", "Kaushambi", "Kheri",
    "Kushinagar", "Lalitpur", "Lucknow", "Maharajganj", "Mahoba", "Mainpuri",
    "Mathura", "Mau", "Meerut", "Mirzapur", "Moradabad", "Muzaffarnagar",
    "Pilibhit", "Pratapgarh", "Prayagraj", "Raebareli", "Rampur",
    "Saharanpur", "Sambhal", "Sant Kabir Nagar", "Shahjahanpur", "Shamli",
    "Shrawasti", "Siddharthnagar", "Sitapur", "Sonbhadra", "Sultanpur",
    "Unnao", "Varanasi",
]

# Representative training / coverage cities inside UP (lat, lon)
# Dense enough for a single robust UP model; expand later if needed.
UP_TRAINING_LOCATIONS: List[Tuple[str, float, float]] = [
    ("Lucknow", 26.8467, 80.9462),
    ("Kanpur", 26.4499, 80.3319),
    ("Varanasi", 25.3176, 82.9739),
    ("Prayagraj", 25.4358, 81.8463),
    ("Agra", 27.1767, 78.0081),
    ("Meerut", 28.9845, 77.7064),
    ("Ghaziabad", 28.6692, 77.4538),
    ("Noida", 28.5355, 77.3910),
    ("Bareilly", 28.3670, 79.4304),
    ("Moradabad", 28.8386, 78.7733),
    ("Aligarh", 27.8974, 78.0880),
    ("Saharanpur", 29.9680, 77.5553),
    ("Gorakhpur", 26.7606, 83.3732),
    ("Jhansi", 25.4484, 78.5685),
    ("Firozabad", 27.1591, 78.3957),
    ("Muzaffarnagar", 29.4727, 77.7085),
    ("Mathura", 27.4924, 77.6737),
    ("Ayodhya", 26.7922, 82.1998),
    ("Sitapur", 27.5705, 80.6780),
    ("Unnao", 26.5471, 80.4878),
    ("Raebareli", 26.2341, 81.2400),
    ("Sultanpur", 26.2648, 82.0730),
    ("Jaunpur", 25.7460, 82.6837),
    ("Azamgarh", 26.0684, 83.1836),
    ("Ballia", 25.7585, 84.1480),
    ("Mirzapur", 25.1460, 82.5690),
    ("Sonbhadra", 24.4675, 83.0948),
    ("Lalitpur", 24.6900, 78.4190),
    ("Hardoi", 27.3943, 80.1311),
    ("Shahjahanpur", 27.8800, 79.9100),
    ("Pilibhit", 28.6300, 79.8000),
    ("Budaun", 28.0400, 79.1200),
    ("Etawah", 26.7755, 79.0150),
    ("Mainpuri", 27.2300, 79.0200),
    ("Farrukhabad", 27.3900, 79.5800),
    ("Kannauj", 27.0500, 79.9200),
    ("Fatehpur", 25.9300, 80.8100),
    ("Banda", 25.4800, 80.3400),
    ("Hamirpur", 25.9500, 80.1500),
    ("Mahoba", 25.2900, 79.8700),
    ("Chitrakoot", 25.2000, 80.9000),
    ("Kaushambi", 25.3400, 81.3900),
    ("Pratapgarh", 25.9000, 81.9900),
    ("Amethi", 26.1600, 81.8100),
    ("Barabanki", 26.9400, 81.1900),
    ("Gonda", 27.1300, 81.9600),
    ("Bahraich", 27.5700, 81.6000),
    ("Balrampur", 27.4300, 82.1800),
    ("Shrawasti", 27.5000, 82.0500),
    ("Siddharthnagar", 27.2500, 82.8200),
    ("Basti", 26.8000, 82.7400),
    ("Sant Kabir Nagar", 26.7800, 83.0700),
    ("Maharajganj", 27.1800, 83.5700),
    ("Kushinagar", 26.7400, 83.8900),
    ("Deoria", 26.5000, 83.7800),
    ("Mau", 25.9400, 83.5600),
    ("Ghazipur", 25.5800, 83.5800),
    ("Chandauli", 25.2600, 83.2700),
    ("Bhadohi", 25.3900, 82.5700),
    ("Hapur", 28.7300, 77.7800),
    ("Bulandshahr", 28.4000, 77.8500),
    ("Gautam Buddha Nagar", 28.5355, 77.3910),
    ("Bagpat", 28.9400, 77.2200),
    ("Shamli", 29.4500, 77.3100),
    ("Bijnor", 29.3700, 78.1300),
    ("Amroha", 28.9000, 78.4700),
    ("Sambhal", 28.5800, 78.5500),
    ("Rampur", 28.8000, 79.0300),
    ("Kasganj", 27.8000, 78.6500),
    ("Etah", 27.5600, 78.6700),
    ("Hathras", 27.6000, 78.0500),
    ("Auraiya", 26.4700, 79.5100),
    ("Jalaun", 26.1400, 79.3400),
    ("Kanpur Dehat", 26.4500, 79.9500),
    ("Kheri", 27.9500, 80.7800),
]


def is_up_state(name: Optional[str]) -> bool:
    if not name:
        return False
    key = str(name).strip().lower().replace(".", "")
    key = " ".join(key.split())
    if key in UP_STATE_ALIASES or key == "uttar pradesh":
        return True
    return key.replace(" ", "") == "uttarpradesh"


def normalize_state(name: Optional[str]) -> Optional[str]:
    return UP_STATE_NAME if is_up_state(name) else (str(name).strip() if name else None)


def normalize_district(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    raw = str(name).strip()
    key = " ".join(raw.lower().split())
    # Common aliases
    aliases = {
        "allahabad": "Prayagraj",
        "faizabad": "Ayodhya",
        "jyotiba phule nagar": "Amroha",
        "panchsheel nagar": "Hapur",
        "bhim nagar": "Sambhal",
        "prabuddh nagar": "Shamli",
        "sant ravidas nagar": "Bhadohi",
        "kashi": "Varanasi",
        "benares": "Varanasi",
        "gautam budh nagar": "Gautam Buddha Nagar",
        "gb nagar": "Gautam Buddha Nagar",
        "lakhimpur kheri": "Kheri",
        "lakhimpur": "Kheri",
        "raebareli": "Raebareli",
        "rae bareli": "Raebareli",
        "siddharth nagar": "Siddharthnagar",
        "sidharthnagar": "Siddharthnagar",
    }
    if key in aliases:
        return aliases[key]
    for d in UP_DISTRICTS:
        if d.lower() == key:
            return d
    # Fuzzy contains
    for d in UP_DISTRICTS:
        if key in d.lower() or d.lower() in key:
            return d
    return raw  # keep Postal name even if not in list


def is_up_district(name: Optional[str]) -> bool:
    n = normalize_district(name)
    if not n:
        return False
    return n in UP_DISTRICTS or any(n.lower() == d.lower() for d in UP_DISTRICTS)


# UP pincode ranges (approx). Used as a fast pre-filter; Postal API is authoritative.
# Most UP PINs are 20xxxx–28xxxx; a few edge cases exist.
def pincode_looks_like_up(pin: str) -> bool:
    pin = (pin or "").strip()
    if len(pin) != 6 or not pin.isdigit():
        return False
    prefix = int(pin[:2])
    # Primary UP series
    if 20 <= prefix <= 28:
        return True
    return False


# Approximate geographic envelope of Uttar Pradesh (small margin for border pins).
UP_BBOX = {
    "south": 23.85,
    "north": 30.45,
    "west": 77.05,
    "east": 84.75,
}


def is_inside_up(lat: float, lon: float, margin: float = 0.05) -> bool:
    """True if coordinates fall inside UP envelope."""
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return False
    return (
        (UP_BBOX["south"] - margin) <= lat_f <= (UP_BBOX["north"] + margin)
        and (UP_BBOX["west"] - margin) <= lon_f <= (UP_BBOX["east"] + margin)
    )


def reject_non_up_message(state: Optional[str] = None, pin: Optional[str] = None) -> str:
    return (
        "SolarFlow is currently available only for Uttar Pradesh. "
        "Enter a UP pincode (e.g. Lucknow 226001, Kanpur 208001, Varanasi 221001). "
    )


def reject_outside_up_coords_message() -> str:
    return (
        "Selected location is outside Uttar Pradesh. "
        "Move the marker inside UP (and inside your verified pincode area)."
    )


def district_list_for_ui() -> List[str]:
    return sorted(UP_DISTRICTS)


def training_location_dicts() -> List[Dict[str, float]]:
    return [
        {"city": c, "lat": lat, "lon": lon, "state": UP_STATE_NAME}
        for c, lat, lon in UP_TRAINING_LOCATIONS
    ]
