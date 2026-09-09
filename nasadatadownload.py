#!/usr/bin/env python3
"""
NASA POWER hourly downloader for Uttar Pradesh (SolarFlow UP).

Saves CSVs under:
  data/nasa/up/<district_or_city>/<slug>_lat_lon.csv

Usage:
  python nasadatadownload.py
  python nasadatadownload.py --start 20210101 --end 20251231
  python nasadatadownload.py --only lucknow,varanasi,adwa_mirzapur
  python nasadatadownload.py --force   # re-download even if file exists

Notes:
  - Free public API, no key required.
  - Built-in delay + retries to avoid rate limits.
  - Skips existing files unless --force.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Points: Tier A (your UP SW solar stations) + Tier B (main cities)
# name used for folder = district / city key (lowercase, underscores)
# ---------------------------------------------------------------------------
SITES: List[Dict] = [
    # Tier A – match UP telemetry solar stations
    {"key": "adwa_mirzapur", "district": "mirzapur", "lat": 24.7695, "lon": 82.3303, "label": "ADWA AWS"},
    {"key": "jirgo_mirzapur", "district": "mirzapur", "lat": 25.0510, "lon": 82.9381, "label": "Jirgo AWS"},
    {"key": "moosakhand_mirzapur", "district": "mirzapur", "lat": 24.9780, "lon": 83.2993, "label": "Moosakhand AWS"},
    {"key": "obra_mirzapur", "district": "mirzapur", "lat": 24.4390, "lon": 82.9645, "label": "OBRA AWS"},
    # Tier B – major UP cities / districts
    {"key": "lucknow", "district": "lucknow", "lat": 26.8467, "lon": 80.9462, "label": "Lucknow"},
    {"key": "kanpur", "district": "kanpur_nagar", "lat": 26.4499, "lon": 80.3319, "label": "Kanpur"},
    {"key": "varanasi", "district": "varanasi", "lat": 25.3176, "lon": 82.9739, "label": "Varanasi"},
    {"key": "prayagraj", "district": "prayagraj", "lat": 25.4358, "lon": 81.8463, "label": "Prayagraj"},
    {"key": "agra", "district": "agra", "lat": 27.1767, "lon": 78.0081, "label": "Agra"},
    {"key": "meerut", "district": "meerut", "lat": 28.9845, "lon": 77.7064, "label": "Meerut"},
    {"key": "ghaziabad", "district": "ghaziabad", "lat": 28.6692, "lon": 77.4538, "label": "Ghaziabad/Noida"},
    {"key": "bareilly", "district": "bareilly", "lat": 28.3670, "lon": 79.4304, "label": "Bareilly"},
    {"key": "gorakhpur", "district": "gorakhpur", "lat": 26.7606, "lon": 83.3732, "label": "Gorakhpur"},
    {"key": "jhansi", "district": "jhansi", "lat": 25.4484, "lon": 78.5685, "label": "Jhansi"},
    {"key": "moradabad", "district": "moradabad", "lat": 28.8386, "lon": 78.7733, "label": "Moradabad"},
    {"key": "aligarh", "district": "aligarh", "lat": 27.8974, "lon": 78.0880, "label": "Aligarh"},
    {"key": "saharanpur", "district": "saharanpur", "lat": 29.9680, "lon": 77.5553, "label": "Saharanpur"},
    {"key": "ayodhya", "district": "ayodhya", "lat": 26.7922, "lon": 82.1998, "label": "Ayodhya"},
    # Extra useful districts
    {"key": "mathura", "district": "mathura", "lat": 27.4924, "lon": 77.6737, "label": "Mathura"},
    {"key": "firozabad", "district": "firozabad", "lat": 27.1591, "lon": 78.3958, "label": "Firozabad"},
    {"key": "sitapur", "district": "sitapur", "lat": 27.5706, "lon": 80.6780, "label": "Sitapur"},
    {"key": "unnao", "district": "unnao", "lat": 26.5464, "lon": 80.4870, "label": "Unnao"},
    {"key": "raebareli", "district": "raebareli", "lat": 26.2340, "lon": 81.2400, "label": "Raebareli"},
    {"key": "sultanpur", "district": "sultanpur", "lat": 26.2570, "lon": 82.0727, "label": "Sultanpur"},
    {"key": "jaunpur", "district": "jaunpur", "lat": 25.7530, "lon": 82.6860, "label": "Jaunpur"},
    {"key": "azamgarh", "district": "azamgarh", "lat": 26.0680, "lon": 83.1840, "label": "Azamgarh"},
    {"key": "ballia", "district": "ballia", "lat": 25.7600, "lon": 84.1470, "label": "Ballia"},
    {"key": "sonbhadra", "district": "sonbhadra", "lat": 24.4680, "lon": 83.0940, "label": "Sonbhadra"},
    {"key": "lalitpur", "district": "lalitpur", "lat": 24.6900, "lon": 78.4190, "label": "Lalitpur"},
    {"key": "hardoi", "district": "hardoi", "lat": 27.3940, "lon": 80.1310, "label": "Hardoi"},
    {"key": "shahjahanpur", "district": "shahjahanpur", "lat": 27.8830, "lon": 79.9110, "label": "Shahjahanpur"},
    {"key": "muzaffarnagar", "district": "muzaffarnagar", "lat": 29.4730, "lon": 77.7040, "label": "Muzaffarnagar"},
    {"key": "bijnor", "district": "bijnor", "lat": 29.3720, "lon": 78.1360, "label": "Bijnor"},
    {"key": "etah", "district": "etah", "lat": 27.5600, "lon": 78.6630, "label": "Etah"},
    {"key": "farrukhabad", "district": "farrukhabad", "lat": 27.3880, "lon": 79.5810, "label": "Farrukhabad"},
    {"key": "mainpuri", "district": "mainpuri", "lat": 27.2280, "lon": 79.0250, "label": "Mainpuri"},
]

# NASA POWER hourly parameters (Renewable Energy community)
# ALLSKY_SFC_SW_DWN = GHI (W/m2)
# ALLSKY_SFC_SW_DNI / DIFF may be missing for some hours → API still returns rest
PARAMS = ",".join(
    [
        "ALLSKY_SFC_SW_DWN",  # GHI
        "ALLSKY_SFC_SW_DNI",  # DNI
        "ALLSKY_SFC_SW_DIFF", # DHI
        "T2M",                # temp C
        "RH2M",               # relative humidity %
        "WS10M",              # wind m/s
        "PRECTOTCORR",        # precip mm
        "PS",                 # pressure kPa
    ]
)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
DEFAULT_START = "20210101"
DEFAULT_END = "20251231"
SLEEP_SEC = 2.5          # polite delay between points
MAX_RETRIES = 4
TIMEOUT = 120


def out_path(root: Path, site: Dict) -> Path:
    district = site["district"].strip().lower().replace(" ", "_")
    fname = f"{site['key']}_{site['lat']:.4f}_{site['lon']:.4f}.csv"
    return root / "data" / "nasa" / "up" / district / fname


def build_url(lat: float, lon: float, start: str, end: str) -> str:
    # format=CSV returns header + rows; community=RE is renewable energy
    q = (
        f"{BASE_URL}"
        f"?parameters={PARAMS}"
        f"&community=RE"
        f"&longitude={lon:.4f}"
        f"&latitude={lat:.4f}"
        f"&start={start}"
        f"&end={end}"
        f"&format=CSV"
    )
    return q


def download_one(
    site: Dict,
    start: str,
    end: str,
    root: Path,
    force: bool,
    session: requests.Session,
) -> Tuple[str, str]:
    """Returns (status, message). status in ok|skip|fail."""
    path = out_path(root, site)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 500 and not force:
        return "skip", f"exists ({path.stat().st_size // 1024} KB)"

    url = build_url(site["lat"], site["lon"], start, end)
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 429:
                wait = 15 * attempt
                print(f"    rate-limited, sleep {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                time.sleep(5 * attempt)
                continue
            r.raise_for_status()
            text = r.text
            # NASA CSV has metadata lines before the actual header
            if "ALLSKY_SFC_SW_DWN" not in text and "YEAR" not in text:
                last_err = "unexpected response body"
                time.sleep(3)
                continue
            path.write_text(text, encoding="utf-8")
            kb = path.stat().st_size // 1024
            return "ok", f"saved {kb} KB → {path}"
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(4 * attempt)
    return "fail", last_err


def main() -> int:
    ap = argparse.ArgumentParser(description="Download NASA POWER hourly data for UP")
    ap.add_argument("--start", default=DEFAULT_START, help="YYYYMMDD (default 20210101)")
    ap.add_argument("--end", default=DEFAULT_END, help="YYYYMMDD (default 20251231)")
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated site keys to download (default: all)",
    )
    ap.add_argument("--force", action="store_true", help="Re-download even if file exists")
    ap.add_argument(
        "--root",
        default=".",
        help="Project root (default: current dir). Files go to <root>/data/nasa/up/...",
    )
    ap.add_argument("--sleep", type=float, default=SLEEP_SEC, help="Seconds between sites")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    only = {x.strip().lower() for x in args.only.split(",") if x.strip()}

    sites = SITES
    if only:
        sites = [s for s in SITES if s["key"] in only]
        missing = only - {s["key"] for s in sites}
        if missing:
            print("Unknown keys (ignored):", ", ".join(sorted(missing)))
        if not sites:
            print("No matching sites. Available keys:")
            for s in SITES:
                print(f"  {s['key']:25s}  {s['district']:18s}  {s['lat']},{s['lon']}")
            return 1

    print("=" * 60)
    print("NASA POWER hourly downloader — Uttar Pradesh")
    print(f"  period : {args.start} → {args.end}")
    print(f"  sites  : {len(sites)}")
    print(f"  output : {root / 'data' / 'nasa' / 'up'}")
    print(f"  force  : {args.force}")
    print("=" * 60)

    session = requests.Session()
    session.headers.update({"User-Agent": "SolarFlow-UP/1.0 (research; contact local)"})

    ok = skip = fail = 0
    failed: List[str] = []

    for i, site in enumerate(sites, 1):
        print(f"\n[{i}/{len(sites)}] {site['label']}  ({site['key']})  {site['lat']},{site['lon']}")
        status, msg = download_one(site, args.start, args.end, root, args.force, session)
        print(f"    {status.upper()}: {msg}")
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            fail += 1
            failed.append(site["key"])
        if i < len(sites):
            time.sleep(args.sleep)

    print("\n" + "=" * 60)
    print(f"Done.  ok={ok}  skip={skip}  fail={fail}")
    if failed:
        print("Failed keys (re-run with --only):")
        print("  " + ",".join(failed))
        print("Example:")
        print(f"  python nasadatadownload.py --only {','.join(failed)}")
    print(f"Folder: {root / 'data' / 'nasa' / 'up'}")
    print("=" * 60)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
