"""
Generate the bundled sample scene for OILTRACE.

Writes four files into backend/sample_data/:
  scene.tif        1000x1000 greyscale GeoTIFF over REGION_BOUNDS
  wind.json        hourly wind for the 24 h before the scene
  ships.json       6 fake vessels with timestamped tracks
  scene_meta.json  scene id, acquisition time, bounds

Everything is seeded so the fixture is identical on every run. The one
"culprit" vessel is placed by replaying the exact same backward-drift model
that rewind.py uses, so the pipeline genuinely re-discovers it rather than us
hand-waving a match.

Run:  python -m scripts.make_sample_data   (from backend/)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# Allow "python scripts/make_sample_data.py" as well as module form.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

RNG = np.random.default_rng(config.RANDOM_SEED)

MIN_LAT, MIN_LON, MAX_LAT, MAX_LON = config.REGION_BOUNDS
WIDTH = HEIGHT = 1000

SCENE_TIME = datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
SCENE_ID = "S1A_OILTRACE_20260910T1000"

# Constant wind direction keeps the fixture predictable. Wind blows FROM the
# south-west (225 deg), which pushes the slick toward the north-east — exactly
# the direction the drawn streak points.
WIND_FROM_DEG = 225.0


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- geo helpers ------------------------------------------------------------

def step_latlon(lat: float, lon: float, bearing_deg: float, dist_m: float):
    """Move a point dist_m metres along bearing_deg (0=N, clockwise)."""
    b = math.radians(bearing_deg)
    dlat = (dist_m * math.cos(b)) / 111_320.0
    dlon = (dist_m * math.sin(b)) / (111_320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


# --- 1. scene.tif -----------------------------------------------------------

def make_scene_tif(path: Path):
    # Bright sea with mild speckle; oil and look-alikes are dark.
    SEA, SEA_NOISE = 200, 12
    img = RNG.normal(SEA, SEA_NOISE, size=(HEIGHT, WIDTH))
    img = np.clip(img, 0, 255).astype(np.uint8)

    # OpenCV is only needed to draw here; import locally so the generator has
    # the same dependency surface as the detector.
    import cv2

    # One dark linear streak ~40 px wide running north-east. In image space
    # north-east is up (smaller row) and right (larger col): draw from the
    # lower-left toward the upper-right, centred on the image middle so its
    # centroid sits at the centre of REGION_BOUNDS.
    cv2.line(img, (350, 650), (650, 350), color=40, thickness=40, lineType=cv2.LINE_AA)

    # Two compact dark blobs elsewhere. They are NOT oil — look-alikes. Kept
    # deliberately small so the detector's minimum-area rule rejects them.
    cv2.circle(img, (200, 820), 16, color=55, thickness=-1)   # bottom-left
    cv2.circle(img, (830, 180), 15, color=50, thickness=-1)   # top-right

    transform = from_bounds(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, WIDTH, HEIGHT)
    profile = {
        "driver": "GTiff",
        "height": HEIGHT,
        "width": WIDTH,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(img, 1)
    return transform


# --- 2. wind.json -----------------------------------------------------------

def make_wind():
    """Hourly wind for the 24 h before the scene, speed kept in 5-8 m/s."""
    samples = []
    for h in range(24, -1, -1):  # 24 h before .. scene time, inclusive
        t = SCENE_TIME - timedelta(hours=h)
        # Smooth-ish speed in [5, 8].
        speed = 6.5 + 1.3 * math.sin(h / 3.0) + RNG.normal(0, 0.15)
        speed = float(np.clip(speed, 5.0, 8.0))
        samples.append({"time": iso(t), "speed_ms": round(speed, 2),
                        "dir_deg": WIND_FROM_DEG})
    return samples


# --- 3. ships.json ----------------------------------------------------------

def backtrack_center(slick_lat, slick_lon, wind_samples):
    """Replay rewind.py's model on the slick centroid to find the release
    centre, so the culprit vessel can be planted exactly there."""
    lat, lon = slick_lat, slick_lon
    # The SLICK_AGE_HOURS wind samples immediately before the scene.
    recent = wind_samples[-(config.SLICK_AGE_HOURS + 1):-1]
    for w in reversed(recent):
        dist = config.WIND_DRIFT_FACTOR * w["speed_ms"] * 3600.0
        # Backward step: move upwind, i.e. toward where the wind comes FROM.
        lat, lon = step_latlon(lat, lon, w["dir_deg"], dist)
    return lat, lon


def track(mmsi, name, start_lat, start_lon, bearing, speed_ms, gap_after=None):
    """Build a vessel reporting every 20 minutes across the 24 h window.

    gap_after: if set, drop reports for 40 minutes starting at this fraction
    (0-1) through the window, simulating a dark-vessel AIS gap.
    """
    positions = []
    lat, lon = start_lat, start_lon
    total_min = 24 * 60
    step_min = 20
    gap_start = None if gap_after is None else int(gap_after * total_min)
    minute = 0
    while minute <= total_min:
        t = SCENE_TIME - timedelta(hours=24) + timedelta(minutes=minute)
        in_gap = gap_start is not None and gap_start <= minute < gap_start + 40
        if not in_gap:
            positions.append({"time": iso(t), "lat": round(lat, 5), "lon": round(lon, 5)})
        dist = speed_ms * step_min * 60.0
        lat, lon = step_latlon(lat, lon, bearing, dist)
        minute += step_min
    return {"mmsi": mmsi, "name": name, "positions": positions}


def make_ships(release_lat, release_lon):
    ships = []

    # Ship A — the culprit. Placed so that at the release time (scene - 12 h)
    # it sits at the release centre, travelling north-east (bearing ~45), the
    # same direction as the slick's long axis.
    culprit_bearing = 45.0
    culprit_speed = 4.0  # m/s
    # Wind back from release time to the start of the window (12 h) so the
    # track passes through the release centre at the right moment.
    a_lat, a_lon = release_lat, release_lon
    for _ in range(int(12 * 60 / 20)):
        a_lat, a_lon = step_latlon(a_lat, a_lon, culprit_bearing + 180, culprit_speed * 20 * 60)
    ships.append(track("219000001", "Nordfjord", a_lat, a_lon, culprit_bearing, culprit_speed))

    # Ship B — dark vessel. Passes near the release area (so it becomes a
    # candidate) but on a wrong heading (~135, south-east) and with a 40-min
    # AIS reporting gap.
    b_lat, b_lon = release_lat - 0.02, release_lon - 0.02
    for _ in range(int(11 * 60 / 20)):
        b_lat, b_lon = step_latlon(b_lat, b_lon, 135.0 + 180, 3.0 * 20 * 60)
    ships.append(track("235000045", "Kattegat Star", b_lat, b_lon, 135.0, 3.0, gap_after=0.5))

    # Ships C-F — genuinely elsewhere or on unrelated headings.
    ships.append(track("311000078", "Baltic Trader", MIN_LAT + 0.3, MIN_LON + 0.3, 90.0, 5.0))
    ships.append(track("257000112", "Skagen Pride", MAX_LAT - 0.2, MAX_LON - 0.4, 270.0, 4.5))
    ships.append(track("244000203", "Fyn Runner", MIN_LAT + 0.5, MAX_LON - 0.3, 315.0, 5.5))
    ships.append(track("265000330", "Oresund Bell", MAX_LAT - 0.4, MIN_LON + 0.5, 180.0, 3.5))

    return ships


# --- 4. scene_meta.json -----------------------------------------------------

def make_scene_meta():
    return {
        "scene_id": SCENE_ID,
        "acquired_at": iso(SCENE_TIME),
        "bounds": {
            "min_lat": MIN_LAT, "min_lon": MIN_LON,
            "max_lat": MAX_LAT, "max_lon": MAX_LON,
        },
    }


def main():
    out = config.SAMPLE_DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    make_scene_tif(config.SCENE_TIF)

    wind = make_wind()
    config.WIND_JSON.write_text(json.dumps(wind, indent=2))

    # Slick centroid is the centre of the region (streak is drawn centred).
    slick_lat = (MIN_LAT + MAX_LAT) / 2.0
    slick_lon = (MIN_LON + MAX_LON) / 2.0
    release_lat, release_lon = backtrack_center(slick_lat, slick_lon, wind)

    ships = make_ships(release_lat, release_lon)
    config.SHIPS_JSON.write_text(json.dumps(ships, indent=2))

    config.SCENE_META_JSON.write_text(json.dumps(make_scene_meta(), indent=2))

    print(f"Wrote sample data to {out}")
    print(f"  scene centroid   ~ ({slick_lat:.4f}, {slick_lon:.4f})")
    print(f"  release centre   ~ ({release_lat:.4f}, {release_lon:.4f})")
    print(f"  vessels: {len(ships)}  wind samples: {len(wind)}")


if __name__ == "__main__":
    main()
