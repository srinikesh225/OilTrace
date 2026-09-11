"""
Generate the bundled sample scenes for OILTRACE.

Writes three self-contained fixture directories under backend/sample_data/:

  scene_normal/     the original confident-attribution demo (one clear suspect)
  scene_ambiguous/  scene_normal + a third, genuinely plausible vessel whose
                    score lands close to the top candidate, so the candidate-
                    separation rule refuses to rank one above the other
  scene_calm/       scene_normal with ~1.8 m/s wind, so the existing wind gate
                    rejects the slick before attribution

Each directory holds scene.tif, wind.json, ships.json, scene_meta.json.

Everything is seeded (seed 42) so the fixtures are identical on every run. The
"culprit" vessel is placed by replaying the exact backward-drift model that
rewind.py uses, so the pipeline genuinely re-discovers it. The ambiguous
scene's third vessel is placed the same honest way — a plausible straight
track through the release area on a heading close to (but not identical to) the
slick axis; its score is whatever the existing scorer computes, never tuned.

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

MIN_LAT, MIN_LON, MAX_LAT, MAX_LON = config.REGION_BOUNDS
WIDTH = HEIGHT = 1000

SCENE_TIME = datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
SCENE_ID_NORMAL = "S1A_OILTRACE_20260910T1000"
SCENE_ID_AMBIGUOUS = "S1A_OILTRACE_20260910T1000_AMBIG"
SCENE_ID_CALM = "S1A_OILTRACE_20260910T1000_CALM"

# Constant wind direction keeps the fixture predictable. Wind blows FROM the
# south-west (225 deg), which pushes the slick toward the north-east — exactly
# the direction the drawn streak points.
WIND_FROM_DEG = 225.0

# Calm-scene wind speed (m/s). Below config.WIND_MIN_MS so the existing wind
# gate rejects the slick. We do not touch the gate — only the wind.
CALM_WIND_MS = 1.8

# --- Third vessel (ambiguous scene) -----------------------------------------
# The detected slick's long axis is ~33.6 deg. The third vessel travels on a
# heading close to that axis but not identical, and passes through the release
# area at the release time. Its 22:00 position is offset perpendicular to the
# axis so it sits partway (not maximally) inside the release polygon, which
# naturally lowers its spatial score just enough to open a small gap. These are
# trajectory parameters only; the score is the scorer's own output.
THIRD_HEADING_DEG = 50.0            # ~16 deg off the slick axis — close, not identical
THIRD_PERP_BEARING_DEG = 123.6      # slick axis (33.6) + 90
THIRD_PERP_OFFSET_M = 2500.0        # metres offset from the release centre
THIRD_SPEED_MS = 4.0


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- geo helpers ------------------------------------------------------------

def step_latlon(lat: float, lon: float, bearing_deg: float, dist_m: float):
    """Move a point dist_m metres along bearing_deg (0=N, clockwise)."""
    b = math.radians(bearing_deg)
    dlat = (dist_m * math.cos(b)) / 111_320.0
    dlon = (dist_m * math.sin(b)) / (111_320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


# --- scene image ------------------------------------------------------------

def build_scene_image(rng):
    """Return the 1000x1000 greyscale array and its geotransform.

    Identical draw order to the original generator so scene_normal reproduces
    the existing fixture byte for byte."""
    SEA, SEA_NOISE = 200, 12
    img = rng.normal(SEA, SEA_NOISE, size=(HEIGHT, WIDTH))
    img = np.clip(img, 0, 255).astype(np.uint8)

    import cv2  # local import: same dependency surface as the detector

    # One dark linear streak ~40 px wide running north-east, centred so its
    # centroid sits at the centre of REGION_BOUNDS.
    cv2.line(img, (350, 650), (650, 350), color=40, thickness=40, lineType=cv2.LINE_AA)

    # Two compact dark blobs — look-alikes small enough to be rejected on area.
    cv2.circle(img, (200, 820), 16, color=55, thickness=-1)
    cv2.circle(img, (830, 180), 15, color=50, thickness=-1)

    transform = from_bounds(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, WIDTH, HEIGHT)
    return img, transform


def write_tif(path: Path, img, transform):
    profile = {
        "driver": "GTiff", "height": HEIGHT, "width": WIDTH, "count": 1,
        "dtype": "uint8", "crs": "EPSG:4326", "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(img, 1)


# --- wind -------------------------------------------------------------------

def make_wind(rng):
    """Hourly wind for the 24 h before the scene, speed kept in 5-8 m/s.
    Same draw order as the original so scene_normal is unchanged."""
    samples = []
    for h in range(24, -1, -1):
        t = SCENE_TIME - timedelta(hours=h)
        speed = 6.5 + 1.3 * math.sin(h / 3.0) + rng.normal(0, 0.15)
        speed = float(np.clip(speed, 5.0, 8.0))
        samples.append({"time": iso(t), "speed_ms": round(speed, 2),
                        "dir_deg": WIND_FROM_DEG})
    return samples


def make_calm_wind():
    """Hourly wind at CALM_WIND_MS. Deterministic (no randomness), so the calm
    scene differs from normal only in wind speed."""
    samples = []
    for h in range(24, -1, -1):
        t = SCENE_TIME - timedelta(hours=h)
        samples.append({"time": iso(t), "speed_ms": round(CALM_WIND_MS, 2),
                        "dir_deg": WIND_FROM_DEG})
    return samples


# --- ships ------------------------------------------------------------------

def backtrack_center(slick_lat, slick_lon, wind_samples):
    """Replay rewind.py's model on the slick centroid to find the release
    centre, so vessels can be planted where the backtrack lands."""
    lat, lon = slick_lat, slick_lon
    recent = wind_samples[-(config.SLICK_AGE_HOURS + 1):-1]
    for w in reversed(recent):
        dist = config.WIND_DRIFT_FACTOR * w["speed_ms"] * 3600.0
        lat, lon = step_latlon(lat, lon, w["dir_deg"], dist)
    return lat, lon


def track(mmsi, name, start_lat, start_lon, bearing, speed_ms, gap_after=None):
    """Build a vessel reporting every 20 minutes across the 24 h window."""
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


def track_through(mmsi, name, at_lat, at_lon, bearing, speed_ms):
    """Like track(), but placed so the 22:00 (release-time) report sits exactly
    at (at_lat, at_lon): step back 12 h from that point to find the start."""
    lat, lon = at_lat, at_lon
    for _ in range(int(config.SLICK_AGE_HOURS * 60 / 20)):
        lat, lon = step_latlon(lat, lon, bearing + 180, speed_ms * 20 * 60)
    return track(mmsi, name, lat, lon, bearing, speed_ms)


def make_ships(release_lat, release_lon):
    """The original six vessels (unchanged)."""
    ships = []

    # Ship A — the culprit. At the release time it sits at the release centre,
    # travelling north-east (~45), the direction of the slick's long axis.
    ships.append(track_through("219000001", "Nordfjord", release_lat, release_lon, 45.0, 4.0))

    # Ship B — dark vessel. Near the release area on a wrong heading (~135) with
    # a 40-min AIS gap.
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


def make_third_vessel(release_lat, release_lon):
    """The extra plausible vessel for the ambiguous scene. Passes through the
    release area at the release time on a heading close to the slick axis, but
    offset perpendicular so it sits partway inside the polygon."""
    at_lat, at_lon = step_latlon(release_lat, release_lon,
                                 THIRD_PERP_BEARING_DEG, THIRD_PERP_OFFSET_M)
    return track_through("370000999", "Storebaelt", at_lat, at_lon,
                         THIRD_HEADING_DEG, THIRD_SPEED_MS)


# --- scene metadata / writing -----------------------------------------------

def scene_meta(scene_id):
    return {
        "scene_id": scene_id,
        "acquired_at": iso(SCENE_TIME),
        "bounds": {
            "min_lat": MIN_LAT, "min_lon": MIN_LON,
            "max_lat": MAX_LAT, "max_lon": MAX_LON,
        },
    }


def write_scene(name, img, transform, wind, ships, scene_id):
    d = config.SAMPLE_DATA_DIR / f"scene_{name}"
    d.mkdir(parents=True, exist_ok=True)
    write_tif(d / "scene.tif", img, transform)
    (d / "wind.json").write_text(json.dumps(wind, indent=2))
    (d / "ships.json").write_text(json.dumps(ships, indent=2))
    (d / "scene_meta.json").write_text(json.dumps(scene_meta(scene_id), indent=2))
    return d


def main():
    config.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # One RNG, consumed in the same order as the original generator (image then
    # wind) so scene_normal reproduces the existing fixture exactly.
    rng = np.random.default_rng(config.RANDOM_SEED)
    img, transform = build_scene_image(rng)
    wind = make_wind(rng)

    slick_lat = (MIN_LAT + MAX_LAT) / 2.0
    slick_lon = (MIN_LON + MAX_LON) / 2.0
    release_lat, release_lon = backtrack_center(slick_lat, slick_lon, wind)

    ships = make_ships(release_lat, release_lon)

    # NORMAL — the existing clear-ranking demo, unchanged.
    write_scene("normal", img, transform, wind, ships, SCENE_ID_NORMAL)

    # AMBIGUOUS — normal image + wind, existing six vessels plus a third.
    third = make_third_vessel(release_lat, release_lon)
    write_scene("ambiguous", img, transform, wind, ships + [third], SCENE_ID_AMBIGUOUS)

    # CALM — normal image + vessels, wind dropped to ~1.8 m/s.
    write_scene("calm", img, transform, make_calm_wind(), ships, SCENE_ID_CALM)

    print(f"Wrote 3 scenes to {config.SAMPLE_DATA_DIR}")
    print(f"  release centre    ~ ({release_lat:.4f}, {release_lon:.4f})")
    print(f"  normal vessels    : {len(ships)}")
    print(f"  ambiguous vessels : {len(ships) + 1}  (added Storebaelt)")
    print(f"  calm wind         : {CALM_WIND_MS} m/s")


if __name__ == "__main__":
    main()
