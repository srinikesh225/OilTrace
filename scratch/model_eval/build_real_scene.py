#!/usr/bin/env python
"""Build backend/sample_data/scene_real_skagerrak/ from the real S1C OCEAN_01
tile: preprocessed 8-bit GeoTIFF with a GCP-derived geo-transform, real scene
metadata, synthetic (labelled) ships, and a synthetic fallback wind series.

Reuses the already-verified pieces: preprocess_grd (calibration/Lee/dB/8-bit)
and open_ocean_audit (GCP geolocation + tile selection). Does not modify them.
The .SAFE product / weights / credentials are never written into the repo.
"""
from __future__ import annotations
import json, os, sys, shutil
from datetime import datetime, timedelta, timezone
import numpy as np
import rasterio
from rasterio.transform import Affine
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from preprocess_grd import Safe, CalLUT                       # noqa: E402
from open_ocean_audit import parse_gcps, GeoMap, preprocess_tile, SAFE_ZIP, TH, TW  # noqa: E402

# OCEAN_01: centre (row 325, col 22465) -> top-left (row 0, col 21840), 650x1250.
CENTRE_ROW, CENTRE_COL = 325, 22465
FR, FC = CENTRE_ROW - TH // 2, CENTRE_COL - TW // 2      # 0, 21840
SCENE_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "backend",
                                         "sample_data", "scene_real_skagerrak"))
NORMAL_SHIPS = os.path.abspath(os.path.join(HERE, "..", "..", "backend",
                               "sample_data", "scene_normal", "ships.json"))

def fit_affine(gm: GeoMap):
    """Least-squares affine mapping LOCAL tile (col,row) -> (lon,lat) using the
    product GCP interpolation. Returns (Affine, corners, rms_error_m)."""
    rows = np.linspace(0, TH - 1, 12)
    cols = np.linspace(0, TW - 1, 24)
    A, LON, LAT = [], [], []
    for r in rows:
        for c in cols:
            lat, lon = gm(FR + r, FC + c)         # absolute product pixel -> lat/lon
            A.append([c, r, 1.0]); LON.append(lon); LAT.append(lat)
    A = np.array(A); LON = np.array(LON); LAT = np.array(LAT)
    (a, b, cc), *_ = np.linalg.lstsq(A, LON, rcond=None)   # lon = a*col + b*row + cc
    (d, e, ff), *_ = np.linalg.lstsq(A, LAT, rcond=None)   # lat = d*col + e*row + ff
    # residual in metres
    lon_hat = A @ [a, b, cc]; lat_hat = A @ [d, e, ff]
    dlon = (lon_hat - LON) * 111320.0 * np.cos(np.deg2rad(LAT))
    dlat = (lat_hat - LAT) * 111320.0
    rms = float(np.sqrt(np.mean(dlon**2 + dlat**2)))
    transform = Affine(a, b, cc, d, e, ff)
    corners = {"top_left": gm(FR, FC), "top_right": gm(FR, FC + TW - 1),
               "bottom_left": gm(FR + TH - 1, FC), "bottom_right": gm(FR + TH - 1, FC + TW - 1)}
    return transform, corners, rms

def real_meta():
    r = etree.parse(os.path.join(HERE, "out", "safe_meta", "annotation",
        "s1c-iw-grd-vv-20260914t053126-20260914t053151-009439-012c67-001.xml")).getroot()
    start = r.findtext(".//adsHeader/startTime")
    return datetime.fromisoformat(start).replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def synth_wind(acq):
    """Synthetic FALLBACK-only hourly wind around the real scene time. Only used
    if ERA5 is unavailable; provenance will then read FILE_FALLBACK."""
    st = datetime.fromisoformat(acq.replace("Z", "+00:00"))
    base = st.replace(minute=0, second=0, microsecond=0)
    out = []
    for h in range(13, -1, -1):
        t = base - timedelta(hours=h)
        out.append({"time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "speed_ms": 6.0, "dir_deg": 225.0})   # placeholder, clearly not real
    return out

def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except Exception: pass
    safe = Safe.open(SAFE_ZIP)
    cal = CalLUT.parse(safe.read_cal_xml())
    lines, pixels, latg, long, _ = parse_gcps(safe)
    gm = GeoMap(lines, pixels, latg, long)

    u8, valid = preprocess_tile(safe, cal, FR, FC)          # 650x1250 uint8, dB[-30,0]
    transform, corners, rms = fit_affine(gm)
    lats = [c[0] for c in corners.values()]; lons = [c[1] for c in corners.values()]
    bounds = {"min_lat": round(min(lats), 6), "min_lon": round(min(lons), 6),
              "max_lat": round(max(lats), 6), "max_lon": round(max(lons), 6)}
    acq = real_meta()

    os.makedirs(SCENE_DIR, exist_ok=True)
    # scene.tif — single-band 8-bit, EPSG:4326, GCP-derived transform
    with rasterio.open(os.path.join(SCENE_DIR, "scene.tif"), "w", driver="GTiff",
                       height=TH, width=TW, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=transform, compress="deflate") as dst:
        dst.write(u8, 1)
    # scene_meta.json — real id/time/bounds
    meta = {"scene_id": "S1C_IW_GRDH_20260914T053126_OCEAN01",
            "acquired_at": acq, "bounds": bounds,
            "provenance": "REAL Sentinel-1C OCEAN_01 tile; ships.json is SYNTHETIC.",
            "geoloc_approx_rms_m": round(rms, 1)}
    json.dump(meta, open(os.path.join(SCENE_DIR, "scene_meta.json"), "w"), indent=2)
    # ships.json — synthetic vessels carried over (clearly not real AIS)
    ships = json.load(open(NORMAL_SHIPS))
    json.dump(ships, open(os.path.join(SCENE_DIR, "ships.json"), "w"), indent=2)
    # wind.json — synthetic fallback only
    json.dump(synth_wind(acq), open(os.path.join(SCENE_DIR, "wind.json"), "w"), indent=2)

    size = os.path.getsize(os.path.join(SCENE_DIR, "scene.tif"))
    print("scene_real_skagerrak built:")
    print(f"  tile top-left (row,col) = ({FR},{FC}) size {TH}x{TW}")
    print(f"  centre lat/lon = {gm(CENTRE_ROW, CENTRE_COL)}")
    print(f"  bounds = {bounds}")
    print(f"  acquired_at = {acq}")
    print(f"  geo-transform RMS residual vs GCP interpolation = {rms:.1f} m")
    print(f"  scene.tif size = {size/1024:.0f} KiB ({'<5MB, committable' if size<5*1024*1024 else '>=5MB, gitignore'})")
    print(f"  ships.json = synthetic ({len(ships)} vessels), wind.json = synthetic fallback")

if __name__ == "__main__":
    main()
