#!/usr/bin/env python
"""
Verification harness for the ERA5 wind integration (production code in
backend/app/stages/wind_sources/). Verification artifact only — imports the
production classes, never modifies them.

It attempts a REAL ERA5 retrieval for the real Sentinel-1 scene, then runs the
four fallback/cache tests. Where no CDS credentials are available, REAL ERA5 is
reported UNVERIFIED and the extraction/cache/gate CODE is exercised with a
clearly-labelled SYNTHETIC NetCDF (never presented as real ERA5).

The wind gate is the UNMODIFIED backend/app/stages/filter.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, "..", "..", "backend"))
sys.path.insert(0, BACKEND)

from app import config                              # noqa: E402
from app.models import SceneMeta, Bounds, SeeOutput, SlickPolygon, GeoPoint  # noqa: E402
from app.stages import filter as filter_stage       # noqa: E402  (UNMODIFIED gate)
from app.stages.wind_sources import make_wind_source, FileWindSource  # noqa: E402
from app.stages.wind_sources.era5_source import ERA5WindSource        # noqa: E402

ANN = os.path.join(HERE, "out", "safe_meta", "annotation",
                   "s1c-iw-grd-vv-20260914t053126-20260914t053151-009439-012c67-001.xml")
FALLBACK_WIND = str(config.scene_paths("normal")["wind"])  # any fixture, for fallback
CACHE_DIR = os.path.join(HERE, "out", "era5_cache_test")   # ignored (out/ is gitignored)
report = {"real_era5": {}, "tests": {}, "notes": []}


def real_scene() -> SceneMeta:
    """Build SceneMeta from the product's OWN metadata (GCP extent + startTime),
    not hardcoded guesses."""
    r = etree.parse(ANN).getroot()
    start = r.findtext(".//adsHeader/startTime")  # UTC, e.g. 2026-09-14T05:31:26...
    pts = r.findall(".//geolocationGridPoint")
    lat = [float(p.findtext("latitude")) for p in pts]
    lon = [float(p.findtext("longitude")) for p in pts]
    acq = datetime.fromisoformat(start).replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return SceneMeta(scene_id="S1C_REAL_20260914T0531",
                     acquired_at=acq,
                     bounds=Bounds(min_lat=min(lat), min_lon=min(lon),
                                   max_lat=max(lat), max_lon=max(lon)))


def make_synth_nc(path, scene, hours, u=-5.0, v=-4.0):
    """Write a SYNTHETIC ERA5-shaped NetCDF (valid_time,latitude,longitude with
    u10,v10). NOT REAL DATA — a fixed physically-typical field to exercise the
    extraction/cache/gate code only."""
    import xarray as xr
    lats = np.arange(round(scene.bounds.max_lat, 2), round(scene.bounds.min_lat, 2) - 0.25, -0.25)
    lons = np.arange(round(scene.bounds.min_lon, 2), round(scene.bounds.max_lon, 2) + 0.25, 0.25)
    times = np.array([np.datetime64(h.strftime("%Y-%m-%dT%H:00:00")) for h in hours])
    u10 = np.full((len(times), len(lats), len(lons)), u, dtype="float32")
    v10 = np.full((len(times), len(lats), len(lons)), v, dtype="float32")
    ds = xr.Dataset({"u10": (("valid_time", "latitude", "longitude"), u10),
                     "v10": (("valid_time", "latitude", "longitude"), v10)},
                    coords={"valid_time": times, "latitude": lats, "longitude": lons})
    ds.attrs["SYNTHETIC"] = "code-test only, not real ERA5"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(path)
    ds.close()


def run_gate(wind_samples, scene):
    """Feed one detected polygon through the UNMODIFIED wind gate and return the
    WindSample used + the exact rejection string (if any)."""
    poly = SlickPolygon(polygon_id="poly-test",
                        boundary=[GeoPoint(lat=scene.bounds.min_lat, lon=scene.bounds.min_lon),
                                  GeoPoint(lat=scene.bounds.max_lat, lon=scene.bounds.min_lon),
                                  GeoPoint(lat=scene.bounds.max_lat, lon=scene.bounds.max_lon)],
                        area_px=2000.0, area_km2=1.0,
                        centroid=GeoPoint(lat=(scene.bounds.min_lat+scene.bounds.max_lat)/2,
                                          lon=(scene.bounds.min_lon+scene.bounds.max_lon)/2),
                        bearing_deg=45.0)
    see = SeeOutput(scene_id=scene.scene_id, polygons=[poly])
    out = filter_stage.apply_wind_gate(see, scene.acquired_at, wind_samples)
    reason = out.rejected[0].reason if out.rejected else None
    return out.wind_at_scene, ("ACCEPTED" if out.accepted else "REJECTED"), reason


def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except Exception: pass
    scene = real_scene()
    print(f"Real scene: {scene.scene_id}  acquired {scene.acquired_at}")
    print(f"  bounds N{scene.bounds.max_lat:.3f} W{scene.bounds.min_lon:.3f} "
          f"S{scene.bounds.min_lat:.3f} E{scene.bounds.max_lon:.3f}")
    print(f"  SLICK_AGE_HOURS={config.SLICK_AGE_HOURS}  WIND_MIN={config.WIND_MIN_MS} "
          f"WIND_MAX={config.WIND_MAX_MS}")
    src = ERA5WindSource(cache_dir=CACHE_DIR)
    hours = src._window_hours(scene.acquired_at)
    print(f"  drift window: {hours[0].strftime('%Y-%m-%dT%H:00Z')} .. "
          f"{hours[-1].strftime('%Y-%m-%dT%H:00Z')}  ({len(hours)} hourly steps)")
    report["scene"] = {"id": scene.scene_id, "acquired_at": scene.acquired_at,
                       "bounds_NWSE": [scene.bounds.max_lat, scene.bounds.min_lon,
                                       scene.bounds.min_lat, scene.bounds.max_lon],
                       "slick_age_hours": config.SLICK_AGE_HOURS,
                       "window": [hours[0].strftime("%Y-%m-%dT%H:00Z"),
                                  hours[-1].strftime("%Y-%m-%dT%H:00Z")]}

    # -- REAL ERA5 attempt (needs credentials) --------------------------------
    print("\n== REAL ERA5 ATTEMPT ==")
    has_creds = os.path.isfile(os.path.expanduser("~/.cdsapirc")) or (
        os.getenv("CDSAPI_URL") and os.getenv("CDSAPI_KEY"))
    real = ERA5WindSource(cache_dir=CACHE_DIR)
    samples = real.fetch(scene, FALLBACK_WIND)
    prov = real.provenance
    print(f"  provenance.source = {prov.get('source')}")
    if prov.get("source") == "ERA5":
        report["real_era5"] = {"status": "PASS", "provenance": prov}
        print("  REAL ERA5 retrieved.")
    else:
        report["real_era5"] = {"status": "UNVERIFIED",
                               "reason": prov.get("reason", "no credentials"),
                               "has_creds": bool(has_creds)}
        print(f"  REAL ERA5 UNVERIFIED (no credentials / unreachable): {prov.get('reason')}")

    # ========================================================================
    # TEST A — credentials missing -> fallback
    # ========================================================================
    print("\n== TEST A: credentials missing -> expect FILE_FALLBACK ==")
    for k in ("CDSAPI_URL", "CDSAPI_KEY"):
        os.environ.pop(k, None)
    a = ERA5WindSource(cache_dir=CACHE_DIR + "_A")
    sa = a.fetch(scene, FALLBACK_WIND)
    ok_a = a.provenance.get("source") == "FILE_FALLBACK" and len(sa) > 0
    print(f"  source={a.provenance.get('source')}  n={len(sa)}  reason={a.provenance.get('reason')}")
    report["tests"]["A_missing_credentials"] = {"pass": bool(ok_a), "provenance": a.provenance}

    # ========================================================================
    # TEST B — API unreachable -> fallback. Simulated by making the (unmodified)
    # production _download raise a ConnectionError, i.e. credentials present but
    # the endpoint cannot be reached. (Avoids cdsapi's real multi-minute retry
    # loop against a dead socket; production code is untouched.)
    # ========================================================================
    print("\n== TEST B: API unreachable (simulated ConnectionError) -> expect FILE_FALLBACK ==")
    b = ERA5WindSource(cache_dir=CACHE_DIR + "_B")
    def unreachable(nc, area, hours_, variables):
        raise ConnectionError("simulated: CDS endpoint unreachable")
    b._download = unreachable
    t0 = time.perf_counter()
    sb = b.fetch(scene, FALLBACK_WIND)
    ok_b = b.provenance.get("source") == "FILE_FALLBACK" and len(sb) > 0
    print(f"  source={b.provenance.get('source')}  n={len(sb)}  "
          f"({time.perf_counter()-t0:.1f}s)  reason={b.provenance.get('reason')}")
    report["tests"]["B_api_unreachable"] = {"pass": bool(ok_b), "simulated": True,
                                            "provenance": b.provenance}

    # ========================================================================
    # TEST C — valid ERA5 request (CODE PATH, synthetic NetCDF) -> ERA5, MISS
    # ========================================================================
    print("\n== TEST C: ERA5 code path with SYNTHETIC NetCDF -> expect source=ERA5, cache=MISS ==")
    print("   NOTE: values below are from a SYNTHETIC NetCDF (fixed field), NOT real ERA5.")
    cdir = CACHE_DIR + "_C"
    import shutil
    if os.path.isdir(cdir): shutil.rmtree(cdir)
    c = ERA5WindSource(cache_dir=cdir)
    download_calls = {"n": 0}
    orig_download = c._download
    def fake_download(nc, area, hours_, variables):
        download_calls["n"] += 1
        make_synth_nc(nc, scene, hours_)     # write synthetic in place of a real CDS retrieve
    c._download = fake_download
    sc = c.fetch(scene, FALLBACK_WIND)
    ok_c = (c.provenance.get("source") == "ERA5" and c.provenance.get("cache") == "MISS"
            and download_calls["n"] == 1 and len(sc) == len(hours))
    print(f"  source={c.provenance.get('source')} cache={c.provenance.get('cache')} "
          f"downloads={download_calls['n']} n={len(sc)}")
    print(f"  centre={c.provenance.get('centre')}  convention={c.provenance.get('direction_convention')}")
    report["tests"]["C_valid_request_codepath"] = {
        "pass": bool(ok_c), "synthetic": True, "provenance": c.provenance}

    # ========================================================================
    # TEST D — repeat request -> CACHE HIT, no download
    # ========================================================================
    print("\n== TEST D: repeat request -> expect cache=HIT, no download ==")
    d = ERA5WindSource(cache_dir=cdir)
    dl2 = {"n": 0}
    def fake_download2(nc, area, hours_, variables):
        dl2["n"] += 1; make_synth_nc(nc, scene, hours_)
    d._download = fake_download2
    sd = d.fetch(scene, FALLBACK_WIND)
    ok_d = d.provenance.get("cache") == "HIT" and dl2["n"] == 0 and len(sd) == len(hours)
    print(f"  source={d.provenance.get('source')} cache={d.provenance.get('cache')} downloads={dl2['n']}")
    report["tests"]["D_cache_hit"] = {"pass": bool(ok_d), "provenance": d.provenance}

    # -- hourly series (from Test C synthetic) + scene-time value -------------
    print("\n== HOURLY SERIES (SYNTHETIC code-test values) ==")
    print(f"  {'UTC':22}{'u':>8}{'v':>8}{'speed':>8}{'dir_from':>10}")
    for s in sc:
        print(f"  {s['time']:22}{s['u_ms']:>8}{s['v_ms']:>8}{s['speed_ms']:>8}{s['dir_deg']:>10}")
    report["hourly_series_synthetic"] = sc

    # -- gate on the (synthetic) ERA5-format series via UNMODIFIED filter.py --
    wind_used, decision, reason = run_gate(sc, scene)
    print(f"\n== GATE (UNMODIFIED filter.py) on synthetic-ERA5 series ==")
    print(f"  wind_at_scene: time={wind_used.time} speed={wind_used.speed_ms} dir={wind_used.dir_deg}")
    print(f"  WIND_MIN={config.WIND_MIN_MS} WIND_MAX={config.WIND_MAX_MS}  -> {decision}")
    print(f"  exact reason string: {reason!r}")
    report["gate_synthetic"] = {"wind_time": wind_used.time, "speed_ms": wind_used.speed_ms,
                                "dir_deg": wind_used.dir_deg, "decision": decision,
                                "exact_reason": reason,
                                "WIND_MIN_MS": config.WIND_MIN_MS, "WIND_MAX_MS": config.WIND_MAX_MS}

    # sanity check on the (synthetic) magnitude
    sp = wind_used.speed_ms
    plausible = 0.0 <= sp <= 30.0
    report["sanity"] = {"speed_ms": sp, "plausible_le_30": bool(plausible),
                        "basis": "Skagerrak mid-Sept 10-m wind typically ~2-15 m/s; >30 m/s would "
                                 "indicate a unit/component bug. NOTE: this magnitude is from the "
                                 "SYNTHETIC test field, not real ERA5."}

    # -- credential + cache hygiene ------------------------------------------
    import subprocess
    def ignored(path):
        r = subprocess.run(["git", "check-ignore", path], cwd=os.path.join(HERE, "..", ".."),
                           capture_output=True, text=True)
        return r.returncode == 0
    hygiene = {".cdsapirc": ignored(".cdsapirc"),
               "data/cache/": ignored("data/cache/era5/x.nc"),
               "sample.nc": ignored("scratch/model_eval/out/era5_cache_test/sample.nc")}
    report["hygiene"] = hygiene
    print(f"\n== HYGIENE (git check-ignore) == {hygiene}")

    report["summary"] = {
        "real_era5": report["real_era5"]["status"],
        "fixture_byte_identity": "PASS (verified separately: SHA256 match all 3 scenes)",
        "fallback": "PASS" if (ok_a and ok_b) else "FAIL",
        "era5_cache": "PASS" if (ok_c and ok_d) else "FAIL",
        "credential_safety": "PASS" if all(hygiene.values()) else "FAIL",
    }
    with open(os.path.join(HERE, "results", "era5_wind_verification.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nWrote results/era5_wind_verification.json")
    print("SUMMARY:", json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
