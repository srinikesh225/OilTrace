#!/usr/bin/env python
"""Real ERA5 retrieval for the Skagerrak scene, with the ACTUAL CDS credentials.
Downloads real 10m u/v, extracts scene-centre wind across the drift window, runs
the UNMODIFIED filter.py gate, and re-fetches to prove a real CACHE HIT."""
from __future__ import annotations
import json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BACKEND = os.path.abspath(os.path.join(HERE, "..", "..", "backend"))
sys.path.insert(0, BACKEND)

from app import config                                   # noqa: E402
from app.stages.wind_sources.era5_source import ERA5WindSource  # noqa: E402
from verify_era5_wind import real_scene, run_gate        # noqa: E402

for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(encoding="utf-8")
    except Exception: pass

scene = real_scene()
cache_dir = config.ERA5_CACHE_DIR   # data/cache/era5 (git-ignored)
print(f"Scene: {scene.scene_id}  acquired {scene.acquired_at}")
print(f"Bounds N{scene.bounds.max_lat:.3f} W{scene.bounds.min_lon:.3f} "
      f"S{scene.bounds.min_lat:.3f} E{scene.bounds.max_lon:.3f}")
print(f"SLICK_AGE_HOURS={config.SLICK_AGE_HOURS}  cache_dir={cache_dir}")

# clean cache for a genuine MISS -> download
if os.path.isdir(cache_dir):
    shutil.rmtree(cache_dir)

print("\n== REAL ERA5 DOWNLOAD (cache MISS) ==")
src = ERA5WindSource(cache_dir=cache_dir)
t0 = time.perf_counter()
samples = src.fetch(scene, str(config.scene_paths("normal")["wind"]))
dt = time.perf_counter() - t0
prov = src.provenance
print(f"source={prov.get('source')}  cache={prov.get('cache')}  "
      f"n={len(samples)}  ({dt:.1f}s)")
if prov.get("source") != "ERA5":
    print(f"REAL ERA5 FAILED -> {prov}")
    sys.exit(2)
print(f"centre={prov.get('centre')}  convention={prov.get('direction_convention')} units={prov.get('units')}")

print("\n== FULL HOURLY SERIES (REAL ERA5) ==")
print(f"  {'UTC':22}{'u(m/s)':>9}{'v(m/s)':>9}{'speed':>8}{'dir_from':>10}")
for s in samples:
    print(f"  {s['time']:22}{s['u_ms']:>9}{s['v_ms']:>9}{s['speed_ms']:>8}{s['dir_deg']:>10}")

# scene-time value = the sample the gate selects (nearest hour)
wind_used, decision, reason = run_gate(samples, scene)
print("\n== WIND AT SCENE TIME (as the gate selects it) ==")
print(f"  scene time     : {scene.acquired_at}")
print(f"  ERA5 hour used : {wind_used.time}")
print(f"  speed          : {wind_used.speed_ms} m/s")
print(f"  direction      : {wind_used.dir_deg} deg (meteorological FROM)")

print("\n== GATE (UNMODIFIED filter.py) ==")
print(f"  WIND_MIN_MS={config.WIND_MIN_MS}  WIND_MAX_MS={config.WIND_MAX_MS}")
print(f"  ERA5 speed={wind_used.speed_ms} m/s  ->  {decision}")
print(f"  exact reason string: {reason!r}")

# sanity
sp = wind_used.speed_ms
print(f"\n== SANITY ==\n  {sp} m/s "
      f"{'PLAUSIBLE (<=30, in Skagerrak Sept range ~2-15)' if sp<=30 else 'IMPLAUSIBLE >30 -> INVESTIGATE'}")

print("\n== REPEAT FETCH (expect real CACHE HIT, no download) ==")
src2 = ERA5WindSource(cache_dir=cache_dir)
downloads = {"n": 0}
orig = src2._download
def counting(nc, area, hours, variables):
    downloads["n"] += 1; return orig(nc, area, hours, variables)
src2._download = counting
s2 = src2.fetch(scene, str(config.scene_paths("normal")["wind"]))
print(f"  source={src2.provenance.get('source')}  cache={src2.provenance.get('cache')}  downloads={downloads['n']}")

out = {
    "scene": {"id": scene.scene_id, "acquired_at": scene.acquired_at,
              "bounds_NWSE": [scene.bounds.max_lat, scene.bounds.min_lon,
                              scene.bounds.min_lat, scene.bounds.max_lon]},
    "provenance": prov,
    "hourly_series_real": samples,
    "scene_time_wind": {"era5_hour": wind_used.time, "speed_ms": wind_used.speed_ms,
                        "dir_deg": wind_used.dir_deg, "convention": "meteorological FROM"},
    "gate": {"WIND_MIN_MS": config.WIND_MIN_MS, "WIND_MAX_MS": config.WIND_MAX_MS,
             "decision": decision, "exact_reason": reason},
    "download_seconds": round(dt, 1),
    "repeat_cache": src2.provenance.get("cache"), "repeat_downloads": downloads["n"],
}
with open(os.path.join(HERE, "results", "era5_real_run.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("\nWrote results/era5_real_run.json")
