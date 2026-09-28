#!/usr/bin/env python
"""Run the OILTRACE pipeline end-to-end on scene_real_skagerrak with
SEGMENTER_BACKEND=model and WIND_SOURCE=era5. Captures each stage and writes
results/real_scene_run.json. Uses the real backend stages (does not modify them)."""
from __future__ import annotations
import json, os, sys, time
os.environ["SEGMENTER_BACKEND"] = "model"
os.environ["WIND_SOURCE"] = "era5"

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, "..", "..", "backend"))
sys.path.insert(0, BACKEND)

from app import config                                    # noqa: E402
from app.models import SceneMeta                          # noqa: E402
from app.stages import filter as filter_stage            # noqa: E402
from app.stages import rewind as rewind_stage            # noqa: E402
from app.stages import name as name_stage                # noqa: E402
from app.stages import explain as explain_stage          # noqa: E402
from app.stages.name import FileShipSource               # noqa: E402
from app.stages.segmenters import make_segmenter         # noqa: E402
from app.stages.wind_sources import make_wind_source     # noqa: E402

for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(encoding="utf-8")
    except Exception: pass

def load(p): return json.load(open(p, encoding="utf-8"))

SCENE = "real_skagerrak"
paths = config.scene_paths(SCENE)
scene = SceneMeta(**load(paths["meta"]))
print(f"Scene {scene.scene_id}  acquired {scene.acquired_at}")
print(f"  bounds {scene.bounds.model_dump()}")
print(f"  MIN_SLICK_AREA_PX={config.MIN_SLICK_AREA_PX}  WIND_MIN={config.WIND_MIN_MS} WIND_MAX={config.WIND_MAX_MS}")

t0 = time.perf_counter()

# ---- wind source (ERA5) ----
wind_source = make_wind_source()
wind_samples = wind_source.fetch(scene, paths["wind"])
wind_prov = wind_source.provenance

# ---- SEE (model) ----
segmenter = make_segmenter()
t_see = time.perf_counter()
see_out, see_rej = segmenter.segment(scene.scene_id, paths["tif"])
see_s = time.perf_counter() - t_see

# ---- FILTER (unmodified wind gate) ----
filter_out = filter_stage.apply_wind_gate(see_out, scene.acquired_at, wind_samples)
accepted = sorted(filter_out.accepted, key=lambda p: p.area_px, reverse=True)
detected = accepted[0] if accepted else None

# ---- REWIND / NAME / EXPLAIN (only if a slick survived) ----
rewind_out = name_out = explain_out = None
if detected is not None:
    rewind_out = rewind_stage.rewind(detected, scene.acquired_at, wind_samples, filter_out.wind_at_scene)
    ship_source = FileShipSource(paths["ships"])
    name_out, _ = name_stage.name_vessels(rewind_out, ship_source)
    explain_out, _ = explain_stage.explain(name_out, rewind_out, scene, see_out, filter_out)

total_s = time.perf_counter() - t0
w = filter_out.wind_at_scene

# ---- console report ----
print(f"\nSEGMENTER = {type(segmenter).__name__}")
print("\n== SEE ==")
print(f"  oil polygons: {len(see_out.polygons)}")
for p in see_out.polygons:
    print(f"    {p.polygon_id}: area {p.area_px:.0f}px / {p.area_km2}km2, bearing {p.bearing_deg}deg")
print(f"  look-alike regions: {len(see_out.look_alike_polygons)}")
for p in see_out.look_alike_polygons[:10]:
    print(f"    {p.polygon_id}: area {p.area_px:.0f}px / {p.area_km2}km2")
print(f"  see-stage rejections: {len(see_rej)}")
for r in see_rej[:6]:
    print(f"    {r.polygon_id}: {r.reason}")
if len(see_rej) > 6: print(f"    ... (+{len(see_rej)-6} more)")

print("\n== FILTER (ERA5 wind) ==")
print(f"  wind source: {wind_prov.get('source')}  cache={wind_prov.get('cache')}")
print(f"  ERA5 centre/grid: {wind_prov.get('centre')}")
print(f"  wind_at_scene: time={w.time}  speed={w.speed_ms} m/s  dir={w.dir_deg} deg (from)")
print(f"  gate: WIND_MIN={config.WIND_MIN_MS} WIND_MAX={config.WIND_MAX_MS} -> "
      f"{'ACCEPTED' if filter_out.accepted else 'REJECTED'}  ({len(filter_out.accepted)} accepted)")
flt = [r for r in filter_out.rejected]
print(f"  filter rejections: {len(flt)}")
for r in flt[:6]:
    print(f'    {r.polygon_id}: "{r.reason}"')

print("\n== REWIND / NAME / EXPLAIN ==")
if detected is None:
    print("  DID NOT RUN — no slick survived the wind gate (nothing to attribute).")
else:
    print(f"  REWIND: release polygon with {len(rewind_out.particle_cloud)} particles, release_time {rewind_out.release_time}")
    print(f"  NAME: {len(name_out.candidates)} candidate vessels (synthetic ships)")
    print(f"  EXPLAIN: {len(explain_out.ranked_candidates)} ranked; sep={explain_out.separation_flag}")

print(f"\nTOTAL RUNTIME: {total_s:.2f}s (SEE model {see_s:.2f}s)")

# ---- record ----
rec = {
    "product_id": "S1C_IW_GRDH_1SDV_20260914T053126_20260914T053151_009439_012C67_22B7",
    "scene_id": scene.scene_id,
    "tile": {"name": "OCEAN_01", "centre_lat": 58.1399, "centre_lon": 10.4062,
             "coast_distance_km": 45.0, "top_left_row_col": [0, 21840], "size": [650, 1250]},
    "bounds": scene.bounds.model_dump(),
    "acquired_at": scene.acquired_at,
    "preprocessing": {"calibration": "sigma0 = DN^2 / sigmaNought_LUT^2 (ESA S1 LUT)",
                      "speckle": "Lee MMSE window 5 ENL 4.4", "db_range": [-30, 0],
                      "scaling": "linear dB->8bit", "geoloc_rms_m": load(paths["meta"]).get("geoloc_approx_rms_m")},
    "segmenter": type(segmenter).__name__,
    "model_weights": os.path.basename(config.MODEL_WEIGHTS_PATH),
    "oil_prob_threshold": config.OIL_PROB_THRESHOLD, "min_slick_area_px": config.MIN_SLICK_AREA_PX,
    "wind": {"source": wind_prov.get("source"), "cache": wind_prov.get("cache"),
             "dataset": wind_prov.get("dataset"), "centre": wind_prov.get("centre"),
             "at_scene_time": w.time, "speed_ms": w.speed_ms, "dir_deg": w.dir_deg,
             "convention": "meteorological FROM"},
    "stages": {
        "SEE": {"oil_polygons": [{"id": p.polygon_id, "area_px": p.area_px, "area_km2": p.area_km2,
                                  "bearing_deg": p.bearing_deg} for p in see_out.polygons],
                "look_alike_regions": len(see_out.look_alike_polygons),
                "see_rejections": len(see_rej)},
        "FILTER": {"wind_min_ms": config.WIND_MIN_MS, "wind_max_ms": config.WIND_MAX_MS,
                   "speed_ms": w.speed_ms, "decision": "ACCEPTED" if filter_out.accepted else "REJECTED",
                   "accepted": len(filter_out.accepted), "rejected": len(filter_out.rejected),
                   "reasons": [r.reason for r in filter_out.rejected]},
        "REWIND": "ran" if detected is not None else "did not run (gate rejected)",
        "NAME": ("ran" if detected is not None else "did not run (gate rejected)"),
        "EXPLAIN": ("ran" if detected is not None else "did not run (gate rejected)"),
    },
    "runtime_seconds": round(total_s, 2), "see_model_seconds": round(see_s, 2),
    "real_vs_synthetic": {
        "real": ["scene.tif pixels (Sentinel-1C VV, calibrated)", "geo-transform (product GCPs)",
                 "acquisition timestamp + bounds", "SEE segmentation (ResNet50 DeepLabV3+ on real SAR)",
                 "FILTER wind (ERA5 reanalysis)" if wind_prov.get("source") == "ERA5"
                 else "FILTER wind FELL BACK to synthetic fixture (ERA5 unavailable)"],
        "synthetic_or_na": ["ships.json (synthetic vessels, not AIS)",
                            "wind.json fallback (unused if ERA5 ran)",
                            "REWIND/NAME/EXPLAIN did not run" if detected is None
                            else "NAME matched against SYNTHETIC ships"],
    },
}
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(rec, open(os.path.join(HERE, "results", "real_scene_run.json"), "w", encoding="utf-8"), indent=2)
print("\nWrote results/real_scene_run.json")
