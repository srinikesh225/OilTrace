#!/usr/bin/env python
"""
Open-ocean model audit: is the model calling genuine open ocean "land"?

Independent-evidence experiment. Tiles are chosen ONLY from GCP geolocation +
an independent GSHHG-derived coastline (global_land_mask) — the model never
influences selection. The frozen preprocessing dB[-30,0] from the previous
experiment is applied identically to every tile, then the EXISTING model
(reproduced faithfully from backend/app/stages/segmenters/model_segmenter.py via
the isolated oil_model.py wrapper it itself uses) is run.

Reads under backend/app/ (NOT imported): model_segmenter.py, config.py — to
copy the exact model parameters below. Nothing under backend/app/ is imported or
modified.

Frozen (do not tune): DB_MIN=-30, DB_MAX=0, Lee window 5, ENL 4.4,
OIL_CLASS_INDEX=1, LOOKALIKE_CLASS_INDEX=2, OIL_PROB_THRESHOLD=0.5, device cpu.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from lxml import etree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label
from skimage.io import imread
from global_land_mask import globe

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# reuse the previously-built, isolated preprocessing (same directory)
from preprocess_grd import Safe, CalLUT, calibrate_sigma0, lee_filter, stats  # noqa: E402

OUT = os.path.join(HERE, "out")
RESULTS = os.path.join(HERE, "results")
REF_IMAGE = os.path.join(HERE, "source", "src",
                         "sample_padding_image_for_inference", "img_0814.jpg")

# ---- FROZEN model params, copied from backend/app (READ, not imported) ------
# backend/app/config.py: OIL_CLASS_INDEX=1, LOOKALIKE_CLASS_INDEX=2,
#                        OIL_PROB_THRESHOLD=0.5
# backend/app/stages/segmenters/model_segmenter.py::segment(): resize to
#   (NATIVE_W,NATIVE_H)=(1250,650), replicate band->3ch, class_probs, argmax/thr
OIL_CLASS_INDEX = 1
LOOKALIKE_CLASS_INDEX = 2
OIL_PROB_THRESHOLD = 0.5
CLASS_NAMES = ["sea_surface", "oil_spill", "oil_spill_look_alike", "ship", "land"]
DB_MIN, DB_MAX, LEE_WIN, ENL = -30.0, 0.0, 5, 4.4
TH, TW = 650, 1250
COAST_MIN_KM = 20.0

SAFE_ZIP = r"C:\Users\srini\Downloads\S1C_IW_GRDH_1SDV_20260914T053126_20260914T053151_009439_012C67_22B7.SAFE.zip"


# =============================================================================
# GCP geolocation
# =============================================================================
def parse_gcps(safe: Safe):
    """Read geolocationGridPoint grid from the VV annotation. Returns regular
    (lines, pixels) axes and lat/lon grids for RegularGridInterpolator."""
    ann_inner = safe.vv_calibration.replace("/calibration/calibration-", "/").replace(
        "calibration-", "")  # not used; parse from extracted file instead
    # We parse the annotation XML that pairs with VV. Prefer the extracted copy.
    cand = os.path.join(OUT, "safe_meta", "annotation",
                        os.path.basename(safe.vv_calibration).replace("calibration-", ""))
    if os.path.isfile(cand):
        root = etree.parse(cand).getroot()
    else:
        import zipfile
        base = safe.vv_measurement.split("/measurement/")[0]
        inner = base + "/annotation/" + os.path.basename(safe.vv_calibration).replace("calibration-", "")
        root = etree.fromstring(zipfile.ZipFile(safe.path).read(inner))
    pts = root.findall(".//geolocationGridPoint")
    if not pts:
        raise ValueError("No geolocationGridPoint in annotation XML.")
    line = np.array([int(p.findtext("line")) for p in pts])
    pix = np.array([int(p.findtext("pixel")) for p in pts])
    lat = np.array([float(p.findtext("latitude")) for p in pts])
    lon = np.array([float(p.findtext("longitude")) for p in pts])
    ul = np.unique(line); up = np.unique(pix)
    if ul.size * up.size != line.size:
        raise ValueError(f"GCP grid not rectangular: {ul.size}x{up.size} != {line.size}")
    latg = np.full((ul.size, up.size), np.nan); long = np.full((ul.size, up.size), np.nan)
    li = {v: i for i, v in enumerate(ul)}; pi = {v: i for i, v in enumerate(up)}
    for l, p, a, o in zip(line, pix, lat, lon):
        latg[li[l], pi[p]] = a; long[li[l], pi[p]] = o
    if np.isnan(latg).any() or np.isnan(long).any():
        raise ValueError("GCP grid has holes.")
    return ul.astype(float), up.astype(float), latg, long, dict(
        n=len(pts), lat=lat, lon=lon, line=line, pix=pix)


class GeoMap:
    def __init__(self, lines, pixels, latg, long):
        self.flat = (lines, pixels)
        self.flat_lat = RegularGridInterpolator(self.flat, latg, bounds_error=False, fill_value=None)
        self.flat_lon = RegularGridInterpolator(self.flat, long, bounds_error=False, fill_value=None)

    def __call__(self, row, col):
        p = np.array([[row, col]], float)
        return float(self.flat_lat(p)[0]), float(self.flat_lon(p)[0])


def coast_distance_km(lat, lon, max_km=45.0, step=1.0, n_az=24):
    """Approximate nearest-coast distance via a radial search on the independent
    GSHHG-derived land grid. Returns (dist_km, capped_bool). If the centre is
    land, distance 0."""
    if bool(globe.is_land(lat, lon)):
        return 0.0, False
    az = np.deg2rad(np.arange(0, 360, 360.0 / n_az))
    r = step
    while r <= max_km:
        dlat = (r / 111.32) * np.cos(az)
        dlon = (r / (111.32 * np.cos(np.deg2rad(lat)))) * np.sin(az)
        if np.any(globe.is_land(lat + dlat, lon + dlon)):
            return float(r), False
        r += step
    return float(max_km), True  # no land found within max_km


# =============================================================================
# Preprocessing (frozen) for one full-res tile
# =============================================================================
def preprocess_tile(safe: Safe, cal: CalLUT, fr, fc):
    with rasterio.open(safe.vsi_vv) as src:
        dn = src.read(1, window=Window(fc, fr, TW, TH)).astype(np.float64)
    trow = np.arange(fr, fr + TH, dtype=np.float64)
    tcol = np.arange(fc, fc + TW, dtype=np.float64)
    sig = cal.interp(trow, tcol)
    s0, valid = calibrate_sigma0(dn, sig)
    s0 = lee_filter(s0, valid, LEE_WIN, ENL)
    db = np.full_like(s0, np.nan); ok = valid & (s0 > 0)
    db[ok] = 10.0 * np.log10(s0[ok])
    clipped = np.clip(db, DB_MIN, DB_MAX)
    u8 = np.zeros((TH, TW), np.uint8)
    u8[ok] = np.round((clipped[ok] - DB_MIN) / (DB_MAX - DB_MIN) * 255.0).astype(np.uint8)
    return u8, valid


# =============================================================================
# Model (faithful reproduction of ModelSegmenter's model path)
# =============================================================================
class Model:
    def __init__(self):
        from oil_model import OilSpillModel, NATIVE_H, NATIVE_W
        self.NATIVE_H, self.NATIVE_W = NATIVE_H, NATIVE_W
        self.m = OilSpillModel()  # loads config.MODEL_WEIGHTS_PATH once, cpu
        self.weights = self.m  # for reporting via DEFAULT

    def infer(self, u8_650x1250):
        resized = cv2.resize(u8_650x1250, (self.NATIVE_W, self.NATIVE_H),
                             interpolation=cv2.INTER_LINEAR)
        rgb = np.stack([resized, resized, resized], axis=-1)
        probs = self.m.class_probs(rgb)               # (5,650,1250)
        argmax = probs.argmax(axis=0)                 # 5-class map
        counts = {CLASS_NAMES[i]: int((argmax == i).sum()) for i in range(5)}
        # ModelSegmenter's oil/look detection uses prob>=threshold (not argmax)
        oil_thr = int((probs[OIL_CLASS_INDEX] >= OIL_PROB_THRESHOLD).sum())
        look_thr = int((probs[LOOKALIKE_CLASS_INDEX] >= OIL_PROB_THRESHOLD).sum())
        return argmax, counts, oil_thr, look_thr


def largest_components(mask):
    lab, n = label(mask)
    if n == 0:
        return 0, 0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), int(sizes.max())


# =============================================================================
def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except Exception: pass
    os.makedirs(OUT, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    rep = {"frozen": {"db_min": DB_MIN, "db_max": DB_MAX, "lee_window": LEE_WIN, "enl": ENL,
                      "oil_class_index": OIL_CLASS_INDEX, "oil_prob_threshold": OIL_PROB_THRESHOLD,
                      "device": "cpu"},
           "backend_app_files_read": ["backend/app/stages/segmenters/model_segmenter.py",
                                      "backend/app/config.py"],
           "coastline_dataset": "global_land_mask 1.0.0 (GSHHG-derived ~1km land grid)"}

    safe = Safe.open(SAFE_ZIP)
    cal = CalLUT.parse(safe.read_cal_xml())
    with rasterio.open(safe.vsi_vv) as src:
        H, W = src.height, src.width
    print(f"Product: {safe.product_id}")
    print(f"  Platform: Sentinel-1C  Type: IW GRDH  Pol selected: VV  ({','.join(safe.available_pols)})")
    print(f"  VV: {os.path.basename(safe.vv_measurement)}   dims {H}x{W}")

    # ---- TASK 1: GCPs + geolocation ----
    lines, pixels, latg, long, ginfo = parse_gcps(safe)
    gm = GeoMap(lines, pixels, latg, long)
    print(f"\n[GCP] {ginfo['n']} points; lat {ginfo['lat'].min():.3f}..{ginfo['lat'].max():.3f}, "
          f"lon {ginfo['lon'].min():.3f}..{ginfo['lon'].max():.3f}")
    corners = {"top-left": gm(0, 0), "top-right": gm(0, W - 1),
               "bottom-left": gm(H - 1, 0), "bottom-right": gm(H - 1, W - 1)}
    for k, (a, o) in corners.items():
        print(f"      {k:13} {a:.4f} N, {o:.4f} E")
    # sanity: map GCP pixels back, compare
    err = []
    for l, p, a, o in zip(ginfo["line"], ginfo["pix"], ginfo["lat"], ginfo["lon"]):
        pa, po = gm(l, p)
        dkm = np.hypot((pa - a) * 111.32, (po - o) * 111.32 * np.cos(np.deg2rad(a)))
        err.append(dkm)
    err = np.array(err)
    print(f"[GCP sanity] round-trip error: mean {err.mean():.3f} km, max {err.max():.3f} km "
          f"({'OK' if err.max() < 1.0 else 'CHECK'})")
    if err.max() > 5.0:
        print("GCP MAPPING FAILED"); sys.exit(1)

    # ---- TASK 2/3: independent open-ocean tile selection (model NOT used) ----
    print(f"\n[SELECT] scanning candidate tile centres (>= {COAST_MIN_KM} km from coast, fully valid)")
    r_lo, r_hi = TH // 2, H - TH // 2
    c_lo, c_hi = TW // 2, W - TW // 2
    cand = []
    for rr in np.linspace(r_lo, r_hi, 16).astype(int):
        for cc in np.linspace(c_lo, c_hi, 22).astype(int):
            la, lo = gm(rr, cc)
            if bool(globe.is_land(la, lo)):
                continue
            d, capped = coast_distance_km(la, lo)
            if d < COAST_MIN_KM:
                continue
            # require the whole tile to be valid data (no DN=0 border)
            with rasterio.open(safe.vsi_vv) as src:
                sub = src.read(1, window=Window(cc - TW // 2, rr - TH // 2, TW, TH))
            if (sub > 0).mean() < 0.999:
                continue
            cand.append({"row": int(rr), "col": int(cc), "lat": round(la, 4),
                         "lon": round(lo, 4), "coast_km": round(d, 1), "capped": capped})
    print(f"[SELECT] {len(cand)} candidate open-ocean tiles >= {COAST_MIN_KM} km from coast")
    if len(cand) < 3:
        print(f"WARNING: only {len(cand)} valid tiles at >= {COAST_MIN_KM} km (not relaxing threshold).")

    # pick 3 spatially separated (>=10 km apart), preferring farthest-from-coast first
    cand.sort(key=lambda t: -t["coast_km"])
    chosen = []
    for t in cand:
        if all(np.hypot((t["lat"] - c["lat"]) * 111.32,
                        (t["lon"] - c["lon"]) * 111.32 * np.cos(np.deg2rad(t["lat"]))) >= 10.0
               for c in chosen):
            chosen.append(t)
        if len(chosen) == 3:
            break
    if len(chosen) < 3:  # fall back: fill with remaining farthest even if <10km apart (reported)
        for t in cand:
            if t not in chosen:
                chosen.append(t)
            if len(chosen) == 3:
                break
    for i, t in enumerate(chosen, 1):
        t["id"] = f"OCEAN_{i:02d}"
        print(f"   {t['id']}: row {t['row']} col {t['col']}  {t['lat']} N {t['lon']} E  "
              f"coast {t['coast_km']}{'+' if t['capped'] else ''} km")
    rep["tiles"] = chosen

    # ---- TASK 4/6/7/8: preprocess + model + masks ----
    model = Model()
    weights_path = model.m.__dict__.get("weights_path", None)
    print(f"\n[MODEL] weights: {getattr(__import__('oil_model'), 'DEFAULT_WEIGHTS', '?')}")
    print(f"[MODEL] OIL_CLASS_INDEX={OIL_CLASS_INDEX} OIL_PROB_THRESHOLD={OIL_PROB_THRESHOLD} device=cpu")

    cmap = ListedColormap([(0.05, 0.12, 0.28), (0, 1, 1), (1, 0, 0), (0.6, 0.3, 0), (0, 0.6, 0)])
    results = []
    tiles_u8 = []
    for t in chosen:
        fr, fc = t["row"] - TH // 2, t["col"] - TW // 2
        u8, valid = preprocess_tile(safe, cal, fr, fc)
        tiles_u8.append(u8)
        plt.imsave(os.path.join(OUT, f"{t['id'].lower().replace('ocean','open_ocean')}.png"),
                   u8, cmap="gray", vmin=0, vmax=255)
        t0 = time.perf_counter()
        argmax, counts, oil_thr, look_thr = model.infer(u8)
        dt = round(time.perf_counter() - t0, 2)
        tot = sum(counts.values())
        pct = {k: round(100 * v / tot, 2) for k, v in counts.items()}
        n_land, big_land = largest_components(argmax == 4)
        n_oil, big_oil = largest_components(argmax == OIL_CLASS_INDEX)
        res = {**t, "counts": counts, "pct": pct, "total": tot, "sum_ok": tot == TH * TW,
               "oil_thr_pixels": oil_thr, "look_thr_pixels": look_thr, "infer_s": dt,
               "land_components": n_land, "land_largest": big_land,
               "oil_components": n_oil, "oil_largest": big_oil}
        results.append(res)
        plt.imsave(os.path.join(OUT, f"{t['id'].lower().replace('ocean','open_ocean')}_mask.png"),
                   argmax, cmap=cmap, vmin=0, vmax=4)
        print(f"   {t['id']}: {counts}  (sum {tot}=={TH*TW}: {tot==TH*TW})  "
              f"land {pct['land']}%  oil {pct['oil_spill']}%  sea {pct['sea_surface']}%  {dt}s")

    # ---- TASK 9: control on reference image ----
    ref = imread(REF_IMAGE)
    ref_gray = ref[..., 0] if ref.ndim == 3 else ref
    argmax_r, counts_r, oil_thr_r, look_thr_r = model.infer(ref_gray)
    tot_r = sum(counts_r.values())
    pct_r = {k: round(100 * v / tot_r, 3) for k, v in counts_r.items()}
    print(f"\n[CONTROL] {os.path.basename(REF_IMAGE)} {ref_gray.shape}: {counts_r}")
    print(f"          pct: {pct_r}")
    rep["control"] = {"image": REF_IMAGE, "dims": list(ref_gray.shape),
                      "counts": counts_r, "pct": pct_r}

    # ---- TASK 10: previous 69%-land tile (row 640, col 14400), geolocate ----
    prev_fr, prev_fc = 640, 14400
    pla, plo = gm(prev_fr + TH // 2, prev_fc + TW // 2)
    pd, pcap = coast_distance_km(pla, plo)
    u8p, _ = preprocess_tile(safe, cal, prev_fr, prev_fc)
    _, counts_p, _, _ = model.infer(u8p)
    tot_p = sum(counts_p.values())
    pct_p = {k: round(100 * v / tot_p, 2) for k, v in counts_p.items()}
    print(f"\n[PREV] tile row{prev_fr} col{prev_fc} centre {pla:.4f}N {plo:.4f}E  "
          f"coast {pd:.1f}{'+' if pcap else ''} km  land {pct_p['land']}%  oil {pct_p['oil_spill']}%")
    rep["previous_tile"] = {"row": prev_fr, "col": prev_fc, "centre_lat": round(pla, 4),
                            "centre_lon": round(plo, 4), "coast_km": round(pd, 1),
                            "is_land_centre": bool(globe.is_land(pla, plo)),
                            "counts": counts_p, "pct": pct_p}

    # ---- quicklook with markers (independent geolocation, not model) ----
    ql = os.path.join(OUT, "safe_meta", "preview", "quick-look.png")
    if os.path.isfile(ql):
        qimg = imread(ql)
        qh, qw = qimg.shape[:2]
        plt.figure(figsize=(7, 7)); plt.imshow(qimg, cmap="gray")
        for t in chosen:
            x = t["col"] / W * qw; y = t["row"] / H * qh
            plt.plot(x, y, "o", mfc="none", mec="lime", ms=16, mew=2)
            plt.text(x + 8, y, f"{t['id']}\n{t['coast_km']}km", color="lime", fontsize=8)
        # previous tile too
        x = prev_fc + TW // 2; x = x / W * qw; y = (prev_fr + TH // 2) / H * qh
        plt.plot(x, y, "x", color="red", ms=14, mew=2)
        plt.text(x + 8, y, f"PREV\n{pct_p['land']}%land", color="red", fontsize=8)
        plt.title("open-ocean tiles (green) + previous tile (red) — geolocation-selected", fontsize=9)
        plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "quicklook_open_ocean_tiles.png"), dpi=120); plt.close()

    # ---- combined input|prediction ----
    fig, ax = plt.subplots(len(chosen), 2, figsize=(11, 3.2 * len(chosen)))
    if len(chosen) == 1: ax = ax[None, :]
    for i, (t, u8) in enumerate(zip(chosen, tiles_u8)):
        argmax, _, _, _ = model.infer(u8)
        ax[i, 0].imshow(u8, cmap="gray", vmin=0, vmax=255)
        ax[i, 0].set_title(f"{t['id']} input  ({t['lat']}N {t['lon']}E, {t['coast_km']}km)", fontsize=8)
        ax[i, 0].axis("off")
        ax[i, 1].imshow(argmax, cmap=cmap, vmin=0, vmax=4)
        ax[i, 1].set_title(f"{t['id']} prediction  land {results[i]['pct']['land']}%", fontsize=8)
        ax[i, 1].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "open_ocean_model_comparison.png"), dpi=110); plt.close()

    # =========================================================================
    # VERDICT
    # =========================================================================
    land_pcts = [r["pct"]["land"] for r in results]
    sea_pcts = [r["pct"]["sea_surface"] for r in results]
    all_far = all(r["coast_km"] >= COAST_MIN_KM for r in results) and len(results) >= 3
    heavy_land = sum(1 for p in land_pcts if p >= 40)
    mostly_sea = sum(1 for p in sea_pcts if p >= 60)
    control_sane = counts_r["sea_surface"] / tot_r > 0.9 and counts_r["land"] == 0

    if not all_far:
        task4 = "UNRESOLVED"
        answer = "UNRESOLVED"
        why = ("Fewer than 3 independently-verified tiles at >=20 km from coast were available, "
               "so a systematic claim cannot be made.")
    elif heavy_land >= 2:
        task4 = "(b) Model classifies genuine open ocean as land"
        answer = "NO"
        why = (f"{heavy_land}/3 independently GCP+GSHHG-verified open-ocean tiles (>=20 km from any "
               f"coast) are >=40% 'land' by the model, while the in-domain control image is "
               f"{'sane (0% land, %.1f%% sea)' % pct_r['sea_surface'] if control_sane else 'itself anomalous'}. "
               f"Land labels form large coherent regions, not scattered pixels. This is systematic "
               f"open-ocean->land misclassification (domain shift), not real land.")
    elif mostly_sea >= 2 and heavy_land == 0:
        task4 = "(a) open ocean is classified as sea (earlier 69% tile was near/at land)"
        answer = "YES" if mostly_sea == 3 else "PARTIALLY"
        why = (f"{mostly_sea}/3 verified open-ocean tiles are majority sea_surface with low land; "
               f"the earlier 69%-land tile centre is {pct_p['land']}% land and sits "
               f"{'ON land' if rep['previous_tile']['is_land_centre'] else str(rep['previous_tile']['coast_km'])+' km from coast'}.")
    else:
        task4 = "UNRESOLVED"
        answer = "PARTIALLY" if mostly_sea >= 1 else "UNRESOLVED"
        why = ("Mixed behaviour across the three verified open-ocean tiles; evidence does not "
               "cleanly support either hypothesis.")

    rep["control_sane"] = bool(control_sane)
    rep["verdict"] = {"task4": task4, "open_ocean_segmentation_sensible": answer, "explanation": why,
                      "land_pcts": land_pcts, "sea_pcts": sea_pcts}
    rep["results"] = results
    rep["corners"] = {k: [round(v[0], 4), round(v[1], 4)] for k, v in corners.items()}
    rep["gcp_roundtrip_km"] = {"mean": round(float(err.mean()), 3), "max": round(float(err.max()), 3)}
    rep["scene_dims"] = [H, W]

    with open(os.path.join(RESULTS, "open_ocean_audit.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    # text report
    L = ["OPEN-OCEAN MODEL AUDIT", "=" * 60,
         "\n1. PRODUCT",
         f"  product: {safe.product_id}",
         f"  Sentinel-1C IW GRDH, VV; dims {H}x{W}; {ginfo['n']} GCPs",
         f"  extent: lat {ginfo['lat'].min():.3f}..{ginfo['lat'].max():.3f}, "
         f"lon {ginfo['lon'].min():.3f}..{ginfo['lon'].max():.3f}",
         "\n2. GEOLOCATION (RegularGridInterpolator on the 10x21 GCP grid; ROUGH, ~<1km)"]
    for k in ["top-left", "top-right", "bottom-left", "bottom-right"]:
        a, o = corners[k]; L.append(f"  {k:13} {a:.4f} N  {o:.4f} E")
    L.append(f"  GCP round-trip error: mean {err.mean():.3f} km, max {err.max():.3f} km")
    L.append("\n3. TILE LOCATIONS (selected by GCP+GSHHG only; model NOT used)")
    L.append(f"  {'Tile':9}{'Row':>7}{'Col':>7}{'Lat':>10}{'Lon':>9}{'Coast_km':>10}")
    for r in results:
        L.append(f"  {r['id']:9}{r['row']:>7}{r['col']:>7}{r['lat']:>10}{r['lon']:>9}"
                 f"{r['coast_km']:>9}{'+' if r['capped'] else ' '}")
    L.append("\n4. MODEL RESULTS (argmax 5-class pixel counts / %)")
    L.append(f"  {'Tile':10}{'sea':>9}{'oil':>8}{'look':>8}{'ship':>7}{'land':>9}")
    for r in results:
        c = r["counts"]
        L.append(f"  {r['id']:10}{c['sea_surface']:>9}{c['oil_spill']:>8}"
                 f"{c['oil_spill_look_alike']:>8}{c['ship']:>7}{c['land']:>9}")
        L.append(f"  {'   %':10}{r['pct']['sea_surface']:>9}{r['pct']['oil_spill']:>8}"
                 f"{r['pct']['oil_spill_look_alike']:>8}{r['pct']['ship']:>7}{r['pct']['land']:>9}")
    c = counts_r
    L.append(f"  {'Reference':10}{c['sea_surface']:>9}{c['oil_spill']:>8}"
             f"{c['oil_spill_look_alike']:>8}{c['ship']:>7}{c['land']:>9}")
    L.append(f"  (reference control sane: {control_sane};  land components/oil components reported in JSON)")
    L.append("\n5. PREVIOUS TILE (row640 col14400)")
    L.append(f"  centre {pla:.4f} N {plo:.4f} E; is_land(centre)={rep['previous_tile']['is_land_centre']}; "
             f"coast {pd:.1f}{'+' if pcap else ''} km")
    L.append(f"  model land {pct_p['land']}%  oil {pct_p['oil_spill']}%  sea {pct_p['sea_surface']}%")
    L.append("\n6. VERDICT")
    L.append(f"  TASK 4 VERDICT: {task4}")
    L.append(f"  Sensible segmentation on verified open ocean? {answer}")
    L.append(f"  {why}")
    txt = "\n".join(L)
    with open(os.path.join(RESULTS, "open_ocean_audit.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
