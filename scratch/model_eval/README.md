# Sentinel-1 GRD preprocessing / distribution diagnostic

`preprocess_grd.py` — an **isolated** utility (does not import or modify
`backend/app/`, the model, or production config). It takes a **real** Sentinel-1
IW GRDH product and answers one question with measurements:

> Does scientifically defensible VV preprocessing put the input in the same
> distribution the ResNet50 DeepLabV3+ model was trained on?

## Run

```bash
# accepts an extracted .SAFE directory OR a .SAFE.zip (auto-detected)
python preprocess_grd.py "C:/path/S1C_..._.SAFE.zip"
python preprocess_grd.py "C:/path/S1C_..._.SAFE"
python preprocess_grd.py "<PATH>" --db-min -30 --db-max 0 --window 5 --enl 4.4
python preprocess_grd.py --help
```

CPU-only. The 446 Mpixel scene is never held at float precision: a **decimated
overview** drives global stats and ocean-tile selection; a single **full-res
650×1250 tile** is calibrated/filtered for the model-compatible output.

## Pipeline (ESA S1 product spec + calibration ATBD)

1. **Read VV** (uint16 DN) — measurement selected by `-vv-` filename, verified against available pols.
2. **Radiometric calibration** — `sigma0 = DN² / sigmaNought_LUT²`, the product's own `sigmaNought` LUT (27×670 grid) bilinearly interpolated. `DN==0` → nodata/masked. No invented constants.
3. **Speckle filter** — Lee MMSE (`W = 1 − Cu²/Ci²`, `Cu=1/√ENL`), window 5, ENL 4.4. Local box stats via `uniform_filter` (documented sliding mean).
4. **dB** — `10·log10(sigma0)`, invalid excluded.
5. **dB clip + 8-bit** — linear `[DB_MIN, DB_MAX]→[0,255]`; up to 5 physically-defensible ranges evaluated, never fitted to the reference.
6. **Land mask** — **NOT AVAILABLE offline**: GRD is in radar geometry (CRS=None); a real coastline mask needs GCP geolocation + an external dataset (GSHHG) or terrain correction. A valid-data (DN>0) mask is applied; it is **not** a land mask.
7. **Crop/tile** — full-res tile at the model input size (650×1250, from `results/reference_report.json`).

## What "match" means

Measured, not vague. Two references:
- **Single reference image** `img_0814.jpg` (8-bit) — tolerances on mean/std/p05/p50/p95.
- **Dataset-wide anchor** from `image_stats.json` (mean 132 / std 50, 8-bit) — the true training distribution; the single image is a bright outlier of it.

## Outputs

- `results/preprocessing_report.json`, `results/preprocessing_report.txt`
- `out/reference.png`, `out/preprocessed.png`, `out/*_histogram.png`, `out/distribution_comparison.png`, `out/quicklook.png`

`out/`, `weights/`, `*.SAFE`, `*.zip`, `source/` are git-ignored (see `.gitignore`).

## Honest findings (on the S1C 20260914 Danish-waters product)

- Calibration + dB are correct; ocean VV lands at a physically sensible **−25…−8 dB**.
- With `dB[-30,0]`, a representative (scene-median) tile gives 8-bit **mean 126 / std 48**, which **matches the dataset anchor** (132/50) but **not** the bright single reference image (mean 205, 0/5).
- Verdict: **PARTIALLY** — mean/std can reach the training distribution, but (a) the reference image is a bright outlier, (b) the dataset's exact dB→8-bit stretch is undocumented, (c) **no land mask** offline means clean ocean cannot be isolated (the chosen tile is ~69% land per the model). A clean YES/NO is not supportable.
- Model implication: the real preprocessed SAR tile yields **non-zero** oil pixels vs **0** on the synthetic bundled scenes — evidence that input domain/preprocessing was a **material** contributor to the original zero-detection. Correctness is **UNVERIFIED** (no ground truth); land contamination inflates class counts.
