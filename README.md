# OILTRACE

Trace a satellite-radar oil slick back to the ship that likely released it.

This repository is the **day-one skeleton**. The whole pipeline runs end to
end today on bundled fake data. The numbers are not meant to be correct yet —
what is real is that every stage exists, hands typed data to the next, and
returns real JSON. Every simplification is marked in the code with a
`# PROTOTYPE:` comment naming what will replace it.

> **Detection is not attribution. Attribution is not proof.**

---

## The pipeline

A satellite radar scene comes in. Five stages run in order:

| Stage | File | What it does | Today's stand-in |
| ----- | ---- | ------------ | ---------------- |
| **SEE** | `stages/see.py` | Find dark slick polygons in the image | Fixed threshold + OpenCV contours (real: U-Net) |
| **FILTER** | `stages/filter.py` | Wind gate — reject slicks when wind is too calm or too rough | Nearest hourly wind sample |
| **REWIND** | `stages/rewind.py` | Drift the slick backward to a release area | 500 particles, wind-only drift (real: OpenDrift with currents) |
| **NAME** | `stages/name.py` | Find AIS vessels inside the release area at the release time | Point-in-polygon + time window |
| **EXPLAIN** | `stages/explain.py` | Score, rank, build the evidence bundle | Weighted time / spatial / course score |

Each stage is a function taking one Pydantic model in and returning another
(`app/models.py`), so any stage can be swapped without touching the others.
The two stages that reach outside are behind interfaces: the detector is a
`Segmenter` (`see.py`) and the AIS lookup a `ShipSource` (`name.py`). Swap
`FileShipSource` for a live feed, or the threshold `Segmenter` for a U-Net,
and nothing else changes.

Nothing is ever silently dropped. A detection that fails a gate is carried
through to the API response as a `Rejection` with a human-readable reason.

---

## What's in the bundled scene

`scripts/make_sample_data.py` writes a self-consistent fixture into
`backend/sample_data/` (committed, so the app runs offline out of the box):

- **`scene.tif`** — a 1000×1000 greyscale GeoTIFF over Danish waters
  (`REGION_BOUNDS`). One dark ~40 px streak running north-east (the "oil"),
  plus two small dark blobs that are **look-alikes** the detector rejects.
- **`wind.json`** — hourly wind for the 24 h before the scene, 5–8 m/s so it
  passes the wind gate.
- **`ships.json`** — six vessels. **One** (`Nordfjord`) passes through the
  backtracked release area on a matching heading — it should rank #1. **One**
  (`Kattegat Star`) has a 40-minute AIS gap — the dark-vessel case — and gets
  flagged. The other four are elsewhere or on wrong headings.
- **`scene_meta.json`** — scene id, acquisition time, bounds.

The culprit vessel is planted by replaying the exact backward-drift model that
`rewind.py` uses, so the pipeline genuinely re-discovers it.

Expected result of a run: `Nordfjord` #1 (~0.96), `Kattegat Star` #2 (flagged),
two blobs rejected by the detector's minimum-area rule.

---

## Requirements

- Python **3.11+** (developed against 3.13; targets 3.11 in Docker)
- Node **18+** for the frontend
- No database, no GPU, no network at runtime

---

## Run the backend

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# Generate the sample data (already committed, but this regenerates it):
python -m scripts.make_sample_data

# Start the API:
uvicorn app.main:app --reload --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/analyze      # full result incl. rejections
curl http://localhost:8000/candidates            # just the ranked list
curl http://localhost:8000/evidence/S1A_OILTRACE_20260910T1000   # evidence bundle
```

Interactive API docs: <http://localhost:8000/docs>

A full run completes in well under 10 seconds (typically ~0.2 s) and is
deterministic — the fixed `RANDOM_SEED` in `config.py` means the same scene
always gives the same answer.

### Run the backend with Docker

```bash
cd backend
docker build -t oiltrace-backend .
docker run -p 8000:8000 oiltrace-backend
```

---

## Run the frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

The page auto-runs `/analyze` on load and draws, on one Leaflet map:

- the scene footprint,
- the detected slick polygon (amber),
- the backtracked release area (teal),
- every vessel track, with the top-ranked one highlighted in red.

Beside the map is the ranked candidate list — click a candidate to expand its
time / spatial / course score breakdown — and a separate panel listing every
rejected detection with its reason.

> **Offline note:** the map deliberately draws vectors on a plain geographic
> canvas with **no external tile basemap**, because the system must run with no
> network at runtime. Wiring in a self-hosted tile source is a later task.

The backend URL defaults to `http://localhost:8000`; override with
`NEXT_PUBLIC_API_BASE` when building.

---

## API

| Method | Path | Returns |
| ------ | ---- | ------- |
| `POST` | `/analyze` | Full result: scene, primary `detected_slick`, `other_detections` (extra polygons that passed the wind gate but aren't the primary slick — surfaced, never dropped), release area, ranked candidates, rejections, all vessel tracks, wind, evidence id, `data_source`, `provenance_note`. Includes `_runtime_seconds`. |
| `GET` | `/candidates` | Just the ranked candidate list from the latest run. |
| `GET` | `/evidence/{id}` | The full evidence bundle: scene id + timestamp, `data_source`, `provenance_note`, `score_model` caveat, config used, random seed, every stage's output, per-candidate score breakdown. |

Every response carries, verbatim:
`Detection is not attribution. Attribution is not proof.`

Missing or corrupt sample files return a clean HTTP 500 with a readable
`detail` message, not a raw traceback.

### Honesty / provenance

This is a skeleton and it says so in the product itself, not just here:

- Every response carries `data_source: "SYNTHETIC_FIXTURE"` and a
  `provenance_note`. The frontend shows a persistent banner: **"Synthetic
  sample data."** The top-ranked vessel is *planted* by inverting the drift
  model, so results demonstrate pipeline mechanics — not a real detection.
- The scoring in `explain.py` is an **illustrative linear heuristic**, marked
  `# PROTOTYPE:`, with a `score_model` caveat in the evidence bundle. Scores
  are reported to two decimals only (`config.SCORE_DECIMALS`) to avoid
  implying precision the model does not have.
- Every stage carries a `# PROTOTYPE:` marker naming its real replacement, and
  every tunable — including the drift jitter and the spatial reference scale —
  lives in `config.py`, not scattered through the stages.

---

## Configuration

All tunables live at the top of `backend/app/config.py`:

```
SLICK_AGE_HOURS   = 12       single fixed timeframe
WIND_MIN_MS       = 3.0      below this the sea is flat, reject
WIND_MAX_MS       = 12.0     above this the slick mixes away
WIND_DRIFT_FACTOR = 0.03     slick moves at 3% of wind speed
N_PARTICLES       = 500
REGION_BOUNDS     = (54.5, 10.0, 57.5, 13.5)   Danish waters
```

plus the detector threshold, scoring weights, and match windows.

---

## Project layout

```
oiltrace/
  backend/
    app/
      main.py            FastAPI app + the three endpoints
      config.py          all tunable settings
      models.py          Pydantic models for every stage boundary
      stages/
        see.py           Segmenter — image -> slick polygons
        filter.py        wind gate
        rewind.py        backward drift -> release area
        name.py          vessels in the release area
        explain.py       score, rank, evidence bundle
    scripts/
      make_sample_data.py
    sample_data/         generated fixture (committed)
    requirements.txt
    Dockerfile
  frontend/              Next.js + Leaflet single page
```

---

## Not in scope for day one

No U-Net, no external APIs, no database, no ocean currents. Those are the named
replacements behind each `# PROTOTYPE:` marker.
